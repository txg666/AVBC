import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch import nn
from tqdm import tqdm

from utils import CBFocalLoss, FocalLoss, LDAMLoss


def _infer_cls_num_list(loader):
    """Fallback: count class frequencies from the loader's dataset."""
    try:
        dataset = loader.dataset
        if hasattr(dataset, "y"):
            labels = dataset.y.cpu().numpy().tolist()
            if isinstance(labels, int):
                labels = [labels]
            counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
            return counts.tolist()
    except Exception:
        pass

    try:
        counts = np.zeros(2, dtype=int)
        for batch in loader:
            y = batch.y.detach().cpu().numpy().astype(int)
            for label in y:
                if 0 <= int(label) < 2:
                    counts[int(label)] += 1
        return counts.tolist()
    except Exception:
        return None


def train_one_epoch(
    model, loader, optimizer, device,
    loss_type="ce",
    extra=None,
    class_prior=None,
    avbc_db=None,
):
    model.train()

    total_loss = 0
    total_grad_norm = 0
    total_lambda = 0
    total_boundary_gap = 0
    total_boundary_gap_raw = 0

    loss_names = {
        "ce": "Cross Entropy",
        "weighted_ce": "Weighted Cross Entropy",
        "focal": "Focal Loss",
        "logit_adj": "Logit Adjustment",
        "ldam": "LDAM Loss",
        "cb_focal": "Class-Balanced Focal Loss",
    }
    loss_display_name = loss_names.get(loss_type, loss_type)

    pbar = tqdm(loader, desc=f"Training [{loss_display_name}]", leave=False)

    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)

        # AVBCM calibration: logits + lambda_t * log(prior)
        if avbc_db is not None:
            out, lambda_t, gap = avbc_db.calibrate(
                out, data.y, class_prior=class_prior
            )
            if lambda_t is not None:
                total_lambda += lambda_t.item() if hasattr(lambda_t, 'item') else lambda_t
            if gap is not None:
                total_boundary_gap += gap.item() if hasattr(gap, 'item') else gap
            raw_gap = getattr(avbc_db, 'last_raw_gap', None)
            if raw_gap is not None:
                total_boundary_gap_raw += raw_gap.item() if hasattr(raw_gap, 'item') else raw_gap

        # ===== Loss =====
        if loss_type == "ce":
            loss = F.cross_entropy(out, data.y)

        elif loss_type == "weighted_ce":
            if extra is None or "weights" not in extra:
                raise ValueError("weighted_ce requires 'weights' in extra")
            loss = F.cross_entropy(out, data.y, weight=extra["weights"])

        elif loss_type == "focal":
            loss = FocalLoss()(out, data.y)

        elif loss_type == "ldam":
            extra_dict = {} if extra is None else extra
            cls_num_list = extra_dict.get("cls_num_list")
            if cls_num_list is None:
                cls_num_list = _infer_cls_num_list(loader)
            if cls_num_list is None:
                raise ValueError("ldam requires 'cls_num_list' in extra or a labeled dataset on the loader")
            weight = extra_dict.get("weights")
            criterion = LDAMLoss(
                cls_num_list=cls_num_list,
                weight=weight.to(out.device) if weight is not None else None,
            ).to(out.device)
            loss = criterion(out, data.y)

        elif loss_type == "cb_focal":
            extra_dict = {} if extra is None else extra
            cls_num_list = extra_dict.get("cls_num_list")
            if cls_num_list is None:
                cls_num_list = _infer_cls_num_list(loader)
            if cls_num_list is None:
                raise ValueError("cb_focal requires 'cls_num_list' in extra or a labeled dataset on the loader")
            criterion = CBFocalLoss(cls_num_list=cls_num_list).to(out.device)
            loss = criterion(out, data.y)

        elif loss_type == "logit_adj":
            if extra is None or "logit_adj" not in extra:
                raise ValueError("logit_adj requires 'logit_adj' in extra")
            out = out + extra["logit_adj"]
            loss = F.cross_entropy(out, data.y)

        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        loss.backward()
        optimizer.step()

        # ===== Statistics =====
        grad_norm = sum(
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        )
        total_grad_norm += grad_norm
        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = len(loader) if len(loader) > 0 else 1
    return {
        "loss": total_loss / n,
        "grad_norm": total_grad_norm / n,
        "lambda": total_lambda / n,
        "boundary_gap": total_boundary_gap / n,
        "boundary_gap_raw": total_boundary_gap_raw / n,
        "loss_type": loss_display_name,
    }


def evaluate(model, loader, device, return_details=False):
    """Evaluate accuracy, precision, recall, F1, FPR and loss."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Evaluating", leave=False, unit="batch")
    with torch.no_grad():
        for data in pbar:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)

            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            total_loss += loss.item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            pbar.set_postfix({
                'acc': f'{correct / total:.4f}',
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'samples': total,
            })

    accuracy = correct / total
    avg_loss = total_loss / len(loader)

    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if return_details:
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss,
            'fpr': fpr,
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'predictions': all_preds,
            'labels': all_labels,
        }

    return accuracy, precision, recall, f1, fpr, avg_loss
