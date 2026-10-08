import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from avbcm import AVBCM
from config import add_args
from data_loader import LOGGER, load_dataset
from model import GNNModel
from train import evaluate, train_one_epoch
from utils import (check_label_distribution, compute_class_weights,
                   compute_logit_adjustment, save_results_to_excel_and_plots,
                   set_seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = add_args().parse_args()

    print("Using device:", DEVICE)

    # ===== Seed =====
    set_seed(args.seed)

    # ===== Load dataset =====
    dataset = load_dataset(args.data_path)
    if dataset is None:
        LOGGER.error("Failed to load dataset.")
        return

    train_data, temp_data = train_test_split(dataset, train_size=0.8, random_state=42)
    val_data, test_data = train_test_split(temp_data, train_size=0.5, random_state=42)
    class_distribution = check_label_distribution(train_data, val_data, test_data)
    class_prior = torch.tensor(
        [class_distribution['train_counts'].get(i, 0) / len(train_data) for i in range(2)],
        device=args.device,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # ===== Build model =====
    model = GNNModel(
        gnn_type=args.model_name,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=2,
        num_relations=args.num_relations,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=2,
    ).to(DEVICE)

    # ===== Run tag -> output directory =====
    init_methods = []
    if args.skip_init:
        print(" Skipping initialization step")
        if args.loss_function != "ce":
            init_methods.append(args.loss_function)
    if args.use_avbc:
        avbc_tag = "AVBCM"
        if args.fixed_lambda is not None:
            avbc_tag += f"_fx{args.fixed_lambda:g}"
        init_methods.append(avbc_tag)
    init_name = "_".join(init_methods) if init_methods else "NoInit"

    dataset_name = os.path.basename(args.data_path.rstrip('/\\'))
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, dataset_name, args.model_name, init_name)
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f"{args.model_name}_best.pth")

    # ===== Optimizer =====
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ===== AVBCM calibrator =====
    if args.use_avbc:
        avbc_db = AVBCM(
            beta=0.9,
            use_ema=True,
            use_boundary=True,
            use_prior_gate=False,
            use_prior_bias=True,
            gate_strength=0.2,
            fixed_lambda=args.fixed_lambda,
        )
    else:
        avbc_db = None

    # ===== Per-epoch loss extras (computed once) =====
    logit_adj = compute_logit_adjustment(train_loader, DEVICE)
    class_weights = compute_class_weights(train_loader, DEVICE)
    extra = {"weights": class_weights}
    if logit_adj is not None:
        extra["logit_adj"] = logit_adj
    if args.loss_function in {"ldam", "cb_focal"}:
        extra["cls_num_list"] = [
            class_distribution['train_counts'].get(i, 0) for i in range(2)
        ]

    # ===== Training loop =====
    train_loss_history = []
    val_loss_history = []
    grad_norm_history = []
    lambda_history = []
    boundary_gap_history = []
    gap_raw_history = []
    best_val_acc = -1.0
    best_val_f1 = -1.0
    counter = 0
    patience = args.patience
    start_time = time.time()

    print("\n" + "=" * 60)
    print("Start Training")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Patience: {patience}")
    print("=" * 60)

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training Progress", unit="epoch")
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch:03d}/{args.epochs}")

        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            DEVICE,
            class_prior=class_prior,
            loss_type=args.loss_function,
            extra=extra,
            avbc_db=avbc_db,
        )
        train_loss_history.append(stats['loss'])
        grad_norm_history.append(stats['grad_norm'])
        lambda_history.append(stats['lambda'])
        boundary_gap_history.append(stats['boundary_gap'])
        gap_raw_history.append(stats.get('boundary_gap_raw', 0.0))
        torch.cuda.empty_cache()

        # Validation
        val_acc, val_prec, val_rec, val_f1, val_fpr, avg_loss = evaluate(model, val_loader, DEVICE)
        val_loss_history.append(avg_loss)
        torch.cuda.empty_cache()

        epoch_pbar.set_postfix({
            'loss': f'{stats["loss"]:.4f}',
            'acc': f'{val_acc:.4f}',
            'f1': f'{val_f1:.4f}',
            'fpr': f'{val_fpr:.4f}',
            'best_f1': f'{best_val_f1:.4f}',
        })

        tqdm.write(
            f"\n[Epoch {epoch:03d}/{args.epochs}] "
            f"Loss: {stats['loss']:.4f} | "
            f"Grad Norm: {stats['grad_norm']:.4f} | "
        )
        tqdm.write(
            f"  Val: Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, "
            f"F1={val_f1:.4f}, FPR={val_fpr:.4f}"
        )
        tqdm.write(
            f"  Gap: raw={float(stats.get('boundary_gap_raw', 0.0)):.6f}, "
            f"ema={float(stats['boundary_gap']):.6f} | "
            f"lambda_t={float(stats['lambda']):.6f}"
        )

        # Best-model checkpoint
        if val_f1 > best_val_f1:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'train_loss': stats['loss'],
            }, model_path)
            counter = 0
            tqdm.write(f" Best model updated (F1: {val_f1:.4f})")
        else:
            counter += 1
            tqdm.write(f" No improvement ({counter}/{patience})")

        # Early stopping
        if counter >= patience:
            tqdm.write(f"\n Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time

    # ===== Load best checkpoint & test =====
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded best model from epoch {best_epoch} (F1: {checkpoint['val_f1']:.4f})")

    test_acc, test_prec, test_rec, test_f1, test_fpr, avg_loss = evaluate(model, test_loader, DEVICE)

    print(f"\n Final Test Results:")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  FPR:       {test_fpr:.4f}")

    # ===== Save results =====
    results = {
        'test_acc': f"{test_acc:.4f}",
        'test_prec': f"{test_prec:.4f}",
        'test_rec': f"{test_rec:.4f}",
        'test_f1': f"{test_f1:.4f}",
        'test_fpr': f"{test_fpr:.4f}",
        'best_val_acc': f"{best_val_acc:.4f}",
        'best_val_f1': f"{best_val_f1:.4f}",
        'best_epoch': best_epoch,
        'train_losses': train_loss_history,
        'val_losses': val_loss_history,
        'grad_norm_history': grad_norm_history,
        'lambda_history': lambda_history,
        'boundary_gap_history': boundary_gap_history,
        'boundary_gap_raw_history': gap_raw_history,
        'training_time': f"{total_time:.2f}",
        'param_shift': "0.0000",
    }

    # Per-epoch history dump (data source for the mechanism figure)
    def _arr(h):
        return np.array([np.nan if v is None else float(v) for v in h], dtype=np.float64)

    history_npz = os.path.join(results_dir, 'history.npz')
    np.savez(
        history_npz,
        train_losses=_arr(train_loss_history),
        val_losses=_arr(val_loss_history),
        grad_norm=_arr(grad_norm_history),
        lambda_t=_arr(lambda_history),
        boundary_gap_ema=_arr(boundary_gap_history),
        boundary_gap_raw=_arr(gap_raw_history),
    )
    print(f"  History saved to {history_npz}")

    save_results_to_excel_and_plots(
        results=results,
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        args=args,
        save_dir=results_dir,
    )
    return results


if __name__ == "__main__":
    main()
