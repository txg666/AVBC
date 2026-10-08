import re
import torch
import gzip
import pickle
import numpy as np
from torch_geometric.data import Data
import logging, traceback, os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("DataLoader")


def load_init_params(model, path):
    init_state = torch.load(path)

    for name, param in model.named_parameters():
        param.data.copy_(init_state[name])

    print("[✓] Init params loaded")

def nx_to_pyg_data(G, label=0):

    node_list = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    x = torch.zeros((num_nodes, 128), dtype=torch.float32)
    missing = []

    for i, n in enumerate(node_list):
        emb = G.nodes[n].get('embedding')
        if isinstance(emb, (list, tuple, np.ndarray)) and len(emb) == 128:
            x[i] = torch.tensor(emb, dtype=torch.float32)
        else:
            missing.append(i)

    if missing:
        mean_emb = x.sum(dim=0) / max(1, (num_nodes - len(missing)))
        for i in missing:
            x[i] = mean_emb

    edge_index = [[], []]
    edge_type_list = []
    edge_weight_list = []

    edge_type_map = {'DDG': 0, 'AST': 1, 'CDG': 2, 'CFG': 3}

    for u, v, data in G.edges(data=True):
        if u not in node_idx or v not in node_idx:
            continue

        edge_index[0].append(node_idx[u])
        edge_index[1].append(node_idx[v])

        rel = data.get('type', 'DDG')
        edge_type_list.append(edge_type_map.get(rel, 0))

        w = data.get('weight_norm', data.get('weight', 1.0))
        edge_weight_list.append(float(w))

    if edge_index[0]:
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)

    y = torch.tensor([label], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        y=y
    )

def load_compressed_graph(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

def process_single_file(file_path):
    """处理单个文件，返回 (pyg_data_list, error_count)"""
    try:
        file_path = str(file_path)
        
        # 检查文件
        if not os.path.exists(file_path):
            LOGGER.error(f"File not found: {file_path}")
            return [], 1
        
        if os.path.getsize(file_path) == 0:
            LOGGER.error(f"Empty file: {file_path}")
            return [], 1
        
        # 从文件名中提取标签（最后一个字符）
        filename = os.path.basename(file_path)
        # 支持格式：FFmpeg-00a1e1337f22376909338a5319a378b2e2afdde8-1
        # 提取最后一个 - 后面的数字
        label = int(re.search(r'\d+', filename.split('-')[-1]).group())      
        # 加载数据
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取 NetworkX 图并转换为 PyG Data
        graphs = []
        
        if isinstance(data, list):
            for item in data:
                if hasattr(item, 'number_of_nodes'):  # NetworkX graph
                    graphs.append(item)
        elif hasattr(data, 'number_of_nodes'):  # NetworkX graph
            graphs.append(data)
        else:
            LOGGER.warning(f"Unexpected type {type(data)} in {file_path}")
            return [], 1
        
        if not graphs:
            LOGGER.warning(f"No NetworkX graphs found in {file_path}")
            return [], 1
        
        # 转换为 PyG Data 对象，传入标签
        pyg_graphs = []
        for G in graphs:
            pyg_data = nx_to_pyg_data(G, label=label)  # 传入标签
            pyg_graphs.append(pyg_data)
        
        return pyg_graphs, 0
            
    except gzip.BadGzipFile as e:
        LOGGER.error(f"Bad gzip file {file_path}: {e}")
        return [], 1
    except pickle.UnpicklingError as e:
        LOGGER.error(f"Pickle error in {file_path}: {e}")
        return [], 1
    except Exception as e:
        LOGGER.error(f"Error in {file_path}: {e}")
        LOGGER.debug(traceback.format_exc())
        return [], 1

def load_npz_as_pyg_data_dir(npz_dir):
    """加载目录中的所有 .npz 文件"""
    all_data = []
    
    # 获取所有 .npz 文件
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    print(f"Found {len(npz_files)} .npz files")
    
    for npz_file in tqdm(npz_files, desc="Loading npz files"):
        npz_path = os.path.join(npz_dir, npz_file)
        data_list = load_npz_as_pyg_data(npz_path)
        all_data.extend(data_list)
    
    print(f"Total loaded: {len(all_data)} graphs")
    return all_data

# 在 load_npz_as_pyg_data 函数中
def load_npz_as_pyg_data(npz_path):
    """加载 .npz 文件并还原为 PyG Data 对象（自动分离 edge_type 和 edge_weight）"""
    loaded = np.load(npz_path, allow_pickle=True)
    data_list = []
    
    for data_dict in loaded['data_list']:
        data = Data()
        
        # 处理字典数据
        if isinstance(data_dict, np.ndarray):
            dict_data = data_dict.item() if data_dict.ndim == 0 else data_dict
        else:
            dict_data = data_dict
        
        for key, value in dict_data.items():
            if value is None:
                continue
            
            if key == 'edge_index':
                setattr(data, key, value.clone().detach().long())
            elif key == 'y':
                if value not in [0, 1]:
                    raise ValueError(f"Invalid label {value} in file {npz_path}")
                setattr(data, key, value.clone().detach().long())

            elif key == 'edge_attr':
                # 将 edge_attr 分离为 edge_type 和 edge_weight
                edge_attr = value.clone().detach().float()
                if edge_attr.size(1) >= 2:
                    data.edge_type = edge_attr[:, 0].long()   # 第一列：边类型
                    data.edge_weight = edge_attr[:, 1]        # 第二列：边权重
                else:
                    data.edge_type = edge_attr[:, 0].long()
                    data.edge_weight = None
                # 保留原始 edge_attr（可选）
                data.edge_attr = edge_attr
            elif key == 'x':
                setattr(data, key, value.clone().detach().float())
            elif key == 'edge_type':
                # 如果已经有 edge_type，确保是 long 类型
                setattr(data, key, value.clone().detach().long())
            elif key == 'edge_weight':
                setattr(data, key, value.clone().detach().float())
            else:
                # 其他字段
                setattr(data, key, value.clone().detach().float())

        # 验证必要字段
        assert hasattr(data, 'x'), "Missing x field"
        assert hasattr(data, 'edge_index'), "Missing edge_index field"
        assert hasattr(data, 'y'), "Missing y field"
        
        data_list.append(data)
    
    return data_list

def check_label_distribution(train_data, val_data, test_data):
    """检查标签分布"""
    import numpy as np
    
    train_labels = [d.y.item() for d in train_data]
    val_labels = [d.y.item() for d in val_data]
    test_labels = [d.y.item() for d in test_data]
    
    print("\n" + "="*50)
    print("Label Distribution")
    print("="*50)
    
    # 获取所有唯一的标签值
    train_unique = set(train_labels)
    val_unique = set(val_labels)
    test_unique = set(test_labels)
    all_unique = train_unique | val_unique | test_unique
    
    # print(f"\nUnique labels found:")
    # print(f"  Train unique labels: {sorted(train_unique)} (count: {len(train_unique)})")
    # print(f"  Val unique labels:   {sorted(val_unique)} (count: {len(val_unique)})")
    # print(f"  Test unique labels:  {sorted(test_unique)} (count: {len(test_unique)})")
    # print(f"  All unique labels:   {sorted(all_unique)} (count: {len(all_unique)})")
    
    # 检查是否有异常标签
    invalid_labels = [l for l in all_unique if l not in [0, 1]]
    if invalid_labels:
        print(f"\n ERROR: Invalid labels found: {invalid_labels}")
        
        # 找出异常样本的具体位置
        print("\n  Samples with invalid labels:")
        for i, d in enumerate(train_data):
            if d.y.item() not in [0, 1]:
                print(f"    Train index {i}: label={d.y.item()}")
        for i, d in enumerate(val_data):
            if d.y.item() not in [0, 1]:
                print(f"    Val index {i}: label={d.y.item()}")
        for i, d in enumerate(test_data):
            if d.y.item() not in [0, 1]:
                print(f"    Test index {i}: label={d.y.item()}")
    else:
        print(f"\n All labels are valid (0 or 1)")
    
    # 统计每个类别的数量
    print(f"\nLabel counts:")
    for label in sorted(all_unique):
        train_count = train_labels.count(label)
        val_count = val_labels.count(label)
        test_count = test_labels.count(label)
        total = train_count + val_count + test_count
        
        print(f"  Label {label}: Train={train_count}, Val={val_count}, Test={test_count}, Total={total}")
    
    # 计算百分比（只对0和1）
    if 0 in all_unique and 1 in all_unique:
        train_0_pct = train_labels.count(0)/len(train_labels)*100
        train_1_pct = train_labels.count(1)/len(train_labels)*100
        val_0_pct = val_labels.count(0)/len(val_labels)*100
        val_1_pct = val_labels.count(1)/len(val_labels)*100
        test_0_pct = test_labels.count(0)/len(test_labels)*100
        test_1_pct = test_labels.count(1)/len(test_labels)*100
        
        print(f"\nPercentage distribution:")
        print(f"  Train: 0={train_0_pct:.1f}%, 1={train_1_pct:.1f}%")
        print(f"  Val:   0={val_0_pct:.1f}%, 1={val_1_pct:.1f}%")
        print(f"  Test:  0={test_0_pct:.1f}%, 1={test_1_pct:.1f}%")
    
    # 检查类别不平衡
    if 0 in all_unique and 1 in all_unique:
        train_ratio = max(train_labels.count(0), train_labels.count(1)) / min(train_labels.count(0), train_labels.count(1))
        if train_ratio > 5:
            print(f"\n WARNING: Class imbalance in training set (ratio={train_ratio:.1f}:1)")
    
    # 检查是否只有一个类别
    if len(train_unique) == 1:
        print("\n WARNING: Training set has only one class!")
    if len(val_unique) == 1:
        print(" WARNING: Validation set has only one class!")
    if len(test_unique) == 1:
        print(" WARNING: Test set has only one class!")
    
    return {
        'train_unique': train_unique,
        'val_unique': val_unique,
        'test_unique': test_unique,
        'train_counts': {l: train_labels.count(l) for l in train_unique},
        'val_counts': {l: val_labels.count(l) for l in val_unique},
        'test_counts': {l: test_labels.count(l) for l in test_unique}
    }

def save_results_to_excel_and_plots(results, model, test_loader, device, args, save_dir="results"):
    """保存测试结果到Excel（仅指标和超参数）并绘制图表"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== 1. 收集预测结果（用于绘图） ==========
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ========== 2. 计算混淆矩阵和ROC ==========
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(all_labels, all_probs)

    # ========== 3. 保存到 Excel（测试指标统计表） ==========
    excel_path = os.path.join(save_dir, f"results.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Sheet 1: 测试指标
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR', 'AUC-ROC', 'PR-AUC'],
            'Value': [results['test_acc'], results['test_prec'],
                      results['test_rec'], results['test_f1'], results['test_fpr'], roc_auc, pr_auc]
        })
        metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
        
        # Sheet 2: 超参数配置
        hyperparams_df = pd.DataFrame({
            'Parameter': [
                'Batch Size', 'Learning Rate', 'Weight Decay', 'Dropout',
                'Hidden Dim', 'Num Layers', 'Epochs', 'Patience',
                'Model Name', 'Dataset Size', 'Fixed Lambda'
            ],
            'Value': [
                args.batch_size,
                args.lr,
                args.weight_decay if hasattr(args, 'weight_decay') else 'N/A',
                args.dropout,
                args.hidden_dim,
                args.num_layers,
                args.epochs,
                args.patience if hasattr(args, 'patience') else 'N/A',
                args.model_name if hasattr(args, 'model_name') else 'GCN',
                results.get('dataset_size', 'N/A'),
                args.fixed_lambda if getattr(args, 'fixed_lambda', None) is not None else 'adaptive'
            ]
        })
        hyperparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
        
        # Sheet 3: 训练信息
        training_df = pd.DataFrame({
            'Metric': ['Best Val Accuracy', 'Best Val F1', 'Best Epoch', 'Training Time (s)'],
            'Value': [results['best_val_acc'], results['best_val_f1'], 
                      results['best_epoch'], results['training_time']]
        })
        training_df.to_excel(writer, sheet_name='Training Info', index=False)
    # Sheet 4: 测试信息
        test_df = pd.DataFrame({
            'Metric': ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score', 'Test FPR', 'Parameter Shift'],
            'Value': [results['test_acc'], results['test_prec'], 
                      results['test_rec'], results['test_f1'], results['test_fpr'], results['param_shift']]
        })
        test_df.to_excel(writer, sheet_name='Test Info', index=False)

    print(f" Results saved to {excel_path}")
    
    # ========== 4. 绘制混淆矩阵图 ==========
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean (0)', 'Vulnerable (1)'],
                yticklabels=['Clean (0)', 'Vulnerable (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\nAccuracy: {results["test_acc"]} | F1: {results["test_f1"]}')
    plt.tight_layout()
    
    cm_path = os.path.join(save_dir, f"confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Confusion matrix saved to {cm_path}")
    
    # ========== 5. 绘制 ROC 曲线 ==========
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve\nAUC = {roc_auc:.4f}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(save_dir, f"roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ROC curve saved to {roc_path}")
    
    # ========== 6. 绘制 Loss 曲线（如果有历史） ==========
    if 'train_losses' in results and results['train_losses']:
        plt.figure(figsize=(10, 6))
        
        plt.plot(results['train_losses'], label='Train Loss', marker='o', markersize=3, linewidth=1.5)
        if 'val_losses' in results and results['val_losses']:
            plt.plot(results['val_losses'], label='Validation Loss', marker='s', markersize=3, linewidth=1.5)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_path = os.path.join(save_dir, f"loss_curve.png")
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Loss curve saved to {loss_path}")
    # ========== 绘制 Gradient Norm 曲线（如果有历史） ==========
    if 'grad_norm_history' in results and results['grad_norm_history']:
        plt.figure(figsize=(10, 6))
        
        plt.plot(results['grad_norm_history'], label='Gradient Norm', marker='o', markersize=3, linewidth=1.5, color='green')
        
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        grad_norm_path = os.path.join(save_dir, f"grad_norm_curve.png")
        plt.savefig(grad_norm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Gradient norm curve saved to {grad_norm_path}")

# ========== 绘制 boundary_gap 曲线（如果有历史） ==========
    if 'boundary_gap_history' not in results:
            print("No boundary_gap recorded.")
            return
    gaps_all = results['boundary_gap_history']
    raw_all = results.get('boundary_gap_raw_history', [])
    # remove invalid values (keep raw aligned)
    gaps = []
    raw_gaps = []
    for i, g in enumerate(gaps_all):
        if g is not None:
            gaps.append(g)
            if i < len(raw_all):
                raw_gaps.append(raw_all[i])

    epochs = np.arange(
        1,
        len(gaps)+1
    )

    plt.figure(
        figsize=(6,4)
    )

    if len(raw_gaps) == len(gaps) and len(gaps) > 0:
        plt.plot(
            epochs,
            raw_gaps,
            linewidth=1.2,
            alpha=0.7,
            label=r'Raw $g_t$'
        )

    plt.plot(
        epochs,
        gaps,
        linewidth=2,
        label=r'EMA $\bar{g}_t$'
    )
    plt.legend()

    plt.xlabel(
        "Epoch",
        fontsize=12
    )
    plt.ylabel(
        "Boundary Gap",
        fontsize=12
    )
    plt.title(
        "Evolution of Vulnerability Boundary Gap",
        fontsize=13
    )
    plt.grid(
        linestyle="--",
        alpha=0.4
    )
    plt.tight_layout()

    boundary_gap_path = os.path.join(save_dir, f"boundary_gap_curve.png")
    plt.savefig(boundary_gap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Boundary gap curve saved to {boundary_gap_path}")

# ========== 绘制 Lambda curve 曲线（如果有历史） ==========

    if 'lambda_history' not in results:
        print("No lambda recorded.")
        return
    lambdas = results['lambda_history']
    lambdas = [
        l for l in lambdas
        if l is not None
    ]
    epochs = np.arange(
        1,
        len(lambdas)+1
    )

    plt.figure(
        figsize=(6,4)
    )

    plt.plot(
        epochs,
        lambdas,
        linewidth=2
    )
    plt.xlabel(
        "Epoch",
        fontsize=12
    )
    plt.ylabel(
        r"Adaptive Calibration Strength $\lambda_t$",
        fontsize=12
    )

    plt.title(
        "Evolution of Adaptive Calibration Strength",
        fontsize=13
    )
    plt.ylim(
        0,
        1
    )
    plt.grid(
        linestyle="--",
        alpha=0.4
    )
    plt.tight_layout()
    lambda_path = os.path.join(save_dir, f"lambda_curve.png")
    plt.savefig(lambda_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Lambda curve saved to {lambda_path}")

# ========== 绘制 avbc_dynamic 曲线（如果有历史） ==========

    gaps = results['boundary_gap_history']
    lambdas = results['lambda_history']
    length = min(
        len(gaps),
        len(lambdas)
    )

    gaps = gaps[:length]
    lambdas = lambdas[:length]

    epochs=np.arange(
        1,
        length+1
    )
    fig, ax1 = plt.subplots(
        figsize=(6,4)
    )

    # boundary gap
    ax1.plot(
        epochs,
        gaps,
        linewidth=2,
        label="Boundary Gap"
    )
    ax1.set_xlabel(
        "Epoch"
    )
    ax1.set_ylabel(
        "Boundary Gap"
    )

    # lambda
    ax2=ax1.twinx()
    ax2.plot(
        epochs,
        lambdas,
        linewidth=2,
        linestyle="--",
        label="Lambda"
    )
    ax2.set_ylabel(
        r"$\lambda_t$"
    )

    plt.title(
        "Adaptive Boundary Calibration Dynamics"
    )

    ax1.grid(
        linestyle="--",
        alpha=0.3
    )
    plt.tight_layout()
    avbc_dynamic = os.path.join(save_dir, f"avbc_dynamic.png")
    plt.savefig(avbc_dynamic, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Adaptive Boundary Calibration dynamics saved to {avbc_dynamic}")

    # ========== 7. 打印摘要 ==========
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Test Accuracy:   {results['test_acc']} ")
    print(f"Test Precision:  {results['test_prec']}")
    print(f"Test Recall:     {results['test_rec']}")
    print(f"Test F1-Score:   {results['test_f1']}")
    print(f"Test FPR:        {results['test_fpr']}")
    print(f"AUC-ROC:         {roc_auc:.4f}")
    print(f"\nBest Val Acc:    {results['best_val_acc']}")
    print(f"Best Val F1:     {results['best_val_f1']}")
    print(f"Best Epoch:      {results['best_epoch']}")
    print(f"Training Time:   {results['training_time']} seconds ({float(results['training_time'])/60:.2f} minutes)")
    print("="*60)
    
    return {
        'excel_path': excel_path,
        'cm_path': cm_path,
        'roc_path': roc_path,
        'loss_path': loss_path if 'train_losses' in results else None,
        'auc': roc_auc
    }

def set_seed(seed):

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证完全可复现（可能稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to {seed}")

# 计算参数偏移量
def compute_param_shift(model, init_state):
    shift = 0
    for name, p in model.named_parameters():
        if name in init_state:
            shift += (p - init_state[name]).norm().item()
    return shift

    #===============================处理不平衡数据========================================
# 计算类别权重（用于处理不平衡数据）
def compute_class_weights(loader, device):
        # 收集所有标签
    y_train = []
    for data in loader:
        y = data.y
        y_train.extend(y.cpu().numpy().tolist())
    y_train = np.array(y_train)
    
    # 使用 sklearn 官方函数
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    return torch.tensor(class_weights, dtype=torch.float).to(device)

import torch.nn.functional as F

# 定义 Focal Loss（处理类别不平衡）
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        
        loss = ((1 - pt) ** self.gamma) * ce
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        return loss.mean()
    
# 计算 logit 调整（用于处理类别不平衡）    
def compute_logit_adjustment(loader, device):
    pos, neg = 0, 0
    
    for data in loader:
        y = data.y
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    
    total = pos + neg
    p_pos = pos / total
    p_neg = neg / total

    adjustment = torch.log(torch.tensor([p_neg, p_pos], device=device) + 1e-12)
    return adjustment

import torch
import torch.nn as nn
class CBFocalLoss(nn.Module):

    def __init__(
        self,
        cls_num_list,
        beta=0.9999,
        gamma=2.0
    ):
        super().__init__()

        cls_num_list = torch.as_tensor(
            cls_num_list,
            dtype=torch.float32
        )

        # Effective number
        effective_num = (
            1.0 - torch.pow(
                beta,
                cls_num_list
            )
        )

        # Class-balanced weights
        weights = (
            1.0 - beta
        ) / effective_num

        # Same normalization convention
        weights = (
            weights
            / weights.sum()
            * len(cls_num_list)
        )

        self.register_buffer(
            "weights",
            weights
        )

        self.gamma = gamma

    def forward(self, x, target):
        weights = self.weights.to(x.device)

        # Standard CE for every sample
        ce = F.cross_entropy(
            x,
            target,
            weight=weights,
            reduction="none"
        )

        # p_t
        pt = torch.exp(-ce)

        # Focal modulation
        loss = (
            (1.0 - pt) ** self.gamma
            * ce
        )

        return loss.mean()

class LDAMLoss(nn.Module):
    """
    LDAM Loss
    Based on:
        Cao et al., Learning Imbalanced Datasets
        with Label-Distribution-Aware Margin Loss,
        NeurIPS 2019.

    Equivalent to the official LDAM implementation.

    Args:
        cls_num_list:
            Number of training samples for each class.

        max_m:
            Maximum LDAM margin. Official default = 0.5.

        weight:
            Optional class weights.

        s:
            Logit scaling factor. Official default = 30.
    """

    def __init__(
        self,
        cls_num_list,
        max_m=0.5,
        weight=None,
        s=30
    ):
        super().__init__()

        cls_num_list = torch.as_tensor(
            cls_num_list,
            dtype=torch.float32
        )

        # m_c = 1 / n_c^(1/4)
        m_list = 1.0 / torch.sqrt(
            torch.sqrt(cls_num_list)
        )

        # Normalize maximum margin to max_m
        m_list = (
            m_list
            * (max_m / m_list.max())
        )

        self.register_buffer(
            "m_list",
            m_list
        )

        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):

        target = target.long()
        m_list = self.m_list.to(x.device)

        # ----------------------------------------
        # Create target-class mask
        # ----------------------------------------

        index = torch.zeros_like(
            x,
            dtype=torch.bool
        )

        index.scatter_(
            1,
            target.view(-1, 1),
            True
        )

        # ----------------------------------------
        # Get margin for each sample
        # ----------------------------------------

        batch_m = m_list[target]
        batch_m = batch_m.view(-1, 1)

        # ----------------------------------------
        # Subtract margin from target logit
        # ----------------------------------------

        x_m = x - batch_m

        output = torch.where(
            index,
            x_m,
            x
        )

        # ----------------------------------------
        # Official LDAM scaling
        # ----------------------------------------

        return F.cross_entropy(
            self.s * output,
            target,
            weight=self.weight
        )    