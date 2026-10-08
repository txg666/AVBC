# Adaptive Boundary Calibration Mechanism (AVBCM)

GNN-based source-code vulnerability detection under class imbalance, with the
**Adaptive Boundary Calibration Mechanism (AVBCM)**: a training-time module that
reads a boundary-state signal (gap between predicted class mass and the true
prior) and translates it into a calibration strength `λ_t`, which rescales the
log-prior bias added to the logits:

```
logits ← logits + λ_t · log(prior)
λ_t = σ(−ḡ_t) · (1 + 1/(1 + |ḡ_t|))       (adaptive)
λ_t = const                                 (fixed-λ ablation, e.g. 0.5 / 1.0)
```

`ḡ_t` is the EMA-smoothed boundary gap (`g_t`), updated every batch
(`avbcm.py`). All variants share the same boundary-state trajectory and
differ only in the state → strength mapping.

## Layout

| File | Role |
|---|---|
| `main.py` | training / evaluation entry point |
| `run_ablation.py` | batch runner over datasets × models × method configs |
| `train.py` | epoch loop, loss dispatch, AVBCM hooks |
| `avbcm.py` | **AVBCM mechanism** (boundary gap, EMA, λ_t, prior gate) |
| `model.py` | GNN backbones (GCN / GraphSAGE / GAT / RGCN) |
| `config.py` | CLI arguments |
| `data_loader.py`, `utils.py` | dataset loading, metrics, Excel/PNG reporting |

## Requirements

Python ≥ 3.10, `torch`, `torch-geometric`, `scikit-learn`, `pandas`,
`openpyxl`, `matplotlib`, `seaborn`, `tqdm`. CUDA optional but recommended.

## Data

Place each dataset as a directory of per-graph files (PyG `Data` objects) under
`datas/<name>` (e.g. `datas/devign`, `datas/reveal`), with labels `y ∈ {0,1}`.
Splits are fixed (`train/val/test = 80/10/10`, `random_state=42`).

## Quick start

```bash
# AVBCM (adaptive λ)
python main.py --data_path datas/devign --model_name gcn --use_avbc

# Fixed-λ ablation
python main.py --data_path datas/devign --model_name gcn --use_avbc --fixed_lambda 0.5

# Baseline (no AVBCM)
python main.py --data_path datas/devign --model_name gcn --skip_init
```

Batch experiments: edit `methods` / `loss_functions` in `run_ablation.py`,
then `python run_ablation.py`.

Key flags: `--model_name {gcn,graphsage,gat,rgcn}`, `--loss_function {ce,weighted_ce,focal,logit_adj,ldam,cb_focal}`,
`--epochs` (100), `--batch_size` (32), `--lr` (1e-4), `--patience` (100),
`--seed` (36), `--output_dir` (`results`).

## Outputs

Each run writes to `results/<dataset>/<model>/<method>/`:

- `<model>_best.pth` — best-validation checkpoint
- `results.xlsx` — metrics, training info, per-epoch curves
- `history.npz` — per-epoch `lambda_t`, `boundary_gap_raw`, `boundary_gap_ema`
  (the data behind the mechanism figures)

Directory ↔ variant: `NoInit` (baseline), `AVBCM` (adaptive λ_t),
`AVBCM_fx0.5` (fixed λ; any constant passed to `--fixed_lambda`).
