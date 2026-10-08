import os
import subprocess
import sys
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Method configurations (name -> CLI flags)
methods = {
    "AVBCM": ["--use_avbc"],
    # "AVBCM_fx0.5": ["--use_avbc", "--fixed_lambda", "0.5"],
    # "NoInit": ["--skip_init"],
}

loss_functions = {
    "ce": ["--loss_function", "ce"],
    # "focal": ["--loss_function", "focal"],
}

datasets = [
    os.path.join(SCRIPT_DIR, "datas", "devign"),
    os.path.join(SCRIPT_DIR, "datas", "reveal"),
]

models = [
    "gcn",
    # "graphsage",
    # "gat",
    # "rgcn",
]


def run_experiment(config):
    dataset_path, model_name, method_name, method_args, loss_name, loss_args = config
    dataset_name = os.path.basename(dataset_path.rstrip('/\\'))

    log_dir = os.path.join(SCRIPT_DIR, "ablation_results", f"{dataset_name}_{model_name}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{method_name}_{loss_name}.log")

    print(f"Running: Dataset={dataset_name} | Model={model_name} | "
          f"Method={method_name} | Loss={loss_name}")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "main.py"),
        "--model_name", model_name,
        "--data_path", dataset_path,
    ] + method_args + loss_args

    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)

    if result.returncode == 0:
        print(f"  Done: {dataset_name}/{model_name}/{method_name}/{loss_name}")
        return True
    print(f"  Failed: {dataset_name}/{model_name}/{method_name}/{loss_name} "
          f"(exit code {result.returncode})")
    return False


def batch_test_all():
    experiments = [
        (dataset_path, model_name, method_name, method_args, loss_name, loss_args)
        for dataset_path in datasets
        for model_name in models
        for method_name, method_args in methods.items()
        for loss_name, loss_args in loss_functions.items()
    ]

    total_exp = len(experiments)
    print(f"\n{'=' * 80}")
    print(f"Total experiments: {total_exp} (Datasets: {len(datasets)}, "
          f"Models: {len(models)}, Methods: {len(methods)}, Losses: {len(loss_functions)})")
    print(f"{'=' * 80}\n")

    n_workers = min(len(experiments), os.cpu_count())
    print(f"Starting with {n_workers} workers...\n")

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_experiment, experiments)

    successful = sum(results)
    print(f"\n{'=' * 80}")
    print(f"All experiments completed!")
    print(f"Successful: {successful}, Failed: {total_exp - successful}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    batch_test_all()
