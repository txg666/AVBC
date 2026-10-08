import logging
from pathlib import Path

import torch
from tqdm import tqdm
from torch_geometric.data import Data

from config import add_args

args = add_args().parse_args()
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("PTDataLoader")


def load_pt_file(file_path: Path):
    """Load a single .pt file as a PyG Data object."""
    try:
        data = torch.load(file_path, weights_only=False)
        return data
    except Exception as e:
        LOGGER.error(f"Failed to load {file_path}: {e}")
        return None


def load_dataset(root_dir=args.data_path):
    """
    Load all .pt graph files under a dataset directory.

    Args:
        root_dir: dataset directory (default: --data_path).

    Returns:
        List[Data]: PyG Data objects.
    """
    root = Path(root_dir)

    files = list(root.rglob("*.pt"))
    if not files:
        LOGGER.warning(f"No .pt files found in {root_dir}")
        return []

    print(f"Found {len(files)} files, loading...")

    all_data = []
    for f in tqdm(files):
        data = load_pt_file(f)
        if data is not None:
            all_data.append(data)

    print(f"Loaded {len(all_data)} graphs")

    if all_data:
        labels = [d.y.item() for d in all_data]
        unique, counts = torch.tensor(labels).unique(return_counts=True)
        print(f"Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    return all_data


if __name__ == "__main__":
    data_list = load_dataset("datas/devign")
    if data_list:
        print(f"\nFirst graph: {data_list[0]}")
        print(f"  Nodes: {data_list[0].x.shape[0]}")
        print(f"  Edges: {data_list[0].edge_index.shape[1]}")
        print(f"  Label: {data_list[0].y.item()}")
