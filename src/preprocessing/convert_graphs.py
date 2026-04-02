"""
Convert MediaPipe keypoint JSON files into PyTorch Geometric graph objects
and split them class-wise into train/val/test using an 80/10/10 ratio.

Input structure:
keypoints/
    ClassA/
        img1.json
        img2.json
    ClassB/
        img3.json

Output structure:
data/processed/graphs/
    label_map.json
    split_summary.json
    train/
        ClassA/
            img1.pt
        ClassB/
            img3.pt
    val/
        ...
    test/
        ...
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch_geometric.data import Data


# --------------------------------------------------
# MediaPipe Pose skeleton connections (33 landmarks)
# --------------------------------------------------
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]


def make_undirected(edges):
    undirected = []
    for src, dst in edges:
        undirected.append((src, dst))
        undirected.append((dst, src))
    return undirected


EDGE_INDEX = torch.tensor(
    make_undirected(EDGES), dtype=torch.long
).t().contiguous()


# --------------------------------------------------
# Keypoint normalization
# --------------------------------------------------
def normalize_keypoints(keypoints):
    """
    Input:
        keypoints: list of 33 landmarks, each [x, y, visibility]

    Output:
        x: torch tensor of shape [33, 3]
    """
    x = torch.tensor(keypoints, dtype=torch.float)

    if x.shape != (33, 3):
        raise ValueError(f"Expected keypoints of shape (33, 3), got {tuple(x.shape)}")

    # Center pose using midpoint of left and right hip
    left_hip = x[23, :2]
    right_hip = x[24, :2]
    hip_center = (left_hip + right_hip) / 2.0

    x[:, :2] = x[:, :2] - hip_center

    return x


# --------------------------------------------------
# Convert one JSON file into a graph
# --------------------------------------------------
def json_to_graph(json_path, label_map):
    with open(json_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    label_name = sample["label"]
    keypoints = sample["keypoints"]

    x = normalize_keypoints(keypoints)
    y = torch.tensor([label_map[label_name]], dtype=torch.long)

    data = Data(x=x, edge_index=EDGE_INDEX, y=y)

    # optional metadata
    data.image_path = sample.get("image", "")
    data.label_name = label_name
    data.sample_name = json_path.stem

    return data


# --------------------------------------------------
# Collect JSON files from keypoints folder
# --------------------------------------------------
def collect_files_by_class(input_root):
    input_root = Path(input_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")

    class_dirs = [d for d in input_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found inside {input_root}")

    file_dict = {}
    for class_dir in sorted(class_dirs):
        json_files = sorted(class_dir.glob("*.json"))
        if len(json_files) == 0:
            print(f"Warning: class '{class_dir.name}' has no JSON files")
        file_dict[class_dir.name] = json_files

    return file_dict


# --------------------------------------------------
# Class-wise 80/10/10 split
# --------------------------------------------------
def stratified_split(file_dict, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits each class separately into train/val/test.

    Keeps all data.
    Does NOT downsample classes.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    random.seed(seed)

    split_dict = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    summary = {}

    for class_name, files in file_dict.items():
        files_copy = files[:]
        random.shuffle(files_copy)

        n = len(files_copy)
        if n < 3:
            raise ValueError(
                f"Class '{class_name}' has only {n} samples. "
                f"Need at least 3 for train/val/test."
            )

        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)

        # ensure at least 1 sample in val and test if possible
        if n_train >= n:
            n_train = n - 2
        if n_train + n_val >= n:
            n_val = n - n_train - 1

        n_test = n - n_train - n_val

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError(
                f"Invalid split for class '{class_name}': "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

        train_files = files_copy[:n_train]
        val_files = files_copy[n_train:n_train + n_val]
        test_files = files_copy[n_train + n_val:]

        split_dict["train"][class_name] = train_files
        split_dict["val"][class_name] = val_files
        split_dict["test"][class_name] = test_files

        summary[class_name] = {
            "total": n,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        }

        print(
            f"{class_name}: total={n}, "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )

    return split_dict, summary


# --------------------------------------------------
# Save graph objects as .pt files
# --------------------------------------------------
def save_graphs(split_dict, output_root, label_map):
    output_root = Path(output_root)

    for split_name, class_dict in split_dict.items():
        for class_name, json_files in class_dict.items():
            out_dir = output_root / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for json_path in json_files:
                try:
                    graph = json_to_graph(json_path, label_map)
                    out_file = out_dir / f"{json_path.stem}.pt"
                    torch.save(graph, out_file)
                except Exception as e:
                    print(f"Error processing {json_path}: {e}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    input_root = "keypoints"
    output_root = "data/processed/graphs"
    seed = 42

    file_dict = collect_files_by_class(input_root)

    class_names = sorted(file_dict.keys())
    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    # save label map
    with open(output_root_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    print("Label map:")
    for class_name, idx in label_map.items():
        print(f"  {class_name}: {idx}")

    split_dict, summary = stratified_split(
        file_dict,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed
    )

    # save graphs
    save_graphs(split_dict, output_root_path, label_map)

    # save split summary
    with open(output_root_path / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nDone.")
    print(f"Graphs saved to: {output_root_path}")
    print(f"Label map saved to: {output_root_path / 'label_map.json'}")
    print(f"Split summary saved to: {output_root_path / 'split_summary.json'}")


if __name__ == "__main__":
    main()