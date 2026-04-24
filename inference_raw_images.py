"""Inference script for raw yoga pose images using MediaPipe + graph conversion."""

import argparse
from pathlib import Path

import cv2
import torch
import mediapipe as mp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.utils.helpers import load_config, load_checkpoint
from src.models.gcn_model import PoseGCN
from src.models.sage_model import PoseGraphSAGE
from src.models.gin_model import PoseGIN
from src.datasets.pose_dataset import PoseDataset


mp_pose = mp.solutions.pose


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


def build_model(config, num_classes, input_dim):
    model_name = config["model"]["name"]

    if model_name == "GCN":
        return PoseGCN(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=num_classes,
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )

    elif model_name == "GraphSAGE":
        return PoseGraphSAGE(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=num_classes,
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )

    elif model_name == "GIN":
        return PoseGIN(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=num_classes,
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def extract_keypoints_from_image(image_path, pose):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append([lm.x, lm.y, lm.visibility])

    return keypoints


def normalize_keypoints(keypoints):
    x = torch.tensor(keypoints, dtype=torch.float)

    if x.shape != (33, 3):
        raise ValueError(f"Expected keypoints of shape (33, 3), got {tuple(x.shape)}")

    left_hip = x[23, :2]
    right_hip = x[24, :2]
    hip_center = (left_hip + right_hip) / 2.0

    x[:, :2] = x[:, :2] - hip_center

    return x


def image_to_graph(image_path, pose, use_confidence=True):
    keypoints = extract_keypoints_from_image(image_path, pose)

    if keypoints is None:
        return None

    x = normalize_keypoints(keypoints)

    if not use_confidence:
        x = x[:, :2]

    data = Data(
        x=x,
        edge_index=EDGE_INDEX
    )

    data.image_path = str(image_path)
    data.sample_name = Path(image_path).stem
    data.true_label = Path(image_path).parent.name

    return data


def collect_images(image_dir):
    image_dir = Path(image_dir)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    image_files = [
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in image_extensions
    ]

    return sorted(image_files)


def main(checkpoint_path, config_path, image_path=None, image_dir=None, split="test"):
    config = load_config(config_path)

    use_conf = config["model"].get("use_confidence", True)
    input_dim = 3 if use_conf else 2
    device = config["training"]["device"]

    dataset = PoseDataset(
        config["data"]["processed_dir"],
        split=split,
        use_confidence=use_conf
    )

    idx_to_label = {idx: label for label, idx in dataset.label_map.items()}

    model = build_model(config, len(dataset.label_map), input_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(model, optimizer, checkpoint_path)

    model.eval()

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False
    )

    graphs = []

    if image_path is not None:
        image_path = Path(image_path)
        graph = image_to_graph(image_path, pose, use_confidence=use_conf)

        if graph is None:
            print(f"No pose detected in image: {image_path}")
            return

        graphs.append(graph)

    elif image_dir is not None:
        image_files = collect_images(image_dir)

        if len(image_files) == 0:
            print(f"No images found in folder: {image_dir}")
            return

        failed_images = []

        for img_path in image_files:
            graph = image_to_graph(img_path, pose, use_confidence=use_conf)

            if graph is None:
                failed_images.append(str(img_path))
                continue

            graphs.append(graph)

        print(f"\nImages found: {len(image_files)}")
        print(f"Successfully converted to graphs: {len(graphs)}")
        print(f"Failed pose detections: {len(failed_images)}")

        if len(graphs) == 0:
            print("No valid graphs were created.")
            return

    else:
        raise ValueError("Please provide either --image or --image_dir")

    loader = DataLoader(
        graphs,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    print("\nInference results:")

    result_index = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)

            for pred in preds.tolist():
                graph = graphs[result_index]
                result_index += 1

                pred_label = idx_to_label[pred]
                true_label = graph.true_label

                if pred_label == true_label:
                    correct += 1

                total += 1

                print(f"Image: {graph.image_path}")
                print(f"Predicted: {pred_label} | True: {true_label}")
                print("-" * 40)

    accuracy = correct / total if total > 0 else 0.0

    print("\nFINAL RESULTS")
    print(f"Total valid samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on raw yoga pose images"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved checkpoint"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single raw image"
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to a folder of raw images"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split used only to load label_map"
    )

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        image_path=args.image,
        image_dir=args.image_dir,
        split=args.split
    )