"""Evaluate a trained model and save a confusion matrix image."""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from src.utils.helpers import load_config, load_checkpoint
from src.models.gcn_model import PoseGCN
from src.models.sage_model import PoseGraphSAGE
from src.models.gin_model import PoseGIN
from src.datasets.pose_dataset import PoseDataset


def build_model(config, num_classes, input_dim):
    """Build the correct model from config."""
    model_name = config["model"]["name"]

    if model_name == "GCN":
        model = PoseGCN(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=num_classes,
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "GraphSAGE":
        model = PoseGraphSAGE(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=num_classes,
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "GIN":
        model = PoseGIN(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=num_classes,
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model


def main(checkpoint_path, config_path, split="test", output_dir="outputs"):
    """Run evaluation and save confusion matrix."""
    config = load_config(config_path)

    use_conf = config["model"].get("use_confidence", True)
    input_dim = 3 if use_conf else 2

    dataset = PoseDataset(
        config["data"]["processed_dir"],
        split=split,
        use_confidence=use_conf
    )

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    idx_to_label = {idx: label for label, idx in dataset.label_map.items()}
    ordered_labels = [idx_to_label[i] for i in range(len(idx_to_label))]

    device = config["training"]["device"]

    model = build_model(config, len(dataset.label_map), input_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(model, optimizer, checkpoint_path)

    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)

            all_true.extend(batch.y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    acc = sum(int(p == t) for p, t in zip(all_pred, all_true)) / len(all_true)

    print(f"\nFinished evaluation on {split} split")
    print(f"Total samples: {len(all_true)}")
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(all_true, all_pred, target_names=ordered_labels, digits=4))

    cm = confusion_matrix(all_true, all_pred)

    os.makedirs(output_dir, exist_ok=True)

    run_name = config["training"]["run_name"]
    output_path = os.path.join(output_dir, f"{run_name}_{split}_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordered_labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title(f"Confusion Matrix - {run_name} ({split})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved confusion matrix to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate confusion matrix for a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the confusion matrix image"
    )
    args = parser.parse_args()

    main(args.checkpoint, args.config, args.split, args.output_dir)