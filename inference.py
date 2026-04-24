"""Inference / evaluation script for yoga pose classification."""

import argparse
import torch
from torch_geometric.loader import DataLoader

from src.utils.helpers import load_config, load_checkpoint
from src.models.gcn_model import PoseGCN
from src.datasets.pose_dataset import PoseDataset
from src.models.sage_model import PoseGraphSAGE
from src.models.gin_model import PoseGIN
from src.models.gat_model import PoseGAT


def build_model(config, num_classes, input_dim):
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
    elif model_name == "GAT":
        model = PoseGAT(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=num_classes,
            num_layers=config["model"]["num_layers"],
            heads=config["model"]["heads"],
            dropout=config["model"]["dropout"],
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model


def main(checkpoint_path, config_path, split="test", num_examples=5, graph_path=None):
    """
    Run inference on either:
    1. a full dataset split, or
    2. a single graph file (.pt)

    Args:
        checkpoint_path: path to saved model checkpoint
        config_path: path to yaml config
        split: which split to run on (train / val / test)
        num_examples: how many example predictions to print
        graph_path: optional path to a single .pt graph file
    """
    config = load_config(config_path)
    use_conf = config["model"].get("use_confidence", True)
    input_dim = 3 if use_conf else 2
    device = config["training"]["device"]

    # Load dataset once to access label_map and number of classes
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

    
    # SINGLE GRAPH MODE
    
    if graph_path is not None:
        graph = torch.load(graph_path, weights_only=False)

        # Ensure batch exists for single-graph inference
        if not hasattr(graph, "batch") or graph.batch is None:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)

        graph = graph.to(device)

        with torch.no_grad():
            out = model(graph.x, graph.edge_index, graph.batch)
            pred = out.argmax(dim=1).item()

        pred_label = idx_to_label[pred]
        print("\nFinished inference on single graph")
        print(f"Graph file: {graph_path}")
        print(f"Predicted class: {pred_label}")

        if hasattr(graph, "y") and graph.y is not None:
            true_idx = graph.y.item() if graph.y.dim() == 0 else graph.y[0].item()
            true_label = idx_to_label[true_idx]
            print(f"True class: {true_label}")

        return

    
    # FULL SPLIT MODE
    
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    total_correct = 0
    total_samples = 0
    example_results = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)

            total_correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

            for pred, true in zip(preds.tolist(), batch.y.tolist()):
                if len(example_results) < num_examples:
                    example_results.append(
                        (idx_to_label[pred], idx_to_label[true])
                    )

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print(f"\nFinished inference on {split} split")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {total_correct}")
    print(f"Accuracy: {accuracy:.4f}")

    if example_results:
        print("\nExample predictions:")
        for i, (pred_label, true_label) in enumerate(example_results, start=1):
            print(f"{i}. Predicted: {pred_label} | True: {true_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on yoga pose graphs")
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
        "--split",
        type=str,
        default="test",
        help="Which split to evaluate on"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="How many example predictions to print"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Optional path to a single graph .pt file"
    )

    args = parser.parse_args()

    main(
        args.checkpoint,
        args.config,
        args.split,
        args.num_examples,
        args.graph
    )