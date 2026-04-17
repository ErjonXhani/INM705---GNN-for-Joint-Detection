"""Inference / evaluation script for yoga pose classification."""

import argparse
import torch
from torch_geometric.loader import DataLoader

from src.utils.helpers import load_config, load_checkpoint
from src.models.gcn_model import PoseGCN
from src.datasets.pose_dataset import PoseDataset
from src.models.sage_model import PoseGraphSAGE
from src.models.gin_model import PoseGIN



def main(checkpoint_path, config_path, split="test", num_examples=5):
    """
    Run inference on a full dataset split.

    Args:
        checkpoint_path: path to saved model checkpoint
        config_path: path to yaml config
        split: which split to run on (train / val / test)
        num_examples: how many example predictions to print
    """
    # load config first
    config = load_config(config_path)
    use_conf = config["model"].get("use_confidence", True)
    input_dim = 3 if use_conf else 2

    # load dataset
    # this gives us graph samples plus the label map
    dataset = PoseDataset(
        config["data"]["processed_dir"],
        split=split,
        use_confidence=use_conf
)
    # PyG batches graphs together and creates a batch vector automatically
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # reverse label map so we can print readable class names
    idx_to_label = {idx: label for label, idx in dataset.label_map.items()}

    # build model with same settings used in training
    device = config["training"]["device"]
    
    model_name = config["model"]["name"]

    if model_name == "GCN":
        model = PoseGCN(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=len(dataset.label_map),
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "GraphSAGE":
        model = PoseGraphSAGE(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=len(dataset.label_map),
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "GIN":
        model = PoseGIN(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=len(dataset.label_map),
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    model.to(device)

    # helper expects an optimizer too, even though we are only doing inference
    optimizer = torch.optim.Adam(model.parameters())

    # load trained weights
    load_checkpoint(model, optimizer, checkpoint_path)

    # switch model to eval mode
    model.eval()

    total_correct = 0
    total_samples = 0

    # store a few examples so we can inspect predictions
    example_results = []

    # no gradients needed during inference
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # forward pass
            out = model(batch.x, batch.edge_index, batch.batch)

            # highest logit = predicted class
            preds = out.argmax(dim=1)

            total_correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

            # keep a few examples to print later
            for pred, true in zip(preds.tolist(), batch.y.tolist()):
                if len(example_results) < num_examples:
                    example_results.append(
                        (
                            idx_to_label[pred],
                            idx_to_label[true]
                        )
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
    args = parser.parse_args()

    main(args.checkpoint, args.config, args.split, args.num_examples)