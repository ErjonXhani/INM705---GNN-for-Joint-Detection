"""Main training script."""

import argparse
import wandb
import torch
from src.utils.helpers import load_config, save_checkpoint
from src.models.gcn_model import PoseGCN
from src.datasets.pose_dataset import PoseDataset
from src.training.trainer import PoseTrainer


def main(config_path):
    """
    Main training function.

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)

    # Start WandB
    wandb.init(
        project="inm705-yoga-gnn",
        config=config,
        name="wandb_test_run",
        mode="offline"
    )

    # Initialize model
    model = PoseGCN(
        input_dim=17,  # OpenPose keypoints
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=13,  # Number of yoga poses
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )

    # Initialize trainer
    device = config["training"]["device"]
    trainer = PoseTrainer(model, device=device)

    # Temporary WandB test log
    wandb.log({
        "test_metric": 1.0,
        "device": device
    })

    print("WandB test run created successfully.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train yoga pose GCN model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)
