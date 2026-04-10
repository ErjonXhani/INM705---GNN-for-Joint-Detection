"""Main training script."""

import argparse
import os
import wandb
import torch
from torch_geometric.loader import DataLoader

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

    # make sure checkpoint folder exists
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize model
    # input_dim is 3 because each node has [x, y, confidence]
    model = PoseGCN(
        input_dim=3,
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=13,  # Number of yoga poses
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )

    # Initialize trainer
    device = config["training"]["device"]
    trainer = PoseTrainer(model, device=device)

    # Load datasets
    train_dataset = PoseDataset(config["data"]["train_path"])
    val_dataset = PoseDataset(config["data"]["val_path"])

    # Graph dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # Main training loop
    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")

        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        val_loss, val_acc = trainer.validate(val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # log results to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # only save if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                filename="checkpoints/best_model.pt"
            )

            print("Saved new best model")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train yoga pose GCN model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)