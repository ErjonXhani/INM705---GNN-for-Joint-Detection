

"""Main training script."""

import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from src.utils.helpers import load_config, save_checkpoint
from src.models.gcn_model import PoseGCN
from src.datasets.pose_dataset import PoseDataset
from src.training.trainer import PoseTrainer


def main(config_path):
    """
    Main training function.
    """
    # load config
    config = load_config(config_path)

    # make checkpoint folder if it does not exist
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)

    # model setup
    # each node has [x, y, confidence]
    model = PoseGCN(
        input_dim=3,
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=14,
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )

    # trainer
    device = config["training"]["device"]
    trainer = PoseTrainer(model, device=device)

    # load train/val splits from the processed keypoints folder
    train_dataset = PoseDataset(config["data"]["processed_dir"], split="train")
    val_dataset = PoseDataset(config["data"]["processed_dir"], split="val")

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

    # optimizer + loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")

        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        val_loss, val_acc = trainer.validate(val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                f"{config['training']['checkpoint_dir']}/best_model.pt"
            )

            print("Saved new best model")


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