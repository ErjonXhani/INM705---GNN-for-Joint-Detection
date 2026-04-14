"""Main training script."""

import argparse
import os
import torch
import wandb
from torch_geometric.loader import DataLoader

from src.utils.helpers import load_config, save_checkpoint
from src.models.gcn_model import PoseGCN
from src.models.sage_model import PoseGraphSAGE   
from src.models.gin_model import PoseGIN          
from src.datasets.pose_dataset import PoseDataset
from src.training.trainer import PoseTrainer


def main(config_path):
    """
    Main training function.
    """
    # load config
    config = load_config(config_path)

    # run name comes from yaml
    run_name = config["training"]["run_name"]
    checkpoint_path = f"{config['training']['checkpoint_dir']}/{run_name}.pt"

    print(f"Running experiment: {run_name}")

    # optional notes section in yaml
    # this does not affect training, it is just there for reference
    if "notes" in config:
        print("Experiment notes:")
        for key, value in config["notes"].items():
            print(f"  {key}: {value}")

    # start wandb
    wandb.login(relogin=True)

    wandb.init(
        entity="omari-march-city-university-of-london",
        project="DeepLearningImages",
        config=config,
        name=run_name
    )

    # make checkpoint folder if it does not exist
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)

    # load train/val splits first so we can get the number of classes
    train_dataset = PoseDataset(config["data"]["processed_dir"], split="train")
    val_dataset = PoseDataset(config["data"]["processed_dir"], split="val")

    # model setup
    # each node has [x, y, confidence]
        
    model_name = config["model"]["name"]

    if model_name == "GCN":
        model = PoseGCN(
            input_dim=3,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=len(train_dataset.label_map),
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "GraphSAGE":
        model = PoseGraphSAGE(
            input_dim=3,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=len(train_dataset.label_map),
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "GIN":
        model = PoseGIN(
            input_dim=3,
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=len(train_dataset.label_map),
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # trainer
    device = config["training"]["device"]
    trainer = PoseTrainer(model, device=device)

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

    # main training loop
    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        val_loss, val_acc = trainer.validate(val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # only save if this is the best validation score so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                checkpoint_path
            )

            print(f"Saved new best model -> {checkpoint_path}")

    print(f"\nFinished training. Best validation accuracy: {best_val_acc:.4f}")
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