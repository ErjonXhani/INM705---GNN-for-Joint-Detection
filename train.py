"""Main training script."""

import argparse
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
    
    # TODO: Load data, initialize optimizer, and run training loop
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train yoga pose GCN model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)
