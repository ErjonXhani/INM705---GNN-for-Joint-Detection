"""Inference script for yoga pose classification."""

import argparse
import torch
from src.utils.helpers import load_config, load_checkpoint
from src.models.gcn_model import PoseGCN


def main(checkpoint_path, image_path, config_path):
    """
    Run inference on a single image.
    
    Args:
        checkpoint_path: Path to model checkpoint
        image_path: Path to input image
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize model
    device = config["training"]["device"]
    model = PoseGCN(
        input_dim=17,
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=13,
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    model.to(device)
    
    # Load checkpoint
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(model, optimizer, checkpoint_path)
    
    # TODO: Load image, extract keypoints, create graph, and run inference
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on yoga pose image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.checkpoint, args.image, args.config)
