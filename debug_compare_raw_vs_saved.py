import torch
import mediapipe as mp
from pathlib import Path

from inference_raw_images import image_to_graph


mp_pose = mp.solutions.pose

saved_graph_path = "data/processed/graphs/test/Plank/00000440.pt"  # change if needed

saved_graph = torch.load(saved_graph_path, weights_only=False)
image_path = saved_graph.image_path

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False
)

raw_graph = image_to_graph(
    image_path,
    pose,
    use_confidence=False
)

print("Saved graph path:", saved_graph_path)
print("Original image path:", image_path)

print("Saved x shape:", saved_graph.x.shape)
print("Raw x shape:", raw_graph.x.shape)

print("Saved first 5 nodes:")
print(saved_graph.x[:5])

print("Raw first 5 nodes:")
print(raw_graph.x[:5])

print("Mean absolute difference:")
print(torch.mean(torch.abs(saved_graph.x[:, :2] - raw_graph.x[:, :2])).item())