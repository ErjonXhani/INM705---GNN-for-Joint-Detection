"""Dataset class for yoga pose data."""

import os
import json

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PoseDataset(Dataset):
    """Dataset class for yoga pose keypoints and graphs."""

    # using full MediaPipe-style skeleton connections (33 keypoints)
    # this defines how joints are connected in the graph
    POSE_EDGES = [
        # face / head
        (0, 1), (0, 4),
        (1, 2), (2, 3),
        (4, 5), (5, 6),
        (9, 10),

        # torso
        (11, 12),
        (11, 23), (12, 24),
        (23, 24),

        # left arm
        (11, 13), (13, 15),
        (15, 17), (15, 19), (15, 21),
        (17, 19),

        # right arm
        (12, 14), (14, 16),
        (16, 18), (16, 20), (16, 22),
        (18, 20),

        # left leg
        (23, 25), (25, 27),
        (27, 29), (27, 31),
        (29, 31),

        # right leg
        (24, 26), (26, 28),
        (28, 30), (28, 32),
        (30, 32),
    ]
    
    def __init__(self, data_dir, split="train", transform=None):
        """
        Args:
            data_dir: path to keypoints folder
            split: which split to load (train / val / test)
            transform: optional feature transform
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # pick the correct split file (these were generated earlier)
        self.split_file = self._get_split_file()

        # load all file paths from that split
        self.samples = self._load_samples()

        # build mapping from class name -> integer label
        # doing this once keeps labels consistent across splits
        self.label_map = self._build_label_map()

        # build edge_index once since it is the same for every graph
        self.edge_index = self._build_edge_index()
    
    def _get_split_file(self):
        """Return the correct split file path."""
        split_files = {
            "train": "data/splits/train_files.json",
            "val": "data/splits/val_files.json",
            "test": "data/splits/test_files.json"
        }

        if self.split not in split_files:
            raise ValueError("split must be one of: train, val, test")

        return split_files[self.split]
    
    def _load_samples(self):
        """Load file paths from the split file."""
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        with open(self.split_file, "r") as f:
            samples = json.load(f)

        if len(samples) == 0:
            raise ValueError(f"No samples found in {self.split_file}")

        # normalise paths just in case (windows paths etc)
        samples = [os.path.normpath(path) for path in samples]

        return samples

    def _build_label_map(self):
        """
        Create mapping from folder names to integer labels.

        Example:
        "Camel" -> 0
        "Tree" -> 1
        etc.
        """
        class_names = [
            folder for folder in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, folder))
        ]

        class_names = sorted(class_names)

        return {label: idx for idx, label in enumerate(class_names)}

    def _build_edge_index(self):
        """Convert edge list into PyG format."""
        edge_index = torch.tensor(self.POSE_EDGES, dtype=torch.long).t().contiguous()

        # make graph undirected by adding reverse edges
        reverse_edges = edge_index.flip(0)
        edge_index = torch.cat([edge_index, reverse_edges], dim=1)

        return edge_index
    
    def __len__(self):
        """Return number of samples in this split."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load one sample and convert it into a graph."""
        file_path = self.samples[idx]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # load json file
        with open(file_path, "r") as f:
            sample = json.load(f)

        keypoints = sample["keypoints"]
        label = sample["label"]

        # node features: [x, y, confidence]
        x = torch.tensor(keypoints, dtype=torch.float)

        if self.transform is not None:
            x = self.transform(x)

        # convert label string to integer
        y = torch.tensor(self.label_map[label], dtype=torch.long)

        # create PyG graph object
        data = Data(
            x=x,
            edge_index=self.edge_index,
            y=y
        )

        return data