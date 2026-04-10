"""Training loop for the pose classification model."""

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class PoseTrainer:
    """Trainer class for yoga pose classification model."""

    def __init__(self, model, device="cuda"):
        """
        Set up the trainer.

        Args:
            model: the GNN model we want to train
            device: "cuda" if using GPU, otherwise "cpu"
        """
        self.model = model
        self.device = device
        self.model.to(device)

    def train_epoch(self, train_loader, optimizer, criterion):
        """
        Run one full training epoch.

        Args:
            train_loader: loader for training graphs
            optimizer: e.g. Adam
            criterion: e.g. CrossEntropyLoss

        Returns:
            average training loss for the epoch
        """
        self.model.train()
        total_loss = 0

        # go through each batch of graphs
        for batch in tqdm(train_loader):
            # send batch to GPU/CPU
            batch = batch.to(self.device)

            optimizer.zero_grad()

            # forward pass
            # batch.x = node features
            # batch.edge_index = graph connections
            # batch.batch = tells us which nodes belong to which graph
            out = self.model(batch.x, batch.edge_index, batch.batch)

            # batch.y should contain one label per graph
            loss = criterion(out, batch.y)

            # backward pass + parameter update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        """
        Run validation without updating the model.

        Args:
            val_loader: loader for validation graphs
            criterion: loss function

        Returns:
            average validation loss, validation accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # no gradients needed for validation
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # same forward pass as training
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)

                total_loss += loss.item()

                # predicted class is just the max logit
                preds = out.argmax(dim=1)

                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)

        return total_loss / len(val_loader), correct / total