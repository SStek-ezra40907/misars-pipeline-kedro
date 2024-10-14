from kedro.io import AbstractDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class PTDataset(AbstractDataset):
    """Kedro Dataset for loading PyTorch Datasets."""

    def __init__(self, filepath: str, transform=None, batch_size=32, shuffle=True):
        self.dataset_path = filepath
        self.transform = transform or transforms.ToTensor()  # Use a default transform if not provided
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _load(self) -> DataLoader:
        """Load the dataset and return a DataLoader."""
        dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader

    def _save(self, data: torch.utils.data.Dataset) -> None:
        """Saving datasets is typically not necessary for PyTorch, so you can pass here."""
        pass

    def _describe(self) -> dict:
        return dict(dataset_path=self.dataset_path, batch_size=self.batch_size, shuffle=self.shuffle)