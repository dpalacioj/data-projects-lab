"""Carga de Fashion-MNIST. Descarga automatica la primera vez (~30 MB)."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import BATCH_SIZE, DATA_DIR

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]
)


def cargar_datasets():
    """Devuelve (train, test) como torchvision datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=TRANSFORM)
    return train, test


def cargar_dataloaders(batch_size: int = BATCH_SIZE):
    """Envuelve los datasets en DataLoaders listos para entrenar."""
    train_ds, test_ds = cargar_datasets()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, test_dl
