"""Bucle de entrenamiento y evaluacion. Ejecutable desde linea de comandos:

    uv run python -m reconocimiento_imagenes.train
"""
from pathlib import Path

import torch
from torch import nn, optim

from .config import EPOCHS, LEARNING_RATE, MODEL_PATH, MODELS_DIR, SEMILLA
from .data import cargar_dataloaders
from .model import SimpleCNN


def entrenar_una_epoca(modelo, dataloader, loss_fn, optimizador, device):
    modelo.train()
    total, correctos, perdida_acum = 0, 0, 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = modelo(X)
        perdida = loss_fn(logits, y)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        perdida_acum += perdida.item() * X.size(0)
        correctos += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return perdida_acum / total, correctos / total


@torch.no_grad()
def evaluar(modelo, dataloader, loss_fn, device):
    modelo.eval()
    total, correctos, perdida_acum = 0, 0, 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = modelo(X)
        perdida_acum += loss_fn(logits, y).item() * X.size(0)
        correctos += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return perdida_acum / total, correctos / total


def entrenar(epochs: int = EPOCHS, lr: float = LEARNING_RATE, destino: Path = MODEL_PATH):
    torch.manual_seed(SEMILLA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entrenando en {device}")

    train_dl, test_dl = cargar_dataloaders()
    modelo = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)

    for epoca in range(1, epochs + 1):
        train_perdida, train_acc = entrenar_una_epoca(modelo, train_dl, loss_fn, optimizador, device)
        test_perdida, test_acc = evaluar(modelo, test_dl, loss_fn, device)
        print(
            f"Epoca {epoca}/{epochs}  "
            f"train_loss={train_perdida:.4f} train_acc={train_acc:.4f}  "
            f"test_loss={test_perdida:.4f} test_acc={test_acc:.4f}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), destino)
    print(f"Modelo guardado en {destino}")
    return modelo


if __name__ == "__main__":
    entrenar()
