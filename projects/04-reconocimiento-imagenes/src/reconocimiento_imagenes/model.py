"""Arquitectura de la CNN.

SimpleCNN es intencionalmente pequena: dos bloques conv + pool, luego
dos capas densas. Entrena en CPU en minutos y es facil de leer.
"""
from torch import nn

from .config import NUM_CLASES


class SimpleCNN(nn.Module):
    """CNN de dos bloques convolucionales para imagenes 28x28 en escala de grises."""

    def __init__(self, num_clases: int = NUM_CLASES):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 1 canal (grises) -> 32 mapas de 28x28 -> pool a 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Bloque 2: 32 -> 64 mapas de 14x14 -> pool a 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_clases),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
