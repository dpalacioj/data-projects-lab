"""
Configuración estandarizada de logging para el proyecto.

Permite que cada módulo cree su propio logger con un formato consistente
y un nivel configurable.
"""

import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Crea (o devuelve) un logger configurado.

    Parameters
    ----------
    name : str
        Nombre del logger, típicamente ``__name__`` del módulo que lo usa.
    level : int, optional
        Nivel de log (DEBUG=10, INFO=20, WARNING=30). Default: INFO.

    Returns
    -------
    logging.Logger
        Logger listo para usar.

    Examples
    --------
    >>> logger = setup_logger(__name__)
    >>> logger.info("Procesando datos...")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar agregar handlers duplicados si el logger ya existe
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
