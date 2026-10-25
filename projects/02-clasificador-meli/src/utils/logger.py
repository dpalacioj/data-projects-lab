"""
Configuración centralizada de logging para el proyecto.

Proporciona funciones para configurar y obtener loggers con formato consistente.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: Path = None
) -> logging.Logger:
    """
    Configura y retorna un logger con formato consistente.

    Args:
        name: Nombre del logger (generalmente __name__ del módulo)
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Si True, guarda logs en archivo además de consola
        log_dir: Directorio donde guardar los logs (default: PROJECT_ROOT/logs)

    Returns:
        Logger configurado

    Ejemplo:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Procesando datos...")
    """
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicar handlers si ya existe
    if logger.handlers:
        return logger

    # Formato de logs
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo (opcional)
    if log_to_file:
        if log_dir is None:
            # Usar directorio del proyecto
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            log_dir = project_root / 'logs'

        # Crear directorio si no existe
        log_dir.mkdir(exist_ok=True)

        # Nombre del archivo con timestamp
        log_filename = f"{name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
        log_file = log_dir / log_filename

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logs guardándose en: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger existente o crea uno nuevo con configuración por defecto.

    Args:
        name: Nombre del logger

    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
