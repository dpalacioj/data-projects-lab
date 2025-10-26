"""
Configuración del proyecto Clasificador MercadoLibre.

Define constantes para preprocesamiento de datos, rutas de archivos
y configuración de modelos de Machine Learning.

Todas las rutas son relativas y portables usando pathlib.Path.
"""
from pathlib import Path

# =============================================================================
# CLASIFICACIÓN DE GARANTÍAS
# =============================================================================
# Keywords para clasificar automáticamente si un producto tiene garantía aplicable

GARANTIA_APLICA = [
    'garantia', 'cubre', 'respaldado', 'siempre que', 'proteccion',
    'defectos de fabricacion', 'aplica', 'si', 'garantizado', 'por fallas',
    'fabricacion', 'defecto'
]

GARANTIA_NO_APLICA = [
    'no', 'desconocido', 'no aplica', 'sin garantia', 'no se aceptan devoluciones',
    'sin', 'no cubre', 'no garantia', 'no respaldado', 'no proteccion',
    'no defectos de fabricacion', 'no aplica', 'no garantizado', 'no por fallas',
    'no fabricacion', 'no defecto'
]


# =============================================================================
# DEFINICIÓN DE TIPOS DE COLUMNAS
# =============================================================================
# Columnas que serán convertidas a tipo 'category' para optimización de memoria
CATEGORY_COLS = [
    'condition', 'warranty', 'buying_mode', 'currency_id', 'seller_country',
    'seller_state', 'seller_city', 'shipping_mode', 'parent_item_id',
    'category_id', 'seller_id', 'official_store_id', 'video_id',
    'status', 'garantia_aplica', 'listing_type_id'
]

# Columnas numéricas (float o int)
NUMERIC_COLS = [
    'initial_quantity', 'available_quantity', 'sold_quantity',
    'original_price', 'base_price', 'price'
]

# Columnas booleanas (True/False)
BOOL_COLS = [
    'tarjeta_de_credito', 'transferencia_bancaria', 'shipping_local_pick_up',
    'efectivo', 'automatic_relist', 'acordar_con_el_comprador'
]

# =============================================================================
# COLUMNAS A ELIMINAR EN PREPROCESAMIENTO
# =============================================================================
# Columnas que se eliminan porque no aportan valor predictivo o ya fueron procesadas
DROP_COLUMNS = [
    'seller_address', 'warranty', 'sub_status', 'seller_contact', 'deal_ids', 'shipping',
    'seller_id', 'variations', 'location', 'attributes', 'tags', 'parent_item_id', 'category_id',
    'descriptions', 'last_updated', 'international_delivery_mode', 'pictures', 'id', 'official_store_id',
    'original_price', 'thumbnail', 'title', 'date_created', 'secure_thumbnail', 'video_id', 'catalog_product_id',
    'start_time', 'stop_time', 'permalink', 'geolocation', 'shipping_tags', 'non_mercado_pago_payment_methods',
    'seller_country', 'site_id'
]


# =============================================================================
# FEATURES FINALES PARA EL MODELO
# =============================================================================
# Columnas seleccionadas tras feature engineering y análisis de importancia
# ORDEN IMPORTANTE: Debe coincidir exactamente con el orden del modelo entrenado
SELECTED_COLUMNS = [
    'initial_quantity', 'listing_type_id', 'price', 'seller_city', 'base_price',
    'available_quantity', 'week_day', 'sold_quantity',
    'seller_state', 'garantia_aplica', 'shipping_mode', 'month_start',
    'month_stop', 'tarjeta_de_credito', 'dragged_bids_and_visits',
    'transferencia_bancaria', 'shipping_local_pick_up', 'efectivo',
    'automatic_relist', 'days_active', 'acordar_con_el_comprador',
    'shipping_free_shipping', 'condition'
]

# =============================================================================
# RUTAS DE ARCHIVOS Y DIRECTORIOS (usando pathlib.Path - PORTABLE)
# =============================================================================

# Ubicación actual de este archivo: projects/clasificador-meli/src/config/config.py
_CONFIG_FILE = Path(__file__).resolve()

# Raíz del proyecto clasificador-meli: projects/clasificador-meli/
PROJECT_ROOT = _CONFIG_FILE.parent.parent.parent

# Raíz del repositorio data-projects-lab (2 niveles arriba de PROJECT_ROOT)
REPO_ROOT = PROJECT_ROOT.parent.parent

# Dataset principal crudo (316 MB, gestionado con Git LFS)
DATA_PATH_RAW = REPO_ROOT / 'datasets'
DATA_PATH_RAW_NAME = 'MLA_100k.jsonlines'

# Datos procesados (generados durante preprocesamiento o cargados como parquet)
DATA_PATH_PROCESSED = PROJECT_ROOT / 'src' / 'data' / 'processed'
DATA_PATH_PROCESSED_NAME = 'data_model.parquet'  # Cambiado a .parquet

# Directorio donde se guardan los modelos entrenados (.pkl)
MODEL_PATH = PROJECT_ROOT / 'models'
MODEL_NAME = 'xgb_model_v1.pkl'

# Archivos de configuración de hiperparámetros (JSON)
BASE_CONFIG_PATH = PROJECT_ROOT / 'src' / 'models'
BASE_CONFIG_XG_NAME = 'base_model_xg_config.json'
BASE_CONFIG_XG_OPTIMIZED_NAME = 'base_model_xg_optimized_config.json'
