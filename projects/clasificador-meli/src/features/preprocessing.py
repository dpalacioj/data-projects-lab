"""
Módulo de preprocesamiento de datos para el Clasificador MercadoLibre.

Contiene la clase Preprocessing que transforma datos crudos del dataset
en features listos para entrenar el modelo XGBoost.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import ast
import unicodedata
from src.config import (
    GARANTIA_APLICA, GARANTIA_NO_APLICA, CATEGORY_COLS, NUMERIC_COLS,
    BOOL_COLS, DROP_COLUMNS, SELECTED_COLUMNS
)
from src.utils.logger import setup_logger

# Configurar logger para este módulo
logger = setup_logger(__name__)


class Preprocessing:
    """
    Clase para preprocesar datos de productos MercadoLibre.

    Realiza transformaciones como:
    - Extracción de campos anidados (seller_address, shipping, etc.)
    - Feature engineering (fechas, garantías, métodos de pago)
    - Conversión de tipos de datos
    - Eliminación de columnas innecesarias
    - Label encoding de variables categóricas
    """

    def __init__(self):
        """Inicializa el preprocesador con configuraciones del módulo config."""
        # Definiciones estáticas que se usan en el procesamiento
        self.category_cols = CATEGORY_COLS
        self.numeric_cols = NUMERIC_COLS
        self.bool_cols = BOOL_COLS
        self.drop_columns = DROP_COLUMNS
        self.selected_columns = SELECTED_COLUMNS

        # Regex y keywords para clasificar la garantía
        self.patron_tiempo = re.compile(
            r'(\d+)\s*(dias|día|mes|meses|año|años|semanas|semana|ano)',
            re.IGNORECASE
        )
        self.keywords_aplica = GARANTIA_APLICA
        self.keywords_no_aplica = GARANTIA_NO_APLICA

    def normalizar_texto(self, texto):
        """
        Normaliza texto eliminando acentos, convirtiendo a minúsculas y limpiando caracteres especiales.

        Args:
            texto: String a normalizar

        Returns:
            String normalizado
        """
        if isinstance(texto, str):
            texto = texto.lower()
            texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
            texto = re.sub(r'[^\w\s]', '', texto)
        return texto

    def clasificar_garantia(self, texto):
        """
        Clasifica si una garantía aplica basándose en el texto del campo warranty.

        Busca keywords y patrones temporales (ej: '6 meses', '1 año') para determinar
        si el producto tiene garantía válida.

        Args:
            texto: Texto del campo warranty

        Returns:
            'aplica', 'no_aplica' o 'indeterminado'
        """
        if isinstance(texto, str):
            texto_norm = self.normalizar_texto(texto)
            if "si" in texto_norm or self.patron_tiempo.search(texto_norm) or any(
                    kw in texto_norm for kw in self.keywords_aplica):
                return "aplica"
            if any(kw in texto_norm for kw in self.keywords_no_aplica):
                return "no_aplica"
        return "indeterminado"

    def convert_to_category(self, df):
        """
        Convierte columnas especificadas en tipo 'category' para optimización de memoria.

        Args:
            df: DataFrame a modificar

        Returns:
            DataFrame con columnas categóricas convertidas
        """
        for col in self.category_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    def convert_to_numeric(self, df):
        """
        Convierte columnas especificadas a tipo numérico (float).

        Args:
            df: DataFrame a modificar

        Returns:
            DataFrame con columnas numéricas convertidas. Valores no convertibles se marcan como NaN.
        """
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def convert_to_bool(self, df):
        """
        Convierte columnas especificadas a tipo booleano.

        Args:
            df: DataFrame a modificar

        Returns:
            DataFrame con columnas booleanas convertidas
        """
        for col in self.bool_cols:
            if col in df.columns:
                df[col] = df[col].astype("bool")

        return df

    @staticmethod
    def clean_column_name(name):
        name = name.lower().strip()
        name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8")
        return re.sub(r"\s+", "_", name)

    def extract_payment_methods(self, payment_list):
        if not isinstance(payment_list, list):
            return {}
        return {self.clean_column_name(p["description"]): True for p in payment_list}

    def ensure_columns_exist(self, df):
        """
        Asegura que todas las columnas necesarias existen en el DataFrame final.
        Rellena con valores por defecto según el tipo de columna.
        """
        for col in self.selected_columns:
            if col not in df.columns:
                if col in self.bool_cols:
                    df[col] = False
                else:
                    df[col] = np.nan
        return df
    
    @staticmethod
    def safe_parse(val):
        """
        Parsea de forma segura strings que representan estructuras Python (dict, list).

        Args:
            val: Valor a parsear (string o cualquier tipo)

        Returns:
            Estructura Python parseada o el valor original si no es string
        """
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception as e:
                logger.warning(f"Error al parsear valor: {str(val)[:100]}... | Error: {e}")
                return None
        return val

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Método principal de transformación de datos.

        Ejecuta el pipeline completo de preprocesamiento:
        1. Extrae campos anidados (seller_address, shipping)
        2. Crea features de métodos de pago
        3. One-hot encoding de tags
        4. Clasifica garantías
        5. Crea features temporales (mes, día de la semana, días activo)
        6. Rellena valores nulos
        7. Elimina columnas innecesarias
        8. Convierte tipos de datos
        9. Selecciona columnas finales
        10. Aplica label encoding a categóricas

        Args:
            data: DataFrame con datos crudos del dataset MercadoLibre

        Returns:
            DataFrame procesado listo para entrenar/predecir
        """
        logger.info(f"Iniciando preprocesamiento | Shape inicial: {data.shape}")

        # Crear columnas desde campos anidados si existen
        if 'seller_address' in data.columns:
            logger.debug("Extrayendo campos de seller_address")
            data['seller_address'] = data['seller_address'].apply(self.safe_parse)
            data['seller_country'] = data['seller_address'].apply(lambda x: x.get('country', {}).get('name'))
            data['seller_state'] = data['seller_address'].apply(lambda x: x.get('state', {}).get('name'))
            data['seller_city'] = data['seller_address'].apply(lambda x: x.get('city', {}).get('name'))

        if 'shipping' in data.columns:
            logger.debug("Extrayendo campos de shipping")
            data['shipping'] = data['shipping'].apply(self.safe_parse)
            data['shipping_local_pick_up'] = data['shipping'].apply(lambda x: x.get('local_pick_up'))
            data['shipping_free_shipping'] = data['shipping'].apply(lambda x: x.get('free_shipping'))
            data['shipping_tags'] = data['shipping'].apply(lambda x: x.get('tags'))
            data['shipping_mode'] = data['shipping'].apply(lambda x: x.get('mode'))

        if 'non_mercado_pago_payment_methods' in data.columns:
            logger.debug("Procesando métodos de pago")
            payment_df = data["non_mercado_pago_payment_methods"].apply(self.extract_payment_methods).apply(pd.Series,
                                                                                                            dtype='bool').fillna(
                False)
            data = pd.concat([data, payment_df], axis=1)

        if 'tags' in data.columns:
            logger.debug("Aplicando one-hot encoding a tags")
            tags_one_hot = data['tags'].str.join(',').str.get_dummies(sep=',')
            data = pd.concat([data, tags_one_hot], axis=1)

        if 'warranty' in data.columns:
            logger.debug("Clasificando garantías")
            data["garantia_aplica"] = data["warranty"].apply(self.clasificar_garantia)

        # Rellenar valores
        logger.debug("Rellenando valores nulos y limpiando columnas vacías")
        # Actualizado: applymap() está deprecado en pandas >= 2.1
        data = data.map(lambda x: x if x else np.nan)
        data = data.dropna(how='all', axis=1)

        if any(col in data.columns for col in ["visa", "visa_electron"]):
            data['visa'] = data['visa_electron'].fillna(data['visa'])
        if any(col in data.columns for col in ["mastercard", "mastercard_maestro"]):
            data['mastercard'] = data['mastercard_maestro'].fillna(data['mastercard'])

        if any(col in data.columns for col in ["visa", "mastercard", "diners", "american_express"]):
            data["tarjeta_de_credito"] = data["tarjeta_de_credito"].fillna(
                data[["visa", "mastercard", "diners", 'american_express']].any(axis=1))

        if 'mercadopago' in data.columns:
            data['accepts_mercadopago'] = data['accepts_mercadopago'].fillna(data['mercadopago'])

        data['accepts_mercadopago'] = data['accepts_mercadopago'].fillna(False)

        # Drop used columns
        pagos_to_drop = ['mercadopago', 'mastercard_maestro', 'visa_electron', 'visa', 'mastercard', 'diners',
                         'american_express']

        # Filtrar columnas que realmente existen en el DataFrame
        existing_cols_to_drop = [col for col in pagos_to_drop if col in data.columns]

        # Hacer el drop solo si hay columnas existentes
        if existing_cols_to_drop:
            data = data.drop(columns=existing_cols_to_drop)

        # Configuracion de valores por defecto para fillna
        fillna_config = {
            'seller_city': 'mode',
            'seller_state': 'mode',
            'sold_quantity': 0,
            'poor_quality_thumbnail': 0,
            'free_relist': 0,
            'dragged_visits': 0,
            'good_quality_thumbnail': 0,
            'dragged_bids_and_visits': 0,
            'transferencia_bancaria': False,
            'efectivo': False,
            'shipping_local_pick_up': False,
            'cheque_certificado': False,
            'contra_reembolso': False,
            'acordar_con_el_comprador': False,
            'automatic_relist': False,
            'giro_postal': False,
            'shipping_free_shipping': False
        }

        # Aplicar fillna segun configuracion
        for col, fill_value in fillna_config.items():
            if col in data.columns:
                if fill_value == 'mode':
                    data[col] = data[col].fillna(data[col].mode()[0])
                else:
                    data[col] = data[col].fillna(fill_value)

        # Variables temporales
        if 'start_time' in data.columns and 'stop_time' in data.columns:
            logger.debug("Creando features temporales (fechas y días activos)")
            data['year_start'] = pd.to_datetime(data['start_time'], unit='ms', errors='coerce').dt.year.astype('category')
            data['month_start'] = pd.to_datetime(data['start_time'], unit='ms', errors='coerce').dt.month.astype('category')
            data['year_stop'] = pd.to_datetime(data['stop_time'], unit='ms', errors='coerce').dt.year.astype('category')
            data['month_stop'] = pd.to_datetime(data['stop_time'], unit='ms', errors='coerce').dt.month.astype('category')
            data['week_day'] = pd.to_datetime(data['stop_time'], unit='ms', errors='coerce').dt.weekday.astype('category')
            data['days_active'] = (pd.to_datetime(data['stop_time'], unit='ms', errors='coerce') - pd.to_datetime(data['start_time'], unit='ms', errors='coerce')).dt.days

        df = data.copy()
        existing_cols_to_drop = [col for col in self.drop_columns if col in df.columns]
        if existing_cols_to_drop:
            logger.debug(f"Eliminando {len(existing_cols_to_drop)} columnas innecesarias")
            df = df.drop(columns=existing_cols_to_drop)

        # Tipos de dato
        logger.debug("Convirtiendo tipos de datos (category, numeric, bool)")
        df = self.convert_to_category(df)
        df = self.convert_to_numeric(df)
        df = self.convert_to_bool(df)

        df = self.ensure_columns_exist(df)

        # Selección de columnas finales
        logger.debug(f"Seleccionando {len(self.selected_columns)} columnas finales")
        df = df[self.selected_columns]

        logger.debug("Aplicando label encoding a variables categóricas")
        cat_vars = list(df.select_dtypes(include=['category']).columns)
        df[cat_vars] = df[cat_vars].apply(LabelEncoder().fit_transform)

        logger.info(f"Preprocesamiento completado | Shape final: {df.shape}")
        return df