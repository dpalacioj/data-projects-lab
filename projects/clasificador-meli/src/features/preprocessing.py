# app/models/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import ast
import unicodedata
from app.src.config import GARANTIA_APLICA, GARANTIA_NO_APLICA, CATEGORY_COLS, NUMERIC_COLS, BOOL_COLS, DROP_COLUMNS, \
    SELECTED_COLUMNS


class Preprocessing:
    def __init__(self):
        # Definiciones estáticas que se usan en el procesamiento
        self.category_cols = CATEGORY_COLS
        self.numeric_cols = NUMERIC_COLS
        self.bool_cols = BOOL_COLS
        self.drop_columns = DROP_COLUMNS
        self.selected_columns = SELECTED_COLUMNS

        # Regex y keywords para clasificar la garantía
        self.patron_tiempo = re.compile(r'(\d+)\s*(dias|día|mes|meses|año|años|semanas|semana|ano)', re.IGNORECASE)
        self.keywords_aplica = GARANTIA_APLICA
        self.keywords_no_aplica = GARANTIA_NO_APLICA

    def normalizar_texto(self, texto):
        if isinstance(texto, str):
            texto = texto.lower()
            texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
            texto = re.sub(r'[^\w\s]', '', texto)
        return texto

    def clasificar_garantia(self, texto):
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
        Convierte columnas en tipo 'category'.
        """
        for col in self.category_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    def convert_to_numeric(self, df):
        """
        Convierte columnas en tipo 'float' (o 'int' si es posible).
        """
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convierte y pone NaN en errores

        return df

    def convert_to_bool(self, df):
        """
        Convierte columnas en tipo 'float' (o 'int' si es posible).
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
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception as e:
                print("Error parsing:", val)
                return None
        return val 

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Crear columnas desde campos anidados si existen
        if 'seller_address' in data.columns:
            data['seller_address'] = data['seller_address'].apply(self.safe_parse)
            data['seller_country'] = data['seller_address'].apply(lambda x: x.get('country', {}).get('name'))
            data['seller_state'] = data['seller_address'].apply(lambda x: x.get('state', {}).get('name'))
            data['seller_city'] = data['seller_address'].apply(lambda x: x.get('city', {}).get('name'))

        if 'shipping' in data.columns:
            data['shipping'] = data['shipping'].apply(self.safe_parse)
            data['shipping_local_pick_up'] = data['shipping'].apply(lambda x: x.get('local_pick_up'))
            data['shipping_free_shipping'] = data['shipping'].apply(lambda x: x.get('free_shipping'))
            data['shipping_tags'] = data['shipping'].apply(lambda x: x.get('tags'))
            data['shipping_mode'] = data['shipping'].apply(lambda x: x.get('mode'))

        if 'non_mercado_pago_payment_methods' in data.columns:
            payment_df = data["non_mercado_pago_payment_methods"].apply(self.extract_payment_methods).apply(pd.Series,
                                                                                                            dtype='bool').fillna(
                False)
            data = pd.concat([data, payment_df], axis=1)

        if 'tags' in data.columns:
            tags_one_hot = data['tags'].str.join(',').str.get_dummies(sep=',')
            data = pd.concat([data, tags_one_hot], axis=1)

        if 'warranty' in data.columns:
            data["garantia_aplica"] = data["warranty"].apply(self.clasificar_garantia)

        # Rellenar valores
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

        if 'seller_city' in data.columns:
            data['seller_city'] = data['seller_city'].fillna(data['seller_city'].mode()[0])
        if 'seller_state' in data.columns:
            data['seller_state'] = data['seller_state'].fillna(data['seller_state'].mode()[0])

        # MAYBE THIS CAN
        if 'sold_quantity' in data.columns:
            data['sold_quantity'] = data['sold_quantity'].fillna(0)  # Fill NaN values with 0
        if 'poor_quality_thumbnail' in data.columns:
            data['poor_quality_thumbnail'] = data['poor_quality_thumbnail'].fillna(0)  # Fill NaN values with 0
        if 'free_relist' in data.columns:
            data['free_relist'] = data['free_relist'].fillna(0)  # Fill NaN values with 0
        if 'dragged_visits' in data.columns:
            data['dragged_visits'] = data['dragged_visits'].fillna(0)  # Fill NaN values with 0
        if 'good_quality_thumbnail' in data.columns:
            data['good_quality_thumbnail'] = data['good_quality_thumbnail'].fillna(0)  # Fill NaN values with 0
        if 'dragged_bids_and_visits' in data.columns:
            data['dragged_bids_and_visits'] = data['dragged_bids_and_visits'].fillna(0)  # Fill NaN values with 0

        if 'transferencia_bancaria' in data.columns:
            data['transferencia_bancaria'] = data['transferencia_bancaria'].fillna(False)
        if 'efectivo' in data.columns:
            data['efectivo'] = data['efectivo'].fillna(False)
        if 'shipping_local_pick_up' in data.columns:
            data['shipping_local_pick_up'] = data['shipping_local_pick_up'].fillna(False)

        if 'cheque_certificado' in data.columns:
            data['cheque_certificado'] = data['cheque_certificado'].fillna(False)
        if 'contra_reembolso' in data.columns:
            data['contra_reembolso'] = data['contra_reembolso'].fillna(False)
        if 'acordar_con_el_comprador' in data.columns:
            data['acordar_con_el_comprador'] = data['acordar_con_el_comprador'].fillna(False)
        if 'automatic_relist' in data.columns:
            data['automatic_relist'] = data['automatic_relist'].fillna(False)
        if 'giro_postal' in data.columns:
            data['giro_postal'] = data['giro_postal'].fillna(False)
        if 'shipping_free_shipping' in data.columns:
            data['shipping_free_shipping'] = data['shipping_free_shipping'].fillna(False)

        # Variables temporales
        if 'start_time' in data.columns and 'stop_time' in data.columns:
            data['year_start'] = pd.to_datetime(data['start_time'], unit='ms', errors='coerce').dt.year.astype('category')
            data['month_start'] = pd.to_datetime(data['start_time'], unit='ms', errors='coerce').dt.month.astype('category')
            data['year_stop'] = pd.to_datetime(data['stop_time'], unit='ms', errors='coerce').dt.year.astype('category')
            data['month_stop'] = pd.to_datetime(data['stop_time'], unit='ms', errors='coerce').dt.month.astype('category')
            data['week_day'] = pd.to_datetime(data['stop_time'], unit='ms', errors='coerce').dt.weekday.astype('category')
            data['days_active'] = (pd.to_datetime(data['stop_time'], unit='ms', errors='coerce') - pd.to_datetime(data['start_time'], unit='ms', errors='coerce')).dt.days

        df = data.copy()
        existing_cols_to_drop = [col for col in self.drop_columns if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)

        # Tipos de dato
        df = self.convert_to_category(df)
        df = self.convert_to_numeric(df)
        df = self.convert_to_bool(df)

        df = self.ensure_columns_exist(df)

        # Selección de columnas finales
        df = df[self.selected_columns]

        cat_vars = list(df.select_dtypes(include=['category']).columns)
        df[cat_vars] = df[cat_vars].apply(LabelEncoder().fit_transform)

        return df