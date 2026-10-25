# Archivos de Prueba para Streamlit UI

Este directorio contiene archivos de prueba para probar la UI de Streamlit del Clasificador MercadoLibre.

## 📁 Archivos Disponibles

### 1. `test_products.jsonlines` (15 KB)
**Uso:** Opción "Subir JSON" en Streamlit
**Formato:** JSONLINES (1 producto por línea)
**Productos:** 5 productos con estructura completa del dataset original

```bash
# Estructura: Un JSON por línea
{"seller_address": {...}, "warranty": null, "condition": "new", ...}
{"seller_address": {...}, "warranty": "6 meses", "condition": "used", ...}
...
```

### 2. `test_products.json` (19 KB)
**Uso:** Opción "Ingreso Manual" en Streamlit (copiar/pegar)
**Formato:** JSON array estándar
**Productos:** 5 productos en un array JSON

```json
[
  {"seller_address": {...}, "condition": "new", ...},
  {"seller_address": {...}, "condition": "used", ...},
  ...
]
```

### 3. `test_products_simple.csv` (1.2 KB)
**Uso:** Opción "Subir CSV" en Streamlit
**Formato:** CSV con campos básicos (sin nested objects)
**Columnas:** 19 columnas principales

⚠️ **Nota:** Este CSV no incluye campos anidados como `seller_address` o `shipping`, por lo que el preprocesamiento puede no funcionar completamente.

### 4. `test_products_complete.csv` (11 KB)
**Uso:** Opción "Subir CSV" en Streamlit
**Formato:** CSV con TODOS los campos del dataset
**Columnas:** 48 columnas

⚠️ **Nota:** Los campos anidados (dicts) están convertidos a strings. El preprocesamiento intentará parsearlos.

### 5. `test_products_processed.parquet` (13 KB)
**Uso:** Solo para referencia / testing directo
**Formato:** Parquet preprocesado
**Shape:** (5, 23)

Este archivo ya está preprocesado y listo para predicción directa, pero NO es compatible con la UI de Streamlit (que espera datos crudos).

---

## 🚀 Cómo Usar en Streamlit

### Opción 1: Subir JSONLINES (Recomendado)
1. Lanzar Streamlit: `streamlit run ui/streamlit_app.py`
2. Seleccionar: **"Subir JSON"**
3. Cargar archivo: `test_products.jsonlines`
4. ✅ Verás 5 productos procesados y sus predicciones

### Opción 2: Ingreso Manual (JSON)
1. Abrir archivo `test_products.json` en un editor
2. Copiar el contenido completo
3. En Streamlit, seleccionar: **"Ingreso Manual"**
4. Pegar el JSON en el área de texto
5. Click en "Predecir"

### Opción 3: Subir CSV
1. Seleccionar: **"Subir CSV"**
2. Cargar `test_products_complete.csv`
3. ⚠️ Puede tener problemas con campos anidados

---

## 📊 Productos de Prueba

Los 5 productos incluidos son:

| # | Título (aprox) | Condición | Precio |
|---|----------------|-----------|--------|
| 1 | Auriculares Samsung... | new | $80 |
| 2 | Botas Texanas... | used | $2650 |
| 3 | Buzo Rusty Negro... | used | $60 |
| 4 | Cinturón De Cuero... | new | $580 |
| 5 | Camisa De Jean... | used | $30 |

---

## 🔧 Regenerar Archivos de Prueba

Si necesitas regenerar estos archivos:

```bash
source .venv/bin/activate
python << 'EOF'
import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, 'projects/02-clasificador-meli')
from src.features.preprocessing import Preprocessing

# Extraer productos del dataset
dataset_file = Path('datasets/MLA_100k.jsonlines')
test_data_dir = Path('projects/02-clasificador-meli/test_data')

products = []
with open(dataset_file) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        products.append(json.loads(line))

# Crear JSONLINES
with open(test_data_dir / 'test_products.jsonlines', 'w') as f:
    for product in products:
        f.write(json.dumps(product) + '\n')

# Crear JSON
with open(test_data_dir / 'test_products.json', 'w') as f:
    json.dump(products, f, indent=2)

print("✅ Archivos regenerados")
EOF
```

---

## 📝 Notas

- Todos los archivos contienen datos **reales** del dataset MLA_100k
- Las predicciones deberían coincidir con las condiciones reales
- Si encuentras errores, verifica que el modelo esté entrenado: `python train.py`
