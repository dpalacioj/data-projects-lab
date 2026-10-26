---
title: "04 · Artefacto y registro"
parent: "S03 · Procesamiento, pipelines, modelos"
grand_parent: Sesiones
nav_order: 4
---
# Artefacto y registro — contratos para U4

> Fecha de referencia: **2026-10-19**. Spec: [`SPEC_procesamiento_U3.md`](../../docs/specs/SPEC_procesamiento_U3.md) §7 (Tabla 7). Código: [`src/segmentacion/models/registro.py`](../../projects/00-segmentacion-clientes/src/segmentacion/models/registro.py).

## La pregunta

**Cuando el modelo pase a la app, ¿qué sabe la app que no está en el
`pipeline.joblib`?** Columnas en orden, umbral acordado, hash de los
datos, versión del motor. Todo eso es **contrato declarativo**.

## Punto de partida: el dolor

Historia habitual: el data scientist entrega un `model.joblib` por
correo. El equipo de app lo levanta, le pasa un `pd.DataFrame`, y:

```
ValueError: X has 7 features, but X was fit with 8 features
```

Nadie documentó que `pais` había que calcularlo antes. Dos días
después, otro error: `AttributeError: 'KMeans' object has no attribute
'_tol'`. Alguien actualizó scikit-learn.

Los dos errores son **de contrato**: lo que la app tiene que saber
nunca estuvo escrito.

## Idea central

Un modelo serializado es un par:

```
models/recompra_v1/
├── pipeline.joblib   # gitignored (binario)
└── metadatos.json    # versionado (humano + CI)
```

**`metadatos.json` es la interfaz pública.** La app U4 lo lee **primero**
y, con eso, sabe qué pasarle a `pipeline.joblib`. Si cambia la versión
de scikit-learn, `cargar()` falla con un mensaje que explica qué pasó.

Los 10 campos obligatorios (Tabla 7 del OVA):

```json
{
  "fecha_corte": "2011-06-01",
  "horizonte": 90,
  "variables_en_orden": [
    "monetario", "frecuencia", "productos_distintos",
    "recencia", "antiguedad", "tasa_devolucion", "pais"
  ],
  "semilla": 42,
  "versiones": {"python": "3.11", "polars": "2.0", "sklearn": "1.9.1"},
  "datos": {
    "ruta": "data/raw/online_retail_II.parquet",
    "hash": "6994df0411eca9ef"
  },
  "auc_validacion": 0.816,
  "umbral_acordado": 0.3,
  "fecha": "2026-10-19",
  "autor": "David Palacio (U3)"
}
```

## Teoría

### Por qué cada campo

| Campo | Para qué sirve |
|---|---|
| `fecha_corte` | U4 valida que no está prediciendo sobre datos pre-corte |
| `horizonte` | La app sabe en qué ventana evaluar el modelo (90 días) |
| `variables_en_orden` | **Orden** importa para el `ColumnTransformer`; la app ensambla el DataFrame así |
| `semilla` | Reproducibilidad de ejercicios posteriores (fine-tuning, análisis) |
| `versiones.sklearn` | `cargar()` rehúsa si no coincide con el entorno |
| `versiones.{python,polars}` | Documentación; no bloqueante |
| `datos.hash` | Verifica integridad del dataset si re-entrenas |
| `auc_validacion` | Línea base para detectar drift (si en producción cae debajo, alerta) |
| `umbral_acordado` | **Decisión de negocio**; U4 lo lee de aquí, no de código |
| `fecha` + `autor` | Trazabilidad |

### El guard de versión

```python
def cargar(ruta: Path) -> tuple[Pipeline, dict]:
    ...
    if metadatos["versiones"]["sklearn"] != sklearn.__version__:
        raise VersionIncompatible(
            "La version de scikit-learn no coincide con la registrada:\n"
            f"  en metadatos.json: {metadatos['versiones']['sklearn']}\n"
            f"  en el entorno    : {sklearn.__version__}\n"
            "Actualiza el entorno (`uv sync`) o re-entrena con la version actual."
        )
```

El mensaje incluye las dos versiones **y** la acción. Nunca es
`"Error loading model"`; siempre dice qué hacer.

### Campos opcionales que agregamos en la práctica

- `metricas_prueba`: AUC, AP y F1 en la cohorte B al cerrar el
  entrenamiento. Permite seguimiento de drift sin re-entrenar.
- `comentario`: una línea con el contexto («primer modelo tras U3»).
- `pipeline_hash`: SHA-256 de `pipeline.joblib`. Doble candado si
  alguien reemplaza el `.joblib` sin tocar el `metadatos.json`.

## Práctica

En el cuaderno 09:

```python
from segmentacion.models.registro import guardar
import hashlib
from datetime import datetime

hash_datos = hashlib.sha256(parquet.read_bytes()).hexdigest()[:16]

metadatos = {
    "fecha_corte": str(FECHA_CORTE_A.date()),
    "horizonte": HORIZONTE,
    "variables_en_orden": NUM_LOG + NUM_STD + CAT,
    "semilla": SEMILLA,
    "versiones": {"python": "3.11", "polars": "2.0"},  # sklearn se autocompleta
    "datos": {"ruta": "data/raw/online_retail_II.parquet", "hash": hash_datos},
    "auc_validacion": round(cv_auc, 3),
    "umbral_acordado": 0.3,
    "fecha": datetime.today().date().isoformat(),
    "autor": "David Palacio (U3)",
}
guardar(pipeline_entrenado, ruta=PROJECT_ROOT / "models" / "recompra_v1", metadatos=metadatos)
```

`guardar()` autocompleta `versiones.sklearn` y verifica que los 10
campos obligatorios estén presentes (`MetadatosIncompletos` si falta
alguno).

## Verificación

```bash
uv run pytest projects/00-segmentacion-clientes/tests/models/test_registro.py
```

Cinco pruebas:

- `test_guardar_y_cargar_hace_roundtrip` — las predicciones coinciden.
- `test_metadatos_json_versionado_con_sklearn_autocompletado` — el
  autocomplete funciona.
- `test_falta_un_campo_obligatorio_lanza_metadatosincompletos` —
  mensaje de error nombra el campo faltante.
- `test_cargar_falla_con_version_sklearn_distinta` — `VersionIncompatible`.
- `test_cargar_falla_con_mensaje_claro_si_no_existe` — `FileNotFoundError`
  con la ruta.

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| Entregar `model.joblib` sin ficha | U4 adivina los contratos | Siempre `metadatos.json` al lado |
| Hardcodear el umbral en la app | Cambios de negocio tocan código | Umbral en `metadatos.json` |
| Guardar el modelo entrenado sobre A ∪ B | Pierdes la cohorte de prueba honesta | `fit(X_A, y_A)`, luego `guardar()` |
| Commitear `pipeline.joblib` | Pesa, los diffs son ilegibles | gitignored (`models/**/*.joblib`) |

## Si falla

- «`cargar()` dice que sklearn no coincide, pero yo instalé con `uv sync`»:
  posible que haya dos entornos. `uv run python -c "import sklearn; print(sklearn.__version__)"`.
- «Mi `metadatos.json` no tiene `sklearn` en `versiones`»: lo autocompleta
  `guardar()`. Si no sale, revisa que `guardar()` sea el del paquete y
  no una copia local.

## Siguiente

- Próximo tema: [`05-llm-transformador.md`](05-llm-transformador.md) (opcional).
- Si saltas el LLM: [`06-taller.md`](06-taller.md).
- Cuaderno asociado: [`09_modelos_y_pipelines.ipynb`](../../projects/00-segmentacion-clientes/notebooks/09_modelos_y_pipelines.ipynb).
