# SPEC — `features/texto_llm.py`: un modelo de lenguaje como transformador de características (agnóstico al proveedor)

**Repositorio:** `dpalacioj/data-projects-lab` · **Carpeta:** `projects/00-segmentacion-clientes/src/segmentacion/features/`
**Unidad que lo usa:** Unidad 3, subtema 3.2 (Figura 9 y recuadros del OVA; pódcast 2)
**Propósito:** implementar el ejemplo del OVA como un módulo real, pequeño y probado, que categoriza las 5 305 descripciones únicas de producto de Online Retail II con un LLM y produce una variable por cliente. El módulo sirve para **enseñar buenas prácticas** (caché, esquema cerrado, inyección de dependencias, línea base, pruebas sin red), no para presumir del modelo.
**Proveedor:** agnóstico. David lo correrá con DeepSeek; el código no debe nombrar ningún proveedor fuera de un único adaptador opcional.
**Estado:** especificación; nada implementado.
**Orden de implementación:** parte de **2 de 3** (`SPEC_procesamiento_U3.md`). Se implementa después de `pipelines/preprocesamiento.py` y antes del cuaderno 09. Esta spec es la fuente de verdad para el módulo LLM; si `SPEC_procesamiento_U3.md` §5 dice algo distinto, manda esta.

---

## 1. Principios de diseño (los que el OVA enuncia y el código debe cumplir)

| # | Principio | Consecuencia en el código |
|---|---|---|
| 1 | Una llamada por valor único, con caché persistente | Se categorizan descripciones únicas, nunca filas; la caché es un Parquet versionado; segunda ejecución = cero llamadas |
| 2 | Salida validada contra un esquema cerrado | Pydantic con `Literal` de 12 categorías; lo inválido → `otro` + registro |
| 3 | Proveedor intercambiable (inyección de dependencias) | La función recibe `cliente: Callable[[str], str]`; existe `FakeLLM` para pruebas y cuaderno |
| 4 | Evaluación frente a una línea base | 200 descripciones etiquetadas a mano; exactitud LLM vs. reglas por palabras clave |
| 5 | Sin fuga de datos | La categoría depende solo del texto del producto; no recibe cliente, fecha ni etiqueta |
| 6 | Sin datos personales hacia fuera | Solo se envían descripciones de producto (CC BY 4.0); se documenta explícitamente |
| 7 | Determinismo del resto del pipeline | `temperature=0` donde el proveedor lo permita; la caché hace que el pipeline aguas abajo sea reproducible aunque el LLM no lo sea |

## 2. Interfaz pública

```python
# src/segmentacion/features/texto_llm.py
from typing import Callable, Literal, Protocol, get_args
from pathlib import Path
import polars as pl
from pydantic import BaseModel, Field

CATEGORIAS = Literal["iluminacion", "cocina", "textil", "papeleria", "decoracion", "bolsas",
                     "juguetes", "jardin", "navidad", "accesorios", "cargo_no_producto", "otro"]   # 12, como dice el OVA
CATEGORIAS_LISTA: list[str] = list(get_args(CATEGORIAS))

class Categoria(BaseModel):
    descripcion: str
    categoria: CATEGORIAS
    confianza: float = Field(ge=0, le=1)

class ClienteLLM(Protocol):
    def __call__(self, prompt: str) -> str: ...        # texto → texto (JSON como cadena)

def categorizar_descripciones(descripciones: pl.Series, *, cache: Path, cliente: ClienteLLM,
                              lote: int = 50, reintentos: int = 2) -> pl.DataFrame:
    """Devuelve un DataFrame con columnas: Description, categoria, confianza, fuente ('cache'|'llm'|'invalido').
    - Lee `cache` si existe y solo consulta las descripciones ausentes.
    - Agrupa en lotes de `lote` descripciones por prompt.
    - Valida cada elemento con `Categoria`; lo que no valida → categoria='otro', fuente='invalido'.
    - Escribe la caché actualizada antes de devolver (idempotente)."""

def proporcion_gasto_por_categoria(ventas: pl.DataFrame, categorias: pl.DataFrame, corte) -> pl.DataFrame:
    """Por cliente y antes de `corte`: proporción del Importe en cada categoría (12 columnas que suman 1)."""

def regla_palabras_clave(descripciones: pl.Series) -> pl.DataFrame:
    """Línea base sin LLM: diccionario de palabras clave → categoría; lo no cubierto → 'otro'."""

def evaluar(categorias: pl.DataFrame, gold: pl.DataFrame) -> dict:
    """Exactitud y matriz de confusión frente a `tests/fixtures/categorias_gold.csv` (200 filas)."""
```

### Adaptadores (en `src/segmentacion/features/llm_clientes.py`)

```python
class FakeLLM:
    """Responde desde tests/fixtures/respuestas_fake.json; sin red. Es el cliente por defecto en cuadernos y tests."""
    def __init__(self, ruta: Path): ...
    def __call__(self, prompt: str) -> str: ...

def cliente_openai_compatible(base_url: str, api_key_env: str, modelo: str, temperature: float = 0.0) -> ClienteLLM:
    """Único adaptador real. Cubre cualquier servicio con API compatible con el formato de chat de OpenAI
    (DeepSeek, OpenAI, Groq, Ollama local, etc.). La clave se lee de la variable de entorno `api_key_env`;
    nunca se escribe en código ni en el repositorio (.env.example documenta el nombre)."""
```

El cuaderno 08 muestra las dos líneas que cambian para pasar del falso al real:

```python
cliente = FakeLLM(Path("tests/fixtures/respuestas_fake.json"))                       # por defecto
# cliente = cliente_openai_compatible(base_url=os.environ["LLM_BASE_URL"], api_key_env="LLM_API_KEY", modelo=os.environ["LLM_MODELO"])
```

## 3. Prompt y contrato de salida

- Un prompt por lote de hasta 50 descripciones, con instrucción de responder **solo** un arreglo JSON de objetos `{descripcion, categoria, confianza}` usando exactamente las 12 etiquetas (en minúsculas, sin tildes).
- El prompt vive en `prompts/categorizar_v1.md` (versionado); el hash del prompt entra en la caché para invalidarla si cambia.
- Parsing tolerante: extraer el primer arreglo JSON de la respuesta; si falla el lote completo, reintentar hasta `reintentos` y luego marcar el lote como `invalido`.

## 4. Caché

| Aspecto | Decisión |
|---|---|
| Ubicación | `data/interim/categorias_llm.parquet` (≈ 5 305 filas, < 200 KB) |
| Columnas | `Description`, `categoria`, `confianza`, `fuente`, `prompt_hash`, `modelo`, `fecha` |
| Versionado | **Sí** se commitea (es pequeño y es lo que hace reproducible el pipeline sin red) |
| Invalidación | Cambio de `prompt_hash` o flag `--recalcular` |

## 5. Evaluación y línea base

- `tests/fixtures/categorias_gold.csv`: 200 descripciones muestreadas con `seed=42` y etiquetadas a mano por David (esfuerzo estimado: 40 minutos). Es el único paso manual y queda documentado.
- `regla_palabras_clave`: diccionario de ~60 palabras (LIGHT, CANDLE, LANTERN → iluminación; MUG, BOWL, JAR → cocina; BAG → bolsas; POST, DOT, M, BANK CHARGES, ADJUST → cargo_no_producto; CHRISTMAS, XMAS → navidad; NECKLACE, EARRINGS, BRACELET, KEYRING → accesorios…).
- Reporte en el cuaderno 08: exactitud y matriz de confusión de ambos; criterio de adopción escrito antes de ver el resultado: «el LLM se adopta si supera a la regla en ≥ 10 puntos de exactitud sobre el gold».
- Costo documentado: n.º de llamadas (≈ 107 lotes), tokens aproximados, tiempo, con fecha.

## 6. Dónde entra en el pipeline

```python
# pipelines/preprocesamiento.py
PROP_CAT = [f"prop_{c}" for c in CATEGORIAS_LISTA]          # 12 columnas en [0, 1]
ColumnTransformer([..., ("prop", "passthrough", PROP_CAT)])  # ya están acotadas; no se escalan
```

- Se compara el AUC del bosque **con y sin** las 12 columnas, en la cohorte B, una sola vez. El OVA no promete que mejore; el cuaderno reporta lo que dé.

## 7. Pruebas (todas sin red)

| Prueba | Qué verifica |
|---|---|
| `test_categorizar_usa_cache` | Segunda llamada con la misma caché → `cliente` no se invoca |
| `test_salida_invalida_cae_en_otro` | Un `FakeLLM` que devuelve una etiqueta inexistente → `otro` + `fuente='invalido'` |
| `test_json_malformado_reintenta` | Respuesta no parseable → reintento → luego `invalido` |
| `test_proporciones_suman_uno` | Por cliente, las 12 columnas suman 1 (± 1e-9) |
| `test_sin_fuga` | La función no acepta ni usa columnas de cliente, fecha o etiqueta (firma y prueba de independencia del corte) |
| `test_regla_vs_gold` | La regla se evalúa sobre el gold y produce una exactitud (umbral mínimo documentado) |

## 8. Criterios de aceptación

- [ ] `pydantic>=2` declarado en el extra `segmentacion-clientes` (el extra `meli` fija `pydantic>=1.8`; no reutilizarlo).
- [ ] `uv run pytest tests/features/test_texto_llm.py` pasa sin variables de entorno ni red.
- [ ] El cuaderno 08 corre de principio a fin con `FakeLLM` y, si existen `LLM_BASE_URL`/`LLM_API_KEY`, con el cliente real; la caché resultante se commitea.
- [ ] `.env.example` documenta `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODELO`; `.gitignore` excluye `.env`.
- [ ] El README del proyecto tiene una sección «Uso de modelos de lenguaje» con: qué se envía (solo descripciones de producto), qué no (datos de clientes), costo de la última corrida y cómo reproducir sin red.
- [ ] Ningún archivo del repositorio nombra un proveedor salvo el adaptador y el README (como ejemplo de configuración).

## 9. Riesgos

| Riesgo | Mitigación |
|---|---|
| El LLM cambia de versión y las categorías cambian | La caché versionada congela el resultado; recalcular es una decisión explícita con `--recalcular` |
| Clave de API en el repositorio | Pre-commit `detect-private-key` + revisión; variable de entorno obligatoria |
| El ejemplo desplaza el foco de la unidad | Un subtema, un cuaderno, evaluación obligatoria contra la línea base; el OVA ya lo encuadra como «un transformador más» |
