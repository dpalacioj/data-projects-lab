---
title: "05 · LLM como transformador"
parent: "S03 · Procesamiento, pipelines, modelos"
grand_parent: Sesiones
nav_order: 5
---
# Un LLM como transformador — con esquema cerrado, caché y línea base

> Fecha de referencia: **2026-10-19**. Cuaderno: [`08_llm_como_transformador.ipynb`](../../projects/00-segmentacion-clientes/notebooks/08_llm_como_transformador.ipynb). Código: [`src/segmentacion/features/{texto_llm,llm_clientes}.py`](../../projects/00-segmentacion-clientes/src/segmentacion/features/).

## La pregunta

**¿Cómo tratas un LLM dentro de un pipeline reproducible?** Las
respuestas que no cuentan: «lo pruebo y veo», «uso el que esté de moda»,
«lo escribo en el notebook y ya». Las respuestas que cuentan: esquema
cerrado, caché, línea base, cliente intercambiable.

## Punto de partida: el dolor

Tres problemas que todo pipeline con LLM arrastra si no se los nombra:

1. **Costo acumulado.** Si no cacheas, cada ejecución llama al proveedor.
   En este dataset son ~107 lotes de 50 descripciones. Un bug que fuerce
   re-ejecutar cuesta tiempo y dinero.
2. **Salida no acotada.** Si el LLM responde `"accesorio"` en vez de
   `"accesorios"`, un `.map(...)` en pandas te dejará nulos silenciosos.
3. **Dependencia del proveedor.** Si el modelo cambia (OpenAI retira
   un endpoint, DeepSeek sube el precio), tu pipeline se rompe sin
   aviso.

Las tres se resuelven con la misma idea: **el LLM es un transformador
más**, con contrato y pruebas, no un servicio mágico.

## Idea central

Tres contratos:

1. **Esquema cerrado**: salida validada con Pydantic + `Literal` de 12
   etiquetas. Cualquier respuesta fuera del vocabulario cae en `otro`
   con `fuente='invalido'`.
2. **Caché versionada**: `data/interim/categorias_llm.parquet` con el
   hash del prompt. Segunda ejecución = cero llamadas. Se **commitea**.
3. **Cliente intercambiable**: `FakeLLM` por defecto (sin red). Un
   único adaptador real (`cliente_openai_compatible`) por variables de
   entorno. **Ningún archivo nombra el proveedor** fuera del adaptador
   y del README.

## Teoría

### Las 12 categorías cerradas

```python
CATEGORIAS = Literal[
    "iluminacion", "cocina", "textil", "papeleria", "decoracion",
    "bolsas", "juguetes", "jardin", "navidad", "accesorios",
    "cargo_no_producto", "otro",
]
```

Elegidas por volumen y accionabilidad para marketing. **12 es el número
acordado del caso guía; no se añade una decimotercera sin re-captura.**

### El esquema de respuesta

```python
class Categoria(BaseModel):
    descripcion: str
    categoria: CATEGORIAS
    confianza: float = Field(ge=0, le=1)
```

Si el LLM devuelve `"iluminacion"` como `"Iluminación"`, Pydantic falla
→ `otro`. Si devuelve una confianza > 1 → `otro`. Si devuelve texto en
lugar de JSON → reintentos (hasta 2 por lote) → `otro`.

### La caché

| Columna | Qué guarda |
|---|---|
| `Description` | Clave; una fila por descripción única |
| `categoria` | Una de las 12 |
| `confianza` | Pydantic la valida en `[0, 1]` |
| `fuente` | `'cache'` / `'llm'` / `'invalido'` |
| `prompt_hash` | SHA-256 del prompt v1 (invalida si cambias el prompt) |
| `modelo` | Identificador del modelo |
| `fecha` | Timestamp de la llamada |

**Reproducibilidad sin red** gracias a la caché: una vez poblada, el
pipeline aguas abajo es determinista sin importar el LLM.

### La línea base

`regla_palabras_clave(descripciones)` recorre un diccionario de ~60
palabras clave con **prioridad explícita**: `navidad` y
`cargo_no_producto` van primero (dominan términos ambiguos como
"CHRISTMAS WREATH" → `navidad`, no `decoracion`).

**Criterio de adopción** escrito antes de ver resultados: el LLM se
adopta solo si supera a la regla en ≥ 10 puntos de exactitud sobre el
gold (200 descripciones etiquetadas a mano, `tests/fixtures/categorias_gold.csv`).

## Práctica

### Cliente por defecto (sin red)

```python
from pathlib import Path
from segmentacion.features.llm_clientes import FakeLLM
from segmentacion.features.texto_llm import categorizar_descripciones

cliente = FakeLLM(Path("tests/fixtures/respuestas_fake.json"))
cache   = Path("data/interim/categorias_llm.parquet")

categorias = categorizar_descripciones(
    ventas["Description"].unique(),
    cache=cache,
    cliente=cliente,
    lote=50,
    reintentos=2,
)
```

### Cliente real (opt-in)

```python
# Descomentar en el cuaderno 08:
# import os
# from segmentacion.features.llm_clientes import cliente_openai_compatible
#
# cliente = cliente_openai_compatible(
#     base_url=os.environ["LLM_BASE_URL"],
#     api_key_env="LLM_API_KEY",
#     modelo=os.environ["LLM_MODELO"],
# )
```

Las tres variables se declaran en `.env` (gitignored). Ver
`.env.example`.

### Comparación bosque con y sin las 12 columnas del LLM

Una vez poblada la caché, en el cuaderno 08:

```python
from segmentacion.features.texto_llm import proporcion_gasto_por_categoria

propA = proporcion_gasto_por_categoria(ventas, categorias, FECHA_CORTE_A)
propB = proporcion_gasto_por_categoria(ventas, categorias, FECHA_CORTE_B)
A_ext = A.join(propA, on="Customer ID", how="left")
B_ext = B.join(propB, on="Customer ID", how="left")

pre_ext = construir_preprocesador(incluir_prop_llm=True)
# Entrena, calcula AUC(B), reporta la diferencia.
```

El curso **no** promete mejora. Se compara **una vez** y se reporta lo
que dé. Si no supera a la regla por el criterio acordado, no entra en
el pipeline final.

## Verificación

```bash
# Todas sin red
uv run pytest projects/00-segmentacion-clientes/tests/features/test_texto_llm.py
uv run pytest projects/00-segmentacion-clientes/tests/features/test_llm_clientes.py
```

Siete pruebas:

- `test_categorizar_usa_cache` — segunda llamada = cero invocaciones.
- `test_salida_invalida_cae_en_otro` — etiqueta no-canon → `otro`.
- `test_json_malformado_reintenta_y_cae_en_invalido` — reintentos y
  `fuente='invalido'`.
- `test_proporciones_suman_uno` — 12 columnas sumadas = 1 ± 1e-9.
- `test_sin_fuga_firma_no_acepta_cliente_fecha_ni_etiqueta` — garantía
  por inspección de signatura.
- `test_regla_palabras_clave_detecta_categorias` — ejemplo con prioridad.
- `test_evaluar_devuelve_exactitud_contra_gold`.

## Auditoría: no se nombra al proveedor

```bash
grep -riE "deepseek|openai|groq|anthropic|ollama" projects/00-segmentacion-clientes \
    --include="*.py" --include="*.ipynb" --include="*.md" \
  | grep -v "cliente_openai_compatible" \
  | grep -v "README.md" \
  | grep -v ".env.example"
```

Debe devolver vacío. `cliente_openai_compatible` es el nombre de la
función del adaptador (término técnico «OpenAI-compatible»); README y
`.env.example` son los documentos donde el proveedor sí se nombra como
ejemplo de configuración.

## Fixture `categorias_gold.csv` — paso manual

El único paso manual del flujo LLM: etiquetar a mano las 200
descripciones del fixture.

```bash
uv run python projects/00-segmentacion-clientes/scripts/generar_gold.py
# Luego: abrir tests/fixtures/categorias_gold.csv y rellenar la columna `categoria`.
```

Tiempo estimado: 40 minutos. Es la **línea base manual** que valida la
regla y, luego, al LLM.

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| Llamar al LLM por cada fila | Costo lineal con las filas | Por valor único (5 305 descripciones, no 779 425 filas) |
| Pasar texto que incluya `Customer ID` | Dato personal al proveedor | Solo descripciones de producto (CC BY 4.0) |
| Hardcodear la URL de un proveedor | Rompe cuando cambias | Variables de entorno; adaptador OpenAI-compatible |
| No tener línea base | No sabes si el LLM añade valor | Regla por palabras clave sobre el gold |
| Confiar en la salida sin validar | Un `"iluminación"` con tilde contamina el pipeline | Pydantic + `Literal` + `fuente='invalido'` |

## Si falla

- `ModuleNotFoundError: fg_data_profiling`: se importa como
  `data_profiling`. Ver [`07-troubleshooting.md`](07-troubleshooting.md).
- «Mi cliente real da `401`»: la variable `LLM_API_KEY` no está
  definida en el entorno activo. Verifica con `env | grep LLM_`.
- «La caché no se usa en la segunda ejecución»: puede ser que el
  `prompt_hash` cambió. Si editaste `prompts/categorizar_v1.md`, la
  caché queda invalidada (comportamiento correcto).

## Siguiente

- [`06-taller.md`](06-taller.md).
- Cuaderno asociado: [`08_llm_como_transformador.ipynb`](../../projects/00-segmentacion-clientes/notebooks/08_llm_como_transformador.ipynb).
