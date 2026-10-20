"""Un LLM como transformador de caracteristicas de producto (subtema 3.2 del OVA).

Firmas citadas por `SPEC_llm_transformador.md`:

    categorizar_descripciones(descripciones, *, cache, cliente, lote=50, reintentos=2) -> pl.DataFrame
    proporcion_gasto_por_categoria(ventas, categorias, corte) -> pl.DataFrame
    regla_palabras_clave(descripciones) -> pl.DataFrame
    evaluar(categorias, gold) -> dict

Principios (seccion 1 de la spec LLM) que el modulo materializa:

1. Una llamada por valor unico, con cache persistente (Parquet versionado).
2. Salida validada contra 12 categorias cerradas (Pydantic + Literal).
3. Proveedor intercambiable (ClienteLLM es un Protocol).
4. Linea base por palabras clave para evaluar la ganancia del LLM.
5. Sin fuga: no se pasan datos de cliente, fecha ni etiqueta.
6. Determinismo aguas abajo gracias a la cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, get_args

import polars as pl
from pydantic import BaseModel, Field, ValidationError

# 12 categorias cerradas (incluye `accesorios`, como pide el OVA y el usuario).
CATEGORIAS = Literal[
    "iluminacion",
    "cocina",
    "textil",
    "papeleria",
    "decoracion",
    "bolsas",
    "juguetes",
    "jardin",
    "navidad",
    "accesorios",
    "cargo_no_producto",
    "otro",
]
CATEGORIAS_LISTA: list[str] = list(get_args(CATEGORIAS))

PROMPT_V1_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "categorizar_v1.md"
)


class Categoria(BaseModel):
    """Esquema cerrado para una descripcion categorizada."""

    descripcion: str
    categoria: CATEGORIAS
    confianza: float = Field(ge=0.0, le=1.0)


class ClienteLLM(Protocol):
    """Protocolo del cliente LLM: texto → texto (JSON como cadena)."""

    def __call__(self, prompt: str) -> str: ...


def _hash_prompt() -> str:
    if not PROMPT_V1_PATH.exists():
        return "no_prompt"
    return hashlib.sha256(PROMPT_V1_PATH.read_bytes()).hexdigest()[:16]


def _construir_prompt(descripciones_lote: list[str]) -> str:
    if not PROMPT_V1_PATH.exists():
        plantilla = "Clasifica: {descripciones}"
    else:
        plantilla = PROMPT_V1_PATH.read_text()
    lista = "\n".join(f"- {d}" for d in descripciones_lote)
    return plantilla.replace("{descripciones}", lista)


def _extraer_json(texto: str) -> list[dict] | None:
    """Extrae el primer arreglo JSON de `texto`; devuelve None si falla."""
    m = re.search(r"\[.*\]", texto, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def _llamar_al_cliente_y_validar(
    cliente: ClienteLLM,
    descripciones_lote: list[str],
    reintentos: int,
) -> list[dict] | None:
    prompt = _construir_prompt(descripciones_lote)
    for _ in range(reintentos + 1):
        respuesta = cliente(prompt)
        parsed = _extraer_json(respuesta)
        if parsed is not None:
            return parsed
    return None


def categorizar_descripciones(
    descripciones: pl.Series,
    *,
    cache: Path,
    cliente: ClienteLLM,
    lote: int = 50,
    reintentos: int = 2,
    modelo: str = "fake",
) -> pl.DataFrame:
    """Devuelve `pl.DataFrame` con `Description`, `categoria`, `confianza`, `fuente`.

    - Lee `cache` (Parquet) si existe y solo consulta las descripciones ausentes.
    - Agrupa en lotes de `lote` descripciones por prompt.
    - Valida cada elemento con `Categoria`; invalidos → `categoria='otro'`,
      `fuente='invalido'`.
    - Escribe la cache actualizada antes de devolver (idempotente).
    - `fuente ∈ {'cache', 'llm', 'invalido'}`.
    """
    descripciones_unicas = descripciones.unique().drop_nulls().to_list()
    prompt_hash = _hash_prompt()

    if cache.exists():
        cacheado = pl.read_parquet(cache)
        descripciones_cache = set(cacheado["Description"].to_list())
    else:
        cacheado = pl.DataFrame(
            schema={
                "Description": pl.Utf8,
                "categoria": pl.Utf8,
                "confianza": pl.Float64,
                "fuente": pl.Utf8,
                "prompt_hash": pl.Utf8,
                "modelo": pl.Utf8,
                "fecha": pl.Datetime("us"),
            }
        )
        descripciones_cache = set()

    faltantes = [d for d in descripciones_unicas if d not in descripciones_cache]

    nuevas: list[dict] = []
    for i in range(0, len(faltantes), lote):
        descripciones_lote = faltantes[i : i + lote]
        parsed = _llamar_al_cliente_y_validar(cliente, descripciones_lote, reintentos)
        if parsed is None:
            for d in descripciones_lote:
                nuevas.append(
                    {
                        "Description": d,
                        "categoria": "otro",
                        "confianza": 0.0,
                        "fuente": "invalido",
                        "prompt_hash": prompt_hash,
                        "modelo": modelo,
                        "fecha": datetime.now(),
                    }
                )
            continue
        parsed_by_desc = {str(e.get("descripcion", "")): e for e in parsed}
        for d in descripciones_lote:
            if d not in parsed_by_desc:
                nuevas.append(
                    {
                        "Description": d,
                        "categoria": "otro",
                        "confianza": 0.0,
                        "fuente": "invalido",
                        "prompt_hash": prompt_hash,
                        "modelo": modelo,
                        "fecha": datetime.now(),
                    }
                )
                continue
            try:
                cat = Categoria(**parsed_by_desc[d])
                nuevas.append(
                    {
                        "Description": d,
                        "categoria": cat.categoria,
                        "confianza": cat.confianza,
                        "fuente": "llm",
                        "prompt_hash": prompt_hash,
                        "modelo": modelo,
                        "fecha": datetime.now(),
                    }
                )
            except ValidationError:
                nuevas.append(
                    {
                        "Description": d,
                        "categoria": "otro",
                        "confianza": 0.0,
                        "fuente": "invalido",
                        "prompt_hash": prompt_hash,
                        "modelo": modelo,
                        "fecha": datetime.now(),
                    }
                )

    if nuevas:
        cacheado = pl.concat([cacheado, pl.DataFrame(nuevas)], how="vertical_relaxed")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cacheado.write_parquet(cache)

    return cacheado.select("Description", "categoria", "confianza", "fuente")


def proporcion_gasto_por_categoria(
    ventas: pl.DataFrame,
    categorias: pl.DataFrame,
    corte: datetime,
) -> pl.DataFrame:
    """Por cliente antes de `corte`: proporcion del Importe en cada categoria.

    Devuelve `Customer ID` + 12 columnas `prop_<cat>` que suman 1 (± 1e-9).
    Un cliente sin compras antes del corte queda fuera del DataFrame.
    """
    ventas_hist = ventas.filter(pl.col("InvoiceDate") < corte).join(
        categorias.select("Description", "categoria"),
        on="Description",
        how="left",
    )
    ventas_hist = ventas_hist.with_columns(pl.col("categoria").fill_null("otro"))

    gasto = ventas_hist.group_by("Customer ID", "categoria").agg(
        pl.col("Importe").sum().alias("gasto")
    )
    pivot = gasto.pivot(
        index="Customer ID",
        on="categoria",
        values="gasto",
        aggregate_function="sum",
    ).fill_null(0.0)

    for cat in CATEGORIAS_LISTA:
        if cat not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0.0).alias(cat))

    cols_cat = [c for c in CATEGORIAS_LISTA]
    pivot = pivot.select("Customer ID", *cols_cat)
    total = pivot.select(cols_cat).sum_horizontal()
    for cat in cols_cat:
        pivot = pivot.with_columns(
            pl.when(total > 0).then(pl.col(cat) / total).otherwise(0.0).alias(f"prop_{cat}")
        )
    return pivot.select("Customer ID", *[f"prop_{c}" for c in cols_cat])


# Diccionario de palabras clave para la linea base.
# ORDEN importa: la primera categoria con un match gana. Primero las etiquetas
# con alta prioridad semantica (navidad gana a decoracion para "CHRISTMAS WREATH";
# cargo_no_producto va antes que todo para capturar POSTAGE, bank charges, etc.).
PALABRAS_CLAVE: dict[str, list[str]] = {
    "cargo_no_producto": [
        "POSTAGE",
        "DOTCOM",
        "BANK CHARGES",
        "ADJUST",
        "MANUAL",
        "AMAZON FEE",
    ],
    "navidad": ["CHRISTMAS", "XMAS", "ADVENT"],
    "iluminacion": ["LIGHT", "CANDLE", "LANTERN", "LAMP", "CANDLES"],
    "cocina": ["MUG", "BOWL", "JAR", "CUP", "KITCHEN", "APRON", "TEAPOT"],
    "textil": ["TOWEL", "CUSHION", "BLANKET", "CLOTH", "PASHMINA"],
    "papeleria": ["CARD", "NOTEBOOK", "PEN", "PENCIL", "WRAP", "GIFT TAG"],
    "accesorios": ["NECKLACE", "EARRING", "BRACELET", "KEYRING", "RING"],
    "bolsas": ["BAG", "TOTE", "PURSE"],
    "juguetes": ["TOY", "DOLL", "PUZZLE", "BLOCK", "GAME"],
    "jardin": ["GARDEN", "PLANT", "POT", "SPADE"],
    "decoracion": ["ORNAMENT", "FLOWER", "FRAME", "WREATH", "VASE"],
}


def regla_palabras_clave(descripciones: pl.Series) -> pl.DataFrame:
    """Linea base sin LLM: diccionario de palabras clave; sin match → `otro`."""
    out = []
    for d in descripciones.to_list():
        if d is None:
            out.append({"Description": None, "categoria": "otro"})
            continue
        texto = d.upper()
        cat_asignada: str | None = None
        for cat, palabras in PALABRAS_CLAVE.items():
            if any(p in texto for p in palabras):
                cat_asignada = cat
                break
        out.append({"Description": d, "categoria": cat_asignada or "otro"})
    return pl.DataFrame(out)


def evaluar(categorias: pl.DataFrame, gold: pl.DataFrame) -> dict:
    """Exactitud + matriz de confusion de `categorias` frente a `gold`.

    `gold` debe tener columnas `Description` y `categoria`. Se emparejan
    por `Description`; filas sin match se ignoran.
    """
    comparacion = gold.join(
        categorias.select("Description", pl.col("categoria").alias("predicha")),
        on="Description",
        how="inner",
    )
    if comparacion.height == 0:
        return {"exactitud": 0.0, "n": 0, "matriz": {}}
    aciertos = comparacion.filter(pl.col("categoria") == pl.col("predicha")).height
    exactitud = aciertos / comparacion.height

    matriz: dict[str, dict[str, int]] = {}
    for fila in comparacion.iter_rows(named=True):
        gold_cat = fila["categoria"]
        pred_cat = fila["predicha"]
        matriz.setdefault(gold_cat, {})
        matriz[gold_cat][pred_cat] = matriz[gold_cat].get(pred_cat, 0) + 1

    return {"exactitud": exactitud, "n": comparacion.height, "matriz": matriz}
