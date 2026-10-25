# Pipeline sin fuga — qué aprende cada bloque

> Fecha de referencia: **2026-10-19**. Spec: [`SPEC_procesamiento_U3.md`](../../docs/specs/SPEC_procesamiento_U3.md) §4 (Tabla 8). Cuaderno: [`07_ingenieria_caracteristicas.ipynb`](../../projects/00-segmentacion-clientes/notebooks/07_ingenieria_caracteristicas.ipynb). Código: [`src/segmentacion/pipelines/preprocesamiento.py`](../../projects/00-segmentacion-clientes/src/segmentacion/pipelines/preprocesamiento.py).

## La pregunta

**Cada transformador tiene parámetros que aprende durante `fit`. ¿Puedes
nombrarlos sin abrir el código, y decir con qué datos los aprendió?**

Si no, hay fuga en potencia.

## Punto de partida: el dolor

Un cuaderno típico:

```python
scaler = StandardScaler()
X = scaler.fit_transform(features)           # ← ¿con qué filas?
X_train, X_test = train_test_split(X)
model.fit(X_train, y_train)
```

El scaler ya vio `X_test` cuando se usa en `model.fit`. **El modelo no
«ve» test, pero el preprocesador ya calibró con test.** Ese scaler es
información del futuro inyectada en el entrenamiento.

La solución es estructural: encapsular preprocesamiento + modelo en
`Pipeline`, y nunca llamar a `fit_transform` sobre los datos
completos.

## Idea central

```python
Pipeline([("pre", construir_preprocesador()), ("mod", MODELOS["Bosque aleatorio"])])
    .fit(X_A, y_A)
    .predict_proba(X_B)[:, 1]
```

- `fit(X_A, y_A)` ajusta **todo** (scaler, OHE, bosque) con A.
- `predict_proba(X_B)` aplica lo aprendido a B.
- No hay forma de que un transformador vea B durante el entrenamiento.

## Teoría

`construir_preprocesador()` devuelve un `ColumnTransformer` con tres
bloques:

```python
NUM_LOG = ["monetario", "frecuencia", "productos_distintos"]
NUM_STD = ["recencia", "antiguedad", "tasa_devolucion"]
CAT     = ["pais"]
```

### Bloque `NUM_LOG`

`SimpleImputer(median) → FunctionTransformer(np.log1p) → StandardScaler`

- Imputa nulos con la mediana (robusta a outliers).
- Comprime la cola larga con `log1p` (que acepta ceros).
- Escala a media 0, varianza 1.

Las tres variables aquí tienen distribución con cola larga confirmada
en U2 (HALLAZGOS.md). Sin `log1p`, el `StandardScaler` queda dominado
por los outliers y la mayoría de los clientes acaba con `z ≈ 0`.

### Bloque `NUM_STD`

`SimpleImputer(median) → StandardScaler`

- `recencia`, `antiguedad` y `tasa_devolucion` tienen distribución
  cercana a uniforme o acotada. `log1p` no aporta.
- La mediana de `tasa_devolucion` es 0 (la mayoría no devuelve); la
  imputación por mediana es segura.

### Bloque `CAT`

`OneHotEncoder(handle_unknown="ignore")` sobre `pais ∈ {UK, otro}`.

- `handle_unknown="ignore"` significa que, si en inferencia aparece un
  país nuevo (p. ej. Nueva Zelanda, no visto en entrenamiento), el
  encoder emite un vector de ceros en vez de levantar excepción.
- Es un contrato con U4: la app no debe caer por un `pais` desconocido.

## Práctica

Inspecciona lo que aprende cada bloque:

```python
from segmentacion.pipelines.preprocesamiento import NUM_LOG, NUM_STD, CAT, construir_preprocesador

pre = construir_preprocesador()
pre.fit(X_A)

# NUM_STD: media y escala del scaler
scaler = pre.named_transformers_["num"].named_steps["sc"]
pd.DataFrame({"columna": NUM_STD, "media": scaler.mean_, "escala": scaler.scale_})

# NUM_LOG: idem pero tras log1p
log_scaler = pre.named_transformers_["log"].named_steps["sc"]
pd.DataFrame({"columna": NUM_LOG, "media_log1p": log_scaler.mean_})

# CAT: categorías aprendidas
ohe = pre.named_transformers_["cat"]
print(ohe.categories_)
```

En el cuaderno 07 se visualiza el efecto del `log1p` sobre `monetario`.

## Verificación

La prueba crítica:

```bash
uv run pytest projects/00-segmentacion-clientes/tests/pipelines/test_sin_fuga.py
```

Construye dos cohortes inventadas con medias de `recencia` **muy
distintas** (100 vs 300). Luego:

- `pipe.fit(A)` → verifica que `scaler.mean_` coincide con `A.mean()`
  (y **no** con `(A + B).mean()`).
- `pipe.fit(A ∪ B).transform(B)` → verifica que la salida es
  **distinta** a `pipe.fit(A).transform(B)`. Si son iguales, el
  diseño de la prueba falló.

## Opción LLM: `incluir_prop_llm`

El preprocesador acepta un flag opcional:

```python
construir_preprocesador(incluir_prop_llm=True)
```

Añade un bloque `prop` con las 12 columnas `prop_<cat>` que produce
`features.texto_llm.proporcion_gasto_por_categoria`. Son proporciones
en [0, 1] y pasan como `passthrough` (no se escalan). Ver
[`llm-transformador.md`](llm-transformador.md).

**Default** = `False`. Las cifras del §9 se calculan sin las 12
columnas; incluirlas requiere la caché del LLM poblada.

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| `StandardScaler().fit_transform(X)` fuera del Pipeline | Fuga estructural | Encapsular en `Pipeline` |
| Imputar con la media antes de partir | Ya viste test | `SimpleImputer` dentro del Pipeline |
| OHE sin `handle_unknown` | Falla en inferencia con categoría nueva | `handle_unknown="ignore"` |
| Pasar `recencia` por `NUM_LOG` «por si acaso» | No tiene cola larga; log1p sobra | Solo `monetario`, `frecuencia`, `productos_distintos` |

## Si falla

- «El scaler aprende la media de A ∪ B»: el `fit` no estaba dentro de
  `Pipeline`. Ver [`troubleshooting.md`](troubleshooting.md).
- «OHE da `ValueError: Found unknown categories`»: olvidaste
  `handle_unknown="ignore"`.

## Siguiente

- Próximo tema: [`modelos-y-umbral.md`](modelos-y-umbral.md).
- Cuaderno asociado: [`07_ingenieria_caracteristicas.ipynb`](../../projects/00-segmentacion-clientes/notebooks/07_ingenieria_caracteristicas.ipynb).
