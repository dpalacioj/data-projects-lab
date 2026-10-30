---
title: "03 · Modelos y umbral"
parent: "S03 · Procesamiento, pipelines, modelos"
grand_parent: Sesiones
nav_order: 3
---
# Modelos y umbral de decisión

> Fecha de referencia: **2026-10-19**. Cuaderno: [`09_modelos_y_pipelines.ipynb`](../../projects/00-segmentacion-clientes/notebooks/09_modelos_y_pipelines.ipynb). Código: [`src/segmentacion/models/entrenar.py`](../../projects/00-segmentacion-clientes/src/segmentacion/models/entrenar.py) + [`evaluation/{metricas,umbral}.py`](../../projects/00-segmentacion-clientes/src/segmentacion/evaluation/).

## La pregunta

**¿Qué modelo colocarías en producción y por qué? ¿Y cuál umbral?** La
decisión no es «el que tenga mayor AUC».

## Punto de partida: el dolor

El reflejo típico:

```python
for name, model in models.items():
    scores = cross_val_score(model, X, y, scoring="roc_auc")
    print(name, scores.mean())
# [el que da mayor número, gana]
```

Dos problemas:

1. **El AUC solo rankinga; no decide.** Un modelo con AUC 0,82 puede
   dar precisión 0,42 al umbral 0,5. Si marketing contacta 2 000
   clientes, 1 160 son falsos positivos.
2. **La línea base.** Una regla «quien compró hace menos de 90 días
   comprará otra vez» da F1 **0,629**. Si tu modelo supera esto en 2
   puntos de F1, el costo de mantenerlo rara vez se justifica.

## Idea central

Decidir con **cuatro números a la vez**, no uno:

| Modelo | AUC CV (A) | AUC (B) | AP (B) | F1 @ 0,5 (B) |
|---|---|---|---|---|
| Línea base (regla recencia < 90) | — | — | — | **0,629** |
| Clase mayoritaria | — | — | — | **0,000** (acc 0,573) |
| Regresión logística | 0,817 ± 0,013 | 0,795 | 0,758 | **0,574** |
| Árbol (profundidad 4) | 0,800 ± 0,015 | 0,780 | 0,723 | **0,595** |
| Bosque (300 árboles, hoja ≥ 5) | 0,816 ± 0,009 | 0,785 | 0,755 | **0,578** |

**El bosque gana en AUC, pero el árbol gana en F1 al umbral 0,5.** La
regla gana en F1 absoluto, pero no ranquea nada (sin AUC).

Interpretación honesta: la regresión logística es el mejor equilibrio
AUC / interpretabilidad / costo. El bosque suma complejidad por un AUC
casi igual. El árbol sobreajusta más allá de profundidad 4.

## Teoría

### Los tres modelos del curso

```python
MODELOS = {
    "Regresión logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árbol de decisión":   DecisionTreeClassifier(max_depth=4, random_state=42),
    "Bosque aleatorio":    RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
}
```

Elegidos deliberadamente:

- **Logreg** = la familia lineal, interpretable, de referencia.
- **Árbol profundidad 4** = la unidad didáctica para enseñar
  sobreajuste.
- **Bosque 300 / hoja ≥ 5** = un ensemble razonable, no obscuro ni
  overfit-prone.

### Validación cruzada: `StratifiedKFold(5, shuffle=True, random_state=42)`

- 5 folds porque A tiene ~5 000 clientes; menos folds subestiman la
  varianza del AUC.
- `shuffle=True` + `random_state=42` para que el ranking de clientes
  no sesgue los folds.
- Estratificada porque la tasa positiva en A (32,9 %) no es balanceada.

### La curva de profundidad del árbol

Para `max_depth ∈ {1, 2, 3, 4, 6, 8, 12, None}`:

| Profundidad | AUC (A) | AUC (B) |
|---|---|---|
| 1 | 0,696 | 0,676 |
| 2 | 0,774 | 0,745 |
| 3 | 0,806 | 0,769 |
| **4** | **0,821** | **0,780** |
| 6 | 0,848 | 0,768 |
| 8 | 0,886 | 0,744 |
| 12 | 0,964 | 0,698 |
| sin límite | 1,000 | 0,656 |

Lectura canónica del sobreajuste: `AUC(A)` crece monótonamente;
`AUC(B)` tiene un **máximo en profundidad 4** y después cae.

## Práctica

### La tabla de umbrales

El umbral 0,5 es un default, no una decisión. La decisión se toma con
la **tabla de umbrales** del bosque:

| Umbral | n_señalados | Precision | Recall | F1 |
|---|---|---|---|---|
| 0,3 | 2 093 | 0,688 | 0,643 | **0,665** |
| 0,4 | 1 653 | 0,751 | 0,554 | 0,638 |
| 0,5 | 1 259 | 0,803 | 0,451 | 0,578 |
| 0,6 | 890 | 0,856 | 0,340 | 0,487 |

**El umbral se elige con marketing**, no con el científico de datos.
Preguntas típicas:

- ¿Cuántos clientes podemos contactar al mes? → acota `n_señalados`.
- ¿Cuánto nos cuesta un falso positivo (contactar al que no iba a
  comprar)? → ponderamos precision.
- ¿Cuánto nos cuesta un falso negativo (perder al que iba a comprar)?
  → ponderamos recall.

En el curso se acuerda **umbral = 0,3** (más recall, menos precision):
marketing señala 2 093 clientes con F1 = 0,665. El umbral queda
congelado en `metadatos.json`.

### Importancia de variables (bosque)

| Variable | Importancia |
|---|---|
| `recencia` | 0,249 |
| `monetario` | 0,218 |
| `frecuencia` | 0,200 |
| `productos_distintos` | 0,147 |
| `antiguedad` | 0,122 |
| `tasa_devolucion` | 0,058 |
| `pais` (UK + otro) | 0,003 |

`pais` es prácticamente ruido en este dataset (91 % UK). Si futurizas a
un retailer más distribuido, revisitarías.

## Verificación

```bash
# Reproduce las métricas del §9
uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_recompra.py

# Reproduce la tabla de umbrales
uv run pytest -m slow -k umbrales projects/00-segmentacion-clientes/tests/test_cifras_recompra.py

# Reproduce la curva de profundidad
uv run pytest -m slow -k profundidad projects/00-segmentacion-clientes/tests/test_cifras_recompra.py
```

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| Elegir modelo por un solo número (AUC) | Decides sin ver recall / costo | Ver AUC + AP + F1 + tabla de umbrales |
| Ignorar la línea base | El modelo puede no superarla | Siempre reporta la regla + mayoría |
| Fijar el umbral en 0,5 por default | No es una decisión de negocio | Decidir con marketing; dejar en `metadatos.json` |
| Entrenar el bosque sin seed | No reproducible | `random_state=SEMILLA` |

## Si falla

- «Mis métricas del bosque difieren en 0,003»: ver [`07-troubleshooting.md`](07-troubleshooting.md)
  §«4 cifras del bosque difieren».
- «El árbol no da curva en forma de U»: puede ser una semilla distinta.
  Comprueba `SEMILLA = 42` en `config.py`.

## Siguiente

- Próximo tema: [`04-artefacto-y-registro.md`](04-artefacto-y-registro.md).
- Cuaderno asociado: [`09_modelos_y_pipelines.ipynb`](../../projects/00-segmentacion-clientes/notebooks/09_modelos_y_pipelines.ipynb).
