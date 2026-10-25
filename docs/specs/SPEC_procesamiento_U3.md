# SPEC — Procesamiento modular para la Unidad 3 (ingeniería de características, modelos y pipelines con scikit-learn)

**Repositorio:** `dpalacioj/data-projects-lab` · **Carpeta:** `projects/00-segmentacion-clientes/` (continuación de la U2). Referencias de solo lectura: `projects/d2-temporal-split/`, `tutorials/07-tecnicas-preprocesamiento.md`.
**Propósito:** implementar en el repositorio el caso supervisado y la modularización que el OVA de la Unidad 3 ya describe (RAE2.1: pipelines que previenen la fuga de datos y garantizan reproducibilidad), de modo que el videotutorial, el OVA y el taller apunten al mismo código y a las mismas cifras.
**Estado:** decisiones tomadas el 16 de octubre de 2026 (§10). El OVA está redactado contra este diseño. El código del caso existe como dos scripts de referencia (`caso_recompra.py`, `figuras_u3.py`) que hay que descomponer en el paquete sin cambiar los resultados.
**Orden de implementación:** **2 de 3.** Requiere que `SPEC_segmentacion-clientes_polars.md` (U2) esté implementada: usa `segmentacion.data.limpieza`, `features/rfm.py` con fecha de corte, el Parquet y `cancelaciones.parquet`. El subtema LLM (§5) se implementa según `SPEC_llm_transformador.md`, que es la fuente de verdad para ese módulo.
**Ubicación:** esta spec en `docs/specs/`; `caso_recompra.py` y `figuras_u3.py` en `projects/00-segmentacion-clientes/scripts/`.

---

## 1. Qué exige el microcurrículo para la Unidad 3 (RAE2.1)

| Contenido | Mediación | Evidencia |
|---|---|---|
| Ingeniería de características con scikit-learn: codificación, estandarización, gestión de nulos | Clase magistral; cuadernos de Jupyter | Taller en clase |
| Creación de modelos predictivos con scikit-learn | | |
| Implementación de pipelines con scikit-learn | | |

Horas: 8 de interacción y 16 de trabajo independiente. Bibliografía del microcurrículo: Kuhn y Johnson (2019), capítulos 5 (codificación), 6 (predictores numéricos) y 8 (datos faltantes).

## 2. Frontera Unidad 2 → Unidad 3

En la U2 se diagnostica, se limpia (qué filas son ventas) y se explora; **no** se imputa, codifica ni escala. La U3 recibe `HALLAZGOS.md` como lista de instrucciones (Tabla 1 del OVA de la U3):

| Hallazgo registrado en U2 | Qué hace la U3 con él |
|---|---|
| Frecuencia y valor monetario muy asociados; 20 % de clientes = 77 % de las ventas | Variables por cliente hasta una fecha de corte: recencia, antigüedad, frecuencia, monetario, productos distintos |
| Precio, cantidad y valor con cola larga | `FunctionTransformer(np.log1p)` antes de `StandardScaler` para monetario, frecuencia y productos distintos |
| Cancelaciones = 1,8 % de las líneas, conservadas aparte | `tasa_devolucion` = facturas canceladas antes del corte / frecuencia |
| Estacionalidad fuerte | Diseño de **cohortes por fecha de corte**: A (corte 2011-06-01) entrena y valida; B (corte 2011-09-01) prueba. Variables con datos < corte; etiqueta = compró en [corte, corte + 90 días) |
| 22,8 % de líneas sin cliente | No entran al modelo por cliente; la ficha del modelo declara la cobertura |
| Reino Unido = 91,9 % de las líneas | `pais` ∈ {UK, otro} antes de `OneHotEncoder(handle_unknown="ignore")` |

Extensión opcional, **no** usada en el caso de referencia ni en el OVA: variable binaria «la factura incluye cargos de envío/ajuste» a partir de los `StockCode` no estándar.

## 3. Objetivos

1. Un paquete `src/segmentacion/` con módulos separados por responsabilidad (`data/`, `features/`, `pipelines/`, `models/`, `evaluation/`), cada uno con pruebas de ≤ 20 filas.
2. Un `Pipeline` de scikit-learn que encapsule imputación, transformación logarítmica, escalado y codificación, ajustado solo con la cohorte A, serializado con `joblib` + `metadatos.json` para la Unidad 4.
3. El problema supervisado del OVA: *¿el cliente volverá a comprar en los 90 días siguientes al corte?*, con tres modelos y una regla simple como línea base. La segmentación no supervisada (k-means, DBSCAN) se conserva como segundo ejemplo (cuaderno 10).
4. Las cifras de §9 reproducibles por prueba automática.
5. El ejemplo del LLM como transformador (subtema 3.2), según `SPEC_llm_transformador.md`.

## 4. Diseño de la modularización (coincide con la Figura 8 y la Tabla 8 del OVA)

```
src/segmentacion/
├── config.py                  FECHA_CORTE_A = 2011-06-01, FECHA_CORTE_B = 2011-09-01, HORIZONTE = 90, SEMILLA = 42, rutas
├── data/                      cargar.py · limpieza.py                                   (de la U2, sin cambios)
├── features/
│   ├── rfm.py                 construir_rfm(ventas, fecha_corte)                       (de la U2)
│   ├── devoluciones.py        tasa_devolucion(cancelaciones, frecuencia_por_cliente, fecha_corte)
│   ├── etiquetas.py           etiqueta_recompra(ventas, fecha_corte, horizonte_dias=90)
│   ├── cohortes.py            cohorte(ventas, cancelaciones, corte, horizonte=90) -> pl.DataFrame
│   │                          compone las tres anteriores + pais ∈ {UK, otro}; termina con .sort("Customer ID")
│   ├── texto_llm.py           ver SPEC_llm_transformador.md
│   └── llm_clientes.py        ver SPEC_llm_transformador.md
├── pipelines/
│   └── preprocesamiento.py    NUM_LOG, NUM_STD, CAT ; construir_preprocesador(num_log=NUM_LOG, num_std=NUM_STD, cat=CAT)
├── models/
│   ├── entrenar.py            MODELOS (reg. logística, árbol prof. 4, bosque 300/hoja 5) ; entrenar(pipeline, X, y, cv)
│   └── registro.py            guardar(pipe, ruta, metadatos) ; cargar(ruta) -> (pipe, meta)  (se niega si cambia la versión de sklearn)
└── evaluation/
    ├── metricas.py            evaluar_clasificacion(y, p, umbral) ; regla_recencia(X, dias=90)
    └── umbral.py              tabla_umbrales(y, p, umbrales=(0.3, 0.4, 0.5, 0.6))
```

Nombres que **citan las guías** y no deben cambiar: `cohorte(ventas, cancelaciones, corte)`, `construir_preprocesador()`, `NUM_LOG`, `NUM_STD`, `CAT`, `guardar()`, `cargar()`, `src/segmentacion/pipelines/preprocesamiento.py`, `src/segmentacion/models/registro.py`, `config.py`, `tests/test_reproducibilidad.py`.

Reglas (Tabla 8 del OVA) y cómo se verifican:

| Regla | Verificación |
|---|---|
| Todo `fit` ocurre dentro del `Pipeline` y después de la partición | `tests/pipelines/test_sin_fuga.py`: tras `pipe.fit(X_A)`, las medias del `StandardScaler` = medias de A (no de A ∪ B) |
| Polars hasta la matriz de variables; scikit-learn desde ahí | `features/` no importa pandas; `pipelines/` recibe `.to_pandas()` |
| Solo `data/cargar.py` y `config.py` conocen rutas | Búsqueda en CI: `read_parquet(`/`open(` fuera de esos módulos falla |
| Una prueba por módulo con ≤ 20 filas inventadas | `tests/features/`, `tests/pipelines/`, `tests/models/`, `tests/evaluation/` |
| Semilla única y orden fijo de filas | `tests/test_reproducibilidad.py`: entrena dos veces el bosque y compara predicciones (igualdad exacta) |
| Los cuadernos solo llaman a `src/` | Revisión: ninguna `def` en cuadernos 06–10 |

## 5. Subtema LLM (resumen; detalle en `SPEC_llm_transformador.md`)

Categorizar las 5 305 descripciones únicas en **12 categorías cerradas** (iluminacion, cocina, textil, papeleria, decoracion, bolsas, juguetes, jardin, navidad, accesorios, cargo_no_producto, otro), con caché versionada, validación Pydantic, cliente inyectable (`FakeLLM` para pruebas; adaptador OpenAI-compatible para uso real) y evaluación contra una regla por palabras clave. No es obligatorio en el taller. Si hay conflicto entre esta sección y `SPEC_llm_transformador.md`, manda esta última.

## 6. Cuadernos de la Unidad 3

Mapa completo de numeración en `SPEC_segmentacion-clientes_polars.md` §5.4.

| Cuaderno | Contenido | Usa / reemplaza |
|---|---|---|
| `06_particion_y_fuga` | Cohortes A y B; la demostración de la fuga temporal en la recencia (0,795 vs 0,955); fuga en el preprocesamiento (186,9 vs 168,7) | Ideas de `projects/d2-temporal-split/` |
| `07_ingenieria_caracteristicas` | `ColumnTransformer` de la Tabla 4 del OVA; qué aprende cada transformador en `fit` | Reemplaza `legacy/05_preprocessing_pipelines`; resume (no copia) `tutorials/07` |
| `08_llm_como_transformador` | `SPEC_llm_transformador.md` con `FakeLLM` | Nuevo |
| `09_modelos_y_pipelines` | Tres modelos + regla; métricas, umbrales, importancia; curva de profundidad; validación cruzada y `GridSearchCV`; `guardar()` | Secuencia de métricas adaptada según `SPEC_material_nicolas.md` |
| `10_segmentacion_no_supervisada` | k-means y DBSCAN sobre el preprocesador | Fusiona `legacy/06`, `07` y `08`; luego se borra `legacy/` |

Los scripts `caso_recompra.py` y `figuras_u3.py` quedan, al final, como envoltorios delgados que importan del paquete y regeneran §9 y las figuras 3, 5, 6 y 7 en `reports/figures/`.

## 7. Criterios de aceptación

- [ ] `uv run pytest` pasa sin red y sin claves de API (incluye `FakeLLM`).
- [ ] `uv run pytest -m slow` ejecuta `tests/test_cifras_recompra.py` y reproduce §9 (tolerancia 0 en conteos; 0,001 en métricas).
- [ ] `tests/test_reproducibilidad.py` y `tests/pipelines/test_sin_fuga.py` existen y pasan.
- [ ] `models/recompra_v1/` contiene `pipeline.joblib` (gitignored) y `metadatos.json` (versionado) con los campos de la Tabla 7 del OVA: fecha de corte, horizonte, variables en orden, semilla, versiones (Python, scikit-learn, Polars), referencia a los datos con hash, AUC de validación, umbral acordado (0,3), fecha y autor.
- [ ] `cargar()` lanza un error claro si la versión de scikit-learn del entorno difiere de la de los metadatos.
- [ ] Los cuadernos 06–10 se ejecutan en orden con `nbconvert --execute` sin errores.
- [ ] `scripts/figuras_u3.py` produce las figuras 3, 5, 6 y 7 con los mismos títulos del OVA.
- [ ] `pydantic>=2` y `scikit-learn>=1.9` declarados en el extra `segmentacion-clientes`.
- [ ] `tutorials/07-tecnicas-preprocesamiento.md` enlaza a los cuadernos 07 y 09 en lugar de duplicar código.
- [ ] Ningún módulo de `features/` importa pandas.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| El ejemplo LLM desvía la unidad hacia «IA generativa» | Un subtema, un cuaderno, evaluación obligatoria contra la línea base |
| Costo de la llamada real | Se hace una vez y se versiona la caché; los estudiantes usan la caché |
| Las cifras cambian al descomponer los scripts | La prueba `slow` es el contrato; si una cifra cambia, se investiga la causa antes de tocar la prueba o el OVA |
| La app de la U4 depende del artefacto | `metadatos.json` es la interfaz: columnas, orden y umbral salen de ahí, no del código de la app |

## 9. Cifras de referencia del caso de recompra (16 de octubre de 2026, scikit-learn 1.9.1, Polars 2.0.0)

Son las que cita el OVA de la U3; `tests/test_cifras_recompra.py` (marcado `slow`) debe reproducirlas.

| Métrica | Valor |
|---|---|
| Cohorte A (corte 2011-06-01): clientes / recompran / tasa | 4 933 / 1 621 / 32,9 % |
| Cohorte B (corte 2011-09-01): clientes / recompran / tasa | 5 249 / 2 241 / 42,7 % |
| Regresión logística: AUC CV (A) / AUC (B) / AP (B) / P / R / F1 @0,5 | 0,817 ± 0,013 / 0,795 / 0,758 / 0,801 / 0,448 / 0,574 |
| Árbol (profundidad 4): AUC CV / AUC (B) / AP / P / R / F1 | 0,800 ± 0,015 / 0,780 / 0,723 / 0,764 / 0,487 / 0,595 |
| Bosque (300 árboles, hoja ≥ 5): AUC CV / AUC (B) / AP / P / R / F1 | 0,816 ± 0,009 / 0,785 / 0,755 / 0,803 / 0,451 / 0,578 |
| Regla recencia < 90: exactitud / P / R / F1 | 0,706 / 0,681 / 0,585 / 0,629 |
| Clase mayoritaria en B: exactitud | 0,573 |
| Fuga temporal en recencia (reg. logística): AUC CV / AUC (B) | 0,881 / 0,955 |
| Media de recencia: todas las cohortes vs. solo A | 186,9 / 168,7 días |
| Umbrales del bosque en B (señalados / P / R / F1): 0,3 · 0,4 · 0,5 · 0,6 | 2 093 / 0,688 / 0,643 / 0,665 · 1 653 / 0,751 / 0,554 / 0,638 · 1 259 / 0,803 / 0,451 / 0,578 · 890 / 0,856 / 0,340 / 0,487 |
| Curva de profundidad del árbol (AUC A / AUC B): 1 · 2 · 3 · 4 · 6 · 8 · 12 · sin límite | 0,696/0,676 · 0,774/0,745 · 0,806/0,769 · 0,821/0,780 · 0,848/0,768 · 0,886/0,744 · 0,964/0,698 · 1,000/0,656 |
| Importancia (bosque): recencia, monetario, frecuencia, productos distintos, antigüedad, tasa devolución, país (cada columna) | 0,249 · 0,218 · 0,200 · 0,147 · 0,122 · 0,058 · 0,003 |
| Reproducibilidad | Determinista solo tras `.sort("Customer ID")` en la cohorte (Polars `unique()` no conserva el orden) |

Validación cruzada: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre el pipeline completo en la cohorte A.

## 10. Decisiones tomadas (16 de octubre de 2026)

1. **Hilo de la U3: «recompra en 90 días»** (supervisado), con la segmentación no supervisada como segundo ejemplo (cuaderno 10).
2. **LLM: subtema 3.2 del OVA, agnóstico al proveedor**, modular y pensado para ejemplificar buenas prácticas. David lo ejecutará con DeepSeek a través del adaptador OpenAI-compatible; el repositorio no nombra proveedores fuera del adaptador. Detalle en `SPEC_llm_transformador.md`.
3. **Material de Nicolás: se reutiliza** la secuencia de métricas/sobreajuste y el código de pipelines, reescritos y con figuras propias, previa autorización. Dónde y cómo: `SPEC_material_nicolas.md` (3 de 3).
4. El componente LLM **no** es obligatorio en el taller de la U3; queda como ejemplo del repositorio y tema del pódcast 2.
