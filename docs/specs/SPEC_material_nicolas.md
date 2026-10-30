# SPEC — Integrar el material de Nicolás Gómez Giraldo en `data-projects-lab`

**Fuente:** `github.com/nicolasggiraldo/especializacion-udem-2026-01` (sin licencia declarada; 2 PDF, 4 cuadernos)
**Destino:** `dpalacioj/data-projects-lab` · nuevas carpetas `tutorials/` y `projects/00-segmentacion-clientes/notebooks/`
**Propósito:** aprovechar lo que su material hace bien —la secuencia explicativa de métricas y sobreajuste, el ejercicio de datos sucios, la lectura de nulos estructurales en House Prices, la base de pipelines— sin duplicar contenidos, sin romper la coherencia (Polars en U2, caso Online Retail II en U2–U4) y respetando autoría.
**Estado:** especificación; requiere su autorización antes de copiar texto o código.
**Orden de implementación:** **3 de 3.** Va al final: las filas marcadas «U2» se apoyan en los cuadernos 00 y 01 de `SPEC_segmentacion-clientes_polars.md`, y las marcadas «U3» en el cuaderno 09 de `SPEC_procesamiento_U3.md`. Sin autorización de Nicolás, solo se ejecuta lo que el §5 permite sin su material (estructura del tutorial 02b y fusión del tutorial 05 redactada desde fuentes primarias).

---

## 1. Condición previa: autorización y atribución

El repositorio de Nicolás no tiene archivo `LICENSE`, de modo que por defecto todos los derechos son suyos. Antes de incorporar cualquier texto, figura o código:

1. Pedirle por escrito autorización para (a) adaptar su material en `data-projects-lab` bajo la licencia MIT del repositorio y (b) usarlo como insumo del material del curso, cuyos derechos quedan reservados a la institución correspondiente.
2. Proponerle que añada un `LICENSE` (MIT o CC BY 4.0) a su repositorio; eso resuelve el punto para siempre.
3. Atribución en cada archivo derivado: encabezado `Adaptado de Gómez Giraldo (2026), github.com/nicolasggiraldo/especializacion-udem-2026-01, con autorización`. En el material público del curso no se cita el repositorio (criterio del módulo: fuentes primarias), pero se agradece en los créditos del entorno virtual si él lo acepta.

## 2. Qué se toma, qué se transforma, dónde queda

| Material de Nicolás | Valor | Transformación | Destino en `data-projects-lab` | Unidad |
|---|---|---|---|---|
| `sesion_2_crisp_dm.pdf` | Secuencia por fases con ejemplos de churn y preguntas de discusión | No se copia (es presentación). Sus **preguntas de discusión** («¿en qué fase fallan más los proyectos?», «¿qué riesgos ven?») se incorporan como preguntas de reflexión en el cuaderno 00 | `notebooks/00_objetivo_de_analisis.ipynb` (sección «Para discutir en clase») | U2 |
| `sesion_3_preparacion_de_datos*.ipynb` — bloque de datos sucios (Bogotá/Bta/bogotá, precios negativos, fechas en dos formatos) | Ejercicio perfecto para enseñar reglas de limpieza | Reescribir en **Polars**; convertir en ejercicio con enunciado y solución separados (patrón de `ejercicio-preprocesamiento`) | `tutorials/04-limpieza-datos-sucios-polars/` con `01_ejercicio.ipynb` y `02_solucion.ipynb` | U2 |
| `sesion_3_*` — House Prices: lectura de nulos estructurales («no hay garaje» ≠ dato faltante) | Mejor ejemplo del repositorio sobre semántica de los nulos | Reescribir en Polars sobre **Online Retail II** (descripción nula = inventario; cliente nulo = venta sin registro). Si se mantiene House Prices, citar De Cock (2011) y verificar licencia de los datos | Sección «Nulos que no son faltantes» en `notebooks/01_comprension_datos.ipynb` | U2 |
| `sesion_6_modelo.ipynb` — métricas de regresión y clasificación, matriz de confusión, ROC, sobreajuste, sesgo-varianza, regularización | Secuencia didáctica clara y completa; fórmulas en LaTeX | Reutilizar la **secuencia y las fórmulas**; reescribir el texto; reemplazar imágenes de terceros (GeeksforGeeks, Wikipedia) por figuras propias generadas desde el caso de recompra (`figuras_u3.py`); añadir precisión media y umbral | `tutorials/09-metricas-ml-regresion-vs-clasificacion.md` (fusionar con el tutorial existente de David, que ya cubre lo mismo: queda **un solo** tutorial) y `notebooks/09_modelos_y_pipelines.ipynb` | U3 |
| `sesion_7_pipelines.ipynb` — código de `Pipeline` sin texto | Base de código correcta | Escribir la explicación que falta; migrar al caso de recompra con `ColumnTransformer` | `notebooks/09_modelos_y_pipelines.ipynb` (sección «Pipeline de principio a fin») | U3 |
| `sesion_1_intro_bi.pdf` — tipos de analítica (descriptiva, diagnóstica, predictiva, prescriptiva) y fechas de entregas de 2026-01 | Contexto de BI útil para el RAE1 (UC1) | Tabla comparativa reescrita, con cita a una fuente primaria por verificar; las fechas no se usan | Recuadro en `notebooks/00_objetivo_de_analisis.ipynb` | U2 |

## 3. Correcciones que se hacen al adaptar (y se le comentan a Nicolás)

| Problema en el original | Corrección en la versión adaptada |
|---|---|
| `LabelEncoder` aplicado a variables explicativas | `OrdinalEncoder` (si hay orden) u `OneHotEncoder`; `LabelEncoder` solo para la etiqueta |
| Imputación por la media sobre todo el conjunto antes de partir | Imputación dentro del `Pipeline`, tras la partición (es la Tabla 2 de la spec de la Unidad 3) |
| `!wget` desde un repositorio de 2025-02 y una línea `!wget !wget` | Script de descarga con hash en `scripts/`; sin comandos de shell en celdas |
| Figuras tomadas de blogs sin licencia clara | Figuras propias (Matplotlib/Mermaid) con nota «Elaboración propia» |
| Sin `pyproject`/dependencias | El extra `segmentacion-clientes` del `pyproject.toml` raíz cubre todo |
| Cuaderno 7 sin texto | Texto explicativo + preguntas de reflexión |

## 4. Estructura resultante en `data-projects-lab`

```
tutorials/
├── 02-configuracion-proyecto-uv.md               (existente)
├── 02b-limpieza-datos-sucios-polars/              NUEVO (adaptado de sesión 3)
│   ├── README.md · 01_ejercicio.ipynb · 02_solucion.ipynb
├── 07-tecnicas-preprocesamiento.md               (existente; enlazar a notebooks U3 en lugar de duplicar código)
├── 09-metricas-ml-regresion-vs-clasificacion.md  (existente; FUSIONAR con sesión 6: fórmulas + figuras propias)
└── ...
projects/00-segmentacion-clientes/notebooks/
├── 00_objetivo_de_analisis.ipynb                  (U2; preguntas de discusión de sesiones 1 y 2)
├── 01_comprension_datos.ipynb                     (U2; sección de nulos estructurales)
├── 06_particion_y_fuga.ipynb                      (U3)
├── 07_ingenieria_caracteristicas.ipynb            (U3)
├── 08_llm_como_transformador.ipynb                (U3; ver SPEC_llm_transformador.md)
├── 09_modelos_y_pipelines.ipynb                   (U3; secuencia de sesión 6 + código de sesión 7, reescritos)
└── 10_segmentacion_no_supervisada.ipynb           (U3; actuales 06–08 renumerados)
```

## 5. Plan

1. Mensaje a Nicolás (autorización + propuesta de `LICENSE`). Sin respuesta positiva, solo se adopta la **estructura** de su secuencia, redactada desde fuentes primarias (Hastie et al., 2009; scikit-learn), que es lo que hoy ya hace el material del curso.
2. `tutorials/04-limpieza-datos-sucios-polars/` (medio día).
3. Fusión del tutorial 05 con la sesión 6 (medio día): un solo documento, figuras propias.
4. Cuaderno 09 con la secuencia métricas → sobreajuste → pipeline sobre el caso de recompra (un día; depende de `SPEC_procesamiento_U3.md`).
5. Preguntas de discusión en el cuaderno 00 (una hora).

## 6. Criterios de aceptación

- [ ] Autorización escrita archivada en `docs/autorizaciones/` (o `LICENSE` en el repo de Nicolás).
- [ ] Cada archivo derivado lleva la línea de atribución del §1.
- [ ] No queda ninguna imagen de terceros sin licencia en los archivos adaptados.
- [ ] No queda `LabelEncoder` sobre variables explicativas ni imputación previa a la partición en ningún cuaderno del proyecto.
- [ ] El tutorial 05 es único (no hay dos documentos sobre métricas).
- [ ] Todos los cuadernos adaptados corren con `uv run jupyter nbconvert --execute` sin red salvo la descarga inicial.

## 7. Qué NO se hace

- No se copian las presentaciones PDF ni sus imágenes.
- No se citan los cuadernos de Nicolás en el material público del curso como fuente bibliográfica (criterio del módulo: fuentes primarias verificables).
- No se mantiene el dataset House Prices como caso del módulo: el hilo es Online Retail II.
