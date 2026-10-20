# Taller S02 — Aplicación independiente

> Entregable: lo que pide §4 de este documento. Tiempo estimado: 4-6 horas de trabajo autónomo.

## Reglas del taller

1. Dataset propio (no Online Retail II). Ejemplos aceptables:
   - [UCI ML Repository](https://archive.ics.uci.edu/datasets) con **licencia explícita**.
   - Un dataset público de tu dominio (siempre con cita y licencia).
   - Un dataset simulado si no encuentras uno real; declara el generador.
2. Mínimo 100 000 filas y 6 columnas (si es menor, pide excepción).
3. Todo en **Polars 2.x**; pandas solo en el borde con una librería que
   lo exija.
4. Entregable en un fork del repo o un repositorio propio con la misma
   estructura.

## Entrega

Un pull request contra tu propia rama (si es un fork) o un repositorio
propio público, con:

- `notebooks/00_objetivo_de_analisis.ipynb` — tu tabla CRISP-DM fase 1.
- `notebooks/01_comprension_datos.ipynb` — lectura con esquema; Tabla
  de nulos; nulos estructurales si aplica.
- `notebooks/02_limpieza.ipynb` — tus reglas de limpieza (al menos 3)
  como funciones separadas, encadenadas con `.pipe` o equivalente,
  con conteos antes/después.
- `notebooks/04_eda.ipynb` — cuartiles, cercas de Tukey, Pearson y
  Spearman para dos pares de variables al menos.
- `notebooks/05_eda_automatizado.ipynb` — `skim` y perfil HTML sobre
  muestra reproducible.
- `HALLAZGOS.md` — mínimo 5 filas (hallazgo | evidencia | decisión).
- `src/<paquete>/data/{cargar,limpieza}.py` — tus funciones.
- `tests/test_limpieza.py` — una prueba por regla con ≤ 20 filas.
- `README.md` — procedencia, licencia, hash SHA-256, cómo descargarlo.

## Criterios de aceptación

- [ ] `uv run pytest` verde en < 10 s.
- [ ] Las cifras del cuaderno 01 están **declaradas** en `HALLAZGOS.md`
      (son auditables).
- [ ] Al menos una fila de `HALLAZGOS.md` menciona un nulo estructural.
- [ ] El cuaderno 05 usa `seed=42` en el muestreo (reproducible).
- [ ] Ningún `!pip` ni `!wget` en celdas.
- [ ] El dataset se descarga por script con hash; nunca se commitea.

## Rúbrica

| Dimensión | Peso | Qué miramos |
|---|---|---|
| CRISP-DM fase 1 | 20 % | Tabla con 6 campos; criterio de negocio medible |
| Limpieza | 25 % | Reglas separadas, test por regla, log por regla |
| EDA manual | 20 % | Al menos 2 asociaciones con Pearson y Spearman discutidas |
| EDA automatizado | 10 % | Muestra reproducible; HTML gitignored |
| HALLAZGOS.md | 15 % | 5+ filas con evidencia cuantificable |
| Reproducibilidad | 10 % | Hash del dataset; `uv run pytest` verde |

## Extensiones opcionales (bonus)

- Añadir una prueba `@pytest.mark.slow` que verifique dos cifras del
  dataset contra el archivo real.
- Comparar la pelea Pearson-vs-Spearman en tus datos con el ejemplo del
  OVA; discute qué dice el patrón sobre tu dominio.
- Generar un `scripts/figuras_tu_proyecto.py` con la paleta propia y
  dos figuras clave.
