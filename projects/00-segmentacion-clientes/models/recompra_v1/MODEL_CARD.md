---
nav_exclude: true
search_exclude: true
---
# Ficha del modelo · recompra_v1

Estructura con base en Mitchell et al. (2019), *Model cards for model reporting*.

## Uso previsto

Priorizar los contactos de una campaña de reactivación: ordenar a los clientes
identificados de un minorista en línea por su probabilidad de volver a comprar
en los 90 días siguientes a la fecha de corte.

## Usos no previstos

Decisiones de crédito, de precio o cualquier decisión individual con efectos
sobre la persona. El modelo no fue evaluado para esos usos.

## Datos

Online Retail II (Chen, 2019), UCI Machine Learning Repository id 502, licencia
CC BY 4.0. Transacciones de un minorista en línea del Reino Unido entre
2009-12-01 y 2011-12-09. Solo clientes identificados (`Customer ID` no nulo);
cancelaciones y líneas con cantidad o precio no positivos excluidas
(`data/limpieza.preparar_ventas`). Procedencia, hash y descarga en
`data/README.md`.

## Variables de entrada (en orden)

`monetario`, `frecuencia`, `productos_distintos`, `recencia`, `antiguedad`,
`tasa_devolucion`, `pais` (`UK` u `otro`), calculadas solo con el pasado de la
fecha de corte (`features/cohortes.cohorte`).

## Evaluación

Bosque aleatorio (300 árboles, hoja mínima 5). AUC 0,816 en validación cruzada
de cinco pliegues sobre la cohorte A (corte 2011-06-01) y 0,785 en la cohorte de
prueba B (corte 2011-09-01). Línea base: «recompra quien compró en los últimos
90 días». Cifras en `metadatos.json` y en el OVA de la Unidad 3.

## Umbral recomendado

0,3: señala 2093 clientes de la cohorte de prueba con precisión 0,688 y recall
0,643. La tabla completa de umbrales (0,3 a 0,6) está en
`metadatos.json["tabla_umbrales"]`.

## Limitaciones

Fuerte estacionalidad (picos de noviembre); el Reino Unido concentra el 91,9 %
de las líneas; datos con más de una década de antigüedad; el país se reduce a
dos categorías; no hay variables de producto ni de campaña.

## Cómo reproducirlo

Desde `projects/00-segmentacion-clientes/`:

```bash
uv sync --extra segmentacion-clientes --extra dev   # desde la raíz del repo
uv run python scripts/descargar_datos.py
uv run python scripts/entrenar_recompra.py
uv run pytest -m slow tests/servicio          # cifras de referencia del OVA U4
```

Semilla, fecha de corte, horizonte y versiones en `metadatos.json`.

## Licencia

Código: MIT (repositorio `dpalacioj/data-projects-lab`). Datos: CC BY 4.0,
atribución a Chen (2019).
