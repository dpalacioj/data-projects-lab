# HALLAZGOS — Unidad 2 (Online Retail II)

Hallazgos consolidados de los cuadernos 01-04. Cada fila es un vinculo
entre lo que observamos (evidencia reproducible), lo que decidimos y
donde se materializa. Esta tabla es insumo directo de la Unidad 3
(seccion 7 del OVA).

| Hallazgo | Evidencia reproducible | Decision |
|---|---|---|
| Hay transacciones **sin cliente** y conviven con precio cero | 243 007 filas sin `Customer ID` (22,77 % del total); dentro de ellas conviven las 4 382 `Description` nulas y todos los `Price = 0` | `exigir_cliente` + `filtrar_validas` en `preparar_ventas` (cuaderno 02). Las descartamos del flujo de RFM pero no las borramos del Parquet crudo. |
| Las **cancelaciones** son un flujo paralelo con valor para U3 | 19 494 facturas que empiezan por `C` (pre-dedup); 19 104 tras `unique` | `separar_cancelaciones` las aparta en `data/processed/cancelaciones.parquet`. En U3 son insumo para calcular tasa de devolucion por cliente. |
| El precio tiene **cola larga** y hay outliers extremos | Cerca de Tukey `Price = 7,50`; 8,40 % de las filas la supera; max = 10 953,50 £ | Documentar la cerca en el cuaderno 04; en U3 aplicar `log1p` en el pipeline sklearn. |
| La cantidad tambien tiene **cola larga** | Cerca de Tukey `Quantity = 27`; 6,56 % de las filas la supera; max = 80 995 | Igual que `Price`: `log1p` en U3; no se recortan outliers en U2 para no sesgar la narrativa. |
| **Pearson ≈ 0 pero Spearman fuerte** entre cantidad y precio | Pearson −0,005; Spearman −0,428 sobre 779 425 filas validas | En U3 preferir features basadas en rangos antes que valores crudos (`log1p` + `StandardScaler` cumple). El cuaderno 04 lo documenta como ejemplo didactico. |
| **Concentracion fuerte** del negocio en pocos clientes | El 20 % de clientes de mayor `monetario` suma el 77,2 % del total | La segmentacion debe producir una ficha de accion diferenciada para ese 20 %; no mezclarlo con "clientes esporadicos" bajo la misma campaña. |
| El negocio es **de dias habiles**: sabado casi inexistente | 400 filas en sabado en dos años (de 779 425 validas) | No modelar estacionalidad semanal con sabado como clase propia; agruparlo con domingo o excluirlo del heatmap final. |

## De aqui pasa a la Unidad 3

- `src/segmentacion/data/limpieza.py` y `src/segmentacion/data/cargar.py`:
  pipeline reutilizable sin cambios.
- `src/segmentacion/features/rfm.py::construir_rfm(ventas, fecha_corte)`:
  misma firma; U3 la llama con cortes anteriores para evaluar recompra.
- `data/processed/ventas_validas.parquet` + `cancelaciones.parquet`:
  insumos canonicos.
- `HALLAZGOS.md` (este archivo): guia las decisiones de ingenieria de
  caracteristicas en el cuaderno `07_ingenieria_caracteristicas.ipynb`
  de U3.
