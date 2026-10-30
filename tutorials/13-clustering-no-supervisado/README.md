---
title: "13 · Clustering no supervisado (K-Means y DBSCAN)"
parent: Tutoriales
nav_order: 13
---

# Clustering no supervisado — paso a paso

Material teórico, pensado para abrirse **antes** del cuaderno
`projects/00-segmentacion-clientes/notebooks/10_segmentacion_no_supervisada.ipynb`.
Objetivo: entender qué hace K-Means, qué hace DBSCAN, y cuándo
conviene cada uno, usando datos sintéticos simples que se pueden
visualizar en 2D.

## Qué contiene

- `01-teoria.ipynb` — notebook con 5 secciones:
  1. ¿Qué significa "no supervisado"? (con un contraste visual
     supervisado vs no supervisado sobre los mismos puntos).
  2. **K-Means paso a paso**: la metáfora de las pizzerías, el
     algoritmo iterativo y una animación de 6 iteraciones sobre
     datos sintéticos.
  3. **Elegir k**: elbow + silhouette, con el mismo dataset mostrado
     a k = 2, 4 y 8.
  4. **DBSCAN**: densidad, ε, `min_samples`, núcleos / frontera /
     ruido. Con un scatter donde cada tipo de punto está marcado.
  5. **¿K-Means o DBSCAN?**: comparación lado a lado sobre tres
     datasets (blobs, half-moons, blobs con ruido).

El cuaderno termina con el "puente" al proyecto 00: qué esperar en
el cuaderno 10 y 3 preguntas para asegurarse de que los conceptos
quedaron claros.

## Cómo usarlo

```bash
# desde la raiz del repo
uv run jupyter lab tutorials/13-clustering-no-supervisado/01-teoria.ipynb
```

Las celdas son autocontenidas: solo usan `numpy`, `matplotlib` y
`scikit-learn` (ya en las dependencias base). No requieren datos
externos ni API keys.

