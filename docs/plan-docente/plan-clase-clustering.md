---
title: "Plan de clase — Clustering no supervisado"
author: "David Palacio · Universidad de Medellín"
date: "2026-10"
---

# Plan de clase — Clustering no supervisado

> **Para el docente, uso privado.** Este documento vive en
> `docs/plan-docente/`, excluido del sitio público (`_config.yml`).
> Es la secuencia que yo (David) sigo para dictar la introducción a
> clustering antes de abrir el proyecto 00. Nadie más necesita verlo.

---

## 0. Antes de la clase (10 min de preparación)

**Checklist:**

- [ ] Entorno activo: `uv sync --all-extras` en la raíz del repo.
- [ ] Terminal abierta en la raíz del repo.
- [ ] Jupyter Lab arrancado: `uv run jupyter lab`.
- [ ] Dos pestañas listas:
  1. `tutorials/13-clustering-no-supervisado/01-teoria.ipynb`
  2. `projects/00-segmentacion-clientes/notebooks/10_segmentacion_no_supervisada.ipynb`
- [ ] Proyector probado. Zoom del navegador al 125% para que se lea
      desde el fondo del aula.
- [ ] Pizarra o iPad para dibujar la metáfora de las pizzerías.

**Mensaje inicial (1 línea):** *"Hasta ahora todo lo que hemos hecho
era supervisado. Hoy quitamos la respuesta correcta y vemos qué pasa."*

---

## 1. Durante la clase — secuencia total ~75 min

### Bloque 1 · Apertura (5 min)

Preguntas abiertas al aula, **sin abrir nada todavía**:

- ¿Qué es "aprendizaje supervisado"?
- Si yo les doy datos de clientes **sin decir** quiénes son "buenos" o
  "malos", ¿cómo encuentran grupos?
- ¿Han escuchado "K-Means"? ¿"clustering"?

Objetivo: activar lo que ya saben y abrir la curiosidad.

---

### Bloque 2 · Supervisado vs no supervisado (5 min)

**Abrir:** `01-teoria.ipynb` → Secciones 0.

**Ejecutar:** la celda del scatter dual (supervisado coloreado vs todo
gris).

**Punto a enfatizar:** *"En la vida real los datos vienen como la
figura de la derecha. El algoritmo propone una estructura, nosotros
decidimos si tiene sentido."*

**Pregunta al aula:** *"Cuántos grupos ven en la figura de la
derecha?"* — casi todos dirán 3. Perfecto, es la intuición humana que
vamos a formalizar.

---

### Bloque 3 · K-Means paso a paso (15 min)

**Abrir:** `01-teoria.ipynb` → Sección 1.

**Antes de ejecutar la celda de las 6 iteraciones**, dibujar en la
pizarra la **metáfora de las pizzerías**:

1. 3 puntos aleatorios = "aquí abriré las pizzerías"
2. Cada cliente al local más cercano = asignación
3. "Me doy cuenta de que mis clientes están más al norte, muevo el
   local" = actualización del centroide
4. Repetir hasta estabilizar

**Ejecutar:** la celda de las 6 iteraciones.

**Punto a enfatizar:** *"Las X negras parten en cualquier lado y se
asientan. Esto son 6 iteraciones; en la práctica sklearn hace esto
muchas veces desde posiciones aleatorias distintas (`n_init=10`) y se
queda con la mejor solución."*

**Pregunta trampa:** *"¿Qué pasaría si eligiéramos los 3 puntos
iniciales todos muy cerca?"* → Puede converger a un mínimo local.
Por eso `n_init=10`.

---

### Bloque 4 · Elegir k (10 min)

**Abrir:** `01-teoria.ipynb` → Sección 2.

**Antes de ejecutar las métricas**, mostrar primero el scatter de
k=2, k=4, k=8.

**Punto a enfatizar:** *"K-Means siempre te devuelve `k` grupos.
Con k=8 parte artificialmente los 3 grupos reales. Elegir `k` es
nuestra responsabilidad, no del algoritmo."*

**Ejecutar:** las curvas de elbow + silhouette.

**Punto a enfatizar:** *"En este dataset tan limpio ambas métricas
coinciden en k=3. En datos reales suelen discrepar — ya veremos en el
cuaderno 10 que silhouette dice una cosa y davies_bouldin dice otra."*

**Esto es el gancho para el proyecto 00.**

---

### Bloque 5 · DBSCAN paso a paso (15 min)

**Abrir:** `01-teoria.ipynb` → Sección 3.

**Antes de ejecutar**, dibujar en pizarra la **metáfora del mapa
nocturno**:

1. Vista aérea de una ciudad en la noche
2. Zonas brillantes = barrios (clusters)
3. Luces aisladas = casas rurales (ruido)
4. DBSCAN pregunta: "¿dónde hay mucha gente cerca?"

Explicar los **dos parámetros** antes de ver código:
- **ε (eps)**: radio del vecindario
- **min_samples**: cuántos vecinos necesita un punto para ser núcleo

Y los **tres tipos de punto**: núcleo, frontera, ruido.

**Ejecutar:** el scatter de DBSCAN sobre half-moons + ruido.

**Punto a enfatizar:** *"Fíjense: encontró las dos medias lunas sin
problema. K-Means no podría — asumiría nubes esféricas. Y los 10
puntos aleatorios quedaron marcados como ruido (-1), no se forzaron a
entrar en algún cluster."*

**Pregunta trampa:** *"¿Qué pasa si bajo ε a 0.05?"* → Casi todo se
vuelve ruido. *"¿Y si lo subo a 1.0?"* → Las dos medias lunas se
fusionan en un solo cluster.

---

### Bloque 6 · Comparación (10 min) — **el bloque más importante**

**Abrir:** `01-teoria.ipynb` → Sección 4.

**Ejecutar:** la grilla 2×3 (K-Means vs DBSCAN sobre blobs, moons, ruido).

**Puntos a enfatizar, en orden:**

1. **Blobs**: ambos funcionan. Preferir K-Means por rapidez.
2. **Half-moons**: K-Means parte las medias lunas por la mitad; DBSCAN
   las encuentra enteras. *"La forma de los datos importa."*
3. **Blobs con ruido**: K-Means fuerza el ruido en algún cluster
   (contamina los centroides). DBSCAN marca ruido como -1. *"Si tus
   datos tienen outliers que no quieres mezclar, DBSCAN primero."*

**Cerrar con la tabla de "cuándo usar cada uno"** del notebook.

---

### Bloque 7 · Puente al proyecto 00 (5 min)

**Abrir:** `01-teoria.ipynb` → Sección 5.

Leer en voz alta los 4 bullets de "Lo que ya sabes antes de abrirlo".
Son **predicciones de lo que verán** en el cuaderno 10. Si los
estudiantes lo leen antes, no se van a sorprender cuando:

- Las métricas discrepen (silhouette = 2, davies_bouldin = 4).
- Elijamos k = 4 por razones de negocio aunque silhouette diga 2.
- DBSCAN marque ~46% de ruido.

Dejar las 3 preguntas de reflexión del cuaderno teórico abiertas. Yo
quiero oír al menos 2 respuestas antes de pasar al proyecto.

---

### Bloque 8 · Abrir el proyecto 00 (10 min + lo que falte)

**Cerrar** `01-teoria.ipynb`. **Abrir:**
`projects/00-segmentacion-clientes/notebooks/10_segmentacion_no_supervisada.ipynb`

Las secciones del cuaderno 10 **espejan** las del cuaderno teórico:

| Cuaderno 10 (sección) | Corresponde a (teoría) |
|---|---|
| 1. k-means con silhouette | Sección 2 (elegir k) |
| 2. DBSCAN como contraste | Sección 3 + 4 (DBSCAN + comparación) |
| 3. Perfilar y nombrar segmentos | Nuevo — RFM específico |
| 4. Fichas de acción | Nuevo — puente a marketing |

**No ejecutar en vivo** si el tiempo apremia — ya está ejecutado y
guardado. Hacer scroll y comentar los resultados.

---

## 2. Preguntas típicas de estudiantes (con respuesta corta)

**P: ¿Por qué K-Means no funciona con los half-moons?**
R: Porque minimiza distancia al centroide; una media luna no tiene un
"centro" que la represente — el centro cae **fuera** de la luna. Falla
estructural.

**P: ¿Cómo elijo ε en DBSCAN sin probar?**
R: Hay una heurística: calcular la distancia al `k`-ésimo vecino de
cada punto (`k = min_samples`), ordenar y buscar el "codo". Pero en la
práctica, probar varios ε es más honesto.

**P: ¿Puedo usar K-Means y DBSCAN juntos?**
R: Sí. Patrón común: DBSCAN primero para marcar outliers → quitarlos
→ K-Means para segmentar el resto. En el proyecto 00 lo mencionamos
pero no lo implementamos.

**P: ¿Qué pasa si mis variables tienen escalas distintas (precio en
dólares, edad en años)?**
R: K-Means y DBSCAN usan distancia euclidiana. **Siempre escalar
antes.** El cuaderno 10 usa `StandardScaler` dentro del preprocesador.

**P: ¿Cuántos clusters hay "realmente"?**
R: No hay una respuesta correcta. Hay k que son más útiles para un
objetivo (ej. 4 segmentos de marketing es manejable, 15 no). La
métrica propone, el negocio decide.

---

## 3. Puntos a NO olvidar enfatizar

1. **No supervisado no significa "sin trabajo humano".** Hay que
   decidir k, interpretar segmentos, ponerles nombres. Es más subjetivo
   que supervisado.
2. **Las métricas discrepan en datos reales.** Mostrar explícitamente
   que en el cuaderno 10 silhouette elige 2 y davies_bouldin elige 4.
3. **DBSCAN no es "mejor" ni "peor" que K-Means.** Son herramientas
   para problemas distintos. La forma de los datos manda.
4. **Siempre escalar antes de clustering.** Si una variable va de 0 a 1
   y otra de 0 a 1 000 000, la segunda domina la distancia.
5. **El ruido en DBSCAN no es un bug.** En el cuaderno 10, ~46% de
   los clientes quedan como ruido → eso dice "DBSCAN no segmenta bien
   acá", no "DBSCAN está roto".

---

## 4. Si tengo poco tiempo (versión 45 min)

Prioridades (en orden):

1. Bloque 1 + 2: supervisado vs no supervisado (10 min)
2. Bloque 3: K-Means paso a paso (15 min)
3. Bloque 6: comparación con DBSCAN (10 min) — **saltar el detalle
   de los parámetros de DBSCAN**; solo decir "existe y detecta formas
   no esféricas"
4. Bloque 8: abrir el cuaderno 10, correr y comentar (10 min)

Dejar las secciones 2 (elegir k) y 5 (puente) como lectura previa a la
siguiente clase. Y asignar las 3 preguntas de reflexión del cuaderno
teórico como entregable rápido.

---

## 5. Material de respaldo

- `projects/00-segmentacion-clientes/HALLAZGOS.md` — contexto RFM que
  los estudiantes **ya deberían haber leído** antes de llegar al
  cuaderno 10.
- Métricas en `src/segmentacion/models/clustering.py`:
  `find_optimal_k`, `fit_kmeans`, `fit_dbscan`, `evaluate_clustering`.
  Si alguien quiere ver el código de las métricas, aquí está.
- `tests/models/` — ejemplos de qué está probado. Útil para preguntas
  tipo "¿cómo saben que esto no está roto?".

---

## 6. Al terminar

- [ ] Las 3 preguntas de reflexión del cuaderno teórico, al menos 2
      respondidas en voz alta.
- [ ] Alguien del aula capaz de decir en voz alta: *"K-Means asume
      formas esféricas, DBSCAN no."*
- [ ] Nadie confundido por por qué silhouette dice 2 pero elegimos 4.
- [ ] Pasada clara al próximo tema (modelos supervisados ya cubiertos
      en 06-09, apps/Streamlit en U4).
