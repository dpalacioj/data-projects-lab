# Sesión S04 · Aplicación y MLOps — Guía para el instructor

> **Estado: 🟡 placeholder.** El esqueleto está en pie; el contenido falta porque la Unidad 4 todavía no tiene spec en `docs/specs/`. El artefacto que esta sesión consumirá — [`models/recompra_v1/metadatos.json`](../../projects/segmentacion-clientes/models/recompra_v1/metadatos.json) — ya existe (S03).

> Revisión: **2026-10-19** · Spec que implementará: pendiente de redacción (SPEC_app_u4.md candidato) · Proyecto que soportará: `projects/segmentacion-clientes/app/streamlit_app.py` + API opcional.

**No asumas que un contenido sin fecha sigue siendo correcto.** La pregunta
que organizará esta sesión será:

> _(Pendiente. Posible dirección: «¿cómo convierte un metadatos.json en
> una app que un usuario de negocio puede abrir, usar y auditar?»)_

---

## TODO para completar

- [ ] Escribir SPEC U4 en `docs/specs/SPEC_app_u4.md` con cifras y
      contratos.
- [ ] Escribir [`contrato-metadatos-y-app.md`](contrato-metadatos-y-app.md):
      cómo la app lee `metadatos.json` y construye el formulario de
      entrada.
- [ ] Escribir [`streamlit-estructura.md`](streamlit-estructura.md):
      secciones de la app (caso de negocio, exploración, clustering,
      perfiles y acciones).
- [ ] Escribir [`inferencia-y-caching.md`](inferencia-y-caching.md):
      `@st.cache_data`, `@st.cache_resource`, invalidación.
- [ ] Escribir [`observabilidad.md`](observabilidad.md): logs de
      inferencia, drift básico, alertas por caída de AUC.
- [ ] Escribir [`deployment.md`](deployment.md): Streamlit Cloud,
      Hugging Face Spaces, o servidor propio; variables de entorno.
- [ ] Escribir [`taller.md`](taller.md): aplicación propia que consuma
      el `metadatos.json` que el estudiante produjo en S03.
- [ ] Escribir [`troubleshooting.md`](troubleshooting.md): errores
      típicos (streamlit cache, versiones de sklearn, parquet inexistente).
- [ ] Escribir [`checklist-dia-de-clase.md`](checklist-dia-de-clase.md).
- [ ] Reemplazar este README por una versión completa siguiendo
      [`_templates/GUION_SESION.md`](../_templates/GUION_SESION.md).

## Material transversal mientras tanto

- `projects/segmentacion-clientes/app/streamlit_app.py` ya existe pero
  consume los artefactos del flujo pandas legacy (`features_clientes.parquet`,
  `build_preprocessing_pipeline`). En U4 se migra para consumir
  `models/recompra_v1/metadatos.json` como fuente de verdad.
- `projects/prediccion-precios-automoviles/` tiene ejemplos de API
  FastAPI + Gradio como referencia.

## Lo que no se improvisa

Antes de escribir el contenido, el instructor debe decidir:

1. **Streamlit vs. Gradio vs. FastAPI puro.** Decisión declarada en la
   spec U4.
2. **Alcance de MLOps.** ¿Solo deployment o también monitoreo + drift?
3. **Datos de inferencia.** ¿La app recibe un `Customer ID` y busca en
   el Parquet, o recibe un cliente completo (JSON)?

Esas decisiones son la Fase 1 de CRISP-DM de la Unidad 4; no se
resuelven en el notebook.

## Puente al final del módulo

**S04** cierra el ciclo: el estudiante toma el modelo que entrenó en
S03, lo monta en una app usable y define el contrato con el usuario
final. La app Streamlit del proyecto Segmentación de Clientes es el
caso de referencia.
