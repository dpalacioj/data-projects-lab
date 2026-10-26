---
title: "93 · Fine-tuning y por qué en otra instancia"
parent: Tutoriales
nav_order: 93
---
# Tutorial 93 · Fine-tuning y por qué se hace en otra instancia

> Complemento opcional. No hace parte de la ruta mínima del curso. Léelo
> si quieres entender qué es fine-tuning, cuándo tiene sentido, y por
> qué el entrenamiento vive siempre en una máquina distinta de la que
> sirve el modelo.
>
> Pre-requisito: haber trabajado con algún modelo entrenado y guardado
> (p. ej. en los tutoriales 07-11 o en el proyecto 04 de reconocimiento
> de imágenes).

Este tutorial resuelve dos preguntas que un estudiante del curso se
hace cuando ve su primer proyecto con LLMs:

1. **¿Qué es fine-tuning y cuándo lo necesito?**
2. **¿Por qué el notebook de entrenamiento no corre en la misma máquina donde corre la app?**

Al final apunta a un **caso de estudio real** que puedes clonar y
explorar: [caficulbot-talk-demo](https://github.com/dpalacioj/caficulbot-talk-demo),
un asistente offline para caficultores con Gemma-3n fine-tuneado.

---

## Parte 1 — ¿Qué es fine-tuning?

### La analogía corta

Un modelo de lenguaje preentrenado es como una persona con educación
general: sabe mucho, pero no es experta en nada específico. **Fine-tuning**
es el proceso de convertirla en especialista entrenándola con datos de
tu dominio.

```
Gemma-3n base (generalista)
    │
    ├─ "¿Qué es la roya?"
    └─ "Es un hongo que afecta plantas."    (genérico)

          ↓ fine-tuning sobre 5000 docs de Cenicafé

Gemma-3n fine-tuneado (especialista en café)
    │
    ├─ "¿Qué es la roya?"
    └─ "Es Hemileia vastatrix. En Colombia, variedad Castillo..."   (específico)
```

### Fine-tuning vs. otras técnicas

Cuando quieres que un modelo responda mejor sobre tu dominio, tienes
cuatro opciones. Fine-tuning es solo una.

| Técnica | Qué hace | Costo | Cuándo usar |
|---|---|---|---|
| **Prompt engineering** | Mejor instrucción en el prompt | $0 | Siempre primero. Si funciona, no hagas más |
| **Few-shot / in-context learning** | Pones 3-5 ejemplos en el prompt | $0 | Si el prompt engineering no basta y tienes pocos ejemplos |
| **RAG (Retrieval-Augmented Generation)** | Buscas en una base de conocimiento y añades al prompt | Medio (vector DB) | Si la info del dominio cambia seguido |
| **Fine-tuning** | Reentrenas pesos del modelo | Alto (GPU + tiempo) | Si necesitas que el modelo **hable** como tu dominio, no solo responda sobre él |

**Regla práctica:** empieza por prompt engineering. Pasa a RAG si
necesitas conocimiento específico. Fine-tunea solo cuando necesites
cambiar el *estilo* o la *estructura* de respuesta, o cuando los
mejores prompts del mundo no te den calidad suficiente.

### Cuándo tiene sentido fine-tunear

- Terminología muy específica de un dominio (clínico, legal, financiero).
- Formato de respuesta estricto (p. ej. siempre JSON con ciertos campos).
- Lengua / variedad regional (castellano colombiano técnico, catalán, etc.).
- Function calling: enseñar al modelo cuándo invocar qué herramienta.
- Reducir el tamaño del prompt (todo lo que estaba en instrucciones ahora está en los pesos).

### Cuándo NO fine-tunear

- Para "añadir conocimiento" nuevo que cambia seguido → usa RAG.
- Para "no decir palabrotas" → usa guardrails o filtros, no fine-tuning.
- Para un prototipo → antes prueba prompt engineering.
- Si no tienes al menos 500-1000 ejemplos de calidad → vas a overfittear.

---

## Parte 2 — ¿Por qué el training vive en otra máquina?

Esta es la pregunta que todos los estudiantes se hacen la primera vez
que miran un proyecto de ML real: *"¿Por qué el notebook de entrenamiento
está en Colab y la app está en mi laptop? ¿No es lo mismo?"*

**No**. Training y serving son dos cargas de trabajo completamente
distintas. Mezclarlas rompe la economía del proyecto.

### Razón 1 — Memoria (la más dura)

Para **inferencia** (producción) con un modelo de 6B parámetros
necesitas: el modelo cargado, ~6 GB de VRAM (cuantizado a 4 bits).

Para **fine-tunear** el mismo modelo necesitas:

- Modelo cargado: ~6 GB.
- Gradientes (copia de los parámetros): ~6 GB más.
- Estados del optimizador (Adam guarda 2 valores por parámetro): ~12 GB más.
- Activaciones intermedias: varios GB más según el batch.
- **Total: 24-30 GB de VRAM, 4-5× más que inferencia.**

Esto es por qué:

- Un modelo que corre bien en un M1 de 8 GB RAM **no puede entrenar** en
  esa misma máquina.
- Para fine-tunear se usan técnicas de reducción (LoRA, QLoRA,
  cuantización) y plataformas con GPU grande.

### Razón 2 — Tiempo

| Operación | Duración típica |
|---|---|
| Inferencia (una consulta) | 1-3 segundos |
| Fine-tuning (dataset de ~10k ejemplos) | 2-8 horas |

Nadie quiere ocupar su laptop 8 horas corriendo un notebook. Nadie
quiere pagar una GPU en la nube 24/7 solo para entrenar algo una vez al
mes.

### Razón 3 — Ciclo de vida distinto

| | Training | Serving |
|---|---|---|
| **Frecuencia** | Ocasional (semanal, mensual) | Continuo (siempre encendido) |
| **Carga** | Picos intensos de pocas horas | Baja constante |
| **Hardware óptimo** | GPU grande por horas | GPU chica (o CPU), 24/7 |
| **Qué optimizas** | Throughput de entrenamiento | Latencia por consulta |
| **Si falla** | Reintentas, no es crítico | App caída |

Es la misma lógica de **separar OLAP y OLTP en bases de datos**:
cargas distintas quieren infraestructura distinta.

### Dónde se fine-tunea en la práctica

| Plataforma | Costo | Cuándo tiene sentido |
|---|---|---|
| **Google Colab (gratis)** | $0 | Prototipos. T4 de 16 GB. Lo más común para empezar |
| **Google Colab Pro** | $10-50/mes | GPUs mejores (A100, L4), sesiones más largas |
| **Kaggle Notebooks (gratis)** | $0 | 30h/semana de P100/T4. Alternativa a Colab |
| **AWS SageMaker** | pay-per-use | Empresarial, integra con infra AWS |
| **Databricks** | pay-per-use | Empresarial, MLflow nativo |
| **Lambda Labs / RunPod** | ~$1-2/hora | Alquiler barato de GPUs sueltas |
| **Local con GPU dedicada** | upfront | Si tienes RTX 4090 (24 GB) o M4 Max |

### El patrón MLOps que esto implementa

```
┌────────────────────────────┐    ┌─────────────────────────┐
│  Instancia de TRAINING     │    │  Instancia de SERVING   │
│  (Colab / SageMaker / ...) │    │  (laptop / server)      │
├────────────────────────────┤    ├─────────────────────────┤
│ notebook_finetuning.ipynb  │    │ api.py (FastAPI)        │
│                            │    │ app.py (Streamlit)      │
│ - Carga modelo base        │    │ - Descarga artefacto    │
│ - Carga dataset            │ ──►│ - Carga en memoria      │
│ - Entrena (LoRA, etc.)     │    │ - Expone /predict       │
│ - Cuantiza                 │    │                         │
│ - Sube a Hugging Face Hub  │    │                         │
└────────────────────────────┘    └─────────────────────────┘
     Una vez al mes                       Siempre encendida
     8 horas, GPU grande                  Barato, GPU chica
```

**El puente entre los dos mundos es un artefacto versionado** (modelo
en Hugging Face Hub, o `.joblib` en S3, o `.pth` en MLflow). El código
de serving no sabe nada del training; solo sabe descargar un artefacto
y cargarlo. El código de training no sabe nada del serving; solo sabe
subir un artefacto.

Esta separación es lo que define **MLOps**: training y serving son
pipelines independientes que se comunican por artefactos, no por
tiempo compartido en la misma máquina.

---

## Parte 3 — Caso de estudio: Caficulbot

[`caficulbot-talk-demo`](https://github.com/dpalacioj/caficulbot-talk-demo)
es un asistente de IA **offline** para caficultores colombianos.
Ilustra todo lo que acabamos de ver en vivo.

> **Autor original del proyecto:** Sergio Quintero. Fork educativo
> adaptado para esta especialización por David Palacio. El trabajo
> técnico (modelo, datasets, arquitectura) es de Sergio.

### Lo que fine-tunearon

- **Modelo base:** [Gemma-3n](https://huggingface.co/google/gemma-3n-E2B)
  (6B parámetros, multimodal: texto + imagen + audio).
- **Técnica:** [Unsloth](https://github.com/unslothai/unsloth) + LoRA.
  Unsloth reduce el consumo de VRAM 2-3× respecto a entrenamiento
  estándar — por eso cabe en una T4 de 16 GB.
- **Datasets** (los 3 también son del autor original):
  - [`sergioq2/coffe`](https://huggingface.co/datasets/sergioq2/coffe): ~5000 preguntas-respuesta extraídas de documentación técnica de Cenicafé.
  - [Roboflow Coffee Pests and Diseases](https://app.roboflow.com/detection-3nbwx/coffe-mw9n0/2/export): 2616 imágenes etiquetadas de plagas y enfermedades.
  - [`sergioq2/functioncalling_coffedata`](https://huggingface.co/datasets/sergioq2/functioncalling_coffedata): 2700 pares de instrucción-función para gestión administrativa.
- **Output:** modelo cuantizado a 4 bits ([`sergioq2/gemma-3N-finetune-coffe_q4_off`](https://huggingface.co/sergioq2/gemma-3N-finetune-coffe_q4_off)) publicado en Hugging Face.

### Dónde se corrió el training

En **Google Colab**, en el notebook `gemma3n_finetuning_coffeagent.ipynb`.
Dura varias horas y requiere la GPU de Colab (T4 de 16 GB).

### Dónde se corre la app

En **la laptop del caficultor**: FastAPI + Streamlit, descarga el modelo
desde Hugging Face Hub (vía `download.py`), y lo sirve con MPS en
macOS o CUDA en Linux. Sin internet después de la descarga inicial.

### Qué recorrer en el repo

| Archivo / carpeta | Qué muestra |
|---|---|
| [`docs/fine-tuning.md`](https://github.com/dpalacioj/caficulbot-talk-demo/blob/main/docs/fine-tuning.md) | Guía completa de fine-tuning, incluyendo sección detallada de por qué otra instancia |
| [`docs/guia-pedagogica.md`](https://github.com/dpalacioj/caficulbot-talk-demo/blob/main/docs/guia-pedagogica.md) | De "hola Docker" a un sistema MLOps real, pensada para estudiantes |
| [`docs/presentacion-tecnica.md`](https://github.com/dpalacioj/caficulbot-talk-demo/blob/main/docs/presentacion-tecnica.md) | Documentación técnica completa (~1000 líneas) |
| `gemma3n_finetuning_coffeagent.ipynb` | El notebook que corre en Colab (en el raíz del repo) |
| `app/` | FastAPI + Streamlit + microservicios por dominio |
| `app/api.py` | Cómo se carga el modelo descargado y se sirve con `/predict` |

### Qué deberías poder responder después de leerlo

1. ¿Por qué eligieron Gemma-3n y no otro modelo?
2. ¿Qué es LoRA y por qué reduce el consumo de VRAM?
3. ¿Qué es Unsloth y qué aporta sobre Transformers de HuggingFace?
4. ¿Qué es cuantización a 4 bits y qué pierden en el proceso?
5. ¿Cómo se decidió qué datos entraban al dataset de fine-tuning?
6. ¿Cómo verificar que el modelo fine-tuneado es mejor que el base? (métricas, no "se ve bien")

Las respuestas están en el `docs/` del repo.

---

## Resumen

- **Fine-tuning** convierte un modelo generalista en especialista.
- **No siempre lo necesitas**: antes prueba prompt engineering y RAG.
- **El training vive en otra máquina** por memoria (4-5× más), tiempo (horas) y ciclo de vida (ocasional vs. continuo).
- **El puente entre training y serving es un artefacto versionado**: subes a HF Hub, descargas en la app. Nunca se tocan directo.
- **Caficulbot** es un caso real donde todo esto se ve en vivo.

Para profundizar en MLOps más allá de este tutorial, el módulo siguiente
de la especialización cubre: orquestación de pipelines, monitoreo en
producción, drift detection, CI/CD de modelos.
