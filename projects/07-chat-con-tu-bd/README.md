---
title: "07 · Chat con tu BD (LangChain + LangGraph + LLM open source)"
parent: Proyectos
nav_order: 7
---
# 07 · Chat con tu base de datos — LangChain + LangGraph + LLM local

> Pregúntale a tu base de datos en español. Un LLM **open source corriendo
> localmente** (Ollama + Qwen o Llama) traduce la pregunta a SQL, la ejecuta
> contra la BD cargada por el proyecto 06, y te devuelve la respuesta en
> lenguaje natural. El proyecto muestra tanto **LangChain** (cadena simple)
> como **LangGraph** (agente con auto-corrección cuando el SQL falla).

## Qué resuelve

**Escenario:** tu equipo cargó una base de datos relacional con el proyecto 06
(1M+ pedidos de e-commerce). Ahora el equipo de negocio quiere hacerle preguntas
sin tener que aprender SQL ni abrir un dashboard. Tú construyes un **chat** que:

1. Recibe una pregunta en español (*"¿cuáles son los 5 productos top por
   ingresos del último mes?"*).
2. Un LLM local la convierte a SQL mirando el schema.
3. Ejecutamos el SQL (con validación de seguridad: solo lectura).
4. Otro paso del LLM convierte el resultado en una respuesta en texto.

**Por qué este caso:** cubre lo que los proyectos 01–06 no tocan:

- **Modelos de lenguaje open source** corriendo localmente (sin API keys, sin
  mandar datos a terceros).
- **LangChain** (cadenas con LCEL, tools, prompts).
- **LangGraph** (estado compartido, loops, routing condicional, agentes con
  auto-corrección).

Y lo conecta con lo que ya hiciste: **reutiliza la BD del proyecto 06**.

---

## Qué aprendes aquí

| Concepto | Dónde lo ves |
|---|---|
| Factory de LLM con 3 proveedores (Ollama / HF / fake) | `src/chat_bd/llm.py` |
| Ejecución segura de SQL generado por LLM | `src/chat_bd/sql_tools.py` (`validar_sql_lectura`) |
| Prompts centralizados con `ChatPromptTemplate` | `src/chat_bd/prompts.py` |
| LangChain Expression Language (LCEL): `prompt \| llm \| parser` | `src/chat_bd/chain_simple.py`, notebook 01 |
| Tools (el LLM decide cuándo llamarlas) | Notebook 02 |
| LangGraph: estado tipado + nodos + aristas condicionales | `src/chat_bd/agent_graph.py`, notebook 03 |
| Agente con loop de auto-corrección | `construir_grafo()` |
| Tests sin red/LLM usando `FakeListChatModel` | `tests/conftest.py` |
| Streamlit con historial de chat y selector de motor | `app/chat_streamlit.py` |

---

## Estructura del proyecto

```
projects/07-chat-con-tu-bd/
├── README.md                        ← este archivo
├── Makefile                         ← comandos (`make help`)
├── .env.example                     ← LLM_PROVIDER, OLLAMA_MODEL, DATABASE_URL
├── .gitignore
│
├── notebooks/
│   ├── 01_langchain_basico.ipynb    ← prompt | llm | parser
│   ├── 02_langchain_con_tools.ipynb ← LLM decide cuándo usar la BD
│   └── 03_langgraph_agente.ipynb    ← grafo con auto-corrección
│
├── src/chat_bd/
│   ├── __init__.py
│   ├── config.py                    ← proveedor, modelo, URL de BD (desde .env)
│   ├── llm.py                       ← factory: ollama / hf / fake
│   ├── sql_tools.py                 ← schema inspector + SQL seguro
│   ├── prompts.py                   ← plantillas centralizadas
│   ├── chain_simple.py              ← LangChain: cadena NL → SQL → respuesta
│   └── agent_graph.py               ← LangGraph: grafo con reintentos
│
├── app/
│   └── chat_streamlit.py            ← chat UI con selector chain/graph
│
└── tests/                           ← 24 tests, usan FakeListChatModel (sin LLM real)
    ├── conftest.py
    ├── test_llm.py
    ├── test_sql_tools.py
    ├── test_chain_simple.py
    └── test_agent_graph.py
```

### Qué hace cada módulo en `src/`

| Archivo | Responsabilidad única |
|---|---|
| `config.py` | Lee `.env`: proveedor de LLM, modelo, URL de BD, max reintentos. |
| `llm.py` | **Factory** que devuelve un `ChatModel`. El resto del paquete no sabe qué LLM hay detrás. |
| `sql_tools.py` | Conexión a la BD, inspección de schema y **ejecución segura** (rechaza INSERT/UPDATE/DELETE/DROP…). |
| `prompts.py` | Plantillas de prompt (sistema + usuario). Un solo sitio para afinarlas. |
| `chain_simple.py` | **LangChain**: cadena directa `pregunta → SQL → ejecuta → respuesta`. Una sola pasada, sin reintentos. |
| `agent_graph.py` | **LangGraph**: grafo con nodos (generar, ejecutar, formatear) y un loop de auto-corrección si el SQL falla. |

---

## Cómo correrlo (receta en 5 pasos)

### 1. Instala las dependencias

Desde la **raíz del repositorio**:

```bash
uv sync --extra chat-bd --extra ingesta-eventos --extra dev
```

O desde esta carpeta con el Makefile:

```bash
make setup
```

### 2. Instala y arranca Ollama

Ollama es una herramienta para correr LLMs localmente. **Cero API keys, cero
datos a terceros.**

```bash
# macOS
brew install ollama
ollama serve &           # arranca el servidor en localhost:11434

# O descarga el instalador desde https://ollama.com/
```

Verifica que responde:

```bash
make check-ollama
```

### 3. Descarga el modelo

```bash
make pull-model          # descarga qwen2.5:7b (~4GB)
```

> **Si no puedes correr Ollama** (RAM limitada, Windows sin WSL), ve a la
> sección **"Fallback: HuggingFace Inference API"** más abajo. Toda la receta
> funciona igual cambiando `LLM_PROVIDER=hf` en `.env`.

### 4. Prepara credenciales y conexión a la BD

```bash
make env                 # copia .env.example -> .env
```

El `.env` por defecto apunta al SQLite del proyecto 06. Si todavía no lo
cargaste:

```bash
cd ../06-ingesta-eventos-ecommerce && make pipeline && cd -
```

Verifica que el 07 ve la BD:

```bash
make check-db
```

### 5. Lanza el chat

```bash
make serve               # Streamlit en http://localhost:8501
```

O desde terminal:

```bash
make chat-chain          # cadena simple (LangChain)
make chat-graph          # agente con auto-corrección (LangGraph)
```

---

## Dos motores: chain (LangChain) vs graph (LangGraph)

| Motor | Qué hace | Cuándo úsalo |
|---|---|---|
| **chain** (`chain_simple.py`) | Pregunta → genera SQL → ejecuta → responde. **Una pasada**. Si el SQL falla, la cadena falla. | Pruebas rápidas, prompts bien tuneados, modelo grande (el error es raro). |
| **graph** (`agent_graph.py`) | Lo mismo + si el SQL falla, el agente lee el error y **reintenta** hasta `max_intentos` veces. | Producción. Los LLMs se equivocan y el auto-correct salva la mayoría. |

El dashboard trae un selector en la barra lateral para alternar entre los dos y
comparar.

### Por qué LangGraph y no sólo LangChain

LangChain Expression Language (LCEL) es excelente para **tuberías lineales**:

```python
chain = prompt | llm | parser | ejecutor | prompt2 | llm | parser
```

Pero no soporta bien:

- **Loops condicionales** (reintentar si el SQL falla).
- **Estado compartido** entre nodos (número de intentos, SQL previo, error).
- **Routing condicional** (si pasa X ir al nodo A, si no al B).

LangGraph añade exactamente esto: un `StateGraph` con nodos y aristas
condicionales. El grafo de este proyecto:

```
START
  ↓
generar_sql
  ↓
ejecutar_sql ─── ok ──→ formatear_respuesta → END
  │
  ├── retry (error, intentos < max) → regenerar_sql ──┐
  │                                                   │
  │                                      ejecutar_sql ←┘
  │
  └── giveup (intentos >= max) → formatear_fallo → END
```

El código del grafo vive en `src/chat_bd/agent_graph.py`. Son ~100 líneas,
léelo, es más corto de lo que parece.

---

## Correr los tests

```bash
make test
```

Deberías ver:

```
tests/test_agent_graph.py ....         [ 16%]
tests/test_chain_simple.py ...         [ 29%]
tests/test_llm.py ....                 [ 45%]
tests/test_sql_tools.py .............  [100%]

24 passed in ~2s
```

**Los tests NO necesitan Ollama corriendo.** Usan `FakeListChatModel` de
LangChain, que devuelve una lista de respuestas predeterminada en orden. Así
testeamos:

- `test_sql_tools.py` — la validación de seguridad rechaza `INSERT`, `UPDATE`,
  `DELETE`, `DROP`, multi-sentencia, etc. y el inspector de schema devuelve
  tablas y muestras.
- `test_llm.py` — la factory devuelve el proveedor correcto y falla con error
  claro si falta `HF_TOKEN`.
- `test_chain_simple.py` — la cadena orquesta las dos llamadas al LLM y la
  ejecución de SQL.
- `test_agent_graph.py` — el grafo recorre el camino feliz, reintenta cuando
  hay error, y se rinde tras `max_intentos`.

---

## Fallback: HuggingFace Inference API

Si no puedes correr Ollama local, HF ofrece un **endpoint gratuito** (con
rate limit pero suficiente para aprender).

1. Crea un token gratuito en <https://huggingface.co/settings/tokens>
2. En `.env`:

   ```
   LLM_PROVIDER=hf
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   HF_MODEL=Qwen/Qwen2.5-7B-Instruct
   ```

3. Vuelve a correr `make serve`. **Ni una línea de código cambia** — la
   factory `obtener_llm()` detecta `LLM_PROVIDER=hf` y usa `ChatHuggingFace`
   en vez de `ChatOllama`.

Lo mismo aplica si mañana quieres usar DeepSeek, Groq o OpenAI: agregas un
`elif proveedor == "groq"` en `llm.py` y listo.

---

## Cómo adaptar esta receta a otro caso

| Quiero... | Toca... |
|---|---|
| Que responda sobre otra BD | Cambiar `DATABASE_URL` en `.env`. El schema se auto-descubre. |
| Cambiar de modelo | Cambiar `OLLAMA_MODEL` en `.env`. Ejemplo: `phi3:mini` (más rápido, menos preciso) o `llama3.1:70b` (más preciso, requiere GPU). |
| Agregar una tool al LLM (p. ej. consultar una API externa) | Ver notebook 02 y añadir un nodo nuevo al grafo en `agent_graph.py`. |
| Agregar memoria de conversación | LangGraph tiene `checkpointers` (SQLite, Redis). Se añade al compilar el grafo con `.compile(checkpointer=...)`. |
| Que el agente reflexione antes de responder | Añade un nodo `evaluar_respuesta` entre `ejecutar_sql` y `formatear_respuesta`. Si falla el criterio, vuelve a `generar_sql`. |

---

## Lo que este proyecto no cubre (y por qué)

- **Fine-tuning** de modelos. Para NL→SQL funcional, un modelo grande general
  (Qwen 2.5 7B o superior) con un buen prompt es suficiente. Fine-tunear es
  caro y pocas veces mejora.
- **RAG sobre documentos**. Este proyecto usa una BD estructurada. Para PDF o
  docs no estructurados, la pieza clave es un vector store (Chroma, pgvector)
  + retriever; es otro proyecto.
- **Despliegue a producción** (rate limiting, auth, observabilidad con
  LangSmith). Fuera de alcance de un proyecto didáctico.
- **Streaming de tokens**. `ChatOllama` lo soporta, pero complica la UI de
  Streamlit; lo dejamos como ejercicio.

---

## Checklist antes de dar por terminado el proyecto

- [ ] `make setup` corre sin error.
- [ ] `make check-ollama` dice que Ollama responde.
- [ ] `make pull-model` descarga el modelo.
- [ ] `make check-db` muestra las tablas del proyecto 06.
- [ ] `make test` pasa los 24 tests en menos de 5 segundos.
- [ ] `make serve` abre el chat y puedes hacer una pregunta.
- [ ] En el chat puedes alternar entre motor `chain` y `graph`.
- [ ] Haz una pregunta mal formulada a propósito y verifica que el motor
      `graph` reintenta (lo verás en la etiqueta *"X intento(s)"*).
- [ ] El `.env` NO está commiteado.

Si todo esto se cumple, el proyecto está sano.
