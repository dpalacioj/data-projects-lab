# Recursos y Herramientas de IA

Guía práctica de plataformas, servicios y herramientas para estudiantes de IA y Data Science.

---

## 1. Plataformas de Chat con LLMs

Servicios para probar modelos, comparar respuestas y experimentar con prompts.

| Plataforma | URL | Casos de Uso |
|------------|-----|--------------|
| **Groq** | https://groq.com | Inferencia ultra-rápida, ideal para prototipar |
| **Mistral Chat** | https://chat.mistral.ai | Modelos europeos, soporte multilingüe |
| **DeepSeek** | https://chat.deepseek.com | Modelos especializados en código |
| **Qwen (Alibaba)** | https://chat.qwen.ai | Multimodal, soporte asiático |

**Ejercicio práctico:**
- Mismo prompt en 3 plataformas diferentes
- Comparar: precisión, estilo, velocidad
- Documentar diferencias observadas

---

## 2. Investigación y Organización de Conocimiento

### NotebookLM (Google)
**URL:** https://notebooklm.google/

**Funcionalidades:**
- Crea cuadernos con fuentes (PDFs, URLs, textos)
- Genera resúmenes automáticos
- Crea mapas conceptuales
- Genera preguntas guía para estudio

**Casos de uso:**
- Revisar papers académicos
- Preparar presentaciones
- Estudiar para exámenes
- Sintetizar documentación técnica

### Manus
**URL:** https://manus.im/app

**Funcionalidades:**
- Editor para estructurar ideas
- Organización de notas y documentos
- Planificación de proyectos

**Casos de uso:**
- Diseño de proyectos de data science
- Documentación técnica
- Reportes de investigación

---

## 3. Benchmarking y Evaluación de Modelos

### MTEB Leaderboard (Hugging Face)
**URL:** https://huggingface.co/spaces/mteb/leaderboard

**Qué mide:**
- Embeddings para búsqueda semántica
- Clasificación de texto
- Clustering
- Reranking

**Cómo usar:**
1. Identifica la **tarea específica** (ej: "Retrieval")
2. Observa la **métrica** (NDCG@10, Recall@100)
3. Compara **tamaño del modelo** vs rendimiento
4. Evalúa **idioma** del benchmark

**⚠️ Advertencia:**
No compares modelos de diferentes tareas o métricas. Un modelo con 90% en clasificación NO es comparable con 85% en retrieval.

---

## 4. Agentes y Automatización

### AITMPL Agents
**URL:** https://www.aitmpl.com/agents?search=playwright

**Qué ofrece:**
- Directorio de agentes especializados
- Patrones de automatización web
- Integración con Playwright (navegación automática)
- Ejemplos de pipelines complejos

**Casos de uso:**
- Web scraping inteligente
- Automatización de tareas repetitivas
- Testing automatizado de UIs

### Awesome Claude Code Subagents
**URL:** https://github.com/VoltAgent/awesome-claude-code-subagents

**Qué ofrece:**
- Lista curada de subagentes para desarrollo
- Patrones de orquestación
- Roles especializados (code-reviewer, test-runner, etc.)
- Ejemplos de workflows multi-agente

**Casos de uso:**
- Code review automático
- Generación de tests
- Refactorización de código
- Documentación automática

---

## 5. Inferencia en Tiempo Real

### Cerebras LiveKit
**URL:** https://cerebras.livekit.io

**Qué demuestra:**
- Streaming de respuestas
- Baja latencia (<100ms)
- Colaboración en tiempo real
- WebRTC para IA conversacional

**Para analizar:**
- Requisitos de infraestructura
- Quality of Service (QoS)
- Trade-offs: latencia vs precisión
- Costos de hosting

---

## 6. Coding Assistants

### ChatGPT Codex
**URL:** https://chatgpt.com/codex

**Funcionalidades:**
- Pair programming con IA
- Generación de código
- Refactorización
- Generación de tests
- Code review

**Mejores prácticas:**
- Proporciona contexto completo
- Especifica lenguaje y versión
- Pide explicaciones, no solo código
- Valida siempre el output

---

## 7. Model Context Protocol (MCP)

### Documentación Oficial
**URL:** https://modelcontextprotocol.io

**Qué es:**
Protocolo estándar para conectar IAs con aplicaciones y datos.

**Servidores MCP populares:**
- `@modelcontextprotocol/server-github` - Gestión de repos
- `@modelcontextprotocol/server-filesystem` - Archivos locales
- `@modelcontextprotocol/server-sqlite` - Bases de datos
- `@modelcontextprotocol/server-brave-search` - Búsqueda web

**Guía completa:** Ver tutorial `09-introduccion-mcp.md`

---

## Buenas Prácticas para el Curso

### 1. Antes de cada laboratorio
- [ ] Prueba el mismo prompt en 2+ plataformas
- [ ] Documenta diferencias en respuestas
- [ ] Identifica fortalezas de cada modelo

### 2. Al trabajar con benchmarks
- [ ] Verifica la **tarea exacta** evaluada
- [ ] Identifica la **métrica** utilizada
- [ ] Considera **idioma** y **dominio** del dataset
- [ ] No compares métricas diferentes

### 3. En proyectos con agentes
- [ ] Define claramente el **rol** de cada agente
- [ ] Especifica **fuentes de datos permitidas**
- [ ] Documenta **flujo de comunicación**
- [ ] Implementa **validación de outputs**

### 4. Documentación obligatoria
Para cada experimento registra:
```markdown
## Experimento: [nombre]

**Objetivo:** [qué quieres lograr]

**Configuración:**
- Modelo: [nombre y versión]
- Temperatura: [valor]
- Max tokens: [valor]
- Otros parámetros: [lista]

**Datos de entrada:**
- Dataset: [nombre/ruta]
- Tamaño: [N ejemplos]
- Preprocesamiento: [pasos]

**Resultados:**
- Métrica 1: [valor]
- Métrica 2: [valor]
- Observaciones: [texto]

**Conclusiones:**
[Qué aprendiste, qué funcionó, qué no]
```

### 5. Gestión de prompts

**Mantén una biblioteca personal:**
```
prompts/
├── data-analysis/
│   ├── exploratory-analysis.md
│   ├── visualization-ideas.md
│   └── statistical-tests.md
├── coding/
│   ├── code-review.md
│   ├── test-generation.md
│   └── refactoring.md
└── writing/
    ├── technical-docs.md
    └── research-summary.md
```

**Formato de cada prompt:**
```markdown
# [Nombre del Prompt]

## Contexto
[Cuándo usarlo]

## Prompt
```
[Texto del prompt con placeholders {variable}]
```

## Ejemplos
- Input: [ejemplo]
- Output: [resultado esperado]

## Notas
- [Tips de uso]
- [Variaciones efectivas]
```

---

## Recursos Adicionales

### Newsletters y Blogs
- **Hugging Face Blog:** https://huggingface.co/blog
- **Anthropic Research:** https://www.anthropic.com/research
- **OpenAI Research:** https://openai.com/research

### Comunidades
- **r/LocalLLaMA:** Comunidad de modelos open-source
- **Hugging Face Discord:** Ayuda con transformers
- **MCP Discord:** Desarrollo con Model Context Protocol

### Papers Fundamentales
1. **Attention Is All You Need** (2017) - Transformers
2. **BERT** (2018) - Embeddings contextuales
3. **GPT-3** (2020) - Few-shot learning
4. **InstructGPT** (2022) - Alignment con RLHF
5. **Constitutional AI** (2022) - Safety y valores

---

## Checklist de Competencias

Al finalizar el curso deberías poder:

### Nivel Básico
- [ ] Usar 3+ plataformas de chat con LLMs
- [ ] Escribir prompts efectivos
- [ ] Interpretar leaderboards de benchmarks
- [ ] Documentar experimentos sistemáticamente

### Nivel Intermedio
- [ ] Comparar modelos con criterios objetivos
- [ ] Diseñar workflows con múltiples agentes
- [ ] Integrar LLMs en aplicaciones (APIs)
- [ ] Optimizar prompts con técnicas avanzadas

### Nivel Avanzado
- [ ] Crear servidores MCP personalizados
- [ ] Implementar RAG (Retrieval Augmented Generation)
- [ ] Fine-tuning de modelos pequeños
- [ ] Evaluar modelos con métricas custom

---

**Última actualización:** 2025-10-29
**Versión:** 1.0
