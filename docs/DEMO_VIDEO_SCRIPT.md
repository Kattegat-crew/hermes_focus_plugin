# Neural Context Plugin — Official Hackathon Demo Video Script
**Target Duration:** 75–85 seconds (Strictly < 90 seconds requirement)  
**Channel:** X (Twitter) post tagging `@NousResearch`  
**Authors:** Jesús Díaz (Jemadiar) & NeuralCrew Labs Architecture Team  
**Event:** Nous Research Hermes Agent Plugin Hackathon (November 11, 2026)

---

## Screen Layout & Visual Setup
- **Visual Style**: Clean, modern dark-mode terminal layout.
- **Top / Left Area**: Interactive Hermes Agent CLI session.
- **Bottom / Right Area**: Diagnostic indicators showing real-time interception, memory probe, and prompt cache status.

---

## Scene 1: The Problem — The Growing Agent Dilemma (00:00 - 00:20)

### Visuals
- An active Hermes Agent session that has been in use over weeks or months.
- Show the agent's growing memory folder (`MEMORY.md`, project logs) and a customized local skill library.
- When the user sends a message, show the agent visibly hesitating: the model is forced to ingest and analyze massive accumulative memory files and large skill catalogs before emitting the first token, driving up wait times and token consumption.

### Audio / Voiceover
> *"As you use your Hermes agent day after day, your system evolves: your memory files accumulate historical facts, and as you personalize your workflows, you install more and more specialized skills.*
> 
> *But over time, you notice a growing penalty: response times slow down, and token consumption climbs higher and higher. Why? Because the agent is forced to read, process, and analyze massive amounts of context and files on every turn—even when only a fraction is relevant to the task at hand."*

---

## Scene 2: The Solution — Intelligent, Selective Context Triage (00:20 - 00:45)

### Visuals
- Introduce **Neural Context Plugin**.
- Show the native `pre_llm_call` hook intercepting incoming messages before the main model runs.
- **Demonstration A (Conversational Greeting)**:
  - User types: `"Hola, ¿cómo estás?"`
  - The Cognitive Arbiter identifies it as `DIRECT_REPLY` in under a millisecond.
  - Hermes replies immediately with **zero unnecessary files loaded and zero extra tokens consumed**.
- **Demonstration B (Task with Shorthand Context)**:
  - User types: `"Revisa lo de AcmeCorp"` (shorthand, doesn't mention the underlying tech stack).

### Audio / Voiceover
> *"Meet Neural Context Plugin.*
> 
> *Instead of blindly dumping your entire memory and skill library into every prompt, Neural Context intercepts the interaction right before inference, performing intelligent, selective triage.*
> 
> *On casual conversation, it exits immediately with zero overhead. But when faced with an actual task, it analyzes true communicative intent and identifies the exact facts and tools needed."*

---

## Scene 3: Two-Phase Enriched Routing & KV-Cache Stability (00:45 - 01:08)

### Visuals
- Highlight the real-time execution log of `"Revisa lo de AcmeCorp"`:
  - **Phase 1 (Surgical Memory Probe)**: Locates `"AcmeCorp"` in memory (whether in standard Markdown or SQLite) and extracts only its specific stack facts: *AcmeCorp uses Meta Conversions API (CAPI) and FastAPI*.
  - **Phase 2 (Grounded Skill Discovery)**: Those domain facts feed the semantic search, instantly ranking the exact winning procedural skill (`meta-pixel-capi`).
  - **Single-Step Ephemeral Assembly**: Both the domain facts and the complete procedural guidance are delivered directly in the user turn.
  - Show the **System Prompt remaining 100% byte-stable**, guaranteeing maximal prompt-cache hits and minimal latency.

### Audio / Voiceover
> *"In two coordinated phases, selective memory grounds tool discovery:*
> 
> *First, it pulls the specific domain facts for AcmeCorp—discovering its backend stack without loading unrelated projects.*
> *Second, it uses those facts to pinpoint the exact winning procedural skill, injecting both into the user turn in a single step.*
> 
> *By keeping the base System Prompt completely stable, it maximizes KV-cache hits, significantly cutting response latency and reducing token consumption on every turn—without ever truncating the procedural depth you need."*

---

## Scene 4: Universal Adaptability, CLI & Outro (01:08 - 01:25)

### Visuals
- Quick terminal run of the built-in management commands:
  ```bash
  hermes neural-context config
  hermes plugins validate .
  hermes plugins doctor .
  ```
- Highlight:
  - **35/35 unit tests passing**
  - **15/15 Hermes plugin validation checks green**
  - Preserves full context by default (`0 = unconstrained`), with customizable ceilings.

### Audio / Voiceover
> *"Universal adaptability across standard Hermes Markdown and SQLite stores, 35 unit tests passing, and 100% compliant with official Hermes plugin standards.*
> 
> *Give your agent surgical context. Install it today with `hermes plugins install Kattegat-crew/neural-context-plugin`."*

---

## Script Timing Breakdown
| Section | Visual Focus | Target Duration |
| :--- | :--- | :--- |
| **Scene 1: The Problem** | Memory growth, accumulating skills, rising latency & token overhead | 00:00 – 00:20 (20s) |
| **Scene 2: The Solution** | Pre-inference triage, fast-exit on greetings, intent detection | 00:20 – 00:45 (25s) |
| **Scene 3: Core Engine** | Two-phase memory-to-skill routing, KV-cache stability, single-step injection | 00:45 – 01:08 (23s) |
| **Scene 4: Outro & Proof** | CLI config, official doctor/validation passing, install command | 01:08 – 01:25 (17s) |
| **Total Duration** | | **85 seconds** |

---

## Guion en Español (Opción para Grabación de Voz / Subtítulos)

### Escena 1: El Dilema del Agente Maduro (00:00 - 00:20)
> *"A medida que usas y personalizas tu agente Hermes día tras día, su memoria se va expandiendo y vas instalando cada vez más skills para cubrir nuevas necesidades.*
> 
> *Pero con el tiempo notas un problema silencioso: las respuestas tardan más en llegar y el consumo de tokens en cada turno es cada vez mayor. ¿Por qué ocurre esto? Porque en cada interacción el agente debe leer, procesar y analizar acumulaciones masivas de memoria y catálogos completos de herramientas que muchas veces ni siquiera necesita para la tarea actual."*

### Escena 2: Análisis y Carga Inteligente (00:20 - 00:45)
> *"Para resolver esto creamos Neural Context Plugin.*
> 
> *En lugar de sobrecargar al modelo con todo el historial y todas las skills en cada consulta, nuestro plugin intercepta la interacción justo antes de la inferencia, realizando un análisis y selección inteligente.*
> 
> *Si el usuario simplemente saluda o conversa, el plugin lo detecta al instante y sale sin inyectar nada innecesario. Y cuando detecta una tarea real, interpreta la intención comunicativa y localiza únicamente los hechos y herramientas esenciales."*

### Escena 3: Enrutamiento en Dos Fases y Estabilidad de Caché (00:45 - 01:08)
> *"La innovación central opera en dos fases:*
> 
> *Primero, extrae de la memoria únicamente los hechos del proyecto mencionado —descubriendo su arquitectura técnica sin traer proyectos ajenos.*
> *Segundo, utiliza ese contexto para encontrar la skill procedural exacta, inyectando ambos elementos en un solo paso directamente en el turno del usuario.*
> 
> *Al mantener el System Prompt completamente inmutable, se aprovecha al máximo el caché del modelo, reduciendo notablemente los tiempos de espera y recortando de forma sustancial el consumo de tokens, sin recortar jamás la profundidad técnica que necesitas."*

### Escena 4: Adaptabilidad Universal y Cierre (01:08 - 01:25)
> *"Adaptable tanto al estándar de Markdown de Hermes como a almacenes SQLite, con 35 pruebas unitarias verificadas y validación oficial aprobada al 100%.*
> 
> *Dale a tu agente contexto quirúrgico. Instálalo hoy con `hermes plugins install Kattegat-crew/neural-context-plugin`."*

---

## Accompanying X (Twitter) Post Copy
```text
Tired of your Hermes Agent slowing down and burning tokens as your memory and skill catalog grow? 🧠⚡

Introducing Neural Context Plugin for @NousResearch Hermes Agent:

Problem:
As you customize your agent, memory files expand and skills multiply. Reading massive context on every turn drives up latency and wastes tokens—even on simple greetings.

Solution:
Neural Context intercepts turns before inference:
✅ Cognitive Arbiter: Instant zero-token fast exit on casual chat
✅ Two-Phase Enriched Routing: Surgical memory facts ground precise skill discovery
✅ Single-Step Ephemeral Delivery: Preserves byte-stable System Prompt for maximum KV-cache hits
✅ Universal Adaptability: Works seamlessly with vanilla Hermes Markdown & SQLite stores
✅ Preserves 100% of procedural depth with zero arbitrary truncation

15/15 Hermes plugin validation checks passed. 35/35 unit tests green.

Install now:
hermes plugins install Kattegat-crew/neural-context-plugin

GitHub: https://github.com/Kattegat-crew/neural-context-plugin

#HermesAgent #NousResearch #AI #AgenticAI #OpenSource #ContextEngineering
```
