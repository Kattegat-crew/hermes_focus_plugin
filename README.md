# Focus (`focus`)

[![Hermes Agent Plugin](https://img.shields.io/badge/hermes--agent-plugin%20v1-blue.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Validation](https://img.shields.io/badge/hermes%20plugins%20validate-15%2F15%20passing-brightgreen.svg)]()
[![Doctor](https://img.shields.io/badge/hermes%20plugins%20doctor-healthy-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/pytest-35%20passed-brightgreen.svg)]()

> **Cognitive context engine for Hermes Agent**: intercepts turns prior to inference, classifies user intent into four cognitive routing quadrants, grounds skill discovery in universal memory facts, and injects winning procedural guidance in a single step with zero KV-cache mutation.

---

## The Problem: The Mature Agent Dilemma

In production LLM agent environments, context provisioning faces compound scaling challenges as deployments evolve:

1. **Static Catalog Bloat**: Loading an entire skill catalog into the System Prompt consumes significant token overhead on every single turn, increasing operational costs and degrading Time-to-First-Token (TTFT).
2. **Cold Start & Blind Routing**: Disabling catalog loading forces the model to guess which skills exist and waste additional tool-call round trips searching for them.
3. **Semantic Mismatches**: User prompts are frequently conversational, cross-lingual, or describe high-level goals without naming specific tools. Naive lexical search fails when prompts omit canonical keywords.
4. **Context Amputation**: Hardcoded token truncations often clip critical step-by-step instructions or architecture facts when agents handle complex tasks.

---

## The Architecture: Two-Phase Context-Enriched Routing

`focus` hooks natively into `pre_llm_call` to resolve context before the primary agent model executes:

```
                          [ Incoming User Message ]
                                     │
                                     ▼
                      [ Native Hook: pre_llm_call ]
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ STEP 1: COGNITIVE ARBITER & TRIAGE                      │
        │ - Model-driven intent interpretation (ctx.llm)          │
        │ - 4-Quadrant Routing Decision:                          │
        │   • DIRECT_REPLY: Casual greeting ➔ Fast exit (0 tokens)│
        │   • SKILL_ONLY: Procedural execution, no history needed │
        │   • MEMORY_ONLY: Fact recall, bypasses skill overhead   │
        │   • ENRICHED_TASK: Complex task ➔ Two-phase routing    │
        │ - Synthesizes broad semantic concepts & entities        │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ STEP 2: TWO-PHASE CONTEXT ENRICHMENT                    │
        │                                                         │
        │ Phase 1: Universal Surgical Memory Probe                │
        │ - Reads vanilla Hermes Markdown (PROJECT_MEMORY, USER)  │
        │ - Recursively scans memories/**/*.md & custom env paths │
        │ - Read-only safe inspection of SQLite stores (mode=ro)  │
        │ - Strict path-traversal & symlink boundary defenses     │
        │                                                         │
        │ Phase 2: Grounded Skill Routing                         │
        │ - Enriches skill search with discovered domain facts    │
        │ - Bipartite Concept Graph spreading activation          │
        │ - Sub-5ms BM25 ranking over SQLite sidecar index        │
        │ - Skill gap detection: suggests missing capabilities    │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ STEP 3: SINGLE-STEP EPHEMERAL INJECTION                 │
        │ - Full procedural depth preserved by default (unlimited)│
        │ - Optional configurable ceilings for constrained models │
        │ - Ingests complete winning SKILL.md immediately         │
        │ - Eliminates secondary tool round-trips                 │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ STEP 4: USER TURN INJECTION & AUDIT RECEIPT             │
        │ - Appends context block strictly to the USER message    │
        │ - System Prompt remains 100% BYTE-STABLE (KV-cache safe)│
        │ - Emits auditable execution record to `receipts.jsonl`  │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                     [ Primary Agent LLM Executes ]
```

---

## Key Differentiators

- **Cognitive Arbiter (4 Quadrants)**: Classifies every user turn into `DIRECT_REPLY`, `SKILL_ONLY`, `MEMORY_ONLY`, or `ENRICHED_TASK`. Casual conversational turns exit in $<1\text{ ms}$ with 0 tokens injected.
- **Universal Memory Adaptability**: Works out of the box with standard vanilla Hermes Markdown (`MEMORY.md`, `USER.md`, `PROJECT_MEMORY.md`, `memories/**/*.md`), supports custom directory configurations (`HERMES_MEMORY_DIR`), and provides read-only probing for SQLite stores (`memories.db`, `state.db`) without locking or schema dependencies.
- **Preserves Full Context by Default**: Does not enforce artificial token ceilings by default (`0` = unlimited). Procedural depth and domain facts are fully preserved, with optional user-configurable budgets for smaller models.
- **Zero KV-Cache Mutation**: Injects context exclusively into the ephemeral user message turn. The System Prompt remains byte-stable, maximizing prompt cache hits and minimizing latency.
- **Single-Step Procedural Ingestion**: Directly provides the winning skill body, eliminating the multi-turn latency and context resending of secondary search tools.
- **Information-Weighted Graph Routing**: Uses bipartite concept spreading activation in memory to resolve multi-hop relationships (e.g. project codename $\to$ tech stack $\to$ matching skill) with zero external API dependencies.
- **Skill Gap Detection**: When an actionable task genuinely requires a capability absent from the catalog, the arbiter avoids false positive injections and returns an explicit capability notice.
- **Zero File Mutation**: User skill repositories and profile files are never altered. Metadata is cached in a local SQLite sidecar (`sidecar_index.db`) keyed by SHA-256 hashes.
- **Auditable Receipts**: Emits structured JSON records to `receipts.jsonl` tracking route decisions, injected tokens, latency, and model provenance.

---

## Installation & CLI Usage

### Install into Hermes Agent
```bash
hermes plugins install Kattegat-crew/neural-context-plugin
```

### CLI Subcommands
```bash
# Check status and health metrics
hermes focus status

# Run plugin diagnostic checks
hermes focus doctor

# Index local and catalog skills into sidecar SQLite cache
hermes focus index

# View or update plugin settings directly from the terminal
hermes focus config
hermes focus config --set max_context_tokens=1500

# Verify recent turn receipts and routing history
hermes focus verify --limit 20
```

---

## Configuration (`config.yaml`)

Configure `focus` within your Hermes `config.yaml`:

```yaml
plugins:
  entries:
    focus:
      enabled: true
      settings:
        router_backend: "host_llm"      # "host_llm" (default, uses ctx.llm) | "graph" | "jev"
        max_context_tokens: 0           # 0 for unlimited (default, preserves full procedural depth)
        memory_token_budget: 0          # 0 for unlimited (default, preserves all matching facts)
        selective_memory_enabled: true  # Surgical memory probing based on extracted entities
        software_inspection_enabled: true
        timeout_ms: 250
```

---

## Verification & Tests

```bash
# Run pytest test suite (35 passing tests)
pytest -v

# Run official Hermes Plugin validation (15/15 checks passed)
hermes plugins validate . --json

# Run official Hermes Plugin doctor (healthy)
hermes plugins doctor .
```

---

## License

MIT License. Designed and authored by Jesús Díaz & NeuralCrew Labs.
