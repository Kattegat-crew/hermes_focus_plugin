# Focus (`focus`)

[![Hermes Agent Plugin](https://img.shields.io/badge/hermes--agent-plugin%20v1-blue.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Validation](https://img.shields.io/badge/hermes%20plugins%20validate-15%2F15%20passing-brightgreen.svg)]()
[![Doctor](https://img.shields.io/badge/hermes%20plugins%20doctor-healthy-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/pytest-60%20passed-brightgreen.svg)]()

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
        │ - BM25 ranking over the local SQLite sidecar index      │
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

- **Cognitive Arbiter (4 Quadrants)**: Classifies turns into `DIRECT_REPLY`, `SKILL_ONLY`, `MEMORY_ONLY`, or `ENRICHED_TASK`. Host model calls can add latency; provider constraints still apply.
- **Sequential Two-Phase Grounding**: When requested by triage, queries memory and history before skill routing. Discovered facts can enrich the semantic skill query.
- **Model-call disclosure**: Host-backed triage and host-LLM skill routing are separate decisions and may issue an extra model call when both stages use the host model.
- **Symmetric Bipartite Personalized PageRank**: Unifies multi-hop concept discovery across both procedural skills (`GraphRouter`) and historical session databases (`SelectiveMemoryRetriever`). Propagates activation across entity seeds and discriminative bridge concepts without cross-topic contamination.
- **Multilingual Text Processing**: Uses Unicode-aware tokenization and normalization; quality depends on the underlying model and catalog.
- **Configurable Budgets**: Memory/context ceilings and plugin timeouts are optional explicit overrides. A zero value leaves token limits to the host/provider and does not impose a plugin deadline. A configured positive model-token limit can trigger one bounded retry after truncation.
- **Zero KV-Cache Mutation**: Injects context exclusively into the ephemeral user message turn. The System Prompt remains byte-stable, maximizing prompt cache hits and minimizing latency.
- **Zero Duplication of Default System Memory**: Leaves default Hermes memory files (`MEMORY.md`, `USER.md`, `SOUL.md`, `AGENTS.md`) untouched so they are never duplicated in context. Inspects Hermes `state.db` (excluding active `session_id`), `engram.db`, and external SQL databases.
- **Atomic Hermes Host Persistence**: Implements `ctx.set_config` for safe, atomic configuration updates in compliance with Hermes plugin lifecycle contracts.
- **Auditable Receipts**: Emits structured JSON records to `receipts.jsonl` with estimated tokens from the rendered context, measured plugin triage elapsed time, configured and executed routing backend, and status/error details. Estimates are not provider token accounting.

---

## Installation & CLI Usage

### Install into Hermes Agent
```bash
hermes plugins install Kattegat-crew/hermes_focus_plugin
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
        router_backend: "graph"         # "graph" (default, bipartite Personalized PageRank) | "host_llm"
        max_context_tokens: 0           # No plugin ceiling; host/provider limits still apply
        memory_token_budget: 0          # No plugin memory ceiling
        selective_memory_enabled: true  # Surgical memory probing based on extracted entities
        software_inspection_enabled: true
        timeout_ms: 0               # 0 leaves timeout control to host/provider
```

---

## Security & Fail-Closed Boundaries

- **Untrusted Retrieved Text**: Skill text is escaped before inclusion in the `<neural_context>` wrapper. Skill guidance and historical memory facts are labeled separately; this is a prompt boundary, not an injection-proof guarantee.
- **Explicit Failure Reporting**: Triage failures retain an error cause and elapsed time and fail closed to a direct reply. A plugin timeout is only applied when explicitly configured; host/provider limits may still apply.
- **Zero File Mutation**: User `SKILL.md` catalogs and workspace files are strictly immutable. Content hashes and procedural descriptions are stored in a local SQLite sidecar (`sidecar_index.db`).
- **Read-Only Memory Sandboxing**: Memory scans enforce strict directory allowlists, symlink boundary checks, and escaped wildcard queries (`ESCAPE '\'`), preventing directory traversal or query injection.

---

## Verification & Tests

```bash
# Run pytest test suite (60 passing tests)
pytest -v

# Run official Hermes Plugin validation (15/15 checks passed)
hermes plugins validate . --json

# Run Hermes Plugin doctor
hermes plugins doctor .
```

---

## License

MIT License. Designed and authored by Jesús Díaz & NeuralCrew Labs.
