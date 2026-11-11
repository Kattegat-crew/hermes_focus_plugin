# Neural Context Plugin (`neural-context-plugin`)

[![Hermes Agent Plugin](https://img.shields.io/badge/hermes--agent-plugin%20v1-blue.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Validation](https://img.shields.io/badge/hermes%20plugins%20validate-passing-brightgreen.svg)]()
[![Doctor](https://img.shields.io/badge/hermes%20plugins%20doctor-healthy-brightgreen.svg)]()

> **Cognitive semantic context engine for Hermes Agent**: intercepts turns prior to inference, extracts true intent and entities, executes Two-Phase Context-Enriched routing, prioritizes repository governance, and injects winning skills in a single step with zero KV-cache mutation.

---

## The Problem: The Mature Agent Dilemma

In standard LLM agent systems, context provisioning faces an architectural dilemma that compounds as the deployment grows:

1. **Static Bloat**: As users install specialized skills over time (from dozens to hundreds of skills), injecting the static skill catalog into the System Prompt costs ~25–30 tokens per skill on **every single turn**. A catalog of 500 skills wastes ~12,000–15,000 tokens per interaction and degrades time-to-first-token (TTFT).
2. **Cold Start / Blind Routing**: Running in `on_demand: true` strips all context, forcing the model to guess which skills exist and burn extra round-trips calling `skill_search`.
3. **The Failure of Raw Lexical Matching**: Real-world user prompts are frequently ambiguous, conversational, or written in another language (e.g., Spanish requests against an English catalog). Users describe **goals**, not skill names. Naive keyword search fails in nearly 100% of ambiguous queries.

---

## The Architecture: Two-Phase Context-Enriched Routing

`neural-context-plugin` operates at the native `pre_llm_call` hook:

```
                          [ Incoming User Message ]
                                     │
                                     ▼
                      [ Native Hook: pre_llm_call ]
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │        STEP 1: COGNITIVE SEMANTIC INTERPRETER           │
        │  - Evaluates message intent with model (ctx.llm)        │
        │  - Synthesizes dual-form English search query           │
        │  - Extracts named entities (brands, projects, clients)  │
        │  - Determines if task requires procedural skills/memory │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │   STEP 2: TWO-PHASE CONTEXT-ENRICHED ROUTING PIPELINE   │
        │                                                         │
        │ Phase 1: Selective Memory Query (Domain Ground Truth)   │
        │  - Queries memory strictly for extracted named entities │
        │  - Extracts factual hints (e.g. "Lucky uses Meta CAPI") │
        │                                                         │
        │ Phase 2: Memory-Enriched Skill Search & Routing         │
        │  - Injects Phase 1 domain hints into skill search query │
        │  - BM25 candidate shortlisting over sidecar SQLite index│
        │  - Single winning skill selected (HostLLM/Heuristic)    │
        │  - Local workspace AGENTS.md/.skills/ sovereignty       │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │     STEP 3: SINGLE-STEP SKILL & MEMORY INGESTION        │
        │  - Ingest winning SKILL.md body immediately             │
        │  - Eliminates secondary round-trip tool calls           │
        │  - Enforces shared token budget (≤ 1,200 tokens total)  │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │      STEP 4: EPHEMERAL INJECTION & AUDIT RECEIPT        │
        │  - Injects context into USER message turn               │
        │  - System Prompt remains 100% BYTE-STABLE (KV-Cache OK) │
        │  - Emits verifiable audit record to `receipts.jsonl`    │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                     [ Primary Agent LLM Executes ]
```

---

## Key Differentiators

- **Two-Phase Context-Enriched Routing**: Memory facts and skill procedures are not isolated silos. Extracted named entities query memory first to establish domain ground truth (e.g. knowing Lucky uses Meta CAPI with FastAPI), which enriches the skill search to pinpoint the exact winning procedural skill.
- **Preserves KV-Cache Stability**: Unlike mechanisms that alter the System Prompt, returning ephemeral context from `pre_llm_call` appends to the user turn, keeping the system prompt byte-stable for maximum prompt-cache hits and lowest TTFT.
- **Single-Step Ingestion**: Unlike existing routers that only emit a tag with the skill name (forcing the agent into a second round-trip costing ~20,000 tokens of re-sent context), Neural Context inlines the winning skill procedure immediately.
- **Repository Governance Sovereignty**: When working in a software repository with `AGENTS.md` and local `.skills/`, local project rules take absolute precedence over global catalog skills.
- **Zero File Mutation & SHA-256 Sidecar Index**: User `SKILL.md` files are never touched or modified. All metadata is cached in SQLite (`sidecar_index.db`) and invalidated deterministically by file hash.
- **Verifiable Audit Receipts**: Every turn writes a structured record to `receipts.jsonl` detailing execution latency, tokens saved, gate scores, and fallback status.

---

## Installation & CLI Usage

### Install into Hermes Agent
```bash
hermes plugins install Kattegat-crew/neural-context-plugin
```

### CLI Subcommands
```bash
# Check status and metrics
hermes neural-context status

# Run system health diagnostics
hermes neural-context doctor

# Index local skills into sidecar SQLite cache
hermes neural-context index

# Verify recent turn receipts and token savings
hermes neural-context verify --limit 20
```

---

## Configuration (`config.yaml`)

```yaml
plugins:
  entries:
    neural-context-plugin:
      enabled: true
      settings:
        router_backend: "host_llm"      # "host_llm" (default, uses ctx.llm) | "heuristic" | "jev"
        max_context_tokens: 1200        # Maximum tokens injected per turn
        selective_memory_enabled: true  # Surgical recall based on detected entities
        memory_token_budget: 400        # Maximum tokens for memory facts
        software_inspection_enabled: true
        timeout_ms: 250
```

---

## Verification & Tests

```bash
# Run test suite
pytest -v

# Run Hermes Plugin validation
hermes plugins validate .

# Run Hermes Plugin doctor
hermes plugins doctor .
```

---

## License

MIT License. Designed and authored by Jesús Díaz & NeuralCrew Labs Architecture Team.
