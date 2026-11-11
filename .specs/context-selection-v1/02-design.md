# Context Selection v1 — Technical Design

**Status:** Approved by Jesús Díaz (Jemadiar); aligned with Two-Phase Context-Enriched Routing architecture.  
**Phase:** 2 of the repository's four-phase SDD lifecycle.  
**Decision recorded:** Technical design approved for Two-Phase Context-Enriched Routing, Hermes Agent integration contracts, and Hackathon operational readiness.

---

## 1. Executive Architecture Summary

`neural-context-plugin` acts as a cognitive context engine for Hermes Agent, intercepting turns at the native `pre_llm_call` hook.
Instead of treating skills and memories as isolated silos, it executes a **Two-Phase Context-Enriched Routing Pipeline**:

```
                             [ Incoming User Message ]
                      (Discord, WhatsApp, CLI, API Gateway)
                                        │
                                        ▼
                         [ Native Hook: pre_llm_call ]
                   (Turn intercepted before primary inference)
                                        │
                                        ▼
           ┌─────────────────────────────────────────────────────────┐
           │            STEP 1: COGNITIVE ARBITER                    │
           │  - Pre-inference evaluation on active model (ctx.llm)   │
           │  - Classifies into 4 Quadrants:                         │
           │    Q1: Direct Reply (Fast Exit, 0 tokens)               │
           │    Q2: Skill Only (No memory dependency)                │
           │    Q3: Memory Only (Fact / agreement recall)            │
           │    Q4: Enriched Task (Memory probe -> Skill route)      │
           │  - Dimensional Query Synthesis:                         │
           │    * Memory Probe: (subject, platform, temporal_anchor) │
           │    * Skill Dispatch: (canonical action, capabilities)   │
           └────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
           ┌─────────────────────────────────────────────────────────┐
           │            STEP 2: CONTEXT BROKER ORCHESTRATION         │
           │                                                         │
           │ Phase 1: Surgical Memory Probe (Domain Ground Truth)    │
           │  - Resolves bounded paths (~/.hermes/memories/, etc.)   │
           │  - Extracts factual anchors (e.g. AcmeCorp stack)       │
           │                                                         │
           │ Phase 2: Information-Weighted Graph Routing             │
           │  - Spreading Activation / Personalized PageRank         │
           │  - Multi-hop traversal: Entity -> Domain -> Winning Skill│
           │  - Ambiguity Gate: Escalates to Host LLM on tie         │
           │  - Local workspace AGENTS.md/.skills/ sovereignty       │
           └────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
           ┌─────────────────────────────────────────────────────────┐
           │        STEP 3: SINGLE-STEP SKILL & MEMORY INGESTION     │
           │  - Ingest winning SKILL.md body immediately             │
           │  - Combine procedural skill with active memory facts    │
           │  - Dynamic token optimization: minimize per turn        │
           │    without sacrificing necessary procedural depth       │
           └────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
           ┌─────────────────────────────────────────────────────────┐
           │         STEP 4: EPHEMERAL INJECTION & AUDIT RECEIPT     │
           │  - Inject context into USER message turn                │
           │  - System Prompt remains 100% BYTE-STABLE (KV-Cache OK) │
           │  - Emit verifiable audit record to receipts.jsonl       │
           │  - Record channel platform and active model provenance  │
           └────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                        [ Primary Agent LLM Executes ]
                 (Zero hallucination, optimal TTFT, minimal cost)
```

---

## 2. Verified Hermes Host Contract & Runtime Seams

The design strictly complies with source-verified contracts in `/opt/hermes-agent`:

| Runtime Surface | Verified File & Line | Contract & Design Implementation |
|---|---|---|
| **Hook Registration** | `/opt/hermes-agent/hermes_cli/plugins_context.py:741` | `ctx.register_hook("pre_llm_call", handler)` and `ctx.register_hook("post_llm_call", handler)`. |
| **Hook Inputs & Injection** | `/opt/hermes-agent/agent/turn_context.py:910-964` | `pre_llm_call` receives `user_message`, `session_id`, `turn_id`, etc. Returning `{"context": "..."}` appends context directly to the user turn, keeping the system prompt byte-stable. |
| **Fail-Closed Boundary** | `/opt/hermes-agent/agent/turn_context.py:920-964` | Returning `None` on errors or fast-exit allows the host turn to continue normally. Never return `{"action": "block"}`. |
| **Host LLM Facade** | `/opt/hermes-agent/agent/plugin_llm.py:486-514` | `ctx.llm.complete(messages=list[dict], max_tokens=int, temperature=float)` returns `PluginLlmCompleteResult` with `.text`. Structured messages list `[{"role": "user", "content": prompt}]` is required. |
| **Tool Registration** | `/opt/hermes-agent/tools/registry.py:832-851` | Tool schemas must be unwrapped dictionaries with top-level `parameters`: `{"name": str, "description": str, "parameters": {"type": "object", ...}}`. |
| **CLI Registration** | `/opt/hermes-agent/hermes_cli/plugins_context.py:442` | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` connects `hermes neural-context <subcommand>` by passing the subparser to `setup_fn`. |
| **Storage Isolation** | `/opt/hermes-agent/plugins/plugin_storage.py:27-46` | Data stored in `<HERMES_HOME>/plugin-data/neural-context-plugin/` isolated per active profile. |

---

## 3. Subsystem Technical Specifications

### 3.1 Cognitive Arbiter (`neural_context/triage.py`)
- **Model-First Evaluation**: Every message is analyzed by calling `ctx.llm.complete(messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=0.0)`.
- **4-Quadrant Decision Tree**:
  - `Q1: DIRECT_REPLY`: Greetings, chit-chat, conceptual questions. Returns `needs_skills=False, needs_memory=False` for immediate sub-5ms exit.
  - `Q2: SKILL_ONLY`: Direct action/task without historical memory dependency. Bypasses memory search and routes skills directly.
  - `Q3: MEMORY_ONLY`: Recalls past agreements, client facts, or preferences. Probes memory and bypasses skill catalog.
  - `Q4: ENRICHED_TASK`: Complex domain tasks tied to history. Probes memory first, synthesizes enriched query, then routes skills.
- **Dimensional Query Synthesis**:
  - `Memory Probe`: Structured entity and temporal/platform scope.
  - `Skill Dispatch Intent`: Canonical English action summary and technical capability nouns (`kebab-case`).

### 3.2 Selective Memory Retriever (`neural_context/memory.py`)
- **Entity-Driven Targeting**: Instead of full memory dumps, only facts matching `extracted_entities` are retrieved.
- **Bounded Search Roots**: Looks strictly in `PROJECT_MEMORY.md`, `MEMORY.md`, `USER.md`, and `~/.hermes/memories/*.md`.
- **Adversarial Content Sanitization**: Neutralizes unauthorized instruction override attempts and malicious system tags from memory lines.
- **Token Ceiling**: Memory injection is strictly capped at `memory_token_budget` (default: 400 tokens).

### 3.3 Sidecar SQLite Index (`neural_context/sidecar.py`)
- **Zero File Mutation**: `SKILL.md` files are read-only; user files are never touched.
- **Deterministic Hash Invalidation**: Indexed entries store `sha256(SKILL.md)`. Files are reindexed only when content changes.
- **BM25 In-Process Scoring**: Evaluates candidate matches over frontmatter description and lead paragraph in sub-5ms with zero external daemon dependencies.

### 3.4 Context Broker & Graph Router (`neural_context/router.py`)
- **Context Broker Orchestration**:
  1. *Phase 1*: Probes memory for target entities. If facts exist, builds domain ground-truth hints.
  2. *Phase 2*: Enriches skill query with domain hints, retrieves BM25 shortlist, and selects the winning skill.
  3. *Phase 3*: Injects winning skill procedure (frontmatter stripped) + active memory facts.
- **Information-Weighted Graph Router (`GraphRouter`)**:
  - Constructs a bipartite concept-skill network.
  - Applies spreading activation (Personalized PageRank) from memory domain concepts and task capability seeds.
  - Penalizes high-frequency hub tokens using Information Content / TF-IDF.
  - Ambiguity Gate: If top candidates have close scores ($\Delta < \tau$), falls back to Host LLM (`ctx.llm`).
  - Zero external API keys, 100% free and local, sub-millisecond execution.
- **Adaptive Token Optimization**: Dynamically bounds injected context (`max_context_tokens`, default: 2,500) with surgical line pruning at step boundaries to preserve complete technical workflows without arbitrary truncations.
- **Workspace Sovereignty**: Sovereign workspace skills (`AGENTS.md` and `.skills/`) take priority over global fleet skills.

### 3.5 Verifiable Audit Receipts (`neural_context/receipt.py`)
- **Location**: Written to `<plugin_data_dir>/receipts.jsonl`.
- **Payload (`ReceiptRecord`)**: Records `turn_id`, `session_id`, `platform` (CLI, Telegram, Discord, WhatsApp, Desktop), `model` (active host model), `timestamp`, `is_task`, `entities_detected`, `selected_skill`, `skill_injected_tokens`, `memory_injected_tokens`, `tokens_saved_estimate`, `triage_latency_ms`, `status`.
- **CLI Commands**:
  - `hermes neural-context status`: Overall summary.
  - `hermes neural-context doctor`: System health and permissions check.
  - `hermes neural-context index`: Re-sync skills into sidecar SQLite cache.
  - `hermes neural-context verify --limit N`: Review recent turn receipts and verified token savings.

---

## 4. Multi-Turn & Concurrency State Isolation

In concurrent sessions (gateway, multi-user web, or background cron):
- State is keyed in `_session_turns: Dict[str, Tuple[TriageAnalysis, RouteResult]]` by `f"{session_id}:{turn_id}"`.
- `on_pre_llm_call` stores the turn analysis and route result under its unique key.
- `on_post_llm_call` pops the matching key to record the receipt, preventing cross-talk across sessions or turns.

---

## 5. Configuration Schema (`plugin.yaml` & `config.py`)

```yaml
config_schema:
  enabled:
    type: bool
    default: true
  pre_hook_enabled:
    type: bool
    default: true
  router_backend:
    type: str
    default: "host_llm"
  max_context_tokens:
    type: int
    default: 1200
  selective_memory_enabled:
    type: bool
    default: true
  memory_token_budget:
    type: int
    default: 400
  software_inspection_enabled:
    type: bool
    default: true
  timeout_ms:
    type: int
    default: 250
```

---

## 6. Security & Fail-Closed Boundaries

1. **Untrusted Markdown Sandboxing**: Skill and memory contents are wrapped inside `<neural_context>` tags with explicit directives that procedural context cannot override safety constraints or execute unauthorized destructive actions.
2. **Fail-Closed on Error**: If any component fails (model timeout, JSON parse error, unreadable file), the hook catches the exception, logs a warning, and returns `None`. Hermes executes normally without interruption.
3. **Zero Third-Party Leaks**: In `host_llm` mode, all calls route through the user's configured Hermes model endpoint. No third-party network egress occurs.

---

## 7. Approval & Traceability

- **Phase 1 Spec (`01-spec.md`)**: Approved.
- **Phase 2 Design (`02-design.md`)**: Approved.
- **Phase 3 Tasks (`03-tasks.md`)**: Ready for authoring.
