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
           │        STEP 1: COGNITIVE SEMANTIC INTERPRETER           │
           │  - Model evaluates communicative intent (ctx.llm)       │
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
           │        STEP 3: SINGLE-STEP SKILL & MEMORY INGESTION     │
           │  - Ingest winning SKILL.md body immediately             │
           │  - Combine procedural skill with active memory facts    │
           │  - Enforce shared token budget (≤ 1,200 tokens total)   │
           └────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
           ┌─────────────────────────────────────────────────────────┐
           │         STEP 4: EPHEMERAL INJECTION & AUDIT RECEIPT     │
           │  - Inject context into USER message turn                │
           │  - System Prompt remains 100% BYTE-STABLE (KV-Cache OK) │
           │  - Emit verifiable audit record to receipts.jsonl       │
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
| **Storage Isolation** | `/opt/hermes-agent/plugins/plugin_storage.py:27-46` | Data stored in `<HERMES_HOME>/plugins/neural-context-plugin/` isolated per active profile. |

---

## 3. Subsystem Technical Specifications

### 3.1 Cognitive Semantic Interpreter (`neural_context/triage.py`)
- **Model-First Evaluation**: Every message is analyzed by calling `ctx.llm.complete(messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=0.0)`.
- **Zero Static Wordlists**: No brittle hardcoded keyword lists or regexes.
- **Structured Schema (`TriageAnalysis`)**:
  ```python
  class TriageAnalysis(BaseModel):
      is_actionable_task: bool
      needs_skills: bool
      needs_memory: bool
      intent_summary_en: str
      search_query: str
      required_capabilities: List[str]
      extracted_entities: List[str]
      domain: DomainClassification
      reasoning: str
      confidence: float
  ```
- **Conversational Fast Exit**: If `is_actionable_task` is False or neither skills nor memory are needed, triage returns `needs_skills=False, needs_memory=False`. The coordinator exits in < 2ms, returning `None`.

### 3.2 Selective Memory Retriever (`neural_context/memory.py`)
- **Entity-Driven Targeting**: Instead of full memory dumps, only facts matching `extracted_entities` (length ≥ 3) are retrieved.
- **Search Roots**: Looks in current workspace `cwd`, `~/.hermes/`, `~/.hermes/memories/`, `~/.hermes/shared/` for `MEMORY.md`, `USER.md`, and `PROJECT_MEMORY.md`.
- **Token Ceiling**: Memory injection is strictly capped at `memory_token_budget` (default: 400 tokens).

### 3.3 Sidecar SQLite Index (`neural_context/sidecar.py`)
- **Zero File Mutation**: `SKILL.md` files are read-only; user files are never touched.
- **Deterministic Hash Invalidation**: Indexed entries store `sha256(SKILL.md)`. Files are reindexed only when content changes.
- **BM25 In-Process Scoring**: Evaluates candidate matches over frontmatter description and lead paragraph in sub-5ms with zero external daemon dependencies.

### 3.4 Skill Router & Coordinator (`neural_context/router.py`)
- **Two-Phase Enriched Routing**:
  1. *Phase 1*: Query memory for `extracted_entities`. If facts exist, build `memory_hints = " ".join([f"{m.entity}: {m.fact}" for m in memories])`.
  2. *Phase 2*: Combine `search_query`, `intent_summary_en`, `required_capabilities`, and `memory_hints` into an enriched query. BM25 shortlists candidates, and `HostLLMRouter` selects the winner.
  3. *Phase 3*: Assemble `<neural_context>` block with winning skill procedure (frontmatter stripped) + active memory facts.
- **Workspace Sovereignty**: Sovereign workspace skills (`AGENTS.md` and `.skills/`) take priority over global fleet skills.

### 3.5 Verifiable Audit Receipts (`neural_context/receipt.py`)
- **Location**: Written to `<plugin_data_dir>/receipts.jsonl`.
- **Payload (`ReceiptRecord`)**: Records `turn_id`, `session_id`, `timestamp`, `is_task`, `entities_detected`, `selected_skill`, `skill_injected_tokens`, `memory_injected_tokens`, `tokens_saved_estimate`, `triage_latency_ms`, `status`.
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
