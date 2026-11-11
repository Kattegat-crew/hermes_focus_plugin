# Context Selection v1 — Specification

**Status:** Approved by Jesús Díaz (Jemadiar); aligned with Two-Phase Context-Enriched Routing architecture.  
**Phase:** 1 of the repository's four-phase SDD lifecycle.  
**Decision recorded:** Gate 1 approved the Two-Phase Context-Enriched Routing architecture and Hermes Agent plugin integration contract.

## 1. Purpose and problem hypothesis

This specification defines the cognitive context engine for Hermes Agent (`neural-context-plugin`).
It intercepts turns prior to model execution at the native `pre_llm_call` hook, extracts true technical intent and named entities, resolves domain ground truth through selective memory recall, and enriches procedural skill routing to inject winning context in a single step with zero KV-cache mutation.

The problem hypothesis:
1. Preloading static skill catalogs into the system prompt wastes 12,000–15,000 tokens per interaction across large catalogs (500+ skills).
2. Pure on-demand tool calling (`on_demand: true`) burns extra round-trips (~20,000 tokens of re-sent context) because the agent is initially blind to available skills.
3. Shorthand or ambiguous user prompts (e.g., *"Revisa lo de AcmeCorp"*) fail with naive lexical matching because the technology stack is unmentioned in the prompt. By querying selective memory first for detected entities (*"AcmeCorp"*), domain facts ground the skill router to select the exact procedural skill (*"meta-pixel-capi"*).

## 2. Actors and operating context

| Actor | Need / responsibility |
|---|---|
| End user | Receives high-precision agent responses with relevant skills and memory loaded instantly, zero overhead on casual turns. |
| Hermes host | Owns the turn, session lifecycle, tool execution, and continues unaffected if the plugin abstains or encounters errors. |
| Context selector (`neural-context-plugin`) | Interprets intent, queries selective memory, enriches skill search, and returns ephemeral context for the user turn. |
| Repository maintainer | Governs SDD lifecycle, approves architectural gates, and ensures hackathon compliance. |

### Representative user scenarios

1. **Entity-driven domain task:** A user says *"Revisa lo de AcmeCorp"*. The selector extracts entity `"AcmeCorp"`, queries memory for AcmeCorp's stack (Meta CAPI + FastAPI), enriches the skill search, and injects `meta-pixel-capi` guidance plus AcmeCorp's memory facts in one step.
2. **Multilingual task:** A user asks in Spanish for help fixing a webhook while the relevant procedure is described in English. The selector derives canonical English search intent and retrieves the webhook procedure without translating away the user's prompt.
3. **Conversational fast exit:** A user greets or acknowledges (*"Hola"*, *"Ok gracias"*). The selector detects non-actionable intent and returns `None` immediately with zero token injection and sub-5ms latency.
4. **Governance precedence:** In a repository with local `AGENTS.md` and `.skills/`, local project skills take absolute priority over global fleet skills.
5. **Fail-closed safety:** If model inference times out or an unexpected error occurs, the plugin returns `None`. Hermes proceeds normally without any block.

### Terms

- **Two-Phase Context-Enriched Routing:** Pipeline where Phase 1 queries selective memory for detected entities to ground technical facts, and Phase 2 uses those facts to enrich skill discovery and single-skill routing.
- **Abstention / Fast Exit:** Returning `None` when no specialized skills or memories are needed, allowing the host turn to proceed with zero injected overhead.
- **Ephemeral Context:** Injected context returned via `{"context": "..."}` appended strictly to the user message turn, leaving the system prompt byte-stable for prompt caching.
- **Verifiable Receipt:** Deterministic JSON record logged per turn to `receipts.jsonl` proving latency, token savings, and execution status.

## 3. Verified reuse inventory and constraints

- **Hermes Hook Seam:** `agent/turn_context.py:910-964` provides `pre_llm_call` which injects ephemeral context into the user message turn while preserving system prompt stability.
- **Hermes Plugin LLM:** `agent/plugin_llm.py:486-514` provides `ctx.llm.complete(messages=list[dict])` for host-owned model inference.
- **Hermes Plugin Storage:** `plugins/plugin_storage.py:27-46` provides profile-isolated storage directories.
- **Sidecar SQLite BM25:** Lightweight, SHA256-invalidated sidecar index over `SKILL.md` frontmatter and lead paragraphs.
- **Selective Memory Recall:** Surgical extraction of entity-matched facts from `MEMORY.md`, `USER.md`, and `PROJECT_MEMORY.md` within strict token bounds.

## 4. Scope

### In scope for the MVP

- Cognitive semantic interpretation of user messages using `ctx.llm` without brittle hardcoded keyword lists.
- Two-Phase Context-Enriched Routing:
  - Phase 1: Entity extraction and selective memory query for technical domain hints.
  - Phase 2: Memory-enriched BM25 search and single-skill selection via Host LLM or fast heuristic.
- Single-step ingestion of winning `SKILL.md` body (stripping YAML frontmatter) combined with memory facts.
- Ephemeral injection into the user message turn preserving KV-cache stability.
- Repository governance sovereignty (`AGENTS.md` and `.skills/` take absolute priority).
- Verifiable JSON receipts logged to `receipts.jsonl` with CLI inspection (`hermes neural-context verify`).
- Hermes plugin integration (`plugin.yaml`, `__init__.py`, inspection tools `neural_context_status` and `neural_context_route`).
- 100% pass on `pytest`, `hermes plugins validate`, and `hermes plugins doctor`.

### Explicitly out of scope

- Unfiltered full-dump memory ingestion.
- Modifying, writing to, or mutating user `SKILL.md` files or memory files.
- Modifying Hermes core code or altering the base system prompt bytes.
- Paid external router requirements (external backends like Jev remain strictly optional).

## 5. Applicable Project Skills

- `.skills/hermes-plugin-development/SKILL.md` — plugin lifecycle, hooks, tools, and CLI wiring.
- `.skills/design-fail-closed-gate/SKILL.md` — fail-closed boundary and graceful abstention principles.
- `.skills/bm25-sidecar-search/SKILL.md` — SQLite BM25 candidate generation and SHA256 cache invalidation.
- `.skills/python-packaging/SKILL.md` — packaging structure and standard dependency constraints.
- `.skills/llm-structured-output/SKILL.md` — structured JSON interpretation and validation.
- `.skills/hackathon-judge/SKILL.md` — rubric optimization for reliability, usefulness, and novelty.

## 6. Functional Requirements and Observable Acceptance

### FR-01 — Cognitive Semantic Interpretation
- **Requirement:** Evaluate user message intent via model inference (`ctx.llm`); do not classify intent using fixed keyword dictionaries.
- **Acceptance:** Informal phrasing, slang, typos, and Spanish prompts are interpreted correctly into canonical English intent and capability nouns.

### FR-02 — Two-Phase Context-Enriched Routing
- **Requirement:** When named entities are present, query selective memory first for technical domain ground truth and use retrieved facts to enrich the skill search query.
- **Acceptance:** A prompt mentioning entity *"AcmeCorp"* retrieves AcmeCorp's tech stack (Meta CAPI + FastAPI) and routes accurately to `meta-pixel-capi`, even when CAPI is not mentioned in the user prompt.

### FR-03 — Single-Step Ingestion
- **Requirement:** Inline the complete procedural body of the winning `SKILL.md` along with memory facts directly into turn context, eliminating secondary tool-call round trips.
- **Acceptance:** Context block `<neural_context>` contains procedural skill instructions and active memory facts in a single turn payload.

### FR-04 — KV-Cache Stability & Ephemeral Turn Injection
- **Requirement:** Injected context must attach strictly to the user message turn; the base system prompt must remain 100% byte-stable.
- **Acceptance:** Hook returns `{"context": "..."}` which Hermes appends to user content without modifying system prompt bytes.

### FR-05 — Conversational Fast Exit
- **Requirement:** For greetings, acknowledgements, or non-actionable queries, exit immediately without touching memory or skill indexes.
- **Acceptance:** Queries like *"Hola"* return `None` in < 5ms with status `skipped_no_context_needed` and zero injected tokens.

### FR-06 — Repository Governance Sovereignty
- **Requirement:** Local project `AGENTS.md` and `.skills/` take absolute precedence over global fleet skills.
- **Acceptance:** When working in a documented workspace, local skills are indexed and preferred; global skills that conflict are excluded.

### FR-07 — Shared Token Budget
- **Requirement:** Enforce a hard ceiling (default: 1,200 tokens) across skill body and memory facts combined.
- **Acceptance:** Long skills are capped at core procedural sections; memory facts are bounded by `memory_token_budget` (400 tokens).

### FR-08 — Fail-Closed Graceful Degradation
- **Requirement:** Any exception, timeout, or parsing failure must return `None` immediately, allowing Hermes to proceed normally without blocking.
- **Acceptance:** Fault injection tests confirm `None` is returned and no `{"action": "block"}` is emitted.

### FR-09 — Verifiable Audit Receipts
- **Requirement:** Log deterministic execution records to `receipts.jsonl` detailing latency, tokens saved, routed skill, and memory facts.
- **Acceptance:** `hermes neural-context verify` outputs audited turns, average latency, and estimated tokens saved.

### FR-10 — Hermes Plugin Compatibility
- **Requirement:** Register cleanly with Hermes runtime contracts.
- **Acceptance:** Passes `hermes plugins validate` and `hermes plugins doctor` with zero errors.

## 7. Approval and Traceability

- **Phase 1 Approval:** Formally approved by Jesús Díaz (Jemadiar).
- **Next Phase:** Phase 2 Technical Design (`02-design.md`) and Phase 3 Tasks (`03-tasks.md`).
