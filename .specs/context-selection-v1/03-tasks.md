# Context Selection v1 — Implementation Tasks

**Status:** Implementation Active & Verified (Tasks 1–9 Complete; Task 10 Hackathon Packaging Pending).  
**Phase:** 3 of the repository's four-phase SDD lifecycle.  
**Governing Documents:**
- [`01-spec.md`](file:///opt/neural-context-plugin/.specs/context-selection-v1/01-spec.md) (Approved)
- [`02-design.md`](file:///opt/neural-context-plugin/.specs/context-selection-v1/02-design.md) (Approved)

---

## Task Matrix & Execution Checklist

### Phase 4.1: Core Data Models & Configuration (COMPLETED)
- [x] **Task 1: Core Pydantic Models**
  - *Skill*: `.skills/llm-structured-output/SKILL.md`
  - Implement `TriageAnalysis`, `DomainClassification`, `SkillCandidate`, `MemoryItem`, `RouteResult`, `ReceiptRecord` in `neural_context/models.py`.
  - Enforce strict typing and boundary fields.
- [x] **Task 2: Dynamic Plugin Configuration**
  - *Skill*: `.skills/hermes-plugin-development/SKILL.md`
  - Implement `PluginConfig.load_from_ctx(ctx)` in `neural_context/config.py` using `ctx.get_config(key, default)`.
  - Add fallback handling for test mocks and environment overrides.

### Phase 4.2: Sidecar Indexing Engine (COMPLETED)
- [x] **Task 3: SQLite Sidecar & In-Memory BM25**
  - *Skill*: `.skills/bm25-sidecar-search/SKILL.md`
  - Implement `SidecarIndex` in `neural_context/sidecar.py`.
  - Invalidate cache entries by `sha256(SKILL.md)`.
  - Fast in-process BM25 scoring over skill name, description, and lead paragraph in sub-5ms.
  - Zero file mutation contract (user `SKILL.md` files are strictly read-only).

### Phase 4.3: Cognitive Triage & Selective Memory (COMPLETED)
- [x] **Task 4: Cognitive Semantic Interpreter**
  - *Skill*: `.skills/llm-structured-output/SKILL.md`
  - Implement `CognitiveTriage` in `neural_context/triage.py`.
  - Eliminate hardcoded keyword lists; evaluate communicative intent via `ctx.llm.complete(messages=[...])`.
  - Implement fast-exit for casual greetings and acknowledgements.
- [x] **Task 5: Selective Memory Retriever**
  - *Skill*: `.skills/hermes-plugin-development/SKILL.md`
  - Implement `SelectiveMemoryRetriever` in `neural_context/memory.py`.
  - Extract entity-specific facts from `MEMORY.md`, `USER.md`, and `PROJECT_MEMORY.md` within a 400-token budget.

### Phase 4.4: Two-Phase Enriched Router & Coordinator (COMPLETED)
- [x] **Task 6: Two-Phase Enriched Routing Pipeline**
  - *Skill*: `.skills/design-fail-closed-gate/SKILL.md`
  - Implement `ContextCoordinator.execute()` and router backends in `neural_context/router.py`.
  - Phase 1: Retrieve memory facts for extracted entities to establish domain ground truth.
  - Phase 2: Enrich BM25 skill search query with memory hints to pinpoint winning skill.
  - Phase 3: Assemble single `<neural_context>` block combining skill procedure and memory facts.
  - Enforce local workspace sovereignty (`AGENTS.md` and `.skills/` take absolute precedence).

### Phase 4.5: Receipts, CLI & Hermes Registration (COMPLETED)
- [x] **Task 7: Verifiable Audit Receipts & CLI**
  - *Skill*: `.skills/hermes-plugin-development/SKILL.md`
  - Implement `ReceiptEmitter` in `neural_context/receipt.py` logging to `receipts.jsonl`.
  - Implement CLI subcommands (`status`, `doctor`, `index`, `verify`) in `neural_context/cli.py`.
- [x] **Task 8: Hermes Plugin Entrypoint & Tools**
  - *Skill*: `.skills/hermes-plugin-development/SKILL.md`
  - Wire `pre_llm_call`, `post_llm_call`, CLI, and tools (`neural_context_status`, `neural_context_route`) in `__init__.py`.
  - Isolate multi-session turn state in `_session_turns` keyed by `f"{session_id}:{turn_id}"`.
  - Unwrap tool schemas to standard `{"name": ..., "description": ..., "parameters": {...}}`.

### Phase 4.6: Verification & Test Suite (COMPLETED)
- [x] **Task 9: Unit Testing & Official Hermes Diagnostics**
  - *Skill*: `.skills/hermes-plugin-development/SKILL.md`
  - 16/16 unit tests passing in `pytest tests/ -v`.
  - Clean pass on `hermes plugins validate /opt/neural-context-plugin`.
  - Clean pass on `hermes plugins doctor /opt/neural-context-plugin` (2 tools, 2 hooks registered).
  - Initial conventional commit `feat(core): initial neural-context-plugin implementation with two-phase enriched routing` on `main`.

---

## Remaining Delivery Tasks (Phase 4.7: Hackathon Packaging)

- [ ] **Task 10: Hackathon Submission Artifacts**
  - *Skill*: `.skills/hackathon-judge/SKILL.md`
  - **10.1 Demo Video Script**: Draft concise side-by-side terminal walkthrough (< 90 seconds) comparing token bloat vs. Neural Context single-step injection and verified token savings.
  - **10.2 Push & Remote Sync**: Run `git push origin main` to `git@github.com:Kattegat-crew/neural-context-plugin.git`.
  - **10.3 Catalog PR**: Prepare pull request into `NousResearch/hermes-agent` under `plugins/` catalog.
