# Context Selection v1 — Technical Design

**Status:** Draft — Gate 1 approved (“Adelante”); awaiting explicit Gate 2 approval from Jesús (Jemadiar). No timestamp was recorded.  
**Scope:** Phase 2 design only. This document proposes implementation structure; it does not authorize Phase 3 tasks, evaluation execution, source changes, dependency additions, provider calls, or rollout.

## 1. Decision summary

Use Hermes' existing `pre_llm_call` hook as the only injection seam. A small adapter validates hook inputs and keyed plugin settings, then delegates to a deep `ContextSelector` that performs structured task interpretation, local BM25 candidate lookup, path/content validation, complete-body budget checks, and safe abstention. It never vetoes the host turn; bounded hook latency remains unproven until the provider timeout behavior is verified.

Two host-contract gaps make the design conditional rather than an assertion of current feasibility:

1. The verified hook payload has session/turn IDs and conversation history but no cwd, workspace root, authorized skill-root set, or repository governance payload. A viable first integration may use explicitly operator-approved profile settings that bind named roots and their governance sources; this does not authorize ambient cwd discovery or make arbitrary configured paths safe. Automatic per-turn workspace discovery is deferred unless Hermes exposes an approved seam. If no explicit binding exists, abstain.
2. The hook exposes no tokenizer. Prefer a host-supported tokenizer API or an already-available tokenizer matched to the active model. Do not add a dependency without approval. If no exact or demonstrably conservative count is available, abstain rather than exceed the cap.
3. `ctx.llm.complete_structured(..., timeout=...)` exposes a timeout argument, but that alone does not prove transport cancellation or a hard bound on synchronous hook latency. No 250 ms guarantee is made. The feature remains disabled by default until a safe timeout/cancellation contract and budgets are approved and verified.

## 2. Verified host contract and feasibility

| Surface | Verified contract | Design consequence |
|---|---|---|
| Registration | `PluginContext.register_hook(hook_name, callback, *, hold_stream=False)` in `/opt/hermes-agent/hermes_cli/plugins_context.py:741-743`; `pre_llm_call` is a valid hook (`hermes_cli/plugins.py:111-125`). | Register one callback in `register(ctx)`; callback accepts `**kwargs` for forward-compatible payloads. |
| Hook inputs | `/opt/hermes-agent/agent/turn_context.py:910-931` sends `session_id`, `task_id`, `turn_id`, `user_message`, `conversation_history=list(messages)`, `is_first_turn`, `model`, `platform`, `parent_session_id`, and `sender_id`. | Use only current-turn and bounded same-session history. Do not persist raw inputs or assume workspace/root/governance fields exist. |
| Hook outputs | `turn_context.py:920-964` accepts a nonempty string or `{"context": ...}`; `{"action":"block"}` vetoes the turn. | Return `None` on abstention/failure; never return a block action. Inject only a fully validated context string. |
| Injection point | `turn_context.py:1469-1482` adds plugin context to user content at API time. `:1526-1534` replays the cached base system prompt and describes it as stable. | Keep the base prompt unmodified. Stable prompt bytes do not guarantee provider KV-cache reuse. |
| Settings | `PluginContext.get_config(key, default=None)` reads the plugin's namespaced `settings` with legacy `config` fallback (`plugins_context.py:73-83`). | Read individual keys. The prototype's `ctx.get_config()` dict call is not the public contract. Do not read keys/secrets from process environment. |
| LLM | `ctx.llm` is host-owned and uses active model/auth (`plugins_context.py:183-191`). `PluginLlm.complete_structured` accepts keyword `instructions`, `input`, `json_schema`, `json_mode`, `max_tokens`, `timeout`, `purpose`, etc. and returns parsed/text/usage fields (`agent/plugin_llm.py:500-514, 561-584`). | One structured interpretation call through host provider; disclose provider traffic. Validate `result.parsed` locally because host schema validation depends on optional `jsonschema`. Do not select provider/model overrides. |
| Storage | `plugins.plugin_storage.plugin_data_dir(name)` returns/creates `<active Hermes home>/plugin-data/<name>/` and resolves home on every call (`plugins/plugin_storage.py:27-33`); `plugin_db` uses shared SQLite utility (`:36-46`). | Lazy-open only after enabled check. Never store in the plugin install tree or cache the data path across profiles. |
| Missing context | Hook has no documented cwd, project roots, authorized roots, instruction text, or tokenizer. | Permit a future explicit operator-approved root/governance binding through supported configuration; do not infer ambient paths. If absent, abstain. A host extension is one possible future seam, not a prerequisite if explicit configuration suffices. |

These are source-verified APIs, not evidence that the feature is installed, active, or compatible with every Hermes version. Recheck against the pinned target version before implementation.

## 3. Proposed pipeline

1. **Register cheaply.** Register `pre_llm_call`; import no model, scan, DB, or filesystem catalog during plugin registration.
2. **Disabled fast path.** Read `ctx.get_config("enabled", False)` and feature-local enablement. If disabled or malformed, return `None` before provider calls, root discovery, DB creation, receipts, or skill reads.
3. **Validate request envelope.** Accept only expected string/text-part user content, current hook IDs as opaque correlation values, and a bounded tail of history. Reject malformed/oversized inputs by abstaining. Do not use `Path.cwd()` as project identity.
4. **Resolve authorized context.** Obtain canonical roots and their governance sources only from explicit per-turn data or operator-approved namespaced settings. Root entries must carry source/precedence (`project`, `profile`, `global`) and explicit authorization; a configured path alone is not proof of approval. Project instructions/skills override fallback roots. No concrete global path is approved here. Do not derive project identity from cwd.
5. **Interpret once.** Call `ctx.llm.complete_structured` with current request and bounded history, `json_mode=True`, a strict schema, capped output, and the remaining monotonic deadline. Ask for actionable status, English intent/capabilities, separate entities, and up to two distinct subtask queries. The model does not authorize paths or actions.
6. **Validate / repair.** Validate required keys, types, bounds, finite values, and allowed selection count locally with strict Pydantic models. Permit at most one repair inside the *same* total deadline. If unsupported structured output, malformed output, or repair failure: abstain. Do not use free-text keyword lists as a substitute classifier.
7. **Generate candidates.** For each subtask, query the already-built BM25 sidecar using canonical English task/capability text. Entities remain separate and are not used as generic query terms. Resolve candidates to authorized roots and rank project-local entries before profile/global entries. If index is absent/stale or candidate fit ambiguous, abstain rather than full-scan in the hook.
8. **Select conservatively.** Select zero to two complete skill bodies. In v1, avoid a second LLM reranker: a second provider call adds latency/cost and the current prototype's name-substring response parsing is not a safe contract. Use evaluation-frozen retrieval/score-margin rules; thresholds are TBD. Do not treat model confidence as calibrated. Distinct multi-intent results must each support a separate subtask; otherwise abstain. If one selected candidate becomes unsafe, changes, or fails the shared budget check, abstain from the whole multi-skill bundle rather than silently dropping one subtask (FR-07).
9. **Read and contain.** Re-open chosen files as UTF-8 from canonical paths, verify containment and content hash, reject symlink escapes or changed files, and treat content as untrusted. Do not execute it, grant permissions, or describe it as authoritative over host/user governance.
10. **Budget complete context.** Build wrapper plus full skill bodies and any supplemental text, then count the entire injected string with a reliable host-supported or already-available model-matched tokenizer. Do not add an unapproved dependency. If exact or demonstrably conservative counting/fit cannot be established, omit the context. Inject `{"context": context}` only after all checks pass.
11. **Return.** Do not keep “last result” global state or use `post_llm_call` for correlation. The host continues normally on no selection, timeout, malformed result, index/read error, or exception.

### Backend boundary

Keep one useful seam around the retrieval/selection behavior, not a forest of pass-through classes:

```python
# Conceptual interface only; not executable or approved implementation code.
class SkillRouterBackend(Protocol):
    def select(self, request: SelectionRequest,
               candidates: Sequence[ScoredSkill]) -> SelectionDecision: ...
```

`SelectionRequest` contains typed subtasks, validated root precedence, and remaining budget; `ScoredSkill` contains a root ID, relative path, hash, metadata, and local score; `SelectionDecision` is a set of at most two candidate IDs or explicit abstention reason. A deterministic BM25/threshold adapter is the only MVP backend. Keep an in-memory fake adapter for tests. Do not ship TypeSafe/Jev or add a provider dependency by default.

### Callback and interaction sketch

```python
# Conceptual adapter signature; exact host callback payload remains extensible.
def pre_llm_call(*, session_id, turn_id, user_message,
                 conversation_history, **kwargs) -> dict[str, str] | None:
    ...
```

```text
Hermes turn_context -> callback: bounded turn inputs
callback -> keyed settings: enabled? / explicit authorized roots?
callback -> ctx.llm: one strict interpretation (optional single repair)
callback -> sidecar snapshot: BM25 candidates for English subtasks
callback -> approved roots: canonicalize, read full bodies, verify hashes
callback -> tokenizer: count complete wrapper + complete skill bodies
callback -> Hermes turn_context: {"context": ...} OR None (never block)
Hermes -> primary model: normal turn proceeds with ephemeral user context
```

## 4. Proposed module map (current prototype → design)

| Proposed responsibility | Existing source | Design direction |
|---|---|---|
| Hermes adapter/lifecycle | `__init__.py:41-62,65-95,125-145`; `neural_context/__init__.py` | Replace eager singleton/scan and global last-result pairing with cheap registration and per-call adapter. |
| Configuration | `neural_context/config.py:19-59` | Keyed `ctx.get_config(key, default)`; strict bounds; feature disabled by default; no raw env secrets or derived paths in install tree. |
| Structured interpretation | `neural_context/triage.py:93-123,136-228`; `neural_context/models.py` | One Pydantic-validated interpretation contract; remove heuristic intent fallback and permissive coercion for semantic selection. |
| Selection coordinator | `neural_context/router.py:14-17,119-228` | Deep pure orchestration boundary; local retrieval only; no memory retrieval, second model call, or user-authoritative wrappers. |
| Catalog / BM25 sidecar | `neural_context/sidecar.py:87-231` | Root-aware composite keys, SHA-256 invalidation/deletion, snapshot reads; explicit maintenance indexing, never full scan during hot hook. |
| Context assembly | `neural_context/router.py:200-227`; proposed new module `neural_context/context.py` | Assemble full untrusted bodies with non-authoritative framing and exact combined token count; no mid-body truncation. |
| Storage | `neural_context/config.py:50-57`; `neural_context/receipt.py` | Use Hermes `plugin_data_dir` / `plugin_db`; optional metadata-only receipt rows in `sidecar_index.db`, off by default. |
| Tests | `tests/test_*.py` | Test adapter/API contracts, selection, budget, root isolation, failure/disabled behavior; planned only, not run in Phase 2. |

This is a future map, not an instruction to preserve every current class or public API. Existing memory selection and Jev router are explicitly excluded from the v1 path.

## 5. Data and persistence design

### Structured contracts (conceptual, not executable)

All models use strict types, `extra="forbid"`, bounded list/string sizes, no implicit string-to-bool or number coercion, and explicit abstention. Exact schema evolution/versioning is a Phase 3 task decision.

```text
Interpretation:
  actionable: bool
  intent_summary_en: str (<=280 chars)
  capabilities_en: list[str] (0..6)
  entities: list[str] (0..8; transient only)
  subtasks: list[{id, task_en, capabilities_en}] (0..2)

SelectionDecision:
  selected_ids: list[str] (0..2, unique)
  abstain_reason: enum | null
```

The following shape is illustrative only; it documents validation intent, not a ready-to-run model or approved public API:

```python
# Conceptual, non-executable sketch; exact Pydantic syntax/version is a later task.
class Interpretation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    actionable: bool
    intent_summary_en: Annotated[str, Field(max_length=280)]
    capabilities_en: Annotated[list[str], Field(max_length=6)]
    entities: Annotated[list[str], Field(max_length=8)]
    subtasks: Annotated[list[Subtask], Field(max_length=2)]
```

No `confidence` field is used as a probability. No output field can provide a root, file path, shell command, permission, provider override, or instruction-precedence change.

### SQLite sidecar

Place `sidecar_index.db` under `plugin_data_dir("neural-context-plugin")`; its per-active-profile home isolates profiles. Open lazily only after enablement and an authorized, already-indexed root exists. Use shared `plugin_db` instead of a handwritten WAL policy. If optional receipts are enabled, add a metadata-only table in this same sidecar; do not create a separate receipt file.

Proposed tables:

- `roots(root_id, canonical_root, source_kind, precedence, profile_scope, index_generation, last_sync)`; root IDs are host/config assigned, not model supplied.
- `skills(root_id, relative_path, skill_name, sha256, description, lead_paragraph, token_vector, updated_at, PRIMARY KEY(root_id, relative_path))`.
- Index keys include root and relative path, not skill name alone, preventing collisions across repositories/catalogs.

Explicit sync/reindex performs full discovery outside the pre-inference hook, canonicalizes permitted paths, hashes contents, upserts changed entries, and deletes records for disappeared files under that root in one transaction/generation. A failed sync never publishes a partially refreshed generation. Hook reads a consistent snapshot and abstains on missing/stale/incomplete roots rather than scanning. Source files remain read-only. Cache invalidation is SHA-256, not mtime alone.

No conversation transcript, raw request, entity, full skill body, credentials, or stable session ID is stored. Optional receipts default off; if later enabled, write aggregate metadata only (outcome code, bounded duration/call/token counts, schema/index version) without prompts, entities, body, or session ID, as a table inside `sidecar_index.db`. Retention and operator visibility remain Gate 2 questions. Do not create `receipts.jsonl` or another receipt file.

## 6. Security, governance, and isolation

- **Authorization:** roots are explicit host-authorized inputs. The candidate path must be relative to its declared root and its canonical target must remain inside the complete authorized root set. Never accept a path emitted by the model or found by arbitrary filesystem search.
- **Symlinks / races:** resolve root and candidate before selection and again when opening; reject escape and content/hash changes. Use descriptor-relative/no-follow opens where supported. If the platform cannot preserve containment across the open/read boundary, abstain. This design still needs platform-specific implementation proof; path string checks alone are insufficient.
- **Precedence:** project-specific governance and skill entries take precedence over profile/global fallbacks. No global root or permission is created by this design. Roots with duplicate logical skill names remain distinct by root ID.
- **Untrusted text:** retrieved skill bodies are data, not system instructions. A wrapper may label provenance and boundaries but XML/delimiters do not neutralize prompt injection. Host instructions and user authorization remain controlling; selected text cannot trigger tools, mutate files, install skills, or disclose secrets.
- **Privacy:** only bounded same-turn history is sent to the active host model; named entities stay transient and separate. Disclose that `ctx.llm` sends provider traffic according to host configuration. Do not claim zero network use.
- **Isolation:** no module-global per-turn state. The synchronous callback returns its own result; if a future receipt needs attribution, use only the provided opaque turn ID and keep it transient, never as a persisted key. Resolve plugin data directory on every profile-scoped operation.
- **System prompt/cache:** use host user-context return only. Byte-stable base system prompts are an architectural constraint; actual KV-cache hit rate is host/provider behavior and is not guaranteed.

## 7. Latency, token, and call budgets

The existing `timeout_ms=250` is historical and unvalidated. It is not adopted as a performance claim or default. `PluginLlm` accepts `timeout` seconds, but the design does not assume cancellation preempts a synchronous transport call unless the exact host/provider implementation proves that contract. Do not spawn per-request threads to simulate timeouts; unbounded or abandoned workers can outlive turns.

Use one monotonic deadline covering interpretation, optional single repair, local lookup, content read, ranking, and assembly. Pass remaining time to each structured call. Before each local stage, check remaining time and abstain if exhausted. If the host call cannot reliably time out/cancel, keep feature disabled by default until the host seam can guarantee bounded hook behavior; an outer deadline cannot force-stop a blocking call.

| Budget | Proposed v1 rule | Approval state |
|---|---|---|
| Injected context | At most 1,200 total tokens including labels/wrappers and up to two complete bodies. | Phase 1 requirement; tokenizer/seam unresolved. |
| Skills | 0–2, each complete, never truncated. | Phase 1 requirement. |
| Provider calls | One structured interpretation; at most one repair on invalid output; no extra reranker call. | Design proposal; requires Gate 2. |
| Latency deadline | Single monotonic total deadline; numeric ceiling TBD before enabling/evaluation. | Not approved; historical 250 ms is not a promise. |
| Token counting | Exact reliable host tokenizer; otherwise abstain. | Host capability not verified. |
| Receipts | Off by default; metadata-only if explicitly enabled later. | Proposed, retention TBD. |

## 8. Failure behavior

| Failure | Hook behavior | Side effects |
|---|---|---|
| Disabled / invalid config | Return `None` immediately. | No feature inference, index/receipt creation, or skill read. |
| Missing authorized root/governance binding or provable tokenizer | Return `None`; report a typed internal reason only if metadata receipts are explicitly enabled. | No ambient cwd fallback or directory scan. |
| Provider denied/unavailable/timeout | Return `None`. | No repair after deadline; host proceeds. |
| Invalid structured result after one repair | Return `None`. | No permissive parser/keyword classifier fallback. |
| Missing/stale sidecar or failed sync | Return `None`. | No full scan in the hot hook; explicit maintenance sync required. |
| Escaping symlink, changed hash, unreadable file | Reject candidate; if no complete safe selection remains, return `None`. | No partial/truncated body injection. |
| Weak/ambiguous retrieval or budget overflow | Return `None`. | No arbitrary winner or destructive truncation. |
| Unexpected exception | Catch at adapter boundary and return `None`. | Host turn must not be vetoed; no `{"action":"block"}`. |

“Fail closed” here means *no feature-selected context*. It does not mean block the host task or claim that all model instructions are safe.

### Per-turn state machine

```text
disabled ────────────────> abstain(None)
enabled + invalid input ─> abstain(None)
enabled + no roots/tokenizer/index ─> abstain(None)
valid request ─> interpret ─> local candidate search ─> validate/read/count
                                                  ├─ safe complete fit -> context
                                                  └─ any uncertainty -> abstain(None)
```

All state is local to one callback invocation. The callback must not block the turn, modify request history, persist the user request, or turn a diagnostic error into an empty “successful” selection. Only a validated empty decision is a normal no-skill result; provider/index/path failures remain internal abstention reasons. The public hook output is `None` on both paths so Hermes proceeds identically.

### Index readiness and freshness

- `unavailable`: no schema or authorized root; abstain without creating storage.
- `ready`: an explicit sync completed for the exact root ID and profile scope; candidate reads use one transaction snapshot.
- `stale_or_partial`: sync failed, generation mismatches, or root authorization changed; do not use cached candidates as current and do not scan in the hook.
- `removed`: next successful explicit sync deletes the root's stale skill rows atomically; a missing root is not permission to keep serving old paths.

The lifecycle/API for explicit sync is not selected by this document. It must be a future approved maintenance operation, not an undocumented background thread or a new mandatory host hook.

## 9. Configuration proposal

Values are design proposals, not configured or approved runtime state. Enablement default is **off**. No concrete global roots are listed.

| Key (plugin settings) | Proposed behavior | Default / status |
|---|---|---|
| `enabled` | Master opt-in; checked before any feature-specific IO/inference. | `false` (proposal). |
| `max_injected_tokens` | Hard shared cap including all wrapper text. | `1200` (Phase 1). |
| `max_skills` | Maximum complete selected skills. | `2` (Phase 1). |
| `max_history_messages` / `max_history_chars` | Bound reference-resolving same-session context. | TBD; no history passed until set. |
| `deadline_ms` | Total monotonic deadline for entire hook. | Required before opt-in; TBD. |
| `authorized_skill_roots` | Explicit operator-approved roots with source kind, precedence, and governance source. | Empty; configuration does not itself prove host authorization. |
| `receipts_enabled` | Optional metadata-only diagnostic record. | `false`. |

Read values through `ctx.get_config(key, default)` and validate strictly. Never fall back to raw environment variables, ambient current directory, or model-proposed config. Actual configuration schema and authorized-root provenance must be approved before implementation.

## 10. Evaluation and planned tests

No evaluation, provider call, private-corpus use, or test execution is authorized by this design. Follow Phase 1's approximately 60 synthetic/sanitized, disjoint development/held-out cases and four arms: native Hermes on-demand; TypeSafe/local router if separately available and authorized; lexical retrieval; proposed candidate. Keep the same primary model/catalog/cases and matched history/token budget where possible; label unavoidable differences and missing arms honestly. The current prototype baseline is not a substitute for TypeSafe/local.

Before seeing held-out results, Jesús must approve/freeze quality delta, allowed regression, cost/call ceiling, latency ceiling, tokenizer/catalog/version, history bounds, root authorization model, and stochastic trial count. Numeric values remain TBD; do not infer thresholds from a desired result. Record quality/selection, abstention, downstream task outcomes, injection errors, end-to-end p50/p95, tokens, calls, and known costs. A zero-failure test set is not a universal safety guarantee.

Future tests should include:

- strict schema acceptance/rejection; one repair maximum; failures after deadline;
- disabled mode with no provider calls, index writes, scans, or receipt writes;
- per-turn/per-profile concurrent isolation with no last-result globals;
- multilingual and multi-intent retrieval, acceptable/forbidden sets, no-skill abstentions, ambiguity thresholds;
- root precedence, duplicate names, traversal, symlink in-root/out-of-root, TOCTOU/hash change, profile separation;
- exact tokenizer ceiling including wrapper, complete-body/no-truncation behavior;
- injected untrusted skill conflicts with host governance and no tool/permission escalation;
- provider denial, timeout, malformed response, stale/missing DB, unreadable/deleted skill, and unexpected exceptions all yield `None` while the host proceeds.

These are verification targets for later approved phases, not executed results.

## 11. Requirement traceability

| Phase 1 requirement | Design coverage |
|---|---|
| FR-01 Cognitive interpretation | §§3, 5; one structured model interpretation, no keyword intent classifier. |
| FR-02 Strict contract | §§3, 5; strict local Pydantic validation and typed abstention. |
| FR-03 Repair / failure | §§3, 8; one repair inside unchanged deadline, else no injection. |
| FR-04 Recent context | §§2–3, 9; bounded history only; limits TBD and no full transcript. |
| FR-05 Entity separation | §§3, 5, 6; entities transient, separate, not default receipt fields. |
| FR-06 Selection / abstention | §§3, 4; max two, retrieval thresholds frozen in evaluation. |
| FR-07 Multi-intent | §§3, 5; up to two distinct subtasks or full abstention. |
| FR-08 Governance | §§2, 6; host-provided precedence and untrusted skill boundary. |
| FR-09 Authorized roots | §§2–3, 5–6; host seam gap blocks project-aware selection until resolved. |
| FR-10 Ephemeral injection | §§2–3; only `pre_llm_call` context, no system/source mutation. |
| FR-11 Token ceiling | §§3, 7, 9; full-wrapper tokenizer required; otherwise abstain. |
| FR-12 Isolation | §§3, 5–6; no global last-turn state; active-profile storage. |
| FR-13 Disabled inactivity | §§3, 8–9; off check before feature IO/provider/index. |
| FR-14 Fail-closed injection | §§3, 8; errors return `None`, never block. |
| FR-15 Transparency/privacy | §§2, 5–6, 10; active provider disclosure, metadata-only optional receipts. |

## 12. Open decisions and Gate 2 checklist

Gate 2 should not be granted until Jesus explicitly accepts or revises:

- Whether explicit operator-approved root/governance bindings are sufficient for MVP or automatic per-turn workspace discovery is deferred pending a host seam; no ambient cwd discovery.
- The reliable model-matched tokenizer source and behavior when exact/conservative counting is unavailable; the 1,200-token ceiling remains mandatory.
- A numeric total deadline and verified transport timeout/cancellation behavior, or a decision to keep feature disabled until Hermes supports a safe bound.
- Bounded history message/character limits, authorized-root provenance and precedence, and root synchronization operation.
- Whether metadata receipts are wanted at all, plus retention/visibility; default remains off.
- Retrieval thresholds and acceptable-set behavior, to be frozen only before authorized evaluation and not chosen retroactively.
- Provider disclosure/enablement, comparison-arm availability/authorization, and numeric evaluation quality/cost/latency thresholds.

**Gate 2 acceptance:** the design is approved as a coherent proposal with the above blocking questions resolved or explicitly deferred into a safe-abstain MVP. Approval authorizes only creation of `03-tasks.md` under the SDD process. It does not authorize evaluation execution, source edits, dependency changes, runtime enablement, or deployment.

## References

- `01-spec.md` — approved Phase 1 scope and FR-01–FR-15.
- `AGENTS.md` — SDD gate policy and architectural invariants.
- Hermes source: `/opt/hermes-agent/hermes_cli/plugins_context.py:73-83,183-191,741-743`; `/opt/hermes-agent/agent/plugin_llm.py:486-514,561-584`; `/opt/hermes-agent/agent/turn_context.py:910-964,1469-1482,1526-1534`; `/opt/hermes-agent/plugins/plugin_storage.py:27-46`.
- Prototype sources: `__init__.py`, `neural_context/config.py`, `neural_context/triage.py`, `neural_context/router.py`, `neural_context/sidecar.py`, `neural_context/receipt.py`.
- Applicable skills: `.skills/hermes-plugin-development/SKILL.md`, `.skills/llm-structured-output/SKILL.md`, `.skills/design-fail-closed-gate/SKILL.md`, `.skills/bm25-sidecar-search/SKILL.md`.
