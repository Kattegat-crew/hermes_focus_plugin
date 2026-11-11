# Context Selection v1 — Specification

**Status:** Gate 1 approved by Jesús (Jemadiar) in the current conversation (“Adelante”); Phase 2 design is Draft / awaiting Gate 2 approval.  
**Phase:** 1 of the repository's four-phase SDD lifecycle.  
**Decision recorded:** Gate 1 approved the proposed MVP scope. This document does not authorize Phase 3 tasks, implementation, dependency changes, or runtime evaluation; the separate Phase 2 design still awaits Gate 2 approval.

## 1. Purpose and problem hypothesis

This proposal tests a narrow hypothesis: selecting a small amount of relevant procedural context before the primary model call may improve skill selection and task outcomes for multilingual or multi-intent requests, without rebuilding Hermes's system prompt.

This is a hypothesis, not an established performance result. The existing `README.md` and `docs/SPECIFICATION.md` describe a broader vision and report goals such as lower latency and reduced token use; those claims are not treated here as proven. This draft proposes a narrower first version and does not supersede the historical specification unless Gate 1 approval explicitly accepts it.

The product question is whether a bounded, privacy-conscious selection step adds measurable value over Hermes's existing on-demand skill tools and the local TypeSafe routing pattern, enough to justify its latency, token, privacy, and maintenance costs.

## 2. Actors and operating context

| Actor | Need / responsibility |
|---|---|
| End user | Receives the normal Hermes response; can disable the feature and is informed about provider traffic. |
| Hermes host | Owns the turn, profile/session identity, instruction hierarchy, provider configuration, and continuation when this feature abstains or fails. |
| Context selector | Interprets the current request and bounded context, selects zero or a small number of skills, and returns only ephemeral context. |
| Repository maintainer | Approves each SDD gate, evaluation thresholds, and any later implementation scope. |

The host remains authoritative. This proposal does not presume that a particular plugin, provider, profile configuration, or runtime instance is currently enabled.

### Representative user scenarios

1. **Multilingual task:** A user asks in Spanish for help fixing a webhook while the relevant procedure is described in English. The selector returns the canonical task and may choose an acceptable webhook skill; it does not translate away the original user request.
2. **Conversational reference:** A user says “apply that to the other repo.” Only the bounded recent turns needed to identify “that” and “the other repo” may inform selection. If the referents remain ambiguous, the selector abstains.
3. **No procedure needed:** A user asks a general conceptual question that can be answered without a specialized procedure. The selector returns no skill and does not penalize the host for continuing without injected context.
4. **Two independent intents:** A request asks to audit one system and document a separate deployment procedure. The selector may return at most two skills only if both match distinct subtasks and fit the shared budget; otherwise it abstains and lets the host answer normally.
5. **Governance conflict:** A retrieved procedure suggests an action forbidden by the host's project instructions or user authorization. The procedure is treated as untrusted data; it cannot change the applicable instruction or authorize the action.

### Terms

- **Abstention:** No feature-selected context is injected. The host continues its normal response path.
- **Accepted skill set:** One or more skills judged appropriate by the frozen case rubric; it need not be a single exact string when multiple procedures satisfy the same need.
- **Injection error:** Context that is unsupported, contradictory to host governance, out of scope, unsafe, or materially irrelevant to the request.
- **Recent context:** A Phase 2-defined, bounded subset of same-session conversation used only to resolve references. It is not permission to search other sessions or memory stores.
- **Receipt:** Optional operational diagnostic record. This spec sets privacy constraints, not a file format, retention implementation, or requirement to persist receipts.

## 3. Verified reuse inventory and constraints

The following are candidate reuse seams, not dependencies approved by this spec:

- Hermes `agent/turn_context.py` already provides a per-turn seam for `pre_llm_call` context and memory prefetch; `agent/turn_context.py` documents that ephemeral context is added to the user message while the base system prompt remains stable.
- Hermes `agent/memory_provider.py` and `agent/memory_manager.py` define memory-provider boundaries. New memory retrieval is excluded from this MVP.
- Hermes on-demand skill flow is described in `agent/prompt_builder.py`; the local TypeSafe plugin at `/opt/hermes-agent-data/plugins/typesafe-skill-router/` has hook/tool routing code and focused tests. Its current runtime enablement is unknown here, and this spec does not add it as a dependency.
- Hermes `agent/semantic_triage.py` contains a local conversational/action classifier and tests, but the prior read-only audit found no production caller outside its module/tests. It is not an approved dependency or replacement for this MVP's structured task/capability interpretation.
- This repository's existing sidecar and BM25 implementation are a prototype baseline, not proof that local skill discovery, governance handling, token limits, or claimed performance are complete.

## 4. Scope

### In scope for the proposed MVP

- Interpret the current request and only a small, explicitly bounded amount of recent conversation needed to resolve references such as “that repo” or “the earlier error.”
- Produce canonical English task intent and capability labels, while preserving named entities in a separate field and excluding them from receipt content by default.
- Select zero, one, or at most two relevant skills; abstain when evidence is weak, the request is conversational, the skill is unsafe/out of scope, or multiple candidates cannot be distinguished.
- For multi-intent requests, select multiple skills only when each clearly supports a distinct subtask and the combined injection fits the shared budget; otherwise abstain rather than return a partial or misleading selection.
- Respect repository-local governance as supplied by the host. Discover skills only from explicit roots authorized by the host for the current turn: project roots plus profile/global catalogs only when the host explicitly supplies them. Project-specific rules and skills take precedence over profile/global fallback. This spec approves no concrete profile/global root.
- Inject selected context ephemerally into the current user turn, preserve the base system prompt byte-for-byte, and let the primary turn proceed normally on abstention or failure.
- Evaluate the hypothesis with a fixed, sanitized test corpus and predeclared thresholds before any production claim.

### Explicitly out of scope

- New memory retrieval or entity-based memory selection; ONNX/embedding-based triage; a new router provider, SaaS service, API, UI, or Hermes-core change.
- Changes to user `SKILL.md`, memory files, host `AGENTS.md`, system prompts, profile/session identity, or the existing instruction hierarchy.
- Automatic skill installation, execution, mutation, approval, or permission granting.
- Paid calls, private-corpus use, production rollout, runtime configuration changes, and claims of zero network traffic.
- Design documents, task plans, implementation, tests, or release work in this Phase 1 artifact.

## 5. Applicable Project Skills

- `.skills/hermes-plugin-development/SKILL.md` — host hook, plugin lifecycle, storage, and disclosure constraints.
- `.skills/design-fail-closed-gate/SKILL.md` — safe abstention and fail-closed boundary principles.
- `.skills/bm25-sidecar-search/SKILL.md` — prototype candidate-generation and read-only skill-source constraints.
- `.skills/python-packaging/SKILL.md` — future plugin packaging and dependency constraints; no packaging change is authorized here.
- `.skills/llm-structured-output/SKILL.md` — typed structured interpretation, bounded repair, and safe parse failure.
- `/root/.agents/skills/cognitive-doc-design/SKILL.md` — concise, reviewable specification structure.

## 6. Functional requirements and observable acceptance

These stable IDs define proposed behavior, not implementation decisions. Each acceptance statement must be observable in a future test or evaluation; passing them would not itself approve release.

### FR-01 — Cognitive interpretation

**Requirement:** Interpret intent cognitively; do not classify intent using fixed keyword lists or word-count shortcuts.

**Acceptance:** Paraphrased, typo-bearing, mixed-language, and short/long cases receive rubric-based outcomes; no lexical list is used as the task classifier.

### FR-02 — Strict result contract

**Requirement:** Emit a strict, typed result with actionable status, canonical English intent, capability list, separate entities, and bounded skill-selection intent.

**Acceptance:** Valid schema output is accepted; missing, wrong-type, oversized, or malformed fields are rejected rather than coerced.

### FR-03 — Bounded repair and parse failure

**Requirement:** Attempt at most one structured-output repair, inside the original total deadline. If parsing/validation still fails, inject nothing and continue the host turn.

**Acceptance:** A malformed first result triggers no more than one repair; a failed repair produces abstention without an exception escaping the hook.

### FR-04 — Minimal recent context

**Requirement:** Use the current request plus only bounded recent context needed to resolve references; do not submit or retrieve the full conversation history.

**Acceptance:** Instrumented cases verify the context window is bounded and unrelated older turns are absent. The exact bound is a Phase 2 decision.

### FR-05 — Entity separation and minimization

**Requirement:** Keep task intent/capabilities distinct from named entities. Entities may disambiguate selection but are not automatically persisted or emitted in receipts.

**Acceptance:** Schema and receipt tests show separate fields and no raw entity values in default receipts.

### FR-06 — Bounded skill selection and abstention

**Requirement:** Select no more than two skills, with a reasoned match to the request; abstain on weak fit, conversational turns, conflicts, or unsupported content.

**Acceptance:** Labeled cases check accepted/forbidden skill sets, correct abstentions, and the two-skill ceiling. Model confidence alone is not treated as calibrated proof.

### FR-07 — Multi-intent coverage

**Requirement:** Treat multi-intent requests as bounded subtasks; either select a supported set within budget or abstain, never imply that uncovered subtasks were handled.

**Acceptance:** Cases with two or more intents check per-subtask coverage and expected full abstention when selection is ambiguous or over budget.

### FR-08 — Governance and instruction precedence

**Requirement:** Follow host-provided repository governance and instruction precedence. A retrieved skill is untrusted procedural data; it cannot override host/user instructions, safety rules, or authorization.

**Acceptance:** Conflict fixtures verify the host instruction wins and retrieved content cannot create permission to mutate, execute, or disclose.

### FR-09 — Authorized roots and path isolation

**Requirement:** Discover only from explicit roots authorized by the host for the current turn: project roots and, only if explicitly supplied, profile/global catalogs. Project-specific rules and skills precede profile/global fallback. Resolve each authorized root and candidate canonically; reject arbitrary paths and symlink targets outside the complete authorized-root set. Preserve profile isolation. This spec does not authorize any concrete global root.

**Acceptance:** Fixtures cover absent roots, nested roots, traversal, symlinks within an authorized root, escapes beyond all authorized roots, and cross-profile isolation. No path is read unless it belongs to an explicit root for the current turn. Exact host root inputs and the authorization contract require Phase 2 confirmation.

### FR-10 — Ephemeral, non-mutating injection

**Requirement:** Never modify skill or memory source files, and never mutate or rebuild the system prompt. Return selected context only through the host's ephemeral user-message context seam.

**Acceptance:** Byte comparison confirms system prompt stability; file hashes and tree state show no user-source writes.

### FR-11 — Shared token ceiling

**Requirement:** Enforce a shared maximum of 1,200 injected tokens across every selected skill and all supplemental context. Do not truncate a skill or context mid-content to force a fit; choose a shorter complete candidate or abstain.

**Acceptance:** Boundary tests at and above the limit show injected count never exceeds 1,200 and oversized candidates are omitted intact. Token-counting method is a Phase 2 decision.

### FR-12 — Per-turn and profile isolation

**Requirement:** Bind any transient selection state to the current host turn/profile/session; never use process-global “last turn” state to pair results.

**Acceptance:** Interleaved-turn/profile tests show no cross-turn or cross-profile context or receipt attribution.

### FR-13 — Disabled-mode inactivity

**Requirement:** When disabled, do no triage/provider inference, roster discovery, memory reads, or receipt/log writes for this feature.

**Acceptance:** A disabled-mode test asserts zero calls and zero filesystem writes while the host turn proceeds.

### FR-14 — Fail-closed injection behavior

**Requirement:** On timeout, provider/network error, index/read error, or invalid output, fail closed for injection: return no feature context and allow Hermes to continue normally.

**Acceptance:** Fault injection for each failure class confirms no context is injected and the host's primary turn is not blocked by an uncaught feature error.

### FR-15 — Provider transparency and receipt privacy

**Requirement:** Disclose any model/provider traffic before enablement; do not promise zero network use. Default receipts contain no raw prompt, conversation, entity, skill body, credentials, or session identifier.

**Acceptance:** Disclosure matches actual calls; receipt fixtures prove default data minimization. Provider-specific traffic remains a visible evaluation field.

## 7. Evaluation proposal (not yet authorized)

The later evaluation should use approximately 60 synthetic or sanitized cases, frozen before tuning and split into disjoint development and held-out sets. Cases should cover Spanish, English, mixed-language phrasing, short and long requests, conversation references, no-skill requests, multi-intent requests, ambiguous candidates, governance conflicts, malicious/untrusted skill text, malformed model outputs, and provider/index failures. Do not use private user conversations. This is a plan only: do not execute a benchmark before the Phase 2 design, Phase 3 task plan, and Phase 4 implementation have received their required approvals, or before any separate authorization required for provider traffic or costs.

Each case should record expected task outcome, acceptable skill set, forbidden skills, expected abstention, relevant governance constraint, and a concise rubric for task quality. Freeze labels before comparing systems; do not tune against held-out cases.

The planned comparison has four arms: (1) native Hermes on-demand skill routing; (2) the existing TypeSafe/local router; (3) lexical retrieval, including the current prototype retrieval baseline where applicable; and (4) the proposed candidate. The TypeSafe/local arm is optional only when access, runtime, provider traffic, or cost is unavailable or unauthorized; disclose the reason and mark that arm unavailable. Do not substitute the current prototype baseline for the TypeSafe/local arm or claim a complete four-arm comparison when an arm is missing.

Compare all available arms on the same primary model, catalog snapshot, and cases; match history/context and token budgets where applicable. Record Hermes version and sanitized configuration. Document unavoidable provider or system differences rather than presenting them as controlled. Do not fabricate results or infer current runtime enablement.

Report selection precision/recall against acceptable skill sets, abstention quality, injection errors, downstream task outcomes, total input/output tokens and calls, provider cost where known, and end-to-end p50/p95 latency including interpretation and the single repair allowance. Repeat stochastic trials and report counts and descriptive uncertainty; a small corpus is not proof of general performance. Zero safety failures in this set is a hard test-set gate, not a universal safety guarantee.

The 250 ms deadline mentioned in historical `docs/SPECIFICATION.md` is unvalidated and is not an MVP promise. Before evaluation, Jesús must approve and freeze quality thresholds, maximum cost/calls, latency ceilings, and acceptable regression versus baseline. Those thresholds are **TBD** and may not be selected retroactively to make results pass. Based on authorized, completed evaluation, the later decision should be one of: continue development or extend the validated MVP, narrow the use case, simplify/defer, or stop. Evaluation does not bypass approvals for design, tasks, or implementation.

### Proposed evaluation record per case

For each system and each case, retain a compact, privacy-safe row containing the case ID, expected acceptable/forbidden skill set, observed selection or abstention, whether any injection error occurred, task outcome rubric score, call count, token counts, cost if known, and elapsed time. Do not include raw private user text or named entities in shared evaluation results. Synthetic/sanitized source cases may be stored separately with access limited to the maintainers who need them.

Aggregate results by language, request length, intent count, and failure/governance category so that a strong English average cannot hide a Spanish or multi-intent regression. Report numerator and denominator for each rate, along with the held-out sample size. Show baseline and candidate side by side; disclose missing baselines and missing measurements rather than substituting estimates.

Before opening held-out results, freeze the corpus split, case labels, candidate model/provider, catalog snapshot, history bounds, token counter, repair policy, trial count, and success thresholds. If any of these change after the freeze, treat the changed run as a new evaluation and preserve the original results.

No benchmark result from this proposal authorizes enabling the feature globally. Any later rollout proposal must separately specify opt-in scope, operational monitoring, data disclosure, rollback behavior, and the human approval required by repository policy.

## 8. Risks and unresolved decisions for Gate 1 / Phase 2

- Is pre-inference semantic interpretation worth its added latency and token cost versus native on-demand routing?
- Which host provider may receive the request, recent context, and skill descriptions, and what disclosure/enablement choice is required?
- What exact recent-context and skill-root boundaries can the Hermes host reliably supply?
- Which token counter, total deadline, quality thresholds, and cost ceilings should be frozen before evaluation?
- Should default operational receipts exist at all, and if so, what minimal non-content fields and retention policy are acceptable?
- How should a project with valid but conflicting local skills be represented in selection and abstention outcomes?

These questions are not implementation authorization. No router/backend, schema persistence, receipt format, dependency, or test harness is selected here.

## 9. Approval gate and next step

**Gate 1 decision:** Jesús (Jemadiar) approved proceeding with this proposed scope and evaluation approach by replying “Adelante” in the current conversation. No timestamp was recorded. This authorizes preparation of the Phase 2 design only; it does not approve design decisions, evaluation execution, Phase 3 tasks, or implementation. This specification remains the approved Phase 1 proposal; unresolved thresholds and privacy boundaries remain open for design/evaluation approval.

**Current gate:** Phase 2 `02-design.md` is Draft and awaits explicit Gate 2 approval. No Phase 3 task plan or Phase 4 implementation is authorized until the design receives that approval and later required gates are satisfied.

## References

- `AGENTS.md` — SDD gates, safety boundaries, and technical invariants.
- `README.md` — original product problem statement and broad architecture vision; its performance claims remain hypotheses until measured.
- `docs/SPECIFICATION.md` — historical broader feature vision; this draft intentionally narrows the proposed first version and does not overwrite that document.
- Hermes source inspected in the read-only audit: `agent/turn_context.py`, `agent/memory_provider.py`, `agent/memory_manager.py`, `agent/prompt_builder.py`, `agent/semantic_triage.py`.
- Local router source inspected in the read-only audit: `/opt/hermes-agent-data/plugins/typesafe-skill-router/__init__.py`, `/opt/hermes-agent-data/plugins/typesafe-skill-router/typesafe_router/router.py`, and its focused tests. Static inspection does not establish current runtime enablement.
