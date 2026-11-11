# Neural Context — Phase 3 Task Plan

**Status:** Draft task plan for review; implementation actions remain subject to the repository approval gates and separate authorization.  
**Deadline context:** Hackathon cutoff is Nov 11, 2026, 5:00 PM EST (UTC-05:00), equivalent to 5:00 PM Bogotá. At plan revision (~4:27 AM Bogotá), about 12.5 hours remained. Schedule below is a critical-path estimate, not a promise; re-estimate after the first integration checkpoint.  
**Governing sources:** Current approved `01-spec.md` and `02-design.md`, `AGENTS.md`, `README.md`, `docs/SPECIFICATION.md`, and human-maintained `docs/DEMO_VIDEO_SCRIPT.md`. The approved architecture is two-phase: cognitive entity/intent extraction → selective memory domain hints → enriched BM25 → one skill → combined memory/skill context ≤1,200 tokens → receipt. Do not revert to the earlier no-memory / two-skill proposal.

## 1. Recovery and current evidence

Tasks 1–9 retain their previously recorded checkmarks and IDs. Their completion statements are historical entries from this file, not independently re-verified in this task. Source spot-check found the corresponding modules and registration path, but did not run tests, Hermes diagnostics, or review receipts. Re-run the relevant checks before claiming current readiness; preserve the prior claims as provenance rather than treating them as fresh proof.

CodeGraph index is absent in this checkout. No initialization was performed; source spot-check used bounded direct reads. Current tree includes the expected `neural_context/` modules and tests. `git status --short` showed only `?? docs/DEMO_VIDEO_SCRIPT.md`; that file is human-maintained and out of this task's edit scope.

Important implementation gaps to resolve in order: (a) `register()` calls `_get_services()` before checking disabled settings; service construction may create storage/index and scan skills; (b) actual Hermes hook payload does not document `cwd`, while current callback reads `kwargs["cwd"]`/`ctx.cwd`; (c) memory reads broad filesystem roots and text matching can overstate factual certainty; (d) turn correlation falls back to `popitem()` when IDs miss; (e) `len(text.split()) + len(text)//4` is an estimate, not proof of the 1,200-token ceiling; (f) the approved receipt JSONL requirement conflicts with `AGENTS.md`'s “all derived data ... in sidecar_index.db” rule. Resolve/record these before release claims; do not silently change the approved architecture. The approved design also describes storage under `<HERMES_HOME>/plugins/...`, while the cited Hermes `plugin_storage` helper returns `<HERMES_HOME>/plugin-data/...`; Task 11 must verify actual per-profile helper behavior and flag the mismatch without editing the approved design.

## 2. Schedule and gates (Bogotá time)

| Window | Critical-path outcome | Stop/re-estimate condition |
|---|---|---|
| 04:30–06:00 | Reconcile API/config/root handling, disabled fast path, and memory privacy/provenance. | If host workspace/governance data is unavailable, use explicitly authorized configuration only; do not infer cwd. |
| 06:00–09:30 | Integrate cognitive triage → entity-targeted memory → memory-enriched BM25 → single winner → context/receipt. | Re-estimate at 08:00; prioritize end-to-end path over optional CLI polish. |
| 09:30–12:00 | Complete integration and deterministic synthetic demo fixtures. | Main path must be runnable by noon; no claim of real provider results without separately authorized calls. |
| 12:00–14:00 | Functional freeze, tests, Hermes validate/doctor, repair only critical blockers. | Reserve 14:00 for freeze; do not keep adding scope. |
| 14:00–15:00 | Record <90s demo, verify claims against observed receipts/output, finalize accompanying copy. | Demo script is human-maintained; seek approval before editing it. |
| 15:00–16:00 | Prepare/publish repo and catalog PR, post approved demo, complete official form. | Push, public post, PR, and form submission each need explicit user authorization. |
| 16:00–17:00 | Submission confirmation, artifact links, contingency buffer. | Do not treat an opened PR as catalog admission or a draft as submitted. |

Full 60-case benchmark is not a deadline blocker under the revised approved docs; retain as later evaluation if time/resources do not support it. Do not cut the memory-enriched routing path, single-step injection, required test gates, or demo/submission deliverables without explicit approval.

## 3. Prior tasks — retained completion record

- [x] **Task 1 — Core Pydantic models** (`neural_context/models.py`). Skill: `.skills/llm-structured-output/SKILL.md`. Historical completion retained; recheck strictness against current approved FRs.
- [x] **Task 2 — Dynamic configuration** (`neural_context/config.py`). Skill: `.skills/hermes-plugin-development/SKILL.md`. Historical completion retained; source currently contains raw environment fallback and must be assessed against host contract.
- [x] **Task 3 — SQLite sidecar and BM25** (`neural_context/sidecar.py`). Skill: `.skills/bm25-sidecar-search/SKILL.md`. Historical sub-5ms claim is not independently verified here.
- [x] **Task 4 — Cognitive semantic triage** (`neural_context/triage.py`). Skill: `.skills/llm-structured-output/SKILL.md`. Historical completion retained; tests/API need current verification.
- [x] **Task 5 — Selective memory retrieval** (`neural_context/memory.py`). Skill: `.skills/hermes-plugin-development/SKILL.md`. Historical completion retained; verify permitted roots, matching, provenance, and bounds.
- [x] **Task 6 — Two-phase enriched routing** (`neural_context/router.py`). Skill: `.skills/design-fail-closed-gate/SKILL.md`. Historical completion retained; verify single-winner and uncertainty behavior.
- [x] **Task 7 — Receipts and CLI** (`neural_context/receipt.py`, `neural_context/cli.py`). Skill: `.skills/hermes-plugin-development/SKILL.md`. Historical completion retained; reconcile JSONL with SQLite-only derived-data rule before changing persistence.
- [x] **Task 8 — Hermes registration and tools** (`__init__.py`). Skill: `.skills/hermes-plugin-development/SKILL.md`. Historical completion retained; check actual hook kwargs and correlation without fallback pairing.
- [x] **Task 9 — Unit tests and Hermes diagnostics** (`tests/`). Skill: `.skills/hermes-plugin-development/SKILL.md`. Earlier file claims 16 tests and clean validate/doctor; commands/results are not freshly observed in this plan. Re-run at Task 14.

## 4. Remaining critical-path tasks

Each task below is planned work, not authorization to execute it now. Exact future source edit surfaces are stated per task; do not broaden them without approval.

| Approved requirement | Planned coverage |
|---|---|
| FR-01 semantic interpretation | Tasks 4, 12 |
| FR-02 memory-first enriched routing | Tasks 5, 6, 12, 14 |
| FR-03 single-step skill + memory injection | Tasks 6, 13, 14 |
| FR-04 ephemeral user-turn / stable system prompt | Tasks 8, 13 |
| FR-05 conversational fast exit | Tasks 4, 12, 14 |
| FR-06 local governance precedence | Tasks 6, 11 |
| FR-07 shared ≤1,200-token budget | Tasks 12–14 |
| FR-08 fail-closed continuation | Tasks 11–14 |
| FR-09 auditable receipts | Tasks 7, 13; resolve SQLite-vs-JSONL conflict explicitly |
| FR-10 Hermes compatibility | Tasks 8, 15 |

### Task 11 — Runtime contract, disabled fast path, and authorized workspace roots

- [ ] **Owner outcome:** Build services only after enabled check; ensure production hook uses verified payload, never assumes `cwd`; bind workspace/local `.skills/` and `AGENTS.md` only through a documented, operator-authorized root source. Preserve user-level fallback order.
- **Skill:** `.skills/hermes-plugin-development/SKILL.md`; `.skills/design-fail-closed-gate/SKILL.md`.
- **Future edit surface:** `__init__.py`; `neural_context/config.py`; `neural_context/sidecar.py`; `tests/test_plugin_entrypoint.py`; `tests/test_sidecar.py`.
- **Acceptance:** Disabled path performs no feature inference, directory scan, receipt write, or DB creation; missing/ambiguous workspace context abstains; no arbitrary outside-root read; fail-closed returns `None`, never `action:block`.
- **Data/receipt gate:** Resolve the `receipts.jsonl` requirement against `AGENTS.md`'s SQLite sidecar-only rule before changing receipt storage. They conflict; obtain explicit maintainer direction rather than selecting a default. Verify `plugin_data_dir()`'s actual `<HERMES_HOME>/plugin-data/<plugin>/` profile behavior; do not copy the design's `/plugins/` path assumption.
- **Checks:** `pytest tests/test_plugin_entrypoint.py tests/test_sidecar.py -v`; inspect callback against Hermes `turn_context.py` source contract and profile-aware `plugins/plugin_storage.py`. No core monkeypatch.

### Task 12 — Two-phase interpretation, memory grounding, and enriched single-skill routing

- [ ] **Owner outcome:** Strictly parse model intent/entities; selective memory lookup first; preserve fact source/provenance and represent ambiguity; enrich canonical-English BM25 query; choose one winning skill via approved HostLLM or heuristic backend. Memory is evidence/hint, not unquestionable ground truth.
- **Skill:** `.skills/llm-structured-output/SKILL.md`; `.skills/bm25-sidecar-search/SKILL.md`.
- **Future edit surface:** `neural_context/models.py`; `neural_context/triage.py`; `neural_context/memory.py`; `neural_context/router.py`; `tests/test_triage.py`; `tests/test_router.py`.
- **Acceptance:** Synthetic ambiguous entity prompt selects the memory-supported skill when it differs from raw-query-only selection; no-entity/no-memory cases retain sensible fallback; malformed extraction receives at most one repair then abstains; uncertain/conflicting memory never asserts unsupported facts; exactly zero or one skill is routed.
- **Checks:** `pytest tests/test_triage.py tests/test_router.py -v`; all LLM interactions use deterministic fakes unless separately authorized. Measure/record total latency for both model stages (triage plus HostLLM routing); compare with historical 250 ms target without presenting it as achieved. Model-first greeting triage and a sub-5 ms fast-exit requirement may conflict; capture actual result and seek approval for any contract change.
- **Traffic disclosure:** The default configured Hermes inference endpoint is still external provider traffic; “no extra skill-tool round trip” does not mean “no model/API calls” or zero network egress. Disclose actual calls and endpoint behavior before any live smoke.

### Task 13 — Injection budget, receipt persistence, and per-turn integrity

- [ ] **Owner outcome:** Combine complete selected skill plus bounded memory facts within ≤1,200 total injected tokens; keep system prompt bytes unchanged; remove unsafe cross-turn fallback; emit per-turn receipt without raw prompt leakage.
- **Skill:** `.skills/design-fail-closed-gate/SKILL.md`; `.skills/hermes-plugin-development/SKILL.md`.
- **Future edit surface:** `neural_context/router.py`; `neural_context/memory.py`; `neural_context/models.py`; `neural_context/receipt.py`; `__init__.py`; `neural_context/config.py`; `tests/test_router.py`; `tests/test_receipt.py`; `tests/test_plugin_entrypoint.py`.
- **Acceptance:** Combined context—not each section separately—stays under 1,200 by a documented conservative method (`len(text)//4` is not proof without a validated safety margin); retain complete skill sections and truncate only at approved core-section boundaries under FR-07; each receipt maps only to exact `session_id:turn_id`, with no `popitem()` fallback.
- **Decision evidence:** Record the `receipts.jsonl` requirement versus `AGENTS.md` SQLite sidecar rule. Do not silently change either contract; obtain maintainer direction if not already recorded in the approved docs. Any design change requires approval.
- **Checks:** `pytest tests/test_router.py tests/test_receipt.py tests/test_plugin_entrypoint.py -v`; synthetic malformed/oversize and interleaved-turn tests.

### Task 14 — Minimal end-to-end deterministic demo path

- [ ] **Owner outcome:** Exercise exact chain: ambiguous entity → memory hint → changed winning skill vs raw query → single-step injection and receipt; include greeting/no-skill and malformed/error abstention case. Capture actual latency/token measurements from the deterministic harness; label them accurately.
- **Skill:** `.skills/hackathon-judge/SKILL.md`; `.skills/hermes-plugin-development/SKILL.md`.
- **Future edit surface:** `tests/test_router.py`; `tests/test_plugin_entrypoint.py`; `tests/fixtures/context_selection/` only if a new fixture directory is needed and approved.
- **Acceptance:** Fake model/catalog/results verify contracts only: selected skill, memory/skill token components, combined ceiling, and graceful failure. They are not evidence of runtime performance, host integration, or a functioning live demo.
- **Checks:** `pytest tests/test_router.py tests/test_plugin_entrypoint.py -v`; save exact command/output for later narration. Separately, a live Hermes smoke must use synthetic “Lucky” memory and synthetic catalog entries, no private memory, and the actual configured host provider; measure total wall latency including every model/API call and record receipt/output. This requires separate provider-call authorization. If unavailable, mark the live demo blocked; never substitute a fake-harness recording as a live demo.

### Task 15 — Functional freeze and submission preflight

- [ ] **Owner outcome:** Reconcile historical task-9 checks with present tree, run the full required test suite, `hermes plugins validate /opt/neural-context-plugin`, and `hermes plugins doctor /opt/neural-context-plugin`; resolve blockers only in already-approved scope. Run the hackathon judge preflight using its documented source-only workflow. Once implementation is separately authorized, these ordinary checks need no additional permission; do not use `--install-deps` or install dependencies without separate authorization.
- **Skill:** `.skills/hermes-plugin-development/SKILL.md`; `.skills/hackathon-judge/SKILL.md`.
- **Future edit surface:** only critical source/test files from Tasks 11–14 if necessary; otherwise no edits. Do not modify human-authored demo file.
- **Acceptance:** Record exact command, exit status, and observed summary; distinguish unit tests from Hermes diagnostics and judge rubric findings; freeze at 14:00 Bogotá or report schedule risk. Human video values `14,500`, `11,480`, `38.2 ms`, `2 ms`, and `16/16 tests` remain placeholders until supported by observed evidence. Stable system-prompt bytes do not prove KV-cache hits; zero extra skill-tool round trips do not mean zero model/API calls; configured host inference is still external provider traffic. Do not auto-edit the human-owned demo script; seek approval first. Any changed tree requires rerunning affected checks.
- **Checks:** `pytest /opt/neural-context-plugin/tests -v`; `hermes plugins validate /opt/neural-context-plugin`; `hermes plugins doctor /opt/neural-context-plugin`; source-only judge preflight command only after confirming installed tool path and scope.

### Task 10 — Hackathon Submission Artifacts (pending; after Tasks 11–15)

- [ ] **10.1 Demo script and video:** The human-authored `docs/DEMO_VIDEO_SCRIPT.md` already exists; do not overwrite or edit it automatically. Treat its numerical claims as placeholders until Tasks 14–15 provide evidence; record a live demo under 90 seconds, and post only after Jesús authorizes. A fake deterministic harness is not a substitute for live Hermes/provider evidence.
- [ ] **10.2 Push / remote sync:** Push only after separate explicit authorization; preserve actual commit/SHA evidence.
- [ ] **10.3 Catalog PR and official form:** Prepare catalog metadata and submit the official form only after separate explicit authorization. Record confirmed links/receipt, never guessed IDs.
- **Skill:** `.skills/hackathon-judge/SKILL.md`; `.skills/hermes-plugin-development/SKILL.md`.
- **Future edit surface:** `README.md`, `plugin.yaml`, `submission/**` only if explicitly authorized; `docs/DEMO_VIDEO_SCRIPT.md` is human-owned and must not be edited without approval. Public X post, push, PR, and official form are external actions requiring separate explicit authorization.
- **Acceptance:** Submitted public repo/commit reference, catalog PR link, X demo URL tagging `@NousResearch`, and official form confirmation are recorded as observed evidence; no blank/guessed identifiers. Opened catalog PR is not maintainer admission, and PR is not proof of form submission.
- **Checks:** Confirm public repo visibility/pinned SHA and links manually through authorized means; confirm video duration <90s and content matches validated runtime. Leave time for contingency before 17:00.

## 5. Deferred after the deadline (not scope deletion)

- Full ~60-case development/held-out benchmark and controlled arm comparison if it cannot be completed responsibly before submission; preserve the plan and do not fabricate metrics.
- Nonessential UI, extra router backends, broader multilingual corpus, and optional paid/external routing. No additional dependency or provider call without approval.

## 6. Execution route

When implementation is separately authorized, Tasks 11–13 use one delegated writer for the bounded source changes; do not parallelize writers. Task 14's synthetic fixtures may stay with that writer if in scope. Task 15 is verification/preflight work; use a separate verifier when the applicable risk/checking policy requires it. Task 10 documentation/package preparation may be inline only if mechanical and understood; publication remains separately authorized. No route here authorizes execution now.

## 7. Next action

Review this deadline-aware task plan against the currently approved Two-Phase architecture. This document authorizes no implementation in this turn. When separately authorized, execute Tasks 11–15 in order, re-estimating at the 08:00 and noon checkpoints; then complete Task 10 packaging only with separate publication approvals, reserving the final three hours for demo/submission and contingency.
