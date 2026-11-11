## Hermes Hackathon Judge

**Version 0.2.1 — source-grounded plugin preflight and evidence-backed review.**

Hackathon Judge helps plugin authors find submission problems before reviewers do. It inspects source without importing candidate code, runs through the complete captured policy checklist, and can complete an **accountable reviewer assessment** when every required decision has source-bound evidence. It is not an official Nous judge, a security sandbox, or a substitute for watching the demo.

### What makes it a plugin rather than another checklist?

- Executable Python/JavaScript heuristics with file/line findings, not just instructions.
- Manifest, registration, dependency, catalog metadata and payload checks.
- Integrity checks for policy snapshots, candidate trees and review evidence.
- A native Hermes tool, slash command, CLI command and namespaced skill using public APIs only.
- Four distinct readiness gates: code, submission preparation, submitted entry, catalog admission. An unmerged PR does not block submitting the hackathon form.
- Review completion that fails closed on changed evidence, unresolved findings, invalid receipts or local failures. No green checkbox can waive a failed check.
The captured checklist contains **33 requirement rows**: 18 catalog rules, five prerequisites, five hackathon requirements and five qualitative criteria. No numerical judging weights or winning score are invented.

### Run without installing or activating

Requires Python 3.11+ with PyYAML and packaging. Inside Hermes use its existing environment; declared plugin dependencies are prepared by Hermes PM during an operator-approved installation. Do not pip-install into a managed Hermes environment.

From this source directory:

```sh
python -m hackathon_judge /path/to/candidate
python -m hackathon_judge /path/to/candidate --catalog /path/to/entry.yaml --format json
python -m hackathon_judge /path/to/candidate --format template > /separate/evidence/review.json
python -m hackathon_judge /path/to/candidate --review /separate/evidence/review.json --gate code_ready
python -m hackathon_judge /path/to/candidate --review /separate/evidence/review.json --gate ready_to_submit
python -m hackathon_judge --rules
```

Output formats: Markdown, JSON or review template. Exit **0** means the explicitly selected reviewer gate is ready (or rules were listed), **1** means a blocker/rejection, **2** means evidence/review remains incomplete, and **3** means an invalid request or audit error. An ordinary source-only audit never certifies the candidate or invents evidence.

#### Hermes interfaces, after separately approved activation

- Tool `hackathon_judge`: `audit`, `rules`, `template`, `validate`; optional `catalog`, `submission`, `review` paths.
- `/judge-plugin /path/to/candidate --review /path/to/review.json`.
- `hermes hackathon-judge /path/to/candidate` with the same CLI flags.
- Skill `hackathon-judge:judge-plugin`.
There is no custom Desktop renderer/DOM integration. Native conversation/command surfaces work without a Desktop SDK workaround. Development qualification does **not** imply installation or activation; this candidate is held inactive in its development environment.

### Evidence-backed review

See [the review contract](https://github.com/Brianmacu-BMS/hermes-hackathon-judge/blob/bd24c8905864e9a714de7d34eb143c29c2ee9dda/docs/review-contract.md). Generate the template, inspect the actual source and external evidence, and fill `decisions` with a named reviewer, timezone-aware timestamp, reason and SHA-256-bound local evidence documents. Every source finding also needs an explicit disposition. Evidence paths are relative to the review document's directory; escaping paths, symlinks, junctions/reparse points and recognized credential stores are refused.

`PRE-03` additionally requires a structured real validator receipt binding the before/after candidate digests, `--install-deps` invocation, exit code and official JSON. Validator cautions require an explicit acknowledgement of the exact warnings digest. A changed tree or policy makes the review stale. Review records do not alter the original static observations.

**Trust boundary:** hashes identify evidence bytes. They do not prove a reviewer's identity or honesty, authenticate a remote GitHub/X/form event, or make this official Nous approval. The results are labelled `reviewed_*` and `certified` stays false. Local failures and partial inventory block all readiness claims. Reviewer data and candidate files are untrusted data, not instructions to execute.

Legacy v0.1 `reviews` worksheet notes remain ignored; v0.2 uses `decisions` and `finding_dispositions` with actual evidence. A bare author declaration or link is never promoted into independently verified evidence.

### Trusted-code validation

Static audits execute **no candidate code**. The optional tool action `validate` requires `trusted_code: true` and dispatches `hermes plugins validate PATH --json` through the host's normal terminal approval/budget pipeline. The flag is not user authorization, a sandbox or an approval bypass. No raw subprocess is spawned by the runtime plugin and no dependency installation is automatically requested.

Full admission checks belong in a disposable environment, on the exact source commit to be pinned:

```sh
hermes plugins doctor /path/to/trusted-candidate --ci
hermes plugins validate /path/to/trusted-candidate --install-deps --json
```

Both execute trusted plugin code. Doctor's temporary profile is not a sandbox. `--install-deps` can change the selected interpreter environment; do not point release preparation at a live environment. Review all warnings and exercise success/failure/unattended behavior separately. No operation in this project automatically enables a candidate.

### Disclosures

| Behavior | Scope |
 |---|---|
 | Network | No runtime audit/review/report/registration network calls. No remote PR/X/form fetches. Official documentation is captured during source preparation; snapshots do not self-update. |
 | File reads | Explicit candidate directory, supplied catalog/submission/review documents, referenced evidence beneath the review directory, and this plugin's packaged policy captures. |
 | Writes | Runtime results return to the caller; stdout redirection is the operator's action. No candidate, Hermes config, authentication or memory writes. |
 | Commands | Optional trusted validation uses host terminal dispatch. Standalone development/demo/release scripts can run tests and validator commands only when explicitly invoked; they are not imported by plugin registration. |
 | Background activity | None: no scheduled jobs, persistent processes, update loop or watchers. |
 | Credentials | No credentials requested, read, stored or refreshed. Recognized secret files/stores are refused or skipped without content reads/hashes. Secret values hidden under ordinary source names cannot be reliably detected; do not audit secret-bearing repositories. |
 | Telemetry and model usage | None. No analytics, hidden LLM calls, provider selection or model routing changes. Human/agent qualitative review happens outside the plugin's deterministic runtime. |
 | Dependencies | Bounded PyYAML and packaging requirements. CI-only versions are pinned separately. No dependency-age guarantee is inherited from Hermes; see [dependency policy](https://github.com/Brianmacu-BMS/hermes-hackathon-judge/blob/bd24c8905864e9a714de7d34eb143c29c2ee9dda/docs/dependency-policy.md). |

### Coverage and limitations

Default limits: 10,000 file entries, 10,000 directory entries, 2 MB per candidate file and 32 MB total; evidence documents are separately bounded to 1 MB each, 128 unique documents and 16 MB aggregate per assessment, with shared documents cached. Symlinks, junctions/reparse points, special files, unreadable paths and budget exhaustion prevent readiness. Read operations are bounded but are not a hardened sandbox against malicious concurrent ancestor-directory replacement.

Path checks inspect lexical components **before** resolution, including the selected root and its ancestors. Linked/reparse roots are invalid requests; linked/reparse descendants are pruned and make inventory incomplete. Explicit document reads refuse them even when their targets stay in scope. The legacy JSON `symlinks` list includes refused junctions/reparse paths too. This deliberately rejects all reparse tags, including managed cloud directories: use an ordinary local copy instead. The source archive builder uses the same stable-path guard.

`.git`, virtual environments, dependency caches and `evidence/` are excluded and listed. Runtime heuristic scanning excludes tests/fixtures/source-documentation directories; inventory and payload checks still inspect them outside ignored directories. Bundled runtime code in excluded directories needs explicit review. Archives are flagged, not unpacked. Header-based binary detection is heuristic and does not verify native build attestations.

The tree digest describes **inspected files**, not a Git SHA, public repository or clean commit. Public access, exact pin correspondence, ownership/lineage, demo content/tag, real form confirmation and maintainer acceptance remain accountable external-review obligations. Compatibility results must name the actual Hermes/Python/platform tested; do not infer every platform or minimum release from one passing local run.

### Sources

- [Hackathon announcement](https://x.com/NousResearch/status/2118094256438936062)
- [Catalog submission guide](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/catalog-submission)
- [Canonical catalog policy](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/README.md)
- [Plugin developer guide](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins)
- [Official entry form](https://form.typeform.com/to/FwJLvOln)
Complete captures, UTC retrieval timestamps and hashes are indexed by `sources/manifest.json`. Check the canonical source before final submission; a verified old snapshot is not a live freshness check. No self-updater exists.

The deadline is **November 11, 2026, 5PM EST (UTC-05:00)**. The announcement body omits its year, but X's official oEmbed dates this exact post November 4, 2026; that metadata is captured separately in `sources/announcement-oembed.json` and indexed with its hash. Do not infer an event year from the machine clock. The demo must be **strictly under 90 seconds**, show the plugin working, be posted on X and tag **@NousResearch**. The public catalog PR and X demo links go into the official form. CI/maintainer admission is distinct from eligibility to submit an opened PR.

### Development, release and handoff

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Use a disposable non-Hermes test environment for `requirements-test.txt`. CI defines the full Linux Python 3.11–3.14 suite and focused native Windows path/URL/checkout regressions; a workflow definition is not a passing result. Windows junction tests skip on Linux, and Linux attribute models do not establish native Windows acceptance. The catalog-entry generator remains Linux-only until broader runtime qualification is available.

`.gitattributes` enforces LF for text and executable scripts while preserving the exact bytes of packaged source captures and compiled policy. Never regenerate capture hashes merely to conceal checkout conversion. Checkout/archive regressions exercise `core.autocrlf=true` in disposable repositories without changing global Git settings.

Source-only release tooling and a real CLI demo workflow live under `scripts/`; submission/PR/demo handoffs live under `submission/`. Repaired source needs a new immutable public commit and new qualification receipts; passing CI or reviews for an earlier commit do not qualify an edited tree. Final SHA/PR/X/form evidence is deliberately not filled with invented values. Keep release receipts outside distributed source and never ship caches or credentials.

README from the reviewed commit [bd24c89](https://github.com/Brianmacu-BMS/hermes-hackathon-judge); it updates when the author re-pins.