---
name: hermes-plugin-development
description: >-
  Build, test, release, and list plugins for Hermes Agent (NousResearch/hermes-agent):
  plugin.yaml manifests, register(ctx) tools and hooks, bundled skills, dashboard and
  desktop panes, secrets, doctor/validate, catalog SHA pins and PRs, deploy/rollback,
  and the pitfalls hit shipping real community plugins. Use whenever a task touches a
  Hermes plugin.
license: MIT
---
# Hermes plugin development

Field guide distilled from shipping several community Hermes plugins (map/OSINT
dashboard, virtual pets game, live-coding music studio, video tooling). Every API
and CLI claim here was checked against hermes-agent source; see
`references/verification.md` for exactly what was checked and on which version.
**When this file and the live Hermes source/docs disagree, the source wins.**

Canonical upstream docs (paths inside `NousResearch/hermes-agent`):

| Topic | Path |
|---|---|
| Build a plugin | `website/docs/developer-guide/plugins/index.md` |
| Public plugin API | `website/docs/developer-guide/plugins/public-api.md` |
| Catalog submission | `website/docs/developer-guide/plugins/catalog-submission.md` |
| Catalog admission rules + entry schema | `plugin-catalog/README.md` |
| Dashboard plugins | `website/docs/user-guide/features/extending-the-dashboard.md` |
| Desktop pane SDK | `website/docs/developer-guide/desktop-plugin-sdk.md` |
| `ctx` implementation | `hermes_cli/plugins_context.py` |
| Validator / doctor | `hermes_cli/plugin_validate.py`, `hermes_cli/subcommands/plugins.py` |

Third-party plugins live in **their own repos**. They reach users by
`hermes plugins install owner/repo[/subdir]` or by a catalog entry, never by a
PR into Hermes core.

## 1. Anatomy

```text
my-plugin/
├── plugin.yaml        # manifest (required for native plugins)
├── __init__.py        # def register(ctx) -> None   (required)
├── pyproject.toml     # deps with upper bounds (wins over python_dependencies)
├── skills/<id>/SKILL.md   # optional bundled skills
├── dashboard/         # optional web dashboard tab: manifest.json, dist/index.js, plugin_api.py
├── desktop/plugin.js  # optional desktop/webapp pane via @hermes/plugin-sdk
├── tests/
├── README.md  LICENSE
```

- Hermes discovers `~/.hermes/plugins/<name>/` (one category level allowed).
  Symlinking a checkout there is the normal dev loop.
- Plugin in a subfolder of a bigger repo? Use `subdir:` in the catalog entry and
  `hermes plugins install owner/repo/sub/path`. Keeps evals, art and fixtures
  out of the installed tree.
- A working skeleton that passes `validate` and `doctor`: `templates/hello-hermes/`.

## 2. Manifest essentials (`references/plugin-yaml.md`)

```yaml
manifest_version: 2
api_version: 1
name: my-plugin            # [a-z0-9_-]{1,64}; the catalog key
version: "0.1.0"           # QUOTE IT
description: What it does. Disclosure: network, subprocesses, threads, credentials.
requires_hermes: ">=0.21.6"   # SemVer floor only
provides_tools: [my_tool]     # must equal what register() registers
provides_hooks: []
requires_env: []
```

- `provides_tools`/`provides_hooks` don't gate loading, but `validate` and
  catalog CI fail if they drift from real registrations.
- Unknown keys are tolerated with a warning; don't rely on invented keys doing anything.
- Duplicate keys are silently "last one wins" in PyYAML. Test with a strict loader.

## 3. Tools (`references/tools-and-hooks.md`)

Verified signature (`hermes_cli/plugins_context.py`):

```text
ctx.register_tool(name, toolset, schema, handler, check_fn=None, requires_env=None,
                  is_async=False, description="", emoji="", override=False)
```

- **Call with keywords only.** Passing an unsupported kw (e.g. `output_schema`)
  raises `TypeError`, aborts `register()`, and the plugin loads with **0 tools**.
  Filter kwargs against `inspect.signature` (template does this).
- `register()` must be fast and offline. Lazy-import heavy deps in handlers.
- **Lean tool surface.** Every schema is sent on every turn. Prefer a few
  composable tools with an `op` enum over dozens of tiny ones (one plugin went
  from ~70 tools to 11). Measure with tiktoken `o200k_base` before/after and
  report the delta (`templates/hello-hermes/scripts/measure_schemas.py`).
- Return structured JSON: `{"ok": false, "code": ..., "error": ..., "fix": ...}`
  so the agent can recover by itself.
- Hook callbacks should accept `**kwargs`; payloads gain fields over time.
- `override=True` needs `plugins.entries.<id>.allow_tool_override: true`.
- `ctx.call_mcp` is default-deny (per-plugin MCP allowlist).

## 4. Skills, state, secrets

- `ctx.register_skill(name, path)` makes a read-only skill resolvable as
  `<plugin_name>:<name>` via `skill_view`. Keep skill text in sync with tool names.
- Persist state with `plugins.plugin_storage.plugin_data_dir(name)` /
  `plugin_db(name)`: profile-aware, survives updates. Never write into the checkout.
- **Secrets:** declare in `requires_env` (or `config_schema` secret), read with
  `agent.secret_scope.get_secret(NAME)`. No `.env` file reads, no raw process-environment
  fallbacks, no vendor CLI/browser/keychain token scraping (catalog rule 11).
  Token files you create yourself: mode `0600`, never logged.
- **Respect user choices.** Never re-enable a layer, feature, or setting the user
  turned off; persist "off" and treat it as authoritative across restarts and updates.

## 5. UI surfaces (`references/panes.md`)

- **Dashboard tab:** `dashboard/manifest.json` + an IIFE bundle using
  `window.__HERMES_PLUGIN_SDK__` (don't bundle React). Optional `plugin_api.py`
  routes mount at `/api/plugins/<name>/`.
- **An already-mounted `/api/plugins/<name>/` route is not re-mounted.** After
  changing `plugin_api.py`, restart the gateway/dashboard process. The
  `/api/dashboard/plugins/rescan` endpoint is for discovering newly added plugins.
- **Desktop pane:** `desktop/plugin.js` importing only `@hermes/plugin-sdk` (and `react`).
  Guest content runs in `SandboxedFrame` (opaque origin). No prototype patching,
  `eval`, app-chunk imports, or core DOM scraping (catalog rule 8).

## 6. Test (`references/testing.md`)

```bash
pytest -q                                  # fake ctx, no Hermes needed
hermes plugins doctor <path|id> --ci       # real runtime contracts
hermes plugins validate <path> --install-deps   # same gate as catalog CI
```

- Fake-ctx test asserting registered names == `provides_tools` is the single
  most valuable test.
- No Hermes available? Build a mock host (fake ctx + in-memory pane/session) and
  label results "mock host" vs "verified in Hermes" honestly in the README.
- The install scanner reads **every file, Markdown included**. Merely mentioning the
  superuser command word (su+do), env-dump or filesystem-format commands, or raw
  environment reads trips `sudo_usage`-style findings and can fail `validate`.
  Rephrase docs and strings, or explain the finding in the PR.

## 7. Release → catalog (`references/release-and-catalog.md`)

1. Tests + doctor + validate green. Secret-scan the tree.
2. Bump `version` in `plugin.yaml` (and `pyproject.toml`); merge to main.
3. Tag `vX.Y.Z` at that commit → GitHub release.
4. On **your fork** of hermes-agent, branch `catalog/<name>-vX.Y.Z`, add or edit
   **only** `plugin-catalog/<name>.yaml`: full 40-hex `sha` (quoted), quoted
   `version`, capabilities matching the pin, images on GitHub raw hosts pinned
   to the same sha and returning HTTP 200.
5. `python scripts/validate_plugin_catalog.py plugin-catalog/<name>.yaml`.
6. Open the PR from the fork (`templates/catalog/PR_BODY.md`). Every SHA bump is a new PR.

**PR gotchas:** contributors without write access may have at most **5 open PRs**
on NousResearch/hermes-agent. Over that, `CreatePullRequest` fails with a
permissions error from the CLI, API and GitHub-App integrations alike, even
though your auth is fine. Close or land one, or open it in the web UI from
the compare link once a slot is free.

## 8. Deploy to a running Hermes (`references/release-and-catalog.md`)

Get the owner's OK first. Back up the checkout and record the current HEAD as
the **rollback SHA**. Check out the target, install deps if changed, restart only
what's needed (gateway for `plugin_api.py`/hook changes), then `hermes plugins
doctor <id>` and exercise one real tool. On failure: check out the rollback SHA,
restart, re-run doctor.

## 9. Top pitfalls (full table: `references/pitfalls.md`)

| Symptom | Fix |
|---|---|
| Plugin loads with 0 tools | Unsupported kw to `register_tool` raised in `register()`. Filter kwargs. |
| validate: declared tools mismatch | Make `provides_tools` identical to registrations. |
| Catalog validator: "sha 0 must be 40 hex" | Quote the sha; an all-digit sha parses as an int. |
| Dashboard API change has no effect | Route already mounted; restart the gateway. |
| Key works in shell, not in plugin | Declare `requires_env`; read via `get_secret`. |
| `CreatePullRequest` permission error | 5-open-PR cap or not from a fork; free a slot. |
| Context bloat / slow turns | Fewer composable tools; measure o200k. |
| validate security scan fails on docs | Scanner reads Markdown too; reword risky command words. |

## 10. Hackathons and demos

See `references/hackathon-submission.md`: open catalog PR (needn't be merged),
demo video under 90 s posted on X tagging the organiser, submission form.
Never post or submit on someone's behalf without their explicit yes.

## Companion tools (when installed as the hermes-plugin-dev Hermes plugin)

- `hpd_scaffold(target_dir, name)`: copy `templates/hello-hermes` into a new or empty
  folder, renamed, with a catalog entry and PR body stub.
- `hpd_check(path, catalog_entry?)`: offline lint covering strict YAML, manifest rules,
  `provides_tools` vs registered, schema o200k tokens, and catalog entry rules. Then run
  `hermes plugins doctor` and `validate` in a terminal for the authoritative gate.

Other agents can get the same checks with `python hpd_core.py` functions from the repo.

## Files in this skill

| Path | Purpose |
|---|---|
| `references/plugin-yaml.md` | Manifest fields |
| `references/tools-and-hooks.md` | `ctx` API, schema design, hooks, skills, secrets, storage |
| `references/panes.md` | Dashboard and desktop panes |
| `references/testing.md` | Fake ctx, mock host, doctor, validate, scanner |
| `references/release-and-catalog.md` | Versioning, tags, catalog entry, PRs, deploy, rollback |
| `references/pitfalls.md` | Symptom → cause → fix |
| `references/hackathon-submission.md` | Demo + submission playbook |
| `references/worked-examples.md` | Public plugin repos to study |
| `references/verification.md` | What was verified against source, and what wasn't |
| `templates/hello-hermes/` | Minimal plugin that passes validate + doctor |
| `templates/catalog/` | Catalog entry + PR body templates |
