from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import neural_context.cli as cli
import neural_context.sidecar as sidecar_mod
from neural_context.config import PluginConfig
from neural_context.i18n import resolve_language
from neural_context.memory import _load_file_fact_nodes
from neural_context.models import CognitiveRoute, SkillCandidate, TriageAnalysis
from neural_context.receipt import ReceiptEmitter
from neural_context.router import ContextCoordinator, GraphRouter
from neural_context.sidecar import SidecarIndex
from neural_context.triage import (
    CognitiveTriage,
    _extract_balanced_json,
    _is_truncated_completion,
    _normalize_turn_content,
)
from neural_context.workspace import inspect_workspace_governance
import __init__ as plugin_mod


def _reset_plugin_globals(monkeypatch):
    monkeypatch.setattr(plugin_mod, "_config", None)
    monkeypatch.setattr(plugin_mod, "_sidecar", None)
    monkeypatch.setattr(plugin_mod, "_triage", None)
    monkeypatch.setattr(plugin_mod, "_coordinator", None)
    monkeypatch.setattr(plugin_mod, "_receipt_emitter", None)
    with plugin_mod._session_turns_lock:
        plugin_mod._session_turns.clear()


def _make_ctx(config_values: dict, llm_complete=None):
    ctx = MagicMock()
    ctx.get_config.side_effect = lambda key, default=None: config_values.get(key, default)
    ctx.session_cwd = None
    if llm_complete is not None:
        ctx.llm = MagicMock()
        ctx.llm.complete.side_effect = llm_complete
    else:
        ctx.llm = None
    return ctx


# ---------------------------------------------------------------------------
# FR-1: Symlink Lexical Tier Attribution, Shadow Persistence & Post-Hook Isolation
# ---------------------------------------------------------------------------


def test_fr1_symlinked_skill_preserves_lexical_tier_zero(tmp_path, monkeypatch):
    project_root = tmp_path / "proj" / ".skills"
    user_root = tmp_path / "user" / "skills"
    external_dir = tmp_path / "external_repo" / "shared-skill"
    project_root.mkdir(parents=True)
    user_root.mkdir(parents=True)
    external_dir.mkdir(parents=True)

    real_skill_md = external_dir / "SKILL.md"
    real_skill_md.write_text("---\nname: shared-skill\ndescription: Shared deploy workflow\n---\n# Body\n")
    symlink_skill_dir = project_root / "shared-skill"
    symlink_skill_dir.symlink_to(external_dir, target_is_directory=True)
    logical_skill_md = symlink_skill_dir / "SKILL.md"

    fake = types.ModuleType("tools.skills_tool")
    fake.skill_search_dirs = lambda: ([(0, project_root), (1, user_root)], user_root)
    fake.find_all_skills = lambda: [{"name": "shared-skill"}]
    fake.locate_skill = lambda name, _cat=None, _roots=None: (None, str(symlink_skill_dir), str(logical_skill_md))
    tools = types.ModuleType("tools")
    tools.skills_tool = fake
    monkeypatch.setitem(sys.modules, "tools", tools)
    monkeypatch.setitem(sys.modules, "tools.skills_tool", fake)

    paths, project_paths, source_info, names, _tiers = sidecar_mod.discover_skill_inventory(include_project=True)

    assert len(paths) == 1
    discovered = paths[0]
    assert discovered in project_paths
    tier, matched_root = source_info[discovered]
    assert tier == 0
    assert matched_root == project_root
    assert names[discovered] == "shared-skill"


def test_fr1_composite_pk_persists_shadowed_tiers_and_precedence(tmp_path):
    db_path = str(tmp_path / "sidecar.db")

    # Seed legacy single-PK table first to test smooth schema migration
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE skills_sidecar (
                skill_name TEXT PRIMARY KEY,
                skill_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                description TEXT,
                lead_paragraph TEXT,
                is_project_local INTEGER DEFAULT 0,
                indexed_tokens INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_root TEXT NOT NULL DEFAULT '',
                source_tier INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.commit()

    idx = SidecarIndex(db_path)

    user_root = tmp_path / "user" / "skills"
    proj_root = tmp_path / "proj" / ".skills"
    user_skill = user_root / "deploy-app" / "SKILL.md"
    proj_skill = proj_root / "deploy-app" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    proj_skill.parent.mkdir(parents=True)

    user_skill.write_text("---\nname: deploy-app\ndescription: Global user deployer\n---\n# User\nGlobal instructions.\n")
    proj_skill.write_text("---\nname: deploy-app\ndescription: Project local deployer\n---\n# Project\nRepo instructions.\n")

    idx.sync_skills(
        [user_skill, proj_skill],
        project_local_paths=[proj_skill],
        source_info={user_skill: (1, user_root), proj_skill: (0, proj_root)},
        skill_names={user_skill: "deploy-app", proj_skill: "deploy-app"},
    )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT skill_name, source_tier, description FROM skills_sidecar ORDER BY source_tier ASC"
        ).fetchall()

    assert len(rows) == 2
    assert rows[0][1] == 0 and "Project local deployer" in rows[0][2]
    assert rows[1][1] == 1 and "Global user deployer" in rows[1][2]
    assert idx.n_docs == 1
    assert idx._candidates["deploy-app"].is_project_local is True
    assert "Project local deployer" in idx._candidates["deploy-app"].description

    # Deleting tier-0 project skill restores tier-1 user skill cleanly
    proj_skill.unlink()
    idx.sync_skills([], project_local_paths=[], replace_tiers={0})
    assert idx.n_docs == 1
    assert idx._candidates["deploy-app"].is_project_local is False
    assert "Global user deployer" in idx._candidates["deploy-app"].description


def test_fr1_on_post_llm_call_skips_skill_discovery(tmp_path, monkeypatch):
    _reset_plugin_globals(monkeypatch)
    receipts = tmp_path / "receipts.jsonl"
    db_path = tmp_path / "sidecar.db"
    ctx = _make_ctx({"sidecar_db_path": str(db_path), "receipts_path": str(receipts)})

    discover_mock = MagicMock(side_effect=AssertionError("discover_skill_inventory must not run on post_llm_call"))
    monkeypatch.setattr(plugin_mod, "discover_skill_inventory", discover_mock)

    analysis = TriageAnalysis(route=CognitiveRoute.DIRECT_REPLY, reasoning="chat")
    route_res = MagicMock()
    route_res.selected_skill = None
    route_res.selected_skills = []
    route_res.selected_memories = []
    route_res.combined_context_block = None
    route_res.suggested_missing_skill = None
    route_res.injected_skill_content = None
    route_res.tokens_saved_estimate = 0
    route_res.triage_latency_ms = 1.0
    route_res.coordination_latency_ms = 1.5
    route_res.router_backend = None
    route_res.configured_router_backend = "graph"
    route_res.status = "no_context_injected"
    route_res.message = "chat"

    with plugin_mod._session_turns_lock:
        plugin_mod._session_turns["s1:t1"] = (analysis, route_res, "telegram", "gpt")

    plugin_mod.on_post_llm_call(response="hi", session_id="s1", turn_id="t1", _ctx=ctx)
    assert discover_mock.call_count == 0
    assert receipts.is_file()


# ---------------------------------------------------------------------------
# FR-2: Hot Configuration Reload
# ---------------------------------------------------------------------------


def test_fr2_hot_config_reload_updates_enabled_timeout_and_paths(tmp_path, monkeypatch):
    _reset_plugin_globals(monkeypatch)
    cfg = {
        "enabled": True,
        "pre_hook_enabled": True,
        "timeout_ms": 1500,
        "router_backend": "graph",
        "sidecar_db_path": str(tmp_path / "s1.db"),
        "receipts_path": str(tmp_path / "r1.jsonl"),
        "software_inspection_enabled": False,
        "selective_memory_enabled": False,
    }
    monkeypatch.setattr(plugin_mod, "discover_skill_inventory", lambda **_kw: ([], [], {}, {}, {0, 1}))
    ctx = _make_ctx(cfg, llm_complete=lambda **_kw: json.dumps({"route": "DIRECT_REPLY", "reasoning": "ok"}))

    assert plugin_mod.on_pre_llm_call("hello there", session_id="s1", turn_id="t1", _ctx=ctx) is None
    assert plugin_mod._config.timeout_ms == 1500
    assert plugin_mod._triage.timeout_ms == 1500
    assert Path(plugin_mod._receipt_emitter.receipts_path) == tmp_path / "r1.jsonl"

    # Mutate host config live and verify singletons refresh
    cfg["timeout_ms"] = 4321
    cfg["receipts_path"] = str(tmp_path / "r2.jsonl")
    assert plugin_mod.on_pre_llm_call("hello again", session_id="s1", turn_id="t2", _ctx=ctx) is None
    assert plugin_mod._config.timeout_ms == 4321
    assert plugin_mod._triage.timeout_ms == 4321
    assert Path(plugin_mod._receipt_emitter.receipts_path) == tmp_path / "r2.jsonl"

    # Flip enabled to False live -> bypass immediately without calling LLM
    llm_calls_before = ctx.llm.complete.call_count
    cfg["enabled"] = False
    assert plugin_mod.on_pre_llm_call("run deploy", session_id="s1", turn_id="t3", _ctx=ctx) is None
    assert ctx.llm.complete.call_count == llm_calls_before


# ---------------------------------------------------------------------------
# FR-3: Omnichannel Extraction, Slash Fast Exit & Conversation History
# ---------------------------------------------------------------------------


def test_fr3_extract_message_text_strips_media_and_never_stringifies_dict():
    payload = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA=="}},
        {"type": "input_audio", "data": "BBB=="},
        {"type": "input_text", "text": "Deploy the FastAPI container"},
    ]
    extracted = plugin_mod.extract_message_text(payload)
    assert extracted == "Deploy the FastAPI container"
    assert "base64" not in extracted

    # Binary-only dict must yield empty string, not "{'type': 'image', ...}"
    assert plugin_mod.extract_message_text({"type": "image", "data": "ZZZZ"}) == ""
    assert plugin_mod.extract_message_text({"caption": "Screenshot error trace"}) == "Screenshot error trace"


def test_fr3_slash_command_fast_exit_and_history_normalization(tmp_path, monkeypatch):
    _reset_plugin_globals(monkeypatch)
    captured_prompts = []

    def fake_complete(**kwargs):
        captured_prompts.append(kwargs["messages"][0]["content"])
        return json.dumps({"route": "DIRECT_REPLY", "reasoning": "ok"})

    ctx = _make_ctx(
        {
            "enabled": True,
            "pre_hook_enabled": True,
            "sidecar_db_path": str(tmp_path / "s.db"),
            "receipts_path": str(tmp_path / "r.jsonl"),
        },
        llm_complete=fake_complete,
    )
    monkeypatch.setattr(plugin_mod, "discover_skill_inventory", lambda **_kw: ([], [], {}, {}, {0, 1}))

    # Slash command fast exit
    assert plugin_mod.on_pre_llm_call("  /model gpt-5", _ctx=ctx) is None
    assert len(captured_prompts) == 0

    # Multimodal & None assistant turns in conversation_history must serialize safely
    history = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,SECRET"}}, {"type": "text", "text": "Inspect staging nginx"}]},
        {"role": "assistant", "content": None},
    ]
    assert _normalize_turn_content(None) == ""
    plugin_mod.on_pre_llm_call("continue", session_id="s", turn_id="1", conversation_history=history, _ctx=ctx)
    assert len(captured_prompts) == 1
    assert "Inspect staging nginx" in captured_prompts[0]
    assert "SECRET" not in captured_prompts[0]


# ---------------------------------------------------------------------------
# FR-4: Balanced JSON, Nested Truncation & max_tokens=0 Retry
# ---------------------------------------------------------------------------


def test_fr4_balanced_json_nested_truncation_and_max_tokens_zero_retry():
    raw = 'Reasoning: {"route": "SKILL_ONLY", "memory_probe": {"entities": ["acme"]}} trailing note {not json}.'
    parsed = _extract_balanced_json(raw)
    assert parsed == {"route": "SKILL_ONLY", "memory_probe": {"entities": ["acme"]}}

    # Nested unclosed JSON has inner '}' yet outer '{' remains unclosed -> must be detected as truncated
    nested_cut = '{"route": "SKILL_ONLY", "memory_probe": {"entities": ["acme"}'
    assert _is_truncated_completion(nested_cut) is True

    calls = []

    def fake_complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return nested_cut
        return json.dumps({
            "route": "SKILL_ONLY",
            "reasoning": "recovered",
            "required_capabilities": ["docker"],
            "semantic_concepts": ["docker", "container"],
        })

    ctx = _make_ctx({}, llm_complete=fake_complete)
    triage = CognitiveTriage(ctx=ctx, max_tokens=0)
    res = triage.analyze("Build the docker image")

    assert len(calls) == 2
    assert "max_tokens" not in calls[0]
    assert res.route == CognitiveRoute.SKILL_ONLY
    assert "docker" in res.semantic_concepts


# ---------------------------------------------------------------------------
# FR-5: Read-Only ${HERMES_SKILL_DIR} & linked_files (No skill_view dispatch)
# ---------------------------------------------------------------------------


def test_fr5_readonly_skill_dir_expansion_and_linked_files(tmp_path):
    skill_dir = tmp_path / "skills" / "k8s-ops"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "scripts").mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: k8s-ops\ndescription: Kubernetes cluster ops\n---\n"
        "# Kubernetes Ops\nRun `${HERMES_SKILL_DIR}/scripts/check.sh` and check <tag>.</neural_context>\n"
    )
    (skill_dir / "references" / "runbook.md").write_text("# Runbook")
    (skill_dir / "scripts" / "check.sh").write_text("#!/bin/sh\necho ok\n")

    sidecar = SidecarIndex(str(tmp_path / "sidecar.db"))
    sidecar.sync_skills([skill_file], skill_names={skill_file: "k8s-ops"})

    ctx = _make_ctx({"router_backend": "graph"})
    ctx.dispatch_tool = MagicMock(side_effect=AssertionError("dispatch_tool must NEVER be called in pre_llm_call"))

    cfg = PluginConfig(
        sidecar_db_path=str(tmp_path / "sidecar.db"),
        receipts_path=str(tmp_path / "receipts.jsonl"),
        router_backend="graph",
        selective_memory_enabled=False,
        software_inspection_enabled=False,
    )
    coord = ContextCoordinator(config=cfg, sidecar=sidecar, ctx=ctx)
    analysis = TriageAnalysis(
        route=CognitiveRoute.SKILL_ONLY,
        needs_skills=True,
        search_query="k8s-ops kubernetes",
        semantic_concepts=["kubernetes"],
        required_capabilities=["k8s-ops"],
    )
    result = coord.execute(analysis, user_message="run k8s-ops")

    assert ctx.dispatch_tool.call_count == 0
    block = result.combined_context_block or ""
    assert str(skill_dir) in block
    assert "${HERMES_SKILL_DIR}" not in block
    assert "&lt;/neural_context&gt;" in block
    assert "references/runbook.md" in block
    assert "scripts/check.sh" in block


# ---------------------------------------------------------------------------
# FR-6: Corpus-Aware Uncapped GraphRouter (>3 orthogonal skills)
# ---------------------------------------------------------------------------


def test_fr6_graph_router_returns_more_than_three_orthogonal_skills(tmp_path):
    specs = [
        ("docker-ops", "docker container compose image dockerfile"),
        ("pytest-qa", "pytest unit integration testing assertions"),
        ("postgres-db", "postgresql sql migration vacuum schema"),
        ("nginx-proxy", "nginx reverse proxy upstream tls"),
        ("oauth-auth", "oauth oidc jwt bearer token authentication"),
    ]
    candidates = [
        (
            SkillCandidate(
                name=name,
                path=str(tmp_path / name / "SKILL.md"),
                sha256="x",
                description=desc,
                lead_paragraph=desc,
            ),
            5.0 - idx * 0.1,
        )
        for idx, (name, desc) in enumerate(specs)
    ]

    analysis = TriageAnalysis(
        route=CognitiveRoute.SKILL_ONLY,
        needs_skills=True,
        semantic_concepts=["docker", "pytest", "postgresql", "nginx", "oauth"],
        required_capabilities=["docker", "pytest", "postgresql", "nginx", "oauth"],
        intent_summary_en="Configure docker, pytest, postgresql, nginx, and oauth",
    )
    winners = GraphRouter().route_all(analysis, candidates)
    winner_names = {w.name for w in winners}
    assert len(winner_names) >= 4
    assert {"docker-ops", "pytest-qa", "postgres-db", "nginx-proxy"}.issubset(winner_names)


# ---------------------------------------------------------------------------
# FR-7: Markdown Fence, Strict ATX & § Section Delimiter Parsing
# ---------------------------------------------------------------------------


def test_fr7_markdown_fences_atx_and_section_delimiters(tmp_path):
    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text(
        "## Client Alpha\n"
        "Primary cluster runs PostgreSQL 16 in production.\n"
        "```python\n"
        "# Fake heading inside code fence\n"
        "def ping():\n"
        "    return 'Client Alpha still owns this fence'\n"
        "```\n"
        "#urgent tag: SLA response under 15 minutes for Client Alpha.\n"
        "§\n"
        "Standalone note separated by section delimiter.\n",
        encoding="utf-8",
    )

    nodes = _load_file_fact_nodes(mem_file)
    facts_by_text = {n.fact: n for n in nodes}

    assert all(n.section_entity != "Fake heading inside code fence" for n in nodes)
    fence_line = next(n for n in nodes if "def ping()" in n.fact)
    assert fence_line.section_entity == "Client Alpha"

    hashtag_line = next(n for n in nodes if "#urgent" in n.fact)
    assert hashtag_line.section_entity == "Client Alpha"

    sep_line = next(n for n in nodes if "Standalone note separated by section delimiter." in n.fact)
    assert sep_line.section_entity is None


# ---------------------------------------------------------------------------
# FR-8: .git Worktree File & Bounded Governance Walk
# ---------------------------------------------------------------------------


def test_fr8_git_worktree_file_detection(tmp_path):
    worktree_dir = tmp_path / "worktree_repo"
    sub = worktree_dir / "services" / "api"
    sub.mkdir(parents=True)
    # Worktree uses a .git file pointing to bare/main repo
    (worktree_dir / ".git").write_text("gitdir: /tmp/main/.git/worktrees/wt\n", encoding="utf-8")

    dom = inspect_workspace_governance(cwd=str(sub), enabled=True)
    assert dom.is_software_task is True
    assert dom.governance_repo_root == str(worktree_dir)
    assert dom.resolved_workspace_path == str(sub)

    # Forbidden OS paths do not run governance walk
    etc_dom = inspect_workspace_governance(cwd="/etc", enabled=True)
    assert etc_dom.governance_repo_root is None


# ---------------------------------------------------------------------------
# FR-9: CLI 15-Key Parity, Symlink index --path, i18n Colon Split & Notice Cap
# ---------------------------------------------------------------------------


def test_fr9_cli_15_key_parity_symlink_index_i18n_colon_and_notice_budget(tmp_path, monkeypatch):
    # 1. All 15 plugin.yaml config_schema keys present in CONFIG_FIELD_TYPES
    expected_keys = {
        "enabled",
        "pre_hook_enabled",
        "router_backend",
        "max_context_tokens",
        "selective_memory_enabled",
        "memory_token_budget",
        "memory_db_uri",
        "extra_memory_dirs",
        "excluded_session_sources",
        "excluded_message_prefixes",
        "software_inspection_enabled",
        "project_scan_paths",
        "sidecar_db_path",
        "receipts_path",
        "timeout_ms",
    }
    assert set(cli.CONFIG_FIELD_TYPES.keys()) == expected_keys

    parsed_list, err = cli._coerce_and_validate("project_scan_paths", '["~/src", "~/work"]')
    assert err is None and parsed_list == ["~/src", "~/work"]

    parsed_csv, err2 = cli._coerce_and_validate("excluded_session_sources", "kanban, subagent")
    assert err2 is None and parsed_csv == ["kanban", "subagent"]

    # 2. index --path includes symlinked skills rooted inside the scanned dir
    scan_dir = tmp_path / "proj" / ".skills"
    ext_dir = tmp_path / "external" / "linked-skill"
    scan_dir.mkdir(parents=True)
    ext_dir.mkdir(parents=True)
    (ext_dir / "SKILL.md").write_text("---\nname: linked-skill\n---\n# Body\n")
    link_dir = scan_dir / "linked-skill"
    link_dir.symlink_to(ext_dir, target_is_directory=True)
    resolved_skill = (ext_dir / "SKILL.md").resolve()

    sidecar = SidecarIndex(str(tmp_path / "sidecar.db"))
    monkeypatch.setattr(
        cli,
        "discover_skill_inventory",
        lambda **_kw: (
            [resolved_skill],
            [resolved_skill],
            {resolved_skill: (0, scan_dir)},
            {resolved_skill: "linked-skill"},
            {0, 1},
        ),
    )
    ret = cli.handle_cli(["index", "--path", str(scan_dir)], PluginConfig(), sidecar, None)
    assert ret == 0
    assert "linked-skill" in sidecar._candidates

    # 3. i18n colon separator splitting (LANGUAGE="es:en")
    monkeypatch.setenv("HERMES_LANGUAGE", "es:en")
    assert resolve_language(None) == "es"

    # 4. Missing-skill notice honors positive max_context_tokens ceiling
    cfg = PluginConfig(
        sidecar_db_path=str(tmp_path / "sidecar.db"),
        receipts_path=str(tmp_path / "receipts.jsonl"),
        max_context_tokens=8,
    )
    coord = ContextCoordinator(config=cfg, sidecar=sidecar)
    tiny_notice = coord._assemble_context_block(
        None,
        None,
        [],
        suggested_skill="very-long-missing-skill-" + ("x" * 200),
        language="en",
    )
    assert tiny_notice is None


# ---------------------------------------------------------------------------
# Pre-PR Preventive Audit Fixes (B1-B5)
# ---------------------------------------------------------------------------


def test_b1_get_hermes_language_uses_secret_scope_in_multiplex_mode(monkeypatch):
    from neural_context.config import get_hermes_language

    monkeypatch.setenv("HERMES_LANGUAGE", "fr")
    fake_scope = types.ModuleType("agent.secret_scope")

    class _FakeUnscoped(Exception):
        pass

    fake_scope.UnscopedSecretError = _FakeUnscoped
    fake_scope.get_secret = lambda name: None  # Scoped profile did not set HERMES_LANGUAGE
    fake_agent = types.ModuleType("agent")
    fake_agent.secret_scope = fake_scope
    fake_cli_cfg = types.ModuleType("hermes_cli.config")
    fake_cli_cfg.load_config_readonly = lambda: {"display": {"language": "es"}}
    fake_cli = types.ModuleType("hermes_cli")
    fake_cli.config = fake_cli_cfg

    monkeypatch.setitem(sys.modules, "agent", fake_agent)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", fake_scope)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake_cli_cfg)

    # Scoped secret missing (None) must NOT fall back to process env "fr"; falls back to profile display.language
    assert get_hermes_language() == "es"


def test_b2_hot_reload_refreshes_memory_db_uri_and_clearing_does_not_leak_stale(tmp_path, monkeypatch):
    _reset_plugin_globals(monkeypatch)
    monkeypatch.setattr(plugin_mod, "discover_skill_inventory", lambda **_kw: ([], [], {}, {}, {0, 1}))
    cfg = {
        "enabled": True,
        "pre_hook_enabled": True,
        "memory_db_uri": "",
        "excluded_session_sources": ["kanban"],
        "sidecar_db_path": str(tmp_path / "s.db"),
        "receipts_path": str(tmp_path / "r.jsonl"),
        "software_inspection_enabled": False,
    }
    ctx = _make_ctx(cfg, llm_complete=lambda **_kw: json.dumps({"route": "DIRECT_REPLY", "reasoning": "ok"}))

    plugin_mod.on_pre_llm_call("hola", session_id="s1", turn_id="t1", _ctx=ctx)
    assert plugin_mod._coordinator.config.memory_db_uri == ""
    assert plugin_mod._coordinator.config.excluded_session_sources == ["kanban"]

    db_uri = f"sqlite:///{tmp_path / 'custom.db'}"
    cfg["memory_db_uri"] = db_uri
    cfg["excluded_session_sources"] = ["tool"]
    plugin_mod.on_pre_llm_call("hola 2", session_id="s1", turn_id="t2", _ctx=ctx)
    assert plugin_mod._coordinator.config.memory_db_uri == db_uri
    assert plugin_mod._coordinator.config.excluded_session_sources == ["tool"]

    # Clearing memory_db_uri="" updates coordinator and passes "" explicitly (not falling back via `or`)
    cfg["memory_db_uri"] = ""
    plugin_mod.on_pre_llm_call("hola 3", session_id="s1", turn_id="t3", _ctx=ctx)
    assert plugin_mod._coordinator.config.memory_db_uri == ""


def test_b3_focus_route_tool_respects_enabled_and_extracts_multimodal_query(tmp_path, monkeypatch):
    _reset_plugin_globals(monkeypatch)
    monkeypatch.setattr(plugin_mod, "discover_skill_inventory", lambda **_kw: ([], [], {}, {}, {0, 1}))
    cfg = {
        "enabled": False,
        "pre_hook_enabled": True,
        "sidecar_db_path": str(tmp_path / "s.db"),
        "receipts_path": str(tmp_path / "r.jsonl"),
        "software_inspection_enabled": False,
    }
    ctx = _make_ctx(cfg, llm_complete=lambda **_kw: json.dumps({"route": "SKILL_ONLY"}))
    registered_tools = {}
    ctx.register_tool = lambda **kwargs: registered_tools.__setitem__(kwargs["name"], kwargs["handler"])

    plugin_mod.register(ctx)
    focus_route = registered_tools["focus_route"]

    # Must honor enabled=False without invoking LLM or routing
    out_disabled = focus_route({"query": "deploy docker"})
    assert "Disabled" in out_disabled
    assert ctx.llm.complete.call_count == 0

    # Multimodal payload with enabled=True extracts text cleanly without image binary
    cfg["enabled"] = True
    out_enabled = focus_route({
        "query": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
            {"type": "input_text", "text": "check status"},
        ]
    })
    assert "Status:" in out_enabled
    sent = ctx.llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "check status" in sent and "SECRET" not in sent


def test_b5_host_llm_router_fallback_sets_last_backend_to_graph(tmp_path):
    from neural_context.router import HostLLMRouter

    ctx = _make_ctx({}, llm_complete=lambda **_kw: "invalid-nonexistent-skill-name")
    candidates = [
        (SkillCandidate(name="docker-ops", path=str(tmp_path / "a.md"), sha256="1", description="docker", lead_paragraph="docker"), 2.0),
        (SkillCandidate(name="pytest-qa", path=str(tmp_path / "b.md"), sha256="2", description="pytest", lead_paragraph="pytest"), 1.0),
    ]
    router = HostLLMRouter(ctx=ctx, fallback=GraphRouter())
    winners = router.route_all(TriageAnalysis(semantic_concepts=["docker"], required_capabilities=["docker"]), candidates)

    assert [w.name for w in winners] == ["docker-ops"]
    assert router.last_backend == "graph"
