from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neural_context.cli import _coerce_and_validate, build_parser, handle_cli
from neural_context.config import PluginConfig
from neural_context.memory import SelectiveMemoryRetriever, is_safe_memory_file
from neural_context.models import CognitiveRoute, DomainClassification, MemoryProbe
from neural_context.router import ContextCoordinator, GraphRouter
from neural_context.sidecar import SidecarIndex
from neural_context.triage import CognitiveTriage
from __init__ import discover_all_skill_paths


def test_triage_empty_model_response_fails_closed_to_direct_reply():
    """Ensure an empty JSON response '{}' or missing route defaults safely to DIRECT_REPLY."""
    triage = CognitiveTriage()
    domain = DomainClassification(is_software_task=False)

    # Empty JSON object from model
    result_empty = triage._parse_structured_output("{}", "Hola, ¿cómo estás?", domain)
    assert result_empty.route == CognitiveRoute.DIRECT_REPLY
    assert result_empty.needs_skills is False
    assert result_empty.needs_memory is False

    # Malformed text
    result_malformed = triage._parse_structured_output("not json", "Hola amigo", domain)
    assert result_malformed.route == CognitiveRoute.DIRECT_REPLY
    assert result_malformed.needs_skills is False


def test_conversational_text_structural_fallback_defaults_to_direct_reply():
    """Pure conversational text without syntax or explicit entities defaults to DIRECT_REPLY."""
    triage = CognitiveTriage()
    domain = DomainClassification(is_software_task=False)

    res = triage._grammatical_intent_induction("buenas tardes que tal todo por alla", domain)
    assert res.route == CognitiveRoute.DIRECT_REPLY
    assert res.is_actionable_task is False
    assert res.needs_skills is False


def test_memory_stopwords_filter_prevents_preposition_matches(tmp_path: Path, monkeypatch):
    """Ensure common grammatical function words ('del', 'de', 'con') do not cause false positive memory matches."""
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: empty_home)

    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text(
        "- Decidimos usar arquitectura hexagonal en el core del sistema\n"
        "- Las credenciales de produccion fueron rotadas ayer\n",
        encoding="utf-8"
    )

    retriever = SelectiveMemoryRetriever(token_budget=0)
    # Query with preposition "del" and "sobre" - should not match unless discriminative terms match
    probe = MemoryProbe(
        query_text="cual es el estado del servidor",
        topic="del"
    )

    results = retriever.retrieve(probe=probe, cwd=str(tmp_path))
    # "del" is filtered out from topic_terms; "servidor" does not appear in the memory lines
    assert len(results) == 0


def test_sqlite_memory_escapes_like_wildcards(tmp_path: Path, monkeypatch):
    """Ensure SQLite memory inspection escapes % and _ wildcards."""
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: empty_home)

    db_path = tmp_path / "memories.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE memories (content TEXT)")
        conn.execute("INSERT INTO memories VALUES ('100% discount promotion campaign')")
        conn.execute("INSERT INTO memories VALUES ('Standard database configuration')")
        conn.commit()

    retriever = SelectiveMemoryRetriever(token_budget=0)
    # Search for entity with wildcard '%'; should match literal '100%' without broad match
    items = retriever.retrieve(entities=["100%"], cwd=str(tmp_path))
    assert len(items) == 1
    assert "100% discount" in items[0].fact


def test_cli_config_type_validation_rejects_invalid_values(tmp_path: Path, capsys):
    """Ensure hermes focus config --set rejects unknown keys or invalid types."""
    parser = build_parser()
    config = PluginConfig()

    # Test unknown key
    args_bad_key = parser.parse_args(["config", "--set", "unknown_key=123"])
    ret1 = handle_cli(args_bad_key, config, None, None)
    assert ret1 == 1
    out1 = capsys.readouterr().out
    assert "Unknown configuration setting" in out1

    # Test invalid int (bananas)
    args_bad_int = parser.parse_args(["config", "--set", "timeout_ms=bananas"])
    ret2 = handle_cli(args_bad_int, config, None, None)
    assert ret2 == 1
    out2 = capsys.readouterr().out
    assert "expected integer, got 'bananas'" in out2

    # Test invalid bool
    args_bad_bool = parser.parse_args(["config", "--set", "enabled=maybe"])
    ret3 = handle_cli(args_bad_bool, config, None, None)
    assert ret3 == 1
    out3 = capsys.readouterr().out
    assert "expected boolean" in out3


def test_multi_tier_skill_discovery(tmp_path: Path):
    """Ensure discover_all_skill_paths finds profile skills, shared skills, and local workspace skills."""
    hermes_home = tmp_path / "hermes_root" / "profiles" / "agent_1"
    hermes_home.mkdir(parents=True)
    profile_skills = hermes_home / "skills" / "profile_tool"
    profile_skills.mkdir(parents=True)
    (profile_skills / "SKILL.md").write_text("# Profile Tool", encoding="utf-8")

    shared_skills = tmp_path / "hermes_root" / "skills" / "shared_tool"
    shared_skills.mkdir(parents=True)
    (shared_skills / "SKILL.md").write_text("# Shared Tool", encoding="utf-8")

    workspace = tmp_path / "workspace"
    local_skills = workspace / ".skills" / "local_tool"
    local_skills.mkdir(parents=True)
    (local_skills / "SKILL.md").write_text("# Local Tool", encoding="utf-8")

    globals_found, locals_found = discover_all_skill_paths(hermes_home, cwd=workspace)

    global_names = [p.parent.name for p in globals_found]
    local_names = [p.parent.name for p in locals_found]

    assert "profile_tool" in global_names
    assert "shared_tool" in global_names
    assert "local_tool" in local_names
