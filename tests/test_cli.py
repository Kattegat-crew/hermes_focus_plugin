from pathlib import Path
import argparse
import pytest

from neural_context.cli import build_parser, handle_cli, register_cli
from neural_context.config import PluginConfig
from neural_context.receipt import ReceiptEmitter
from neural_context.sidecar import SidecarIndex


def test_cli_parser_build():
    parser = build_parser()
    args = parser.parse_args(["status"])
    assert args.subcommand == "status"

    args = parser.parse_args(["verify", "--limit", "5"])
    assert args.subcommand == "verify"
    assert args.limit == 5


def test_cli_doctor_healthy(tmp_path, capsys):
    db_file = tmp_path / "sidecar.db"
    index = SidecarIndex(str(db_file))

    # Add a skill so index is not empty
    skill_dir = tmp_path / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: demo-skill\ndescription: demo\n---\n# Demo\n")
    index.sync_skills([skill_file])

    receipts_file = tmp_path / "receipts.jsonl"
    emitter = ReceiptEmitter(receipts_path=str(receipts_file))

    config = PluginConfig(
        sidecar_db_path=str(db_file),
        receipts_path=str(receipts_file),
        router_backend="host_llm"
    )

    args = argparse.Namespace(subcommand="doctor")
    ret = handle_cli(args, config, index, emitter)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Doctor result: HEALTHY" in captured.out


def test_cli_status(tmp_path, capsys):
    db_file = tmp_path / "sidecar.db"
    index = SidecarIndex(str(db_file))
    receipts_file = tmp_path / "receipts.jsonl"
    emitter = ReceiptEmitter(receipts_path=str(receipts_file))
    config = PluginConfig(sidecar_db_path=str(db_file), receipts_path=str(receipts_file))

    args = argparse.Namespace(subcommand="status")
    ret = handle_cli(args, config, index, emitter)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Plugin Enabled" in captured.out
    assert "Skills Indexed" in captured.out


def test_cli_index_uses_shared_native_inventory_and_inspection_setting(tmp_path, monkeypatch, capsys):
    from unittest.mock import Mock
    import neural_context.cli as cli

    skill = tmp_path / "trusted" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text("---\nname: trusted\n---\n")
    inventory = ([skill], [], {skill: (1, tmp_path)}, {skill: "trusted"}, {0, 1, 2, 3})
    discover = Mock(return_value=inventory)
    monkeypatch.setattr(cli, "discover_skill_inventory", discover)
    sidecar = Mock()
    config = PluginConfig(software_inspection_enabled=False)

    result = handle_cli(["index"], config, sidecar, None)

    assert result == 0
    discover.assert_called_once_with(include_project=False)
    sidecar.sync_skills.assert_called_once_with(
        [skill], project_local_paths=[], source_info={skill: (1, tmp_path)},
        skill_names={skill: "trusted"}, replace_tiers={0, 1, 2, 3}, replace_subtrees=None,
    )


def test_cli_empty_trusted_subtree_still_prunes_removed_skill(tmp_path, monkeypatch):
    import sys
    import types
    from unittest.mock import Mock
    import neural_context.cli as cli

    trusted_root = tmp_path / "profile" / "skills"
    removed_source = trusted_root / "gone"
    removed_source.mkdir(parents=True)
    fake = types.ModuleType("agent.skill_utils")
    fake.get_skill_search_roots = lambda include_project=True: [(1, trusted_root)]
    agent = types.ModuleType("agent")
    agent.skill_utils = fake
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.skill_utils", fake)
    monkeypatch.setattr(cli, "discover_skill_inventory", lambda **_: ([], [], {}, {}, {0, 1, 2, 3}))
    index = Mock()

    result = cli.handle_cli(["index", "--path", str(removed_source)], PluginConfig(), index, None)

    assert result == 0
    assert index.sync_skills.call_args.kwargs["replace_subtrees"] == [removed_source.resolve()]
    assert index.sync_skills.call_args.args == ([],)


def test_cli_config_view_and_set(tmp_path, capsys):
    from unittest.mock import MagicMock

    config = PluginConfig(max_context_tokens=0, memory_token_budget=0)
    args = argparse.Namespace(subcommand="config", set=None)
    ret = handle_cli(args, config, None, None)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Focus Configuration Settings" in captured.out
    assert "max_context_tokens" in captured.out

    # Test setting a parameter via CLI using ctx.set_config
    mock_ctx = MagicMock()
    args_set = argparse.Namespace(subcommand="config", set="max_context_tokens=1500")
    ret_set = handle_cli(args_set, config, None, None, ctx=mock_ctx)
    assert ret_set == 0
    captured_set = capsys.readouterr()
    assert "Reload the plugin to refresh cached services" in captured_set.out
    mock_ctx.set_config.assert_called_once_with("max_context_tokens", 1500)
    assert config.max_context_tokens == 1500

    # Without ctx.set_config, it fails closed without touching disk
    ret_no_ctx = handle_cli(args_set, config, None, None, ctx=None)
    assert ret_no_ctx == 1
    assert "ctx.set_config" in capsys.readouterr().out
