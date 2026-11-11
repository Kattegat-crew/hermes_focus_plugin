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


def test_cli_config_view_and_set(tmp_path, capsys, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setattr("neural_context.cli.get_hermes_home", lambda: hermes_home)

    config = PluginConfig(max_context_tokens=0, memory_token_budget=0)
    args = argparse.Namespace(subcommand="config", set=None)
    ret = handle_cli(args, config, None, None)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Focus Configuration Settings" in captured.out
    assert "max_context_tokens" in captured.out

    # Test setting a parameter via CLI
    args_set = argparse.Namespace(subcommand="config", set="max_context_tokens=1500")
    ret_set = handle_cli(args_set, config, None, None)
    assert ret_set == 0
    captured_set = capsys.readouterr()
    assert "Successfully updated 'max_context_tokens' to 1500" in captured_set.out
    cfg_file = hermes_home / "config.yaml"
    assert cfg_file.is_file()
    assert "max_context_tokens: 1500" in cfg_file.read_text(encoding="utf-8")
