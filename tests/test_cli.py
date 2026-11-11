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
