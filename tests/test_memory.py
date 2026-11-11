from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from neural_context.memory import SelectiveMemoryRetriever, is_safe_memory_file
from neural_context.models import MemoryProbe


def test_safe_memory_file_validation(tmp_path: Path):
    safe_file = tmp_path / "valid.md"
    safe_file.write_text("Valid memory line content here.", encoding="utf-8")
    assert is_safe_memory_file(safe_file) is True

    # System paths must be rejected
    assert is_safe_memory_file(Path("/etc/passwd")) is False
    assert is_safe_memory_file(Path("/etc/shadow")) is False
    assert is_safe_memory_file(Path("/proc/cpuinfo")) is False


def test_vanilla_hermes_markdown_retrieval(tmp_path: Path, monkeypatch):
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)

    # 1. Project Memory
    proj_mem = cwd / "PROJECT_MEMORY.md"
    proj_mem.write_text(
        "# Project Memory\nLucky is the codename for our Meta CAPI ad campaign microservice.\nOther unrelated notes.",
        encoding="utf-8"
    )

    # 2. Global USER.md
    user_md = hermes_home / "USER.md"
    user_md.write_text(
        "# User Facts\nUser preferred deployment is Docker Swarm on Hetzner VPS.\nUser name is Jemadiar.",
        encoding="utf-8"
    )

    retriever = SelectiveMemoryRetriever(token_budget=300)

    # Probe entity "Lucky"
    probe = MemoryProbe(entity="Lucky", topic="campaign")
    items = retriever.retrieve(entities=["Lucky"], probe=probe, cwd=str(cwd))

    assert len(items) >= 1
    assert any("Lucky is the codename" in it.fact for it in items)
    assert any(it.source_file == "PROJECT_MEMORY.md" for it in items)


def test_nested_subdirectories_and_custom_env_memory(tmp_path: Path, monkeypatch):
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    # Nested directory: memories/archive/2026/
    nested_dir = hermes_home / "memories" / "archive" / "2026"
    nested_dir.mkdir(parents=True)
    archive_note = nested_dir / "payment_gateway.md"
    archive_note.write_text(
        "Stripe integration uses Webhook signing secret whsec_live_99482 for invoice payments.",
        encoding="utf-8"
    )

    # Custom memory directory via env var
    custom_dir = tmp_path / "custom_memory_store"
    custom_dir.mkdir()
    custom_note = custom_dir / "infra.md"
    custom_note.write_text(
        "Redis cluster is sharded across nodes redis-01 and redis-02 with TLS enabled.",
        encoding="utf-8"
    )

    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)
    monkeypatch.setenv("HERMES_MEMORY_DIR", str(custom_dir))

    retriever = SelectiveMemoryRetriever(token_budget=300)

    # Search for Stripe
    items_stripe = retriever.retrieve(entities=["Stripe"], cwd=str(cwd))
    assert any("Stripe integration uses Webhook" in it.fact for it in items_stripe)

    # Search for Redis via custom dir
    items_redis = retriever.retrieve(entities=["Redis"], cwd=str(cwd))
    assert any("Redis cluster is sharded" in it.fact for it in items_redis)


def test_sqlite_memory_store_retrieval(tmp_path: Path, monkeypatch):
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)

    # Create a custom pluggable SQLite database: memories.db
    db_path = hermes_home / "memories.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, fact TEXT, entity TEXT)")
        conn.execute(
            "INSERT INTO memories (fact, entity) VALUES (?, ?)",
            ("Kafka brokers run on port 9092 with SASL_SSL authentication.", "Kafka")
        )
        conn.commit()

    retriever = SelectiveMemoryRetriever(token_budget=300)
    items = retriever.retrieve(entities=["Kafka"], cwd=str(cwd))

    assert len(items) >= 1
    assert any("Kafka brokers run on port 9092" in it.fact for it in items)
    assert any("memories.db" in it.source_file for it in items)


def test_memory_fail_closed_on_corrupt_files(tmp_path: Path, monkeypatch):
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)

    # Create corrupted binary file named MEMORY.md
    bad_mem = hermes_home / "MEMORY.md"
    bad_mem.write_bytes(b"\x00\xff\xfe\xfa\x80\x90")

    # Create empty / locked sqlite db
    bad_db = hermes_home / "memories.db"
    bad_db.write_text("not a real sqlite database", encoding="utf-8")

    retriever = SelectiveMemoryRetriever(token_budget=300)
    # Must fail closed gracefully and return empty list without crashing
    items = retriever.retrieve(entities=["Whatever"], cwd=str(cwd))
    assert items == []
