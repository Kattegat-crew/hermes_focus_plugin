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


def test_test_isolation_clears_external_memory_overrides(monkeypatch):
    for name in ("HERMES_MEMORY_DB", "FOCUS_MEMORY_DB", "HERMES_MEMORY_DIR", "HERMES_MEMORY_PATH"):
        assert name not in os.environ


def test_large_sqlite_database_is_safe_to_query_but_large_markdown_is_not(tmp_path: Path, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    sqlite_path = hermes_home / "memories.db"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("CREATE TABLE memories (content TEXT, entity TEXT)")
        conn.execute("INSERT INTO memories VALUES ('The large database sentinel is retained for AcmeCorp.', 'AcmeCorp')")
        conn.execute("CREATE TABLE bulk_data (payload BLOB)")
        conn.execute("INSERT INTO bulk_data VALUES (zeroblob(2000001))")
        conn.commit()

    markdown_path = tmp_path / "large.md"
    markdown_path.write_text("x" * 2_000_001, encoding="utf-8")

    assert sqlite_path.stat().st_size > 2_000_000
    assert is_safe_memory_file(sqlite_path) is True
    assert is_safe_memory_file(markdown_path) is False

    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)
    items = SelectiveMemoryRetriever().retrieve(entities=["AcmeCorp"], cwd=str(tmp_path / "workspace"))
    assert any("large database sentinel" in item.fact for item in items)


def test_profile_history_does_not_fall_back_to_parent_default_database(tmp_path: Path, monkeypatch):
    profiles = tmp_path / "profiles"
    active_home = profiles / "work"
    active_home.mkdir(parents=True)
    root_home = tmp_path
    with sqlite3.connect(root_home / "state.db") as conn:
        conn.execute("CREATE TABLE memories (content TEXT)")
        conn.execute("INSERT INTO memories VALUES ('ProfileLeak account-xyz history')")
        conn.commit()

    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: active_home)
    result = SelectiveMemoryRetriever().retrieve(entities=["ProfileLeak"], cwd=str(tmp_path / "workspace"))

    assert result == []


def test_introspected_sqlite_excludes_rows_with_deleted_at_metadata(tmp_path: Path, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    db_path = hermes_home / "memories.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE records (content TEXT, deleted_at TEXT)")
        conn.execute("INSERT INTO records VALUES (?, NULL)", ("AcmeCorp active invoice backend",))
        conn.execute("INSERT INTO records VALUES (?, '2026-01-01')", ("AcmeCorp deleted invoice backend",))
        conn.commit()

    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)
    result = SelectiveMemoryRetriever().retrieve(entities=["AcmeCorp"], cwd=str(tmp_path / "workspace"))

    assert any("active invoice" in item.fact for item in result)
    assert not any("deleted invoice" in item.fact for item in result)


def test_native_messages_search_excludes_deleted_at_rows(tmp_path: Path):
    db_path = tmp_path / "native-state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, role TEXT, content TEXT, session_id TEXT, timestamp TEXT, deleted_at TEXT)")
        conn.execute("INSERT INTO messages VALUES (1, 'user', 'AcmeCorp active invoice details', 's1', '2026-01-01', NULL)")
        conn.execute("INSERT INTO messages VALUES (2, 'user', 'AcmeCorp deleted invoice details', 's2', '2026-01-02', '2026-01-03')")
        cursor = conn.cursor()
        nodes = SelectiveMemoryRetriever()._query_hermes_state_db(
            cursor, "state.db", "messages", {"messages": "messages"}, ["AcmeCorp"]
        )
    facts = [node.fact for node in nodes]
    assert any("active invoice" in fact for fact in facts)
    assert not any("deleted invoice" in fact for fact in facts)


def test_sqlalchemy_sample_and_search_filter_deleted_rows_when_column_exists(monkeypatch):
    import sys
    import types
    from contextlib import contextmanager

    seen_sql = []

    class Result:
        def __init__(self, rows): self.rows = rows
        def fetchall(self): return self.rows

    class Connection:
        def execute(self, statement, params=None):
            sql = str(statement)
            seen_sql.append(sql)
            if "SELECT *" in sql:
                assert '"deleted_at" IS NULL' in sql
                return Result([("AcmeCorp active account note", "AcmeCorp", None)])
            assert '"deleted_at" IS NULL' in sql
            return Result([("AcmeCorp active account note", "AcmeCorp")])

    class Engine:
        dialect = types.SimpleNamespace(identifier_preparer=types.SimpleNamespace(quote=lambda name, force=True: f'"{name}"'))
        @contextmanager
        def connect(self): yield Connection()
        def dispose(self): pass

    fake = types.ModuleType("sqlalchemy")
    fake.create_engine = lambda *_args, **_kwargs: Engine()
    fake.inspect = lambda _engine: types.SimpleNamespace(
        get_table_names=lambda: ["notes"],
        get_columns=lambda _tbl: [
            {"name": "content", "type": "TEXT"},
            {"name": "entity", "type": "TEXT"},
            {"name": "deleted_at", "type": "TEXT"},
        ],
    )
    fake.text = lambda value: value
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake)

    retriever = SelectiveMemoryRetriever()
    nodes = retriever._extract_external_uri_fact_nodes(
        "postgresql://example", clean_entities=["AcmeCorp"], topic_terms=[]
    )
    assert nodes
    assert len(seen_sql) == 2


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
