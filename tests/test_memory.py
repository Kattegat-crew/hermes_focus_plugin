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
    # Bare env is ignored per security policy: Redis is NOT found without explicit extra_memory_dirs
    monkeypatch.setenv("HERMES_MEMORY_DIR", str(custom_dir))

    retriever = SelectiveMemoryRetriever(token_budget=300)

    # Search for Stripe (from hermes_home/memories/archive/2026)
    items_stripe = retriever.retrieve(entities=["Stripe"], cwd=str(cwd))
    assert any("Stripe integration uses Webhook" in it.fact for it in items_stripe)

    # Bare env is ignored:
    items_redis_ignored = retriever.retrieve(entities=["Redis"], cwd=str(cwd))
    assert not any("Redis cluster is sharded" in it.fact for it in items_redis_ignored)

    # Search for Redis via configured extra_memory_dirs (plugin setting)
    retriever_configured = SelectiveMemoryRetriever(token_budget=300, extra_memory_dirs=[custom_dir])
    items_redis = retriever_configured.retrieve(entities=["Redis"], cwd=str(cwd))
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


def test_hermes_state_db_source_filter_configured_and_dynamic(tmp_path: Path, monkeypatch):
    import sys
    import types

    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, source TEXT)")
        conn.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY, role TEXT, content TEXT, session_id TEXT, "
            "timestamp TEXT, deleted_at TEXT, active INTEGER DEFAULT 1)"
        )
        # Sessions
        conn.executemany(
            "INSERT INTO sessions (id, title, source) VALUES (?, ?, ?)",
            [
                ("s-chat", "Chat Session", "chat"),
                ("s-auto1", "Automation 1", "automation_alpha"),
                ("s-auto2", "Automation 2", "automation_beta"),
                ("s-kanban", "Kanban 1", "kanban"),
                ("s-sub", "Subagent 1", "subagent"),
                ("s-tool", "Tool 1", "tool"),
            ],
        )
        # Messages
        conn.executemany(
            "INSERT INTO messages (id, role, content, session_id, timestamp, deleted_at, active) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "user", "AcmeCorp regular interactive chat discussion", "s-chat", "2026-01-01", None, 1),
                (2, "user", "AcmeCorp automated task payload from alpha", "s-auto1", "2026-01-02", None, 1),
                (3, "user", "AcmeCorp automated worker log from beta", "s-auto2", "2026-01-03", None, 1),
                (4, "user", "AcmeCorp kanban card automation run", "s-kanban", "2026-01-04", None, 1),
                (5, "user", "AcmeCorp subagent delegated background output", "s-sub", "2026-01-05", None, 1),
                (6, "user", "AcmeCorp tool session telemetry event", "s-tool", "2026-01-06", None, 1),
                (7, "assistant", "[CONTEXT COMPACTION] AcmeCorp compacted summary", "s-chat", "2026-01-07", None, 1),
                (8, "assistant", "[CONTEXT SUMMARY]: AcmeCorp older session summary", "s-chat", "2026-01-08", None, 1),
            ],
        )
        cursor = conn.cursor()

        # 1. Configured exclusion filter
        retriever_configured = SelectiveMemoryRetriever(
            excluded_session_sources=["automation_alpha", "automation_beta"]
        )
        nodes = retriever_configured._query_hermes_state_db(
            cursor,
            "state.db",
            "messages",
            {"messages": "messages", "sessions": "sessions"},
            ["AcmeCorp"],
        )
        facts = [node.fact for node in nodes]
        assert any("interactive chat discussion" in fact for fact in facts)
        assert not any("automated task payload" in fact for fact in facts)
        assert not any("automated worker log" in fact for fact in facts)
        assert not any("compacted summary" in fact for fact in facts)
        assert not any("older session summary" in fact for fact in facts)

        # 2. Default config_schema-aligned exclusions apply without private core imports
        assert "tools.session_search_tool" not in sys.modules
        retriever_default = SelectiveMemoryRetriever()
        nodes_default = retriever_default._query_hermes_state_db(
            cursor,
            "state.db",
            "messages",
            {"messages": "messages", "sessions": "sessions"},
            ["AcmeCorp"],
        )
        facts_default = [node.fact for node in nodes_default]
        assert any("interactive chat discussion" in fact for fact in facts_default)
        assert not any("kanban card automation" in fact for fact in facts_default)
        assert not any("subagent delegated" in fact for fact in facts_default)
        assert not any("tool session telemetry" in fact for fact in facts_default)
        assert not any("compacted summary" in fact for fact in facts_default)

        # 3. Explicit empty lists disable filtering (user override is authoritative)
        retriever_unfiltered = SelectiveMemoryRetriever(
            excluded_session_sources=[],
            excluded_message_prefixes=[],
        )
        nodes_unfiltered = retriever_unfiltered._query_hermes_state_db(
            cursor,
            "state.db",
            "messages",
            {"messages": "messages", "sessions": "sessions"},
            ["AcmeCorp"],
        )
        facts_unfiltered = [node.fact for node in nodes_unfiltered]
        assert any("kanban card automation" in fact for fact in facts_unfiltered)
        assert any("compacted summary" in fact for fact in facts_unfiltered)


def test_multilingual_facts_retrieval_without_static_word_blacklists(tmp_path: Path):
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, source TEXT)")
        conn.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY, role TEXT, content TEXT, session_id TEXT, "
            "timestamp TEXT, deleted_at TEXT, active INTEGER DEFAULT 1)"
        )
        conn.execute("INSERT INTO sessions VALUES ('s-multi', 'Multilingual Chat', 'chat')")
        conn.executemany(
            "INSERT INTO messages (id, role, content, session_id, timestamp, deleted_at, active) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "user", "AcmeCorp facturación electrónica en Bogotá con impuestos DIAN validados", "s-multi", "2026-01-01", None, 1),
                (2, "user", "AcmeCorp facture électronique à Paris avec TVA validée", "s-multi", "2026-01-02", None, 1),
                (3, "user", "AcmeCorp 東京オフィスの請求システムと消費税設定が完了しました", "s-multi", "2026-01-03", None, 1),
            ],
        )
        cursor = conn.cursor()
        nodes = SelectiveMemoryRetriever()._query_hermes_state_db(
            cursor,
            "state.db",
            "messages",
            {"messages": "messages", "sessions": "sessions"},
            ["AcmeCorp"],
        )

    facts = [node.fact for node in nodes]
    assert any("facturación electrónica en Bogotá" in fact for fact in facts)
    assert any("facture électronique à Paris" in fact for fact in facts)
    assert any("東京オフィスの請求システム" in fact for fact in facts)


def test_resolve_memory_db_password_honors_secret_scope(monkeypatch):
    import sys
    import types
    import pytest
    from neural_context.memory import _resolve_memory_db_password

    # 1. Active profile secret scope returns scoped secret
    fake_secret_scope = types.ModuleType("agent.secret_scope")
    fake_secret_scope.get_secret = lambda name: "scoped_secret_pwd_42" if name == "FOCUS_MEMORY_DB_PASSWORD" else None
    agent = types.ModuleType("agent")
    agent.secret_scope = fake_secret_scope
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", fake_secret_scope)

    assert _resolve_memory_db_password() == "scoped_secret_pwd_42"

    # 2. When get_secret returns None (e.g. another profile in multiplex gateway), NEVER fall back to os.environ
    monkeypatch.setenv("FOCUS_MEMORY_DB_PASSWORD", "env_secret_pwd_99")
    fake_secret_scope.get_secret = lambda name: None
    assert _resolve_memory_db_password() is None

    # 3. Exceptions from get_secret (e.g. UnscopedSecretError) must propagate, never fall back to os.environ
    def _boom(_name: str):
        raise RuntimeError("UnscopedSecretError")

    fake_secret_scope.get_secret = _boom
    with pytest.raises(RuntimeError, match="UnscopedSecretError"):
        _resolve_memory_db_password()

    # 4. Standalone environment without agent.secret_scope returns None, never reading os.environ
    monkeypatch.delitem(sys.modules, "agent.secret_scope", raising=False)
    monkeypatch.delitem(sys.modules, "agent", raising=False)
    assert _resolve_memory_db_password() is None


def test_plugin_config_loads_memory_settings_from_ctx():
    from neural_context.config import PluginConfig

    # Defaults align with plugin.yaml config_schema
    default_cfg = PluginConfig.load_from_ctx(None)
    assert default_cfg.excluded_session_sources == ["kanban", "subagent", "tool"]
    assert default_cfg.excluded_message_prefixes == ["[CONTEXT COMPACTION", "[CONTEXT SUMMARY]:"]

    class MockCtx:
        def get_config(self, key, default=None):
            if key == "memory_db_uri":
                return "postgresql://user@remote-host:5432/memorydb"
            if key == "extra_memory_dirs":
                return ["/data/extra1", "/data/extra2"]
            if key == "excluded_session_sources":
                return ["kanban", "subagent", "tool"]
            if key == "excluded_message_prefixes":
                return ["[CUSTOM COMPACTION]"]
            return default

    cfg = PluginConfig.load_from_ctx(MockCtx())
    assert cfg.memory_db_uri == "postgresql://user@remote-host:5432/memorydb"
    assert cfg.extra_memory_dirs == ["/data/extra1", "/data/extra2"]
    assert cfg.excluded_session_sources == ["kanban", "subagent", "tool"]
    assert cfg.excluded_message_prefixes == ["[CUSTOM COMPACTION]"]
