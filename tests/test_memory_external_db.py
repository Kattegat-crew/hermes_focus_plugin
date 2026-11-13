"""Ruta real de SQLAlchemy contra una base viva (no mocks).

Cubre el adaptador de bases externas configuradas por URI: si el entorno tiene
SQLAlchemy instalado, el plugin debe crear un engine real, inspeccionar el esquema,
citar identificadores y excluir filas con borrado lógico (``deleted_at``).
"""
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from neural_context.memory import SelectiveMemoryRetriever  # noqa: E402


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        # Nombre con espacio a propósito: obliga al dialecto a citar identificadores.
        conn.execute(
            'CREATE TABLE "Client Records" ('
            '"id" INTEGER PRIMARY KEY, "name" TEXT, "notes" TEXT, "deleted_at" TEXT)'
        )
        conn.executemany(
            'INSERT INTO "Client Records" ("name", "notes", "deleted_at") VALUES (?, ?, ?)',
            [
                ("Acme", "Acme prefiere video vertical para sus contratos de campaña", None),
                ("Acme", "Acme contrato cancelado por impago en 2025", "2025-04-01T10:00:00"),
                ("Globex", "Globex quiere audio en español neutro", None),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_sqlalchemy_uri_adapter_reads_live_db_and_skips_deleted_rows(tmp_path):
    db_path = tmp_path / "crm.db"
    _build_db(db_path)
    uri = f"sqlite+pysqlite:///{db_path}"  # dialecto explícito -> rama SQLAlchemy

    retriever = SelectiveMemoryRetriever()
    nodes = retriever._extract_external_uri_fact_nodes(uri, ["Acme"], ["contrato"])

    facts = [n.fact for n in nodes]
    assert facts, "el adaptador SQLAlchemy no devolvió ninguna fila"
    assert all("Acme" in f for f in facts), facts
    assert not any("cancelado" in f.lower() for f in facts), f"fila con deleted_at se coló: {facts}"
    assert {n.source_file for n in nodes} == {"external_db:Client Records"}


def test_sqlalchemy_uri_adapter_uses_a_real_engine(tmp_path, monkeypatch):
    db_path = tmp_path / "crm2.db"
    _build_db(db_path)
    import sqlalchemy

    calls = []
    real_create_engine = sqlalchemy.create_engine

    def _spy(*args, **kwargs):
        calls.append(args[:1])
        return real_create_engine(*args, **kwargs)

    monkeypatch.setattr(sqlalchemy, "create_engine", _spy)
    nodes = SelectiveMemoryRetriever()._extract_external_uri_fact_nodes(
        f"sqlite+pysqlite:///{db_path}", ["Globex"], ["audio"]
    )
    assert calls, "no se creó engine de SQLAlchemy"
    assert any("Globex" in n.fact for n in nodes), [n.fact for n in nodes]


def test_sqlalchemy_uri_adapter_survives_unreachable_database(tmp_path):
    """Un URI roto no debe propagar excepciones: el plugin degrada a cero hechos."""
    retriever = SelectiveMemoryRetriever()
    nodes = retriever._extract_external_uri_fact_nodes(
        "sqlite+pysqlite:////nonexistent/nope/missing.db", ["Acme"], ["contrato"]
    )
    assert nodes == []
