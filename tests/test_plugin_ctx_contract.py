"""Contrato real de ``PluginContext``: varios miembros son métodos, no atributos.

En Hermes, ``PluginContext.session_cwd`` es un método (``ctx.session_cwd()``). Leerlo con
``getattr`` devolvía un método ligado y ``Path()`` lanzaba ``TypeError``, así que el hook
``pre_llm_call`` fallaba cerrado y el plugin no inyectaba nada en un runtime real.
"""
import pytest

import __init__ as plugin_entry
from neural_context.models import CognitiveRoute, TriageAnalysis


class MethodStyleCtx:
    """Imita ``hermes_cli.plugins_context.PluginContext``."""

    def __init__(self, cwd):
        self._cwd = str(cwd)

    def session_cwd(self):
        return self._cwd

    def get_config(self, key, default=None):
        return default


def test_ctx_cwd_reads_method_style_member(tmp_path):
    assert plugin_entry._ctx_cwd(MethodStyleCtx(tmp_path)) == str(tmp_path)


def test_ctx_cwd_reads_plain_attribute(tmp_path):
    class AttrCtx:
        cwd = str(tmp_path)

    assert plugin_entry._ctx_cwd(AttrCtx()) == str(tmp_path)


def test_ctx_cwd_prefers_session_cwd_over_stale_attribute(tmp_path):
    class BothCtx:
        cwd = "/stale/elsewhere"

        def session_cwd(self):
            return str(tmp_path)

    assert plugin_entry._ctx_cwd(BothCtx()) == str(tmp_path)


def test_pre_hook_injects_with_method_style_ctx(tmp_path, monkeypatch):
    """Regresión: el hook debe inyectar aunque ``session_cwd`` sea método."""
    monkeypatch.setattr(plugin_entry, "_config", None)
    monkeypatch.setattr(plugin_entry, "_sidecar", None)
    monkeypatch.setattr(plugin_entry, "_coordinator", None)
    monkeypatch.setattr(plugin_entry, "_triage", None)
    monkeypatch.setattr(plugin_entry, "_session_turns", {})

    class StubTriage:
        def analyze(self, text, cwd=None, software_inspection_enabled=False):
            assert cwd is None or isinstance(cwd, (str, bytes)), f"cwd llegó como {type(cwd)}"
            return TriageAnalysis(
                route=CognitiveRoute.SKILL_ONLY,
                detected_language="es",
                suggested_missing_skill="quantum-plasma-simulation",
                is_actionable_task=True,
                needs_skills=True,
                needs_memory=False,
            )

    monkeypatch.setattr(plugin_entry, "CognitiveTriage", lambda *a, **k: StubTriage())

    result = plugin_entry.on_pre_llm_call(
        user_message="Implementa un pipeline de simulación de plasma cuántico",
        _ctx=MethodStyleCtx(tmp_path),
        session_id="s-method",
        turn_id="t-method",
    )

    assert result is not None, "el hook falló cerrado con un PluginContext real"
    block = result.get("context") or ""
    assert "neural_context" in block
    assert "quantum-plasma-simulation" in block
