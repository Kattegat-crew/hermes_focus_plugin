"""Regresión: el aviso de skill faltante debe sobrevivir al recorte y hablar el idioma del usuario."""
import os

import pytest

from neural_context.config import PluginConfig
from neural_context.models import MemoryItem, TriageAnalysis, DomainClassification
from neural_context.router import ContextCoordinator


NOTICE_EN = "No procedural skill matched this task"
SUGGESTED = "quantum-plasma-simulation"


def _long_memories(n=60):
    return [MemoryItem(entity="proyecto", fact="dato historico relevante " * 20, token_estimate=80) for _ in range(n)]


# ---------------------------------------------------------------- recorte
def test_notice_survives_low_context_ceiling():
    """Con techo de contexto bajo el aviso NO puede desaparecer en silencio."""
    coordinator = ContextCoordinator(config=PluginConfig(max_context_tokens=100), sidecar=None)
    block = coordinator._assemble_context_block(
        skill=None,
        skill_content=None,
        memories=_long_memories(),
        suggested_skill=SUGGESTED,
    )
    assert block is not None
    assert SUGGESTED in block
    assert "Recommended skill capability to install" in block


def test_notice_survives_ceiling_with_skill_items_present():
    """Si hay items de skill la nota se omite (hay guía real); pero el bloque nunca excede el techo."""
    from neural_context.memory import estimate_tokens
    from neural_context.models import SkillCandidate

    coordinator = ContextCoordinator(config=PluginConfig(max_context_tokens=120), sidecar=None)
    cand = SkillCandidate(name="massive", path="/x")
    block = coordinator._assemble_context_block(
        skill=cand,
        skill_content="\n".join(f"Step {i}: do detailed work" for i in range(400)),
        memories=_long_memories(20),
        suggested_skill=SUGGESTED,
        skill_items=[(cand, "\n".join(f"Step {i}: do detailed work" for i in range(400)))],
    )
    assert block is None or estimate_tokens(block) <= 120


# ---------------------------------------------------------------- idioma
def test_notice_defaults_to_english(monkeypatch):
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    coordinator = ContextCoordinator(config=PluginConfig(), sidecar=None)
    block = coordinator._assemble_context_block(None, None, [], suggested_skill=SUGGESTED)
    assert NOTICE_EN in block


def test_notice_localized_from_explicit_language():
    coordinator = ContextCoordinator(config=PluginConfig(), sidecar=None)
    block = coordinator._assemble_context_block(None, None, [], suggested_skill=SUGGESTED, language="es-CO")
    assert "Ninguna skill procedimental" in block
    assert NOTICE_EN not in block


def test_notice_localized_from_hermes_env(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "pt_BR")
    coordinator = ContextCoordinator(config=PluginConfig(), sidecar=None)
    block = coordinator._assemble_context_block(None, None, [], suggested_skill=SUGGESTED)
    assert "Nenhuma skill procedural" in block


def test_notice_localized_from_os_locale(monkeypatch):
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.setenv("LANG", "fr_FR.UTF-8")
    coordinator = ContextCoordinator(config=PluginConfig(), sidecar=None)
    block = coordinator._assemble_context_block(None, None, [], suggested_skill=SUGGESTED)
    assert "Aucune compétence procédurale" in block


def test_unknown_language_falls_back_to_english():
    coordinator = ContextCoordinator(config=PluginConfig(), sidecar=None)
    block = coordinator._assemble_context_block(None, None, [], suggested_skill=SUGGESTED, language="tlh")
    assert NOTICE_EN in block


def test_mixed_block_header_and_instruction_localized():
    coordinator = ContextCoordinator(config=PluginConfig(), sidecar=None)
    block = coordinator._assemble_context_block(
        None, None,
        [MemoryItem(entity="Acme", fact="hecho historico", token_estimate=3)],
        suggested_skill=SUGGESTED,
        language="es",
    )
    assert "Memoria histórica" in block
    assert "material de referencia no confiable" in block
    assert NOTICE_EN not in block


def test_execute_propagates_analysis_language(tmp_path):
    """El idioma detectado por el triage manda sobre el resto de fuentes."""
    from neural_context.sidecar import SidecarIndex

    index = SidecarIndex(str(tmp_path / "side.db"))
    coordinator = ContextCoordinator(config=PluginConfig(router_backend="heuristic", selective_memory_enabled=False), sidecar=index)
    analysis = TriageAnalysis(
        is_actionable_task=True,
        needs_skills=True,
        needs_memory=False,
        detected_language="ja",
        intent_summary_en="Simulate quantum plasma",
        domain=DomainClassification(is_software_task=False),
        suggested_missing_skill=SUGGESTED,
    )
    result = coordinator.execute(analysis, user_message="プラズマをシミュレーションして")
    assert result.combined_context_block
    assert "一致する手順スキル" in result.combined_context_block
