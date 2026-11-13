"""Regresión: la contabilidad del recibo no depende del idioma del bloque inyectado."""
import json

from neural_context.models import MemoryItem, RouteResult, SkillCandidate, TriageAnalysis
from neural_context.receipt import ReceiptEmitter
from neural_context.router import ContextCoordinator
from neural_context.config import PluginConfig


def _turn(language_block):
    return TriageAnalysis(
        is_actionable_task=True,
        needs_skills=True,
        needs_memory=True,
        detected_language=language_block,
        intent_summary_en="demo",
    )


def test_receipt_parses_localized_block(tmp_path):
    coordinator = ContextCoordinator(config=PluginConfig(), sidecar=None)
    contact = SkillCandidate(name="campaign-generator", path="/skills/campaign-generator")
    block = coordinator._assemble_context_block(
        skill=contact,
        skill_content="# Campaign\nPaso uno: crear el guion.\nPaso dos: renderizar.",
        memories=[MemoryItem(entity="Acme", fact="el cliente prefiere video vertical", token_estimate=8)],
        language="es",
    )
    assert block and "Memoria histórica" in block

    result = RouteResult(
        triage_analysis=_turn("es"),
        combined_context_block=block,
        selected_skill=contact,
        status="success",
    )
    emitter = ReceiptEmitter(str(tmp_path / "receipts.jsonl"))
    emitter.record_turn(turn_id="t1", session_id="s1", analysis=_turn("es"), route_result=result)

    record = json.loads((tmp_path / "receipts.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["memory_injected_tokens"] > 0, record
    assert record["skill_injected_tokens"] > 0, record
    assert record["context_overhead_tokens"] < record["rendered_context_tokens"], record


def test_receipt_parses_english_block_unchanged(tmp_path):
    coordinator = ContextCoordinator(config=PluginConfig(), sidecar=None)
    contact = SkillCandidate(name="campaign-generator", path="/skills/campaign-generator")
    block = coordinator._assemble_context_block(
        skill=contact,
        skill_content="# Campaign\nStep one: write the script.",
        memories=[MemoryItem(entity="Acme", fact="prefers vertical video", token_estimate=6)],
        language="en",
    )
    result = RouteResult(triage_analysis=_turn("en"), combined_context_block=block,
                         selected_skill=contact, status="success")
    emitter = ReceiptEmitter(str(tmp_path / "receipts.jsonl"))
    emitter.record_turn(turn_id="t2", session_id="s2", analysis=_turn("en"), route_result=result)
    record = json.loads((tmp_path / "receipts.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["memory_injected_tokens"] > 0
    assert record["skill_injected_tokens"] > 0
