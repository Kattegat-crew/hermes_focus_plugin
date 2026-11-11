from pathlib import Path
import pytest

from neural_context.models import DomainClassification, RouteResult, SkillCandidate, TriageAnalysis
from neural_context.receipt import ReceiptEmitter


def test_receipt_emitter(tmp_path):
    receipts_file = tmp_path / "receipts.jsonl"
    emitter = ReceiptEmitter(str(receipts_file))

    analysis = TriageAnalysis(
        is_actionable_task=True,
        intent_summary_en="Deploy python application to production VPS",
        required_capabilities=["python", "vps", "docker"],
        domain=DomainClassification(is_software_task=True)
    )

    route_res = RouteResult(
        selected_skill=SkillCandidate(name="vps-deploy", path="/fake/vps-deploy/SKILL.md"),
        injected_skill_content="Run docker-compose up -d",
        tokens_saved_estimate=11200,
        triage_latency_ms=14.5,
        router_backend="host_llm",
        status="success"
    )

    record = emitter.record_turn(
        turn_id="turn-101",
        session_id="session-abc",
        analysis=analysis,
        route_result=route_res
    )

    assert record.turn_id == "turn-101"
    assert record.tokens_saved_estimate == 11200

    # Read back stats
    stats = emitter.get_summary_stats()
    assert stats["total_turns"] == 1
    assert stats["actionable_tasks"] == 1
    assert stats["skills_routed"] == 1
    assert stats["total_tokens_saved"] == 11200
    assert stats["token_reduction_pct"] > 90.0
    assert stats["fast_exit_pct"] == 0.0
