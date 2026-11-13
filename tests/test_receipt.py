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
        configured_router_backend="graph",
        combined_context_block="<neural_context>\n## Historical Memory (Untrusted Facts)\n- [Acme] actual fact\n## Active Procedural Guidance (Skill: vps-deploy)\nRun only the rendered steps\n</neural_context>",
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
    assert record.router_backend == "host_llm"
    assert record.configured_router_backend == "graph"
    assert record.rendered_context_tokens > 0
    assert record.memory_injected_tokens > 0
    assert record.skill_injected_tokens > 0

    # Read back stats
    stats = emitter.get_summary_stats()
    assert stats["total_turns"] == 1
    assert stats["actionable_tasks"] == 1
    assert stats["skills_routed"] == 1
    assert stats["total_tokens_saved"] == 11200
    assert stats["token_reduction_pct"] > 90.0
    assert stats["fast_exit_pct"] == 0.0


def test_memory_only_receipt_preserves_model_error_latency_and_zero_skill_tokens(tmp_path):
    from neural_context.models import MemoryItem

    emitter = ReceiptEmitter(str(tmp_path / "memory-only.jsonl"))
    analysis = TriageAnalysis(model_latency_ms=250.0, error="model_timeout")
    route = RouteResult(
        selected_memories=[MemoryItem(entity="Acme", fact="historic", token_estimate=90)],
        combined_context_block="<neural_context>\n## Historical Memory (Untrusted Facts)\n- [Acme] historic\n</neural_context>",
        triage_latency_ms=250.0,
        coordination_latency_ms=0.002,
        router_backend=None,
    )
    record = emitter.record_turn("t", "s", analysis, route)
    assert record.error == "model_timeout"
    assert record.model_latency_ms == 250.0
    assert record.triage_latency_ms == 250.0
    assert record.coordination_latency_ms == 0.002
    assert record.skill_injected_tokens == 0
    assert record.context_overhead_tokens > 0
