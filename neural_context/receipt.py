from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .memory import estimate_tokens
from .models import ReceiptRecord, RouteResult, TriageAnalysis


class ReceiptEmitter:
    """
    Emits verifiable execution records to receipts.jsonl.
    Enables post-turn audit, metrics, and compliance verification.
    """

    def __init__(self, receipts_path: str):
        self.receipts_path = receipts_path
        os.makedirs(os.path.dirname(os.path.abspath(self.receipts_path)), exist_ok=True)

    def record_turn(
        self,
        turn_id: str,
        session_id: str,
        analysis: TriageAnalysis,
        route_result: RouteResult,
        error: Optional[str] = None,
        platform: str = "cli",
        model: str = ""
    ) -> ReceiptRecord:
        route_val = getattr(analysis, "route", "SKILL_ONLY")
        if hasattr(route_val, "value"):
            route_val = route_val.value

        is_truncated = getattr(analysis, "truncation_detected", False) or (getattr(route_result, "status", "") == "error_truncated")
        status_val = "error_truncated" if is_truncated else (route_result.status if not error else "error")
        error_val = error or analysis.error or ("Model response truncated by token limit" if is_truncated else None)

        rendered_block = route_result.combined_context_block or ""
        rendered_tokens = estimate_tokens(rendered_block) if rendered_block else 0
        memory_marker = "## Historical Memory (Untrusted Facts)"
        memory_tokens = 0
        if memory_marker in rendered_block:
            memory_section = rendered_block.split(memory_marker, 1)[1].split("[Instruction:", 1)[0]
            memory_tokens = estimate_tokens(memory_section)
        skill_lines = []
        in_skill = False
        for line in rendered_block.splitlines():
            if line.startswith("## Active Procedural Guidance"):
                in_skill = True
                skill_lines.append(line)
                continue
            elif line.startswith("## Historical Memory") or line.startswith("[Instruction:") or line == "</neural_context>":
                in_skill = False
            elif in_skill:
                skill_lines.append(line)
        skill_tokens = estimate_tokens("\n".join(skill_lines)) if skill_lines else 0

        record = ReceiptRecord(
            turn_id=turn_id,
            session_id=session_id,
            platform=platform,
            model=model,
            route=str(route_val),
            is_task=analysis.is_actionable_task,
            domain="software" if analysis.domain.is_software_task else "general",
            entities_detected=analysis.extracted_entities,
            memory_injected_tokens=memory_tokens,
            selected_skill=route_result.selected_skill.name if route_result.selected_skill else None,
            suggested_missing_skill=route_result.suggested_missing_skill,
            skill_is_project_local=route_result.selected_skill.is_project_local if route_result.selected_skill else False,
            skill_injected_tokens=skill_tokens,
            router_backend=route_result.router_backend,
            configured_router_backend=route_result.configured_router_backend,
            triage_latency_ms=analysis.model_latency_ms,
            model_latency_ms=analysis.model_latency_ms,
            coordination_latency_ms=route_result.coordination_latency_ms,
            rendered_context_tokens=rendered_tokens,
            context_overhead_tokens=max(0, rendered_tokens - memory_tokens - skill_tokens),
            tokens_saved_estimate=route_result.tokens_saved_estimate,
            status=status_val,
            error=error_val
        )

        try:
            with open(self.receipts_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception:
            pass  # Receipt emission must never crash the turn

        return record

    def get_summary_stats(self, limit: int = 100) -> Dict[str, Any]:
        """Calculates aggregated metrics from the recent receipts."""
        if not Path(self.receipts_path).is_file():
            return {
                "total_turns": 0,
                "actionable_tasks": 0,
                "skills_routed": 0,
                "avg_latency_ms": 0.0,
                "total_tokens_saved": 0
            }

        records: List[ReceiptRecord] = []
        try:
            with open(self.receipts_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    line = line.strip()
                    if line:
                        records.append(ReceiptRecord.model_validate_json(line))
        except Exception:
            pass

        if not records:
            return {"total_turns": 0}

        total_turns = len(records)
        actionable = sum(1 for r in records if r.is_task)
        routed = sum(1 for r in records if r.selected_skill)
        avg_lat = sum(r.triage_latency_ms for r in records) / total_turns
        tokens_saved = sum(r.tokens_saved_estimate for r in records)
        total_injected = sum(r.skill_injected_tokens + r.memory_injected_tokens for r in records)
        total_potential = tokens_saved + total_injected
        token_reduction_pct = round((tokens_saved / total_potential) * 100, 1) if total_potential > 0 else 0.0
        fast_exit_pct = round(((total_turns - actionable) / total_turns) * 100, 1) if total_turns > 0 else 0.0

        return {
            "total_turns": total_turns,
            "actionable_tasks": actionable,
            "skills_routed": routed,
            "avg_latency_ms": round(avg_lat, 2),
            "total_tokens_saved": tokens_saved,
            "token_reduction_pct": token_reduction_pct,
            "fast_exit_pct": fast_exit_pct
        }
