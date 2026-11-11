from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

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
        record = ReceiptRecord(
            turn_id=turn_id,
            session_id=session_id,
            platform=platform,
            model=model,
            is_task=analysis.is_actionable_task,
            domain="software" if analysis.domain.is_software_task else "general",
            entities_detected=analysis.extracted_entities,
            memory_injected_tokens=sum(m.token_estimate for m in route_result.selected_memories),
            selected_skill=route_result.selected_skill.name if route_result.selected_skill else None,
            skill_is_project_local=route_result.selected_skill.is_project_local if route_result.selected_skill else False,
            skill_injected_tokens=len((route_result.injected_skill_content or "").split()),
            router_backend=route_result.router_backend,
            triage_latency_ms=route_result.triage_latency_ms,
            tokens_saved_estimate=route_result.tokens_saved_estimate,
            status=route_result.status if not error else "error",
            error=error
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

        return {
            "total_turns": total_turns,
            "actionable_tasks": actionable,
            "skills_routed": routed,
            "avg_latency_ms": round(avg_lat, 2),
            "total_tokens_saved": tokens_saved
        }
