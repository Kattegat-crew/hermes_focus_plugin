from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import PluginConfig
from .memory import SelectiveMemoryRetriever
from .models import RouteResult, SkillCandidate, TriageAnalysis
from .sidecar import SidecarIndex


class SkillRouterBackend(ABC):
    @abstractmethod
    def route(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> Optional[SkillCandidate]:
        pass


class FastHeuristicRouter(SkillRouterBackend):
    """
    Sub-2ms heuristic ranker based on BM25 score margins and project-local priority.
    """

    def __init__(self, min_score: float = 0.5):
        self.min_score = min_score

    def route(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> Optional[SkillCandidate]:
        if not candidates:
            return None
        top_candidate, score = candidates[0]
        if score < self.min_score:
            return None
        return top_candidate


class HostLLMRouter(SkillRouterBackend):
    """
    Uses the host Hermes model (ctx.llm) to select the single best skill from the shortlist.
    Zero additional API keys required.
    """

    def __init__(self, ctx: Any = None):
        self.ctx = ctx
        self.fallback = FastHeuristicRouter()

    def route(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> Optional[SkillCandidate]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]

        # If ctx.llm is not available, fall back to heuristic
        if self.ctx is None or not hasattr(self.ctx, "llm") or self.ctx.llm is None:
            return self.fallback.route(analysis, candidates, domain_context=domain_context)

        shortlist = candidates[:4]
        options_text = "\n".join([
            f"- {c.name}: {c.description} | {c.lead_paragraph[:120]}"
            for c, _ in shortlist
        ])

        prompt_lines = [
            "You are an AI tool selector. Choose the single most relevant skill from the list below "
            "for the given user task. If none truly fits, reply 'NONE'.\n",
            f"User Task: {analysis.intent_summary_en}",
            f"Required Capabilities: {', '.join(analysis.required_capabilities)}",
        ]
        if domain_context:
            prompt_lines.append(f"Domain Context from Memory: {domain_context}")
        prompt_lines.extend([
            f"\nCandidate Skills:\n{options_text}\n",
            "Reply ONLY with the exact name of the selected skill, or 'NONE':"
        ])
        prompt = "\n".join(prompt_lines)

        try:
            # Hermes Agent PluginLlm.complete expects messages: list[dict[str, Any]]
            resp = self.ctx.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0
            )
            choice = (resp.text if hasattr(resp, "text") else str(resp)).strip()
            for cand, _ in shortlist:
                if cand.name.lower() in choice.lower():
                    return cand
        except Exception:
            return self.fallback.route(analysis, candidates, domain_context=domain_context)

        return None


class JevRouter(SkillRouterBackend):
    """
    Optional external TypeSafe / Jev router backend when TYPESAFE_API_KEY is available.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.typesafe.ai", model: str = "jev-latest"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback = FastHeuristicRouter()

    def route(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> Optional[SkillCandidate]:
        if not candidates:
            return None
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "task": analysis.intent_summary_en,
                "capabilities": analysis.required_capabilities,
                "options": [c.name for c, _ in candidates[:10]]
            }
            if domain_context:
                payload["domain_context"] = domain_context
            res = requests.post(f"{self.base_url}/v1/route", json=payload, headers=headers, timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                winner = data.get("winner")
                for c, _ in candidates:
                    if c.name == winner:
                        return c
        except Exception:
            pass
        return self.fallback.route(analysis, candidates, domain_context=domain_context)


class ContextCoordinator:
    """
    Central coordinator executing Two-Phase Enriched Routing:
    Triage -> Phase 1: Selective Memory Query -> Phase 2: Memory-Enriched Skill Search & Routing -> Phase 3: Ephemeral Assembly.
    """

    def __init__(self, config: PluginConfig, sidecar: SidecarIndex, ctx: Any = None):
        self.config = config
        self.sidecar = sidecar
        self.ctx = ctx
        self.memory_retriever = SelectiveMemoryRetriever(token_budget=config.memory_token_budget)

        if config.router_backend == "jev" and config.jev_api_key:
            self.router: SkillRouterBackend = JevRouter(config.jev_api_key, config.jev_base_url, config.jev_model)
        elif config.router_backend == "host_llm":
            self.router = HostLLMRouter(ctx=ctx)
        else:
            self.router = FastHeuristicRouter()

    def execute(self, analysis: TriageAnalysis, user_message: str, cwd: Optional[str] = None) -> RouteResult:
        start_time = time.perf_counter()

        # Fast exit if prompt is non-actionable or requires neither skills nor memory
        if not analysis.is_actionable_task or (not analysis.needs_skills and not analysis.needs_memory):
            return RouteResult(
                triage_analysis=analysis,
                triage_latency_ms=(time.perf_counter() - start_time) * 1000,
                tokens_saved_estimate=0,
                status="skipped_no_context_needed",
                message=analysis.reasoning or "No skills or memory required for this turn."
            )

        # ---------------------------------------------------------
        # Phase 1: Selective Memory Query (Ground Domain Context)
        # ---------------------------------------------------------
        memories = []
        memory_hints = ""
        if self.config.selective_memory_enabled and (analysis.needs_memory or analysis.extracted_entities):
            if analysis.extracted_entities:
                memories = self.memory_retriever.retrieve(analysis.extracted_entities, cwd=cwd)
                if memories:
                    # Ground domain context with retrieved facts
                    memory_hints = " ".join([f"{m.entity}: {m.fact}" for m in memories])

        selected_skill: Optional[SkillCandidate] = None
        skill_content: Optional[str] = None

        # ---------------------------------------------------------
        # Phase 2: Context-Enriched Skill Search & Single-Skill Routing
        # ---------------------------------------------------------
        if analysis.needs_skills:
            query_parts = []
            if analysis.search_query:
                query_parts.append(analysis.search_query)
            if analysis.intent_summary_en and analysis.intent_summary_en not in analysis.search_query:
                query_parts.append(analysis.intent_summary_en)
            if analysis.required_capabilities:
                query_parts.append(" ".join(analysis.required_capabilities))
            # Enrich skill query with domain facts discovered in Phase 1
            if memory_hints:
                query_parts.append(memory_hints)

            query = " ".join(query_parts) if query_parts else user_message
            candidates = self.sidecar.search_bm25(query=query, top_k=5)
            selected_skill = self.router.route(analysis, candidates, domain_context=memory_hints)

            if selected_skill and Path(selected_skill.path).is_file():
                try:
                    raw_skill = Path(selected_skill.path).read_text(encoding="utf-8", errors="ignore")
                    skill_content = self._format_skill_for_injection(selected_skill.name, raw_skill)
                except Exception:
                    skill_content = None

        # ---------------------------------------------------------
        # Phase 3: Assemble Ephemeral Context Block
        # ---------------------------------------------------------
        context_block = self._assemble_context_block(selected_skill, skill_content, memories)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        injected_tokens = len(context_block.split()) if context_block else 0
        tokens_saved = max(0, 12000 - injected_tokens) if context_block else 12000

        return RouteResult(
            selected_skill=selected_skill,
            injected_skill_content=skill_content,
            selected_memories=memories,
            combined_context_block=context_block,
            triage_analysis=analysis,
            tokens_saved_estimate=tokens_saved,
            triage_latency_ms=elapsed_ms,
            router_backend=self.config.router_backend,
            status="success" if context_block else "no_context_injected",
            message=analysis.reasoning
        )

    def _format_skill_for_injection(self, name: str, raw_content: str) -> str:
        # Strip YAML frontmatter
        body = raw_content
        if raw_content.startswith("---"):
            parts = raw_content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()
        # Cap at 500 lines to protect budget
        lines = body.splitlines()[:500]
        return "\n".join(lines)

    def _assemble_context_block(
        self,
        skill: Optional[SkillCandidate],
        skill_content: Optional[str],
        memories: list
    ) -> Optional[str]:
        if not skill_content and not memories:
            return None

        sections = ["<neural_context>"]
        if skill and skill_content:
            sections.append(f"## Active Procedural Guidance (Skill: {skill.name})")
            sections.append(skill_content.strip())

        if memories:
            sections.append("\n## Active Memory Facts (Target Entities)")
            for m in memories:
                sections.append(f"- [{m.entity}] {m.fact}")

        sections.append("\n[Instruction: Use this context as authoritative procedural guidance for this turn.]")
        sections.append("</neural_context>")
        return "\n".join(sections)
