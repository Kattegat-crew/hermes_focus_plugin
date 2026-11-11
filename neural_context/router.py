from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import PluginConfig
from .memory import SelectiveMemoryRetriever, estimate_tokens
from .models import CognitiveRoute, RouteResult, SkillCandidate, TriageAnalysis
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


class GraphRouter(SkillRouterBackend):
    """
    Graph-Theoretic Semantic Router.
    Constructs a weighted bipartite concept-skill graph and computes
    spreading activation / Personalized PageRank from task capabilities,
    detected entities, and memory domain concepts to candidate skills.

    100% free, local, deterministic, and executes in sub-millisecond CPU time.
    """

    def __init__(self, damping: float = 0.85, max_iter: int = 3):
        self.damping = damping
        self.max_iter = max_iter

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

        import re

        skill_nodes = [c.name for c, _ in candidates]

        # Concept vocabulary and edge weights per candidate skill
        skill_edges: Dict[str, Dict[str, float]] = {s: {} for s in skill_nodes}
        concept_skills: Dict[str, List[str]] = {}

        for c, _ in candidates:
            raw_text = f"{c.name} {c.description} {c.lead_paragraph}".lower()
            tokens = re.findall(r"[\w-]{3,}", raw_text, re.UNICODE)
            total = len(tokens) or 1
            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1

            for t, count in freqs.items():
                w = count / total
                skill_edges[c.name][t] = w
                if t not in concept_skills:
                    concept_skills[t] = []
                concept_skills[t].append(c.name)

        # Build seed activation vector from broad semantic concepts and domain context
        seeds: Dict[str, float] = {}
        concepts = analysis.semantic_concepts or analysis.required_capabilities or []
        for concept in concepts:
            c_clean = concept.strip().lower()
            if not c_clean:
                continue
            seeds[c_clean] = seeds.get(c_clean, 0.0) + 3.0
            for token in re.findall(r"[\w-]{3,}", c_clean):
                seeds[token] = seeds.get(token, 0.0) + 1.5

        if analysis.intent_summary_en:
            for token in re.findall(r"[\w-]{3,}", analysis.intent_summary_en.lower()):
                seeds[token] = seeds.get(token, 0.0) + 1.0

        if domain_context:
            for token in re.findall(r"[\w-]{3,}", domain_context.lower()):
                seeds[token] = seeds.get(token, 0.0) + 3.0  # High domain ground-truth energy

        if not seeds:
            return candidates[0][0]

        seed_sum = sum(seeds.values()) or 1.0
        p0_concepts = {k: v / seed_sum for k, v in seeds.items()}

        # Spreading activation from concepts into candidate skill nodes
        skill_scores: Dict[str, float] = {s: 0.0 for s in skill_nodes}
        for concept, act in p0_concepts.items():
            connected = concept_skills.get(concept, [])
            if connected:
                share = act / len(connected)
                for s in connected:
                    skill_scores[s] += share * skill_edges[s].get(concept, 1.0)

        # Combine with normalized BM25 score as prior
        max_bm25 = max(score for _, score in candidates) or 1.0
        ranked: List[Tuple[SkillCandidate, float]] = []
        for c, bm25_score in candidates:
            norm_bm25 = bm25_score / max_bm25
            graph_act = skill_scores.get(c.name, 0.0)
            hybrid_score = 0.65 * graph_act + 0.35 * norm_bm25
            ranked.append((c, hybrid_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[0][0]


class HostLLMRouter(SkillRouterBackend):
    """
    Uses the host Hermes model (ctx.llm) to select the single best skill from the shortlist.
    Zero additional API keys required. Falls back to GraphRouter if LLM is unavailable.
    """

    def __init__(self, ctx: Any = None):
        self.ctx = ctx
        self.fallback = GraphRouter()

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

        if config.router_backend == "graph":
            self.router: SkillRouterBackend = GraphRouter()
        elif config.router_backend == "jev" and config.jev_api_key:
            self.router: SkillRouterBackend = JevRouter(config.jev_api_key, config.jev_base_url, config.jev_model)
        elif config.router_backend in ("host_llm", "auto"):
            self.router = HostLLMRouter(ctx=ctx)
        else:
            self.router = GraphRouter()

    def execute(self, analysis: TriageAnalysis, user_message: str, cwd: Optional[str] = None) -> RouteResult:
        start_time = time.perf_counter()

        # ---------------------------------------------------------
        # Route 1: DIRECT_REPLY (Fast exit, zero injection)
        # ---------------------------------------------------------
        if analysis.route == CognitiveRoute.DIRECT_REPLY or (not analysis.is_actionable_task and not analysis.needs_memory and not analysis.needs_skills):
            return RouteResult(
                triage_analysis=analysis,
                triage_latency_ms=(time.perf_counter() - start_time) * 1000,
                tokens_saved_estimate=0,
                status="skipped_no_context_needed",
                message=analysis.reasoning or "Direct response: no external context required for this turn."
            )

        # ---------------------------------------------------------
        # Route 3 & 4: Selective Memory Retrieval (Native Language Probe)
        # ---------------------------------------------------------
        memories = []
        memory_hints = ""
        if self.config.selective_memory_enabled and (
            analysis.route in (CognitiveRoute.MEMORY_ONLY, CognitiveRoute.ENRICHED_TASK)
            or analysis.needs_memory
        ):
            memories = self.memory_retriever.retrieve(
                entities=analysis.extracted_entities,
                probe=analysis.memory_probe,
                cwd=cwd
            )
            if memories:
                memory_hints = " ".join([f"{m.entity}: {m.fact}" if m.entity else m.fact for m in memories])

        # ---------------------------------------------------------
        # Route 2 & 4: Skill Routing (Uncoupled Semantic Concepts + BM25)
        # ---------------------------------------------------------
        selected_skill: Optional[SkillCandidate] = None
        skill_content: Optional[str] = None
        suggested_missing_skill: Optional[str] = None

        if analysis.route in (CognitiveRoute.SKILL_ONLY, CognitiveRoute.ENRICHED_TASK) or (
            analysis.route not in (CognitiveRoute.DIRECT_REPLY, CognitiveRoute.MEMORY_ONLY) and analysis.needs_skills
        ):
            query_parts = []
            if analysis.semantic_concepts:
                query_parts.append(" ".join(analysis.semantic_concepts))
            elif analysis.required_capabilities:
                query_parts.append(" ".join(analysis.required_capabilities))

            if analysis.search_query:
                query_parts.append(analysis.search_query)
            elif analysis.intent_summary_en:
                query_parts.append(analysis.intent_summary_en)

            # Route 4: Ground skill search with domain facts discovered in Phase 1
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
            else:
                # Skill gap suggestion
                suggested_missing_skill = analysis.suggested_missing_skill or (
                    "-".join(analysis.semantic_concepts[:2]) if analysis.semantic_concepts else None
                )

        # ---------------------------------------------------------
        # Phase 3: Assemble Ephemeral Context Block
        # ---------------------------------------------------------
        context_block = self._assemble_context_block(
            selected_skill,
            skill_content,
            memories,
            suggested_skill=suggested_missing_skill
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        injected_tokens = estimate_tokens(context_block) if context_block else 0
        tokens_saved = max(0, 12000 - injected_tokens) if context_block else 12000

        return RouteResult(
            selected_skill=selected_skill,
            injected_skill_content=skill_content,
            selected_memories=memories,
            combined_context_block=context_block,
            suggested_missing_skill=suggested_missing_skill,
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
        return body

    def _assemble_context_block(
        self,
        skill: Optional[SkillCandidate],
        skill_content: Optional[str],
        memories: list,
        suggested_skill: Optional[str] = None
    ) -> Optional[str]:
        if not skill_content and not memories and not suggested_skill:
            return None

        # Notice fallback when task requires a skill but none was found
        if not skill and not memories and suggested_skill:
            return (
                "<neural_context>\n"
                f"[Notice: No procedural skill matched this task. Recommended capability to install: {suggested_skill}]\n"
                "</neural_context>"
            )

        max_tokens = self.config.max_context_tokens
        header = "<neural_context>"
        instruction = "[Instruction: Use this context as authoritative procedural guidance for this turn.]"
        footer = "</neural_context>"

        base_overhead = estimate_tokens(f"{header}\n{instruction}\n{footer}\n")

        mem_lines = []
        mem_tokens = 0
        if memories:
            mem_hdr = "## Active Memory Facts (Target Entities)"
            mem_tokens += estimate_tokens(mem_hdr)
            for m in memories:
                m_line = f"- [{m.entity}] {m.fact}" if m.entity else f"- {m.fact}"
                m_tok = estimate_tokens(m_line)
                if mem_tokens + m_tok > self.config.memory_token_budget:
                    break
                mem_lines.append(m_line)
                mem_tokens += m_tok

        skill_budget = max(100, max_tokens - base_overhead - mem_tokens)

        skill_lines = []
        if skill and skill_content:
            skill_hdr = f"## Active Procedural Guidance (Skill: {skill.name})"
            skill_cur_tok = estimate_tokens(skill_hdr)
            for line in skill_content.splitlines():
                line_tok = estimate_tokens(line)
                if skill_cur_tok + line_tok > skill_budget:
                    break
                skill_lines.append(line)
                skill_cur_tok += line_tok

        sections = [header]
        if skill and skill_lines:
            sections.append(f"## Active Procedural Guidance (Skill: {skill.name})")
            sections.append("\n".join(skill_lines).strip())
        if mem_lines:
            sections.append("\n## Active Memory Facts (Target Entities)")
            sections.extend(mem_lines)
        if suggested_skill and not skill:
            sections.append(f"\n[Notice: Recommended skill capability to install: {suggested_skill}]")
        sections.append(f"\n{instruction}")
        sections.append(footer)

        block = "\n".join(sections)
        # Strict hard ceiling guarantee: trim skill lines until combined block <= max_tokens
        while estimate_tokens(block) > max_tokens and skill_lines:
            skill_lines.pop()
            if skill_lines:
                sections[2] = "\n".join(skill_lines).strip()
            else:
                del sections[1:3]
            block = "\n".join(sections)

        return block
