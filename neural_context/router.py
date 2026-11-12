from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import PluginConfig
from .memory import SelectiveMemoryRetriever, estimate_tokens
from .models import CognitiveRoute, RouteResult, SkillCandidate, TriageAnalysis
from .sidecar import SidecarIndex, term_information_weight, tokenize


class SkillRouterBackend(ABC):
    @abstractmethod
    def route(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> Optional[SkillCandidate]:
        pass

    def route_all(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> List[SkillCandidate]:
        winner = self.route(analysis, candidates, domain_context=domain_context)
        return [winner] if winner is not None else []


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
        matched = self.route_all(analysis, candidates, domain_context=domain_context)
        return matched[0] if matched else None

    def route_all(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> List[SkillCandidate]:
        if not candidates:
            return []
        return [cand for cand, score in candidates if score >= self.min_score]


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
        matched = self.route_all(analysis, candidates, domain_context=domain_context)
        return matched[0] if matched else None

    def route_all(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> List[SkillCandidate]:
        if not candidates:
            return []

        skill_nodes = [c.name for c, _ in candidates]
        n_cands = len(candidates)

        # Concept vocabulary and edge weights per candidate skill (zero hardcoded word lists)
        skill_edges: Dict[str, Dict[str, float]] = {s: {} for s in skill_nodes}
        concept_skills: Dict[str, List[str]] = {}

        for c, _ in candidates:
            tokens = tokenize(f"{c.name} {c.description} {c.lead_paragraph}")
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

        def _concept_specificity(tok: str) -> float:
            df = len(concept_skills.get(tok, ()))
            return term_information_weight(tok, df, n_cands)

        # Build seed activation vector weighted by concept specificity (IDF + morphological density)
        seeds: Dict[str, float] = {}
        concepts = analysis.semantic_concepts or analysis.required_capabilities or []
        concept_weight = 1.5 if domain_context else 3.0
        subtoken_weight = 1.0 if domain_context else 1.5
        for concept in concepts:
            c_clean = concept.strip().lower()
            if not c_clean:
                continue
            seeds[c_clean] = seeds.get(c_clean, 0.0) + concept_weight * _concept_specificity(c_clean)
            for token in tokenize(c_clean):
                seeds[token] = seeds.get(token, 0.0) + subtoken_weight * _concept_specificity(token)

        if analysis.intent_summary_en:
            intent_w = 0.5 if domain_context else 1.0
            for token in tokenize(analysis.intent_summary_en):
                seeds[token] = seeds.get(token, 0.0) + intent_w * _concept_specificity(token)

        if domain_context:
            for token in tokenize(domain_context):
                seeds[token] = seeds.get(token, 0.0) + 5.0 * _concept_specificity(token)

        if not seeds:
            return [candidates[0][0]] if candidates[0][1] >= 0.3 else []

        seed_sum = sum(seeds.values()) or 1.0
        p0_concepts = {k: v / seed_sum for k, v in seeds.items()}

        # Iterative Bipartite Personalized PageRank (Random Walk with Restart)
        # Propagates activation: Concept -> Skill -> Co-occurring Bridge Concepts -> Skill
        p_concepts: Dict[str, float] = dict(p0_concepts)
        skill_scores: Dict[str, float] = {s: 0.0 for s in skill_nodes}
        iterations = max(1, int(self.max_iter))

        for step in range(iterations):
            step_skills: Dict[str, float] = {s: 0.0 for s in skill_nodes}
            for concept, act in p_concepts.items():
                if act <= 0.0:
                    continue
                connected = concept_skills.get(concept)
                if connected:
                    share = act / len(connected)
                    for s in connected:
                        step_skills[s] += share * skill_edges[s].get(concept, 1.0)

            skill_scores = step_skills

            # Propagate Skill -> Concept with damping and restart to seed distribution p0
            if step + 1 < iterations:
                feedback_concepts: Dict[str, float] = {}
                for s, s_act in step_skills.items():
                    if s_act <= 0.0:
                        continue
                    for c_tok, weight in skill_edges[s].items():
                        feedback_concepts[c_tok] = feedback_concepts.get(c_tok, 0.0) + (s_act * weight)

                next_concepts: Dict[str, float] = dict(p0_concepts)
                for c_tok, fb_val in feedback_concepts.items():
                    next_concepts[c_tok] = next_concepts.get(c_tok, 0.0) + (self.damping * fb_val)
                p_concepts = next_concepts

        # Require actual concept overlap when only 1 candidate is present
        max_bm25 = max(score for _, score in candidates) or 1.0
        if len(candidates) == 1:
            only_cand, only_bm25 = candidates[0]
            if skill_scores.get(only_cand.name, 0.0) <= 0.0 and only_bm25 < 1.5:
                return []
            return [only_cand] if only_bm25 >= 0.2 else []

        # Combine with normalized BM25 score as prior
        ranked: List[Tuple[SkillCandidate, float]] = []
        for c, bm25_score in candidates:
            norm_bm25 = bm25_score / max_bm25
            graph_act = skill_scores.get(c.name, 0.0)
            hybrid_score = 0.65 * graph_act + 0.35 * norm_bm25
            ranked.append((c, hybrid_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        top_cand, top_score = ranked[0]
        # Require meaningful signal: reject pure noise / residual matches
        if (top_score < 0.15 and max_bm25 < 0.3) or (skill_scores.get(top_cand.name, 0.0) <= 0.0 and max_bm25 < 1.5):
            return []

        # Multi-skill set-cover over distinct requested capabilities and domain concepts:
        # Select the primary winner plus any additional skills that cover distinct requested
        # capability concepts with strong PageRank activation.
        explicit_cap_tokens: Set[str] = set()
        for concept in concepts:
            for t in tokenize(concept):
                if _concept_specificity(t) >= 0.35:
                    explicit_cap_tokens.add(t)
        domain_tokens: Set[str] = set()
        if domain_context:
            for t in tokenize(domain_context):
                if _concept_specificity(t) >= 0.45:
                    domain_tokens.add(t)

        target_concepts = domain_tokens if domain_tokens else explicit_cap_tokens
        if not target_concepts:
            target_concepts = {t for t in p0_concepts if _concept_specificity(t) >= 0.45}

        winners: List[SkillCandidate] = [top_cand]
        covered_concepts: Set[str] = set(skill_edges.get(top_cand.name, {}).keys()) & target_concepts

        for cand, h_score in ranked[1:]:
            if skill_scores.get(cand.name, 0.0) <= 0.0:
                continue
            if h_score < max(0.15, top_score * 0.35):
                continue
            cand_concepts = set(skill_edges.get(cand.name, {}).keys()) & target_concepts
            uncovered = cand_concepts - covered_concepts
            # Include secondary skill if it covers distinct requested capabilities not covered
            # by already-selected skills, or if it has near-equal graph activation on domain concepts.
            if uncovered or (h_score >= top_score * 0.80 and bool(cand_concepts)):
                winners.append(cand)
                covered_concepts.update(cand_concepts)

        return winners


class HostLLMRouter(SkillRouterBackend):
    """
    Uses the host Hermes model (ctx.llm) to select the single best skill from the shortlist.
    Zero additional API keys required. Falls back to GraphRouter if LLM is unavailable.
    """

    def __init__(self, ctx: Any = None, fallback: Optional[SkillRouterBackend] = None, timeout_ms: int = 8000):
        self.ctx = ctx
        self.fallback = fallback or GraphRouter()
        self.timeout_ms = timeout_ms

    def route(
        self,
        analysis: TriageAnalysis,
        candidates: List[Tuple[SkillCandidate, float]],
        domain_context: Optional[str] = None
    ) -> Optional[SkillCandidate]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return self.fallback.route(analysis, candidates, domain_context=domain_context)

        # If ctx.llm is not available, fall back to graph router
        if self.ctx is None or not hasattr(self.ctx, "llm") or self.ctx.llm is None:
            return self.fallback.route(analysis, candidates, domain_context=domain_context)

        shortlist = candidates[:max(6, min(12, len(candidates)))]
        options_text = "\n".join([
            f"- {c.name}: {c.description} | {c.lead_paragraph}"
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
            prompt_lines.append(
                "\nIMPORTANT: 'Domain Context from Memory' contains verified architectural facts "
                "about the target entity discovered in Phase 1. When the User Task is a shorthand "
                "or general request about that entity (e.g., 'Review AcmeCorp', 'Check project X'), "
                "you MUST ground the request in the technical stack and integrations revealed by Memory "
                "and select the specialized skill that governs that stack (rather than a generic review skill or 'NONE')."
            )
        prompt_lines.extend([
            f"\nCandidate Skills:\n{options_text}\n",
            "Reply ONLY with the exact name of the selected skill, or 'NONE':"
        ])
        prompt = "\n".join(prompt_lines)

        try:
            # Allocate token budget with ample headroom for reasoning thought tokens
            budget = max(1000, 500 + len(options_text) // 4)
            complete_kwargs: Dict[str, Any] = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": budget,
                "temperature": 0.0,
            }
            if self.timeout_ms and self.timeout_ms > 0:
                complete_kwargs["timeout"] = self.timeout_ms / 1000.0
            resp = self.ctx.llm.complete(**complete_kwargs)
            choice = (resp.text if hasattr(resp, "text") else str(resp)).strip()
            if "none" in choice.lower():
                return None
            for cand, _ in shortlist:
                if cand.name.lower() in choice.lower():
                    return cand
            if domain_context:
                return self.fallback.route(analysis, candidates, domain_context=domain_context)
        except Exception:
            return self.fallback.route(analysis, candidates, domain_context=domain_context)

        return None


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
        elif config.router_backend in ("host_llm", "auto"):
            self.router = HostLLMRouter(ctx=ctx, timeout_ms=config.timeout_ms)
        else:
            self.router = GraphRouter()

    def execute(
        self,
        analysis: TriageAnalysis,
        user_message: str,
        cwd: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> RouteResult:
        start_time = time.perf_counter()

        # ---------------------------------------------------------
        # Route 1: DIRECT_REPLY (Fast exit, zero injection)
        # ---------------------------------------------------------
        if analysis.route == CognitiveRoute.DIRECT_REPLY or (not analysis.is_actionable_task and not analysis.needs_memory and not analysis.needs_skills):
            is_trunc = getattr(analysis, "truncation_detected", False)
            return RouteResult(
                triage_analysis=analysis,
                triage_latency_ms=(time.perf_counter() - start_time) * 1000,
                tokens_saved_estimate=0,
                status="error_truncated" if is_trunc else "skipped_no_context_needed",
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
                cwd=cwd,
                session_id=session_id,
            )
            if memories:
                memory_hints = " ".join([f"{m.entity}: {m.fact}" if m.entity else m.fact for m in memories])

        # ---------------------------------------------------------
        # Route 2 & 4: Skill Routing (Uncoupled Semantic Concepts + Bipartite Graph)
        # ---------------------------------------------------------
        selected_skills: List[SkillCandidate] = []
        selected_skill: Optional[SkillCandidate] = None
        skill_contents: List[Tuple[SkillCandidate, str]] = []
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

            # Route 4: Ground skill search with domain facts discovered in Phase 1,
            # distilling symbols and low-information stopwords BEFORE querying BM25 and the Graph.
            distilled_memory_hints = ""
            if memory_hints:
                distilled_tokens = self.sidecar.filter_query_terms(memory_hints, min_weight=0.30)
                distilled_memory_hints = " ".join(distilled_tokens)
                if distilled_memory_hints:
                    query_parts.append(distilled_memory_hints)

            raw_query = " ".join(query_parts)
            if not raw_query:
                candidates = []
            else:
                distilled_query_tokens = self.sidecar.filter_query_terms(raw_query, min_weight=0.25)
                query = " ".join(distilled_query_tokens) if distilled_query_tokens else raw_query

                # Evaluate all matching skills across the entire catalog (top_k=0 -> no artificial truncation)
                base_candidates = self.sidecar.search_bm25(query=query, top_k=0)
                if distilled_memory_hints:
                    mem_candidates = self.sidecar.search_bm25(query=distilled_memory_hints, top_k=0)
                    merged_map: Dict[str, Tuple[SkillCandidate, float]] = {}
                    for cand, sc in mem_candidates:
                        boosted = sc * 1.35
                        cand_copy = cand.model_copy()
                        cand_copy.score = boosted
                        merged_map[cand.name] = (cand_copy, boosted)
                    for cand, sc in base_candidates:
                        if cand.name not in merged_map or sc > merged_map[cand.name][1]:
                            merged_map[cand.name] = (cand, sc)
                    candidates = sorted(merged_map.values(), key=lambda x: x[1], reverse=True)
                else:
                    candidates = base_candidates

            selected_skills = self.router.route_all(
                analysis,
                candidates,
                domain_context=distilled_memory_hints or memory_hints,
            )
            selected_skill = selected_skills[0] if selected_skills else None

            for sk in selected_skills:
                if Path(sk.path).is_file():
                    try:
                        raw_skill = Path(sk.path).read_text(encoding="utf-8", errors="ignore")
                        formatted = self._format_skill_for_injection(sk.name, raw_skill)
                        if formatted:
                            skill_contents.append((sk, formatted))
                    except Exception:
                        pass

            if skill_contents:
                skill_content = "\n\n".join(content for _, content in skill_contents)
            else:
                selected_skills = []
                selected_skill = None
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
            suggested_skill=suggested_missing_skill,
            skill_items=skill_contents,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        injected_tokens = estimate_tokens(context_block) if context_block else 0
        if not context_block:
            tokens_saved = 0
        else:
            catalog_tokens = getattr(self.sidecar, "total_catalog_tokens", 0)
            catalog_count = getattr(self.sidecar, "n_docs", 0)
            baseline = max(catalog_tokens, catalog_count * 350, injected_tokens + 500) if catalog_count > 0 else injected_tokens + 500
            tokens_saved = max(0, baseline - injected_tokens)

        is_truncated = getattr(analysis, "truncation_detected", False)
        status_val = "error_truncated" if is_truncated else ("success" if context_block else "no_context_injected")
        msg_val = "Model triage output truncated by token limit; degraded to fallback intent." if is_truncated else analysis.reasoning

        return RouteResult(
            selected_skill=selected_skill,
            selected_skills=selected_skills,
            injected_skill_content=skill_content,
            selected_memories=memories,
            combined_context_block=context_block,
            suggested_missing_skill=suggested_missing_skill,
            triage_analysis=analysis,
            tokens_saved_estimate=tokens_saved,
            triage_latency_ms=elapsed_ms,
            router_backend=self.config.router_backend,
            status=status_val,
            message=msg_val
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
        suggested_skill: Optional[str] = None,
        skill_items: Optional[List[Tuple[SkillCandidate, str]]] = None,
    ) -> Optional[str]:
        items: List[Tuple[SkillCandidate, str]] = list(skill_items or [])
        if not items and skill and skill_content:
            items = [(skill, skill_content)]

        if not items and not memories and not suggested_skill:
            return None

        # Notice fallback when task requires a skill but none was found
        if not items and not memories and suggested_skill:
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
                if self.config.memory_token_budget > 0 and mem_tokens + m_tok > self.token_budget_value():
                    break
                mem_lines.append(m_line)
                mem_tokens += m_tok

        rendered_skill_blocks: List[Tuple[str, List[str]]] = []
        if items:
            skill_budget = max(100, max_tokens - base_overhead - mem_tokens) if max_tokens > 0 else 0
            skill_cur_tok = 0
            for sk, sk_text in items:
                sk_hdr = f"## Active Procedural Guidance (Skill: {sk.name})"
                hdr_tok = estimate_tokens(sk_hdr)
                if max_tokens > 0 and skill_cur_tok + hdr_tok > skill_budget and rendered_skill_blocks:
                    break
                skill_cur_tok += hdr_tok
                cur_lines: List[str] = []
                for line in sk_text.splitlines():
                    line_tok = estimate_tokens(line)
                    if max_tokens > 0 and skill_cur_tok + line_tok > skill_budget:
                        break
                    cur_lines.append(line)
                    skill_cur_tok += line_tok
                if cur_lines:
                    rendered_skill_blocks.append((sk_hdr, cur_lines))

        sections = [header]
        for sk_hdr, cur_lines in rendered_skill_blocks:
            sections.append(sk_hdr)
            sections.append("\n".join(cur_lines).strip())
        if mem_lines:
            sections.append("\n## Active Memory Facts (Target Entities)")
            sections.extend(mem_lines)
        if suggested_skill and not rendered_skill_blocks:
            sections.append(f"\n[Notice: Recommended skill capability to install: {suggested_skill}]")
        sections.append(f"\n{instruction}")
        sections.append(footer)

        block = "\n".join(sections)
        if max_tokens > 0 and rendered_skill_blocks:
            while estimate_tokens(block) > max_tokens and rendered_skill_blocks:
                last_hdr, last_lines = rendered_skill_blocks[-1]
                if last_lines:
                    last_lines.pop()
                if not last_lines:
                    rendered_skill_blocks.pop()
                sections = [header]
                for sk_hdr, cur_lines in rendered_skill_blocks:
                    sections.append(sk_hdr)
                    sections.append("\n".join(cur_lines).strip())
                if mem_lines:
                    sections.append("\n## Active Memory Facts (Target Entities)")
                    sections.extend(mem_lines)
                sections.append(f"\n{instruction}")
                sections.append(footer)
                block = "\n".join(sections)

        return block

    def token_budget_value(self) -> int:
        return self.config.memory_token_budget
