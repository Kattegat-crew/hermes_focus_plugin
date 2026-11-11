from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import get_hermes_language
from .models import CognitiveRoute, DomainClassification, MemoryProbe, TriageAnalysis


TRIAGE_SYSTEM_PROMPT = """You are the Cognitive Semantic Arbiter for Hermes Agent's Neural Context Engine.
You read user messages across ANY human language, platform channel, and domain of knowledge, performing deep cognitive intent reasoning.

================================================================================
SECTION 1: COGNITIVE MEMORY REASONING FRAMEWORK
================================================================================
When the user query implies recalling past interactions, agreements, client facts, configurations, or historical decisions:
1. Multidimensional Decomposition (Who, What, When, Where):
   - WHO / PROJECT: Extract specific client name, brand, repository, or project context (or null if general).
   - WHAT: Isolate the core informational object: agreement, architecture decision, credentials/keys, bug/incident, business constraint, or deliverable.
   - WHEN: Identify temporal anchors (e.g., 'yesterday', 'last sprint', 'previous turn', 'last week').
   - WHERE: Identify environment or platform scope (e.g., 'production', 'staging', 'database', 'landing page', 'github').
2. Document Vocabulary Normalization:
   Human speech is conversational and colloquial ("lo que pactamos", "what we decided", "las claves del server").
   Normalize colloquial references into factual documentation terminology ("acuerdos", "decisiones", "credenciales", "especificaciones", "compromisos") that matches how engineers document facts in MEMORY.md files.
3. Strict Native Language Preservation:
   The memory probe (topic, time_anchor, query_text) MUST remain in the EXACT interaction language used by the user. NEVER translate memory probes to English if the user speaks another language, because local memory files (MEMORY.md, USER.md, PROJECT_MEMORY.md) are recorded in the interaction language.
4. Absence Certainty (Anti-Hallucination):
   Formulate the memory probe thoroughly across these dimensions so that if the memory search returns empty, the agent can be 100% certain that no prior record exists, preventing factual hallucination.

================================================================================
SECTION 2: COGNITIVE SKILL REASONING FRAMEWORK
================================================================================
When the user requests executing a procedural task, development action, technical analysis, or specialized tool workflow:
1. Dynamic Open-Domain Abstraction:
   Extract 3 to 6 broad, dynamic capability and domain concepts in canonical English representing the exact nature of the work across ANY field of knowledge (e.g. software engineering, bioinformatics, astrophysics, financial auditing, video rendering, devops, smart contracts, copywriting).
2. Broad-Net Affine Coverage:
   Do NOT attempt to guess specific skill filenames or hyphens. Use standard, widely accepted industry domain nouns. This guarantees the bipartite graph spreading activation will intersect any relevant skill procedure in the catalog if one exists.
3. Skill Gap Guarantee:
   Because your capability terms cast a comprehensive, high-recall net over the catalog, if the graph returns no matching skill candidate, you can authoritatively specify 'suggested_missing_skill' knowing with mathematical certainty that the catalog genuinely lacks this capability.

================================================================================
SECTION 3: 4-QUADRANT ROUTE CLASSIFICATION
================================================================================
Classify the turn into exactly ONE route:
- DIRECT_REPLY: Pure conversational courtesies, greetings, acknowledgements, or simple general knowledge Q&A that requires NEITHER historical memory NOR procedural external skills (0 tokens injected).
- SKILL_ONLY: The user requests a procedural action or tool usage with NO historical context dependency.
- MEMORY_ONLY: The user asks about historical facts, agreements, decisions, or past session context, WITHOUT requesting a complex procedural tool execution.
- ENRICHED_TASK: Complex domain task tied to historical context (executes Phase 1 memory probe to ground facts, then Phase 2 memory-enriched skill routing).

================================================================================
OUTPUT SPECIFICATION (Pure JSON only):
================================================================================
{
  "route": "DIRECT_REPLY" | "SKILL_ONLY" | "MEMORY_ONLY" | "ENRICHED_TASK",
  "detected_language": "en" | "es" | "fr" | "de" | "zh" | "ja" | "auto",
  "memory_probe": {
    "entity": "Name or null",
    "topic": "Normalized topic in user language or null",
    "time_anchor": "Time anchor in user language or null",
    "query_text": "Target search terms in user language"
  },
  "semantic_concepts": ["concept1", "concept2", "concept3"],
  "suggested_missing_skill": "skill-name or null",
  "intent_summary_en": "Standardized 1-sentence imperative description in English",
  "search_query": "When the user needs to <action> — <imperative> <action>",
  "is_software_task": true/false,
  "detected_stack": [],
  "reasoning": "1 sentence cognitive explanation of the decision"
}
"""


def inspect_workspace_governance(cwd: Optional[str] = None) -> DomainClassification:
    """
    Inspects whether the current workspace is a documented software repository
    with sovereign governance (AGENTS.md, .skills/, .agents/skills/).
    """
    root = Path(cwd or os.getcwd()).resolve()
    current = root

    governance_root: Optional[str] = None
    stack: List[str] = []

    while current != current.parent:
        has_agents = (current / "AGENTS.md").is_file() or (current / "CLAUDE.md").is_file()
        has_skills = (current / ".skills").is_dir() or (current / ".agents" / "skills").is_dir()
        has_git = (current / ".git").is_dir()

        if has_agents or has_skills or has_git:
            governance_root = str(current)
            break
        current = current.parent

    check_dir = Path(governance_root) if governance_root else root
    if (check_dir / "pyproject.toml").is_file() or (check_dir / "setup.py").is_file():
        stack.append("python")
    if (check_dir / "package.json").is_file():
        stack.append("node")
    if (check_dir / "Cargo.toml").is_file():
        stack.append("rust")
    if (check_dir / "Dockerfile").is_file() or (check_dir / "docker-compose.yml").is_file():
        stack.append("docker")

    is_sw = bool(governance_root or stack)
    return DomainClassification(
        is_software_task=is_sw,
        detected_stack=stack,
        governance_repo_root=governance_root
    )


class CognitiveTriage:
    """
    Cognitive Semantic Interpreter.
    Eliminates all keyword lists: every message is evaluated as an interpretive act.
    Synthesizes dual-form queries and capability nouns to maximize skill retrieval accuracy.
    """

    def __init__(self, ctx: Any = None, model_fn: Optional[Callable[[str], str]] = None):
        self.ctx = ctx
        self.model_fn = model_fn

    def analyze(self, user_message: str, cwd: Optional[str] = None) -> TriageAnalysis:
        clean_text = user_message.strip()
        domain_info = inspect_workspace_governance(cwd)

        # 1. Execute cognitive interpretation via model
        llm_output = self._call_model(clean_text)

        # 2. Parse and validate structured output
        analysis = self._parse_structured_output(llm_output, clean_text, domain_info)
        return analysis

    def _call_model(self, user_text: str) -> Optional[str]:
        """Invokes the model facade (ctx.llm, custom model_fn, or structured prompt)."""
        hermes_locale = get_hermes_language()
        locale_directive = f"\n[Hermes Configured Locale: {hermes_locale}]\n" if hermes_locale else ""
        prompt = f"{TRIAGE_SYSTEM_PROMPT}{locale_directive}\n\nUSER MESSAGE:\n\"{user_text}\"\n\nJSON RESPONSE:"

        # Case A: Custom model callable injected (useful for tests or dedicated local LLM)
        if self.model_fn is not None:
            try:
                return self.model_fn(prompt)
            except Exception:
                pass

        # Case B: Host LLM facade from Hermes Agent (ctx.llm)
        if self.ctx is not None and hasattr(self.ctx, "llm") and self.ctx.llm is not None:
            try:
                # Hermes Agent PluginLlm.complete expects messages: list[dict[str, Any]]
                resp = self.ctx.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.0
                )
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    return resp.text
                elif isinstance(resp, str):
                    return resp
                return None
            except Exception:
                pass

        return None

    def _parse_structured_output(
        self,
        raw_output: Optional[str],
        original_text: str,
        domain_info: DomainClassification
    ) -> TriageAnalysis:
        """
        Parses and self-heals structured JSON from the model response.
        If no model output is available (e.g. offline fallback), performs grammatical
        intent induction rather than hardcoded word matching.
        """
        if raw_output:
            match = re.search(r"\{[\s\S]*\}", raw_output)
            if match:
                try:
                    data = json.loads(match.group(0))

                    # Parse route or derive from legacy boolean signals
                    raw_route = str(data.get("route", "")).strip().upper()
                    if raw_route in CognitiveRoute.__members__:
                        route = CognitiveRoute(raw_route)
                    else:
                        legacy_act = bool(data.get("is_actionable_task", True))
                        legacy_skills = bool(data.get("needs_skills", legacy_act))
                        legacy_mem = bool(data.get("needs_memory", False))
                        if not legacy_act or (not legacy_skills and not legacy_mem):
                            route = CognitiveRoute.DIRECT_REPLY
                        elif legacy_skills and legacy_mem:
                            route = CognitiveRoute.ENRICHED_TASK
                        elif legacy_mem:
                            route = CognitiveRoute.MEMORY_ONLY
                        else:
                            route = CognitiveRoute.SKILL_ONLY

                    is_actionable = (route != CognitiveRoute.DIRECT_REPLY)
                    needs_skills = (route in (CognitiveRoute.SKILL_ONLY, CognitiveRoute.ENRICHED_TASK))
                    needs_memory = (route in (CognitiveRoute.MEMORY_ONLY, CognitiveRoute.ENRICHED_TASK))

                    # Extract broad uncoupled semantic concepts
                    raw_concepts = data.get("semantic_concepts") or data.get("required_capabilities") or []
                    semantic_concepts = [str(c).strip().lower() for c in raw_concepts if str(c).strip()]

                    # Extract entities
                    entities = [str(e).strip() for e in data.get("extracted_entities", []) if str(e).strip()]

                    # Parse memory probe
                    memory_probe: Optional[MemoryProbe] = None
                    probe_data = data.get("memory_probe")
                    if isinstance(probe_data, dict):
                        ent = probe_data.get("entity") or (entities[0] if entities else None)
                        memory_probe = MemoryProbe(
                            entity=str(ent).strip() if ent else None,
                            topic=str(probe_data.get("topic") or "").strip() or None,
                            time_anchor=str(probe_data.get("time_anchor") or "").strip() or None,
                            query_text=str(probe_data.get("query_text") or original_text).strip()
                        )
                    elif needs_memory or entities:
                        memory_probe = MemoryProbe(
                            entity=entities[0] if entities else None,
                            topic=original_text[:100],
                            query_text=original_text
                        )

                    detected_lang = str(data.get("detected_language") or "auto").strip().lower()
                    if memory_probe:
                        memory_probe.detected_language = detected_lang

                    intent_en = str(data.get("intent_summary_en", "")).strip()
                    search_query = str(data.get("search_query", "")).strip()
                    suggested_missing = data.get("suggested_missing_skill")
                    suggested_missing_skill = str(suggested_missing).strip().lower() if suggested_missing else None
                    reasoning = str(data.get("reasoning", "")).strip()

                    is_sw = bool(data.get("is_software_task", domain_info.is_software_task))
                    domain_info.is_software_task = is_sw
                    for st in data.get("detected_stack", []):
                        if st not in domain_info.detected_stack:
                            domain_info.detected_stack.append(st)

                    final_query = search_query if needs_skills else ""
                    if needs_skills and not final_query:
                        final_query = intent_en

                    return TriageAnalysis(
                        route=route,
                        detected_language=detected_lang,
                        memory_probe=memory_probe,
                        semantic_concepts=semantic_concepts,
                        suggested_missing_skill=suggested_missing_skill,
                        is_actionable_task=is_actionable,
                        needs_skills=needs_skills,
                        needs_memory=needs_memory,
                        intent_summary_en=intent_en or original_text[:200],
                        search_query=final_query,
                        required_capabilities=semantic_concepts,
                        extracted_entities=entities,
                        domain=domain_info,
                        reasoning=reasoning or f"Classified as {route.value} by cognitive arbiter.",
                        confidence=0.95
                    )
                except Exception:
                    pass

        # Offline / zero-model grammatical induction fallback:
        # Evaluates sentence structure, clause predicates, and communicative purpose
        return self._grammatical_intent_induction(original_text, domain_info)

    def _grammatical_intent_induction(
        self,
        text: str,
        domain_info: DomainClassification
    ) -> TriageAnalysis:
        """
        Structural fallback used ONLY when the model endpoint is offline or unavailable.
        Zero hardcoded keyword dictionaries, word matching, or linguistic assumptions.
        Evaluates structural token length and quoted identifiers, defaulting to safe
        fail-closed behavior when intent cannot be cognitively reasoned.
        """
        stripped = text.strip()
        words = stripped.split()

        # Pure acknowledgements or trivial expressions (< 3 tokens without structure)
        if len(words) <= 2 and not any(c in stripped for c in [":", "/", "\\", "{", "}", "=", "<", ">", "?"]):
            return TriageAnalysis(
                route=CognitiveRoute.DIRECT_REPLY,
                memory_probe=None,
                semantic_concepts=[],
                suggested_missing_skill=None,
                is_actionable_task=False,
                needs_skills=False,
                needs_memory=False,
                intent_summary_en="Short non-actionable token.",
                search_query="",
                required_capabilities=[],
                extracted_entities=[],
                domain=domain_info,
                reasoning="Short conversational expression without task structure.",
                confidence=0.5
            )

        # Extract explicitly quoted identifiers or entities without linguistic assumptions
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        entities = list(quoted)

        # In offline fallback without LLM inference, route structurally based on presence of explicit entities
        if entities:
            route = CognitiveRoute.ENRICHED_TASK
            is_actionable = True
            needs_skills = True
            needs_memory = True
            memory_probe = MemoryProbe(
                entity=entities[0],
                query_text=stripped
            )
        else:
            route = CognitiveRoute.SKILL_ONLY
            is_actionable = True
            needs_skills = True
            needs_memory = False
            memory_probe = None

        return TriageAnalysis(
            route=route,
            memory_probe=memory_probe,
            semantic_concepts=[],
            suggested_missing_skill=None,
            is_actionable_task=is_actionable,
            needs_skills=needs_skills,
            needs_memory=needs_memory,
            intent_summary_en=f"Task: {stripped[:200]}",
            search_query="",
            required_capabilities=[],
            extracted_entities=entities[:5],
            domain=domain_info,
            reasoning=f"Offline structural fallback routed to {route.value} without keyword dictionaries.",
            confidence=0.5
        )
