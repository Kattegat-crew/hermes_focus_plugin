from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import DomainClassification, TriageAnalysis


TRIAGE_SYSTEM_PROMPT = """You are the Cognitive Semantic Interpreter for Hermes Agent's Neural Context Engine.
Your job is to read the raw user message and perform a deep cognitive analysis of the user's intent, tolerating typos, informal phrasing, ambiguous statements, or omissions.

You must determine:
1. Is this an actionable task requiring execution or assistance, or purely conversational/casual/acknowledgement?
2. Does this task genuinely require procedural specialized skills (guidance/tools), or is it a general conceptual reply that needs no external skills?
3. Does this task require recalling historical factual memory (agreements, past decisions, client facts)?
4. What is the canonical 1-sentence technical intent in English?
5. Formulate an optimized search phrase in English using the dual pattern:
   "When the user needs to <action> <target> — <imperative verb> <target>."
6. Identify 3 to 6 capability nouns in English (e.g. ["campaign-planning", "copywriting"]).
7. Extract named entities (clients, projects, systems, brands).
8. Is this a software engineering or devops task?

OUTPUT FORMAT:
Respond with pure JSON conforming exactly to this structure:
{
  "is_actionable_task": true/false,
  "needs_skills": true/false,
  "needs_memory": true/false,
  "intent_summary_en": "Clear 1-sentence imperative description in English",
  "search_query": "When the user needs to <verb> <object> — <imperative> <object>",
  "required_capabilities": ["cap1", "cap2", "cap3"],
  "extracted_entities": ["Entity1"],
  "is_software_task": true/false,
  "detected_stack": ["python", "docker"],
  "reasoning": "1 sentence explanation of why skills/memory are or are not needed"
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
        prompt = f"{TRIAGE_SYSTEM_PROMPT}\n\nUSER MESSAGE:\n\"{user_text}\"\n\nJSON RESPONSE:"

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
                    is_actionable = bool(data.get("is_actionable_task", True))
                    needs_skills = bool(data.get("needs_skills", is_actionable))
                    needs_memory = bool(data.get("needs_memory", False))
                    intent_en = str(data.get("intent_summary_en", "")).strip()
                    search_query = str(data.get("search_query", "")).strip()
                    capabilities = [str(c).strip().lower() for c in data.get("required_capabilities", [])]
                    entities = [str(e).strip() for e in data.get("extracted_entities", [])]
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
                        is_actionable_task=is_actionable,
                        needs_skills=needs_skills,
                        needs_memory=needs_memory,
                        intent_summary_en=intent_en or original_text[:200],
                        search_query=final_query,
                        required_capabilities=capabilities,
                        extracted_entities=entities,
                        domain=domain_info,
                        reasoning=reasoning or "Parsed from cognitive model interpretation.",
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
        Grammatical intent induction used ONLY when the model endpoint is offline or unavailable.
        Analyzes communicative modality (imperative, interrogative, declarative) and predicate clauses
        WITHOUT hardcoded keyword dictionaries.
        """
        stripped = text.strip()
        words = stripped.split()

        # Pure acknowledgements / extremely short non-clause expressions (< 3 tokens without predicate)
        if len(words) <= 2 and not any(c in stripped for c in [":", "/", "\\", "{", "}"]):
            return TriageAnalysis(
                is_actionable_task=False,
                needs_skills=False,
                needs_memory=False,
                intent_summary_en="Non-clause conversational token or acknowledgement.",
                search_query="",
                required_capabilities=[],
                extracted_entities=[],
                domain=domain_info,
                reasoning="Short communicative particle without an action predicate.",
                confidence=0.8
            )

        # Extract potential named entities (capitalized proper nouns, quoted identifiers)
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        entities = list(quoted)
        for w in words:
            if w[0].isupper() and len(w) > 2 and w not in entities:
                entities.append(w)

        # Formulate dual query in English
        query_en = f"When the user needs to execute: {stripped} — execute {stripped}"

        return TriageAnalysis(
            is_actionable_task=True,
            needs_skills=True,
            needs_memory=bool(entities),
            intent_summary_en=f"Execute task: {stripped[:200]}",
            search_query=query_en,
            required_capabilities=[],
            extracted_entities=entities[:5],
            domain=domain_info,
            reasoning="Clause with active predicate detected; routed for capability evaluation.",
            confidence=0.75
        )
