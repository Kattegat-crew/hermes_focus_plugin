from __future__ import annotations

import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import get_hermes_language
from .models import CognitiveRoute, DomainClassification, MemoryProbe, TriageAnalysis
from .workspace import inspect_workspace_governance, resolve_workspace_dir


TRIAGE_SYSTEM_PROMPT = """You are the Cognitive Semantic Arbiter for Hermes Agent's Neural Context Engine.
You read user messages across ANY human language, platform channel, and domain of knowledge, performing deep cognitive intent reasoning.

================================================================================
SECTION 1: COGNITIVE MEMORY REASONING FRAMEWORK
================================================================================
When the user query implies recalling past interactions, agreements, client facts, configurations, or historical decisions:
1. Multidimensional Decomposition (Who, What, When, Where):
   - WHO / PROJECT: Extract specific client name, brand, repository, or project context (or null if general).
   - WHAT: Isolate the core informational object: agreement, architecture decision, credentials/keys, bug/incident, business constraint, or deliverable.
   - WHEN: Capture temporal clues in 'time_anchor' AND infer the language-agnostic integer 'time_window_hours' (approximate lookback horizon in hours, or null if timeless) plus 'recency_bias' (true when asking for latest/recent/current state, else false).
   - WHERE: Identify environment or platform scope when relevant.
2. Reasoning, Typo Repair & Intuitive Query Formulation:
   - Never copy raw user typos, broken grammar, or slang blindly. Reason about what the user genuinely meant and emit correctly spelled, high-signal terms alongside helpful domain synonyms and variants so the retrieval graph finds the record even when the user miswrote names or technical words.
   - Normalize colloquial speech into factual terminology likely stored in historical logs or databases, avoiding empty conversational filler across any language.
   - Choose freely whichever concrete terms and however many concepts provide the strongest discriminative signal.
3. Universal Multilingual & Cross-Lingual Recall:
   - Work seamlessly across ANY human language without restriction. Detect 'detected_language' openly as a BCP-47/ISO code (or 'auto').
   - Keep 'topic', 'time_anchor', and 'query_text' in the user's interaction language (with spelling/grammar cleaned up when needed).
   - Populate 'cross_lingual_terms' with intuitive equivalent technical nouns and key concepts across canonical English and '[Hermes Configured Locale]' (when configured and different from the user's message), plus any likely mixed-language technical terminology stored in historical databases.
4. Absence Certainty (Anti-Hallucination):
   Formulate the memory probe thoroughly across these dimensions so that if the memory search returns empty, the agent can be 100% certain that no prior record exists, preventing factual hallucination.

================================================================================
SECTION 2: COGNITIVE SKILL REASONING FRAMEWORK
================================================================================
When the user requests executing a procedural task, development action, technical analysis, or specialized tool workflow:
1. Concrete Lexical Expansion (Tools, Applications, Protocols & Domain Nouns):
   Reason beyond literal phrasing or user typos to emit concrete capability concepts, tool/application names, frameworks, protocols, and industry-standard technical nouns in canonical English representing the exact nature of the work across ANY field of knowledge. Choose as many or as few capabilities as genuinely match the full scope of the task.
2. Zero Boilerplate in Search Queries:
   Formulate 'search_query' purely as a high-density sequence of concrete skill nouns, application names, technical synonyms, and architectural terms that would appear in a relevant SKILL.md (never include conversational filler like 'When the user needs to...').
3. Skill Gap Guarantee:
   Because your capability terms and technical synonyms cast a comprehensive, high-recall net over the catalog, if the bipartite graph returns no matching skill candidate, you can authoritatively specify 'suggested_missing_skill' knowing with mathematical certainty that the catalog genuinely lacks this capability.

================================================================================
SECTION 3: 4-QUADRANT ROUTE CLASSIFICATION
================================================================================
Classify the turn into exactly ONE route:
- DIRECT_REPLY: The communicative intent is purely phatic, social courtesy, conversational acknowledgement, or open general dialogue that has no procedural task objective, requires no execution of specialized tools, and has no dependency on historical project facts (0 tokens injected). When the utterance serves an interpersonal social function rather than delegating an actionable technical objective, classify strictly as DIRECT_REPLY and leave semantic_concepts and search_query empty.
- SKILL_ONLY: The user explicitly requests executing a generic procedural action, tool workflow, or coding task WITHOUT any reference to a specific client, brand, or named project entity (e.g., 'how do I configure git rebase', 'escribe un script de ffmpeg para cortar video').
- MEMORY_ONLY: The user asks about historical facts, agreements, decisions, or past session context, WITHOUT requesting a new procedural tool execution or software implementation.
- ENRICHED_TASK: Any actionable technical, software, or procedural task that references a specific client, brand, project, repository, or named entity (e.g., 'para AcmeCorp', 'en el proyecto X', 'revisa y arregla el pipeline de Y', 'implementa el pixel CAPI de Meta para AcmeCorp'). Executes Phase 1 to retrieve domain ground-truth and architectural facts about that entity from memory, then executes Phase 2 memory-enriched skill routing to select and inject the winning procedural tool. Whenever an actionable request mentions a specific client or project entity, prioritize ENRICHED_TASK so domain facts ground the execution.

================================================================================
OUTPUT SPECIFICATION (Pure JSON only):
================================================================================
{
  "route": "DIRECT_REPLY" | "SKILL_ONLY" | "MEMORY_ONLY" | "ENRICHED_TASK",
  "detected_language": "<BCP-47 or ISO language code, or 'auto'>",
  "memory_probe": {
    "entity": "Proper noun / client / project name or null",
    "topic": "Normalized topic in interaction language or null",
    "time_anchor": "Temporal clue in interaction language or null",
    "time_window_hours": null,
    "recency_bias": false,
    "query_text": "Typo-repaired substantive search terms in interaction language",
    "cross_lingual_terms": ["equivalent technical terms in English and active Hermes locale when helpful"]
  },
  "extracted_entities": ["ProperNounEntity"],
  "semantic_concepts": ["concept1", "concept2"],
  "suggested_missing_skill": "skill-name or null",
  "intent_summary_en": "Standardized 1-sentence imperative description in English",
  "search_query": "Dense space-separated list of concrete tool names, applications, protocols, and technical domain nouns in English",
  "is_software_task": true/false,
  "detected_stack": [],
  "reasoning": "1 sentence cognitive explanation of the decision"
}
"""


def _normalize_turn_content(content: Any) -> str:
    """Safely extract text from omnichannel/multimodal turn content, dropping binary blobs."""
    if content is None:
        return ""
    if isinstance(content, str):
        stripped = content.strip()
        return "" if stripped.startswith("data:") else stripped
    if isinstance(content, (list, tuple)):
        parts = [_normalize_turn_content(item) for item in content]
        return " ".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        for key in ("text", "input_text", "caption"):
            val = content.get(key)
            if isinstance(val, str):
                stripped = val.strip()
                if stripped and not stripped.startswith("data:"):
                    return stripped
        for container_key in ("content", "parts"):
            if container_key in content and content[container_key] is not None:
                nested = _normalize_turn_content(content[container_key])
                if nested:
                    return nested
        return ""
    return str(content).strip()


def _scan_balanced_json(text: str) -> tuple[Optional[Dict[str, Any]], bool]:
    """
    Scan text for the first valid top-level JSON object using balanced brace/quote tracking.
    Returns (parsed_dict, has_unclosed_brace).
    """
    has_unclosed = False
    idx = text.find("{")
    while idx != -1:
        depth = 0
        in_quote = False
        escape = False
        parsed_here: Optional[Dict[str, Any]] = None

        for pos in range(idx, len(text)):
            ch = text[pos]
            if in_quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_quote = False
                continue

            if ch == '"':
                in_quote = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[idx : pos + 1]
                    try:
                        loaded = json.loads(candidate)
                        if isinstance(loaded, dict):
                            parsed_here = loaded
                            return parsed_here, False
                    except Exception:
                        pass
                    break

        if depth > 0 or in_quote:
            has_unclosed = True
        idx = text.find("{", idx + 1)

    return None, has_unclosed


def _extract_balanced_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract the first valid balanced JSON object from raw model output."""
    if not text:
        return None
    obj, _ = _scan_balanced_json(text)
    return obj


def _is_truncated_completion(text: Optional[str]) -> bool:
    """Returns True if model completion was truncated (empty string or unclosed/nested JSON)."""
    if text is None:
        return False
    stripped = text.strip()
    if not stripped:
        return True
    obj, has_unclosed = _scan_balanced_json(stripped)
    if obj is not None:
        return False
    return has_unclosed


class CognitiveTriage:
    """
    Cognitive Semantic Interpreter.
    Eliminates all keyword lists: every message is evaluated as an interpretive act.
    Synthesizes dual-form queries and capability nouns to maximize skill retrieval accuracy.
    """

    def __init__(
        self,
        ctx: Any = None,
        model_fn: Optional[Callable[[str], str]] = None,
        max_tokens: int = 0,
        timeout_ms: int = 0,
    ):
        self.ctx = ctx
        self.model_fn = model_fn
        self.max_tokens = max_tokens
        self.timeout_ms = timeout_ms

    def analyze(
        self,
        user_message: str,
        cwd: Optional[str] = None,
        software_inspection_enabled: bool = True,
        project_scan_paths: Optional[List[Union[str, Path]]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> TriageAnalysis:
        clean_text = user_message.strip()
        domain_info = inspect_workspace_governance(
            cwd,
            enabled=software_inspection_enabled,
            project_scan_paths=project_scan_paths,
        )

        history_lines: List[str] = []
        if conversation_history:
            recent = list(conversation_history)[-6:]
            for idx, turn in enumerate(recent):
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role") or "user").strip().lower()
                turn_text = _normalize_turn_content(turn.get("content"))
                if not turn_text:
                    continue
                if idx == len(recent) - 1 and role == "user" and turn_text == clean_text:
                    continue
                history_lines.append(f"{role}: {turn_text[:300]}")
        history_context = "\n".join(history_lines)

        # 1. Execute cognitive interpretation via model
        model_started = time.perf_counter()
        llm_output, is_truncated, model_error = self._call_model(clean_text, history_context=history_context)
        model_elapsed_ms = (time.perf_counter() - model_started) * 1000

        # 2. Parse and validate structured output
        analysis = self._parse_structured_output(
            llm_output, clean_text, domain_info, is_truncated=is_truncated, model_error=model_error
        )
        analysis.model_latency_ms = model_elapsed_ms

        # 3. Post-triage multi-project entity workspace resolution
        # If cwd is a static bot directory (Discord/Telegram), resolve named entities against project_scan_paths
        if software_inspection_enabled and project_scan_paths and analysis.extracted_entities:
            resolved_domain = inspect_workspace_governance(
                cwd=cwd,
                enabled=True,
                entities=analysis.extracted_entities,
                project_scan_paths=project_scan_paths,
            )
            if resolved_domain.is_software_task and resolved_domain.resolved_workspace_path:
                analysis.domain = resolved_domain
                # If triage classified as generic procedural task without client reference, but target
                # entity is a verified sovereign workspace, upgrade to ENRICHED_TASK
                if analysis.route == CognitiveRoute.SKILL_ONLY:
                    analysis.route = CognitiveRoute.ENRICHED_TASK
                    analysis.needs_memory = True

        return analysis

    def _call_model(self, user_text: str, history_context: str = "") -> tuple[Optional[str], bool, Optional[str]]:
        """Invokes the model facade (ctx.llm, custom model_fn, or structured prompt) with adaptive deadline & tokens."""
        hermes_locale = get_hermes_language()
        locale_directive = f"\n[Hermes Configured Locale: {hermes_locale}]\n" if hermes_locale else ""
        history_block = f"\nRECENT CONVERSATION:\n{history_context}\n" if history_context else ""
        prompt = f"{TRIAGE_SYSTEM_PROMPT}{locale_directive}{history_block}\n\nUSER MESSAGE:\n\"{user_text}\"\n\nJSON RESPONSE:"
        timeout_s: Optional[float] = (self.timeout_ms / 1000.0) if (self.timeout_ms and self.timeout_ms > 0) else None

        # Compute dynamic token budget based on input complexity (preserves exact base for short prompts)
        effective_max_tokens = self.max_tokens + max(0, len(user_text) - 200) * 2 if self.max_tokens > 0 else 0

        effective_timeout_s = timeout_s
        failure: Optional[str] = None

        # Case A: Custom model callable injected (useful for tests or dedicated local LLM)
        if self.model_fn is not None:
            try:
                if effective_timeout_s is not None:
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(self.model_fn, prompt)
                    try:
                        out = future.result(timeout=effective_timeout_s)
                    except concurrent.futures.TimeoutError:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return None, False, "model_timeout"
                    finally:
                        executor.shutdown(wait=False)
                else:
                    out = self.model_fn(prompt)
                truncated = _is_truncated_completion(out)
                return out, truncated, None
            except Exception as exc:
                failure = f"model_exception:{type(exc).__name__}"

        # Case B: Host LLM facade from Hermes Agent (ctx.llm)
        if self.ctx is not None and hasattr(self.ctx, "llm") and self.ctx.llm is not None:
            try:
                t0 = time.perf_counter()
                complete_kwargs: Dict[str, Any] = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }
                if effective_max_tokens > 0:
                    complete_kwargs["max_tokens"] = effective_max_tokens
                if effective_timeout_s is not None:
                    complete_kwargs["timeout"] = effective_timeout_s
                resp = self.ctx.llm.complete(**complete_kwargs)
                text = resp.text if hasattr(resp, "text") and isinstance(resp.text, str) else (resp if isinstance(resp, str) else None)

                # Retry once on truncation even when max_tokens == 0 (unconstrained budget)
                if _is_truncated_completion(text):
                    try:
                        retry_prompt = (
                            f"{prompt}\n[IMPORTANT: Previous output was truncated. "
                            "Return ONLY compact, single-line valid JSON without reasoning prose or fences.]"
                        )
                        retry_kwargs: Dict[str, Any] = {
                            "messages": [{"role": "user", "content": retry_prompt}],
                            "temperature": 0.0,
                        }
                        if effective_max_tokens > 0:
                            retry_kwargs["max_tokens"] = effective_max_tokens * 2
                        if effective_timeout_s is not None:
                            elapsed = time.perf_counter() - t0
                            remaining_s = effective_timeout_s - elapsed
                            if remaining_s <= 0:
                                return text, True, "model_truncated"
                            retry_kwargs["timeout"] = remaining_s
                        retry_resp = self.ctx.llm.complete(**retry_kwargs)
                        retry_text = retry_resp.text if hasattr(retry_resp, "text") and isinstance(retry_resp.text, str) else (retry_resp if isinstance(retry_resp, str) else None)
                        if not _is_truncated_completion(retry_text):
                            return retry_text, False, None
                        return retry_text, True, "model_truncated"
                    except Exception as exc:
                        return text, True, f"model_retry_exception:{type(exc).__name__}"

                return text, False, None
            except Exception as exc:
                failure = f"model_exception:{type(exc).__name__}"

        return None, False, failure or "model_unavailable"

    def _parse_structured_output(
        self,
        raw_output: Optional[str],
        original_text: str,
        domain_info: DomainClassification,
        is_truncated: bool = False,
        model_error: Optional[str] = None,
    ) -> TriageAnalysis:
        """
        Parses and self-heals structured JSON from the model response.
        If no model output is available (e.g. offline fallback), performs grammatical
        intent induction rather than hardcoded word matching.
        """
        if raw_output:
            data = _extract_balanced_json(raw_output)
            if isinstance(data, dict):
                try:

                    # Parse route or derive from legacy boolean signals
                    raw_route = str(data.get("route", "")).strip().upper()
                    if raw_route in CognitiveRoute.__members__:
                        route = CognitiveRoute(raw_route)
                    elif "is_actionable_task" in data or "needs_skills" in data or "needs_memory" in data:
                        legacy_act = bool(data.get("is_actionable_task", False))
                        legacy_skills = bool(data.get("needs_skills", False))
                        legacy_mem = bool(data.get("needs_memory", False))
                        if not legacy_act or (not legacy_skills and not legacy_mem):
                            route = CognitiveRoute.DIRECT_REPLY
                        elif legacy_skills and legacy_mem:
                            route = CognitiveRoute.ENRICHED_TASK
                        elif legacy_mem:
                            route = CognitiveRoute.MEMORY_ONLY
                        else:
                            route = CognitiveRoute.SKILL_ONLY
                    else:
                        # Fail-closed default: when response is empty, malformed, or ambiguous, default to DIRECT_REPLY
                        route = CognitiveRoute.DIRECT_REPLY

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
                        ent_str = str(ent).strip() if ent else None
                        if ent_str and ent_str not in entities:
                            entities.insert(0, ent_str)
                        raw_win = probe_data.get("time_window_hours")
                        time_window_hours: Optional[int] = None
                        if raw_win is not None and not isinstance(raw_win, bool):
                            try:
                                parsed_win = int(float(raw_win))
                                if parsed_win > 0:
                                    time_window_hours = parsed_win
                            except (ValueError, TypeError):
                                pass
                        raw_xling = probe_data.get("cross_lingual_terms") or []
                        xling_terms: List[str] = []
                        if isinstance(raw_xling, list):
                            for item in raw_xling:
                                term_s = str(item).strip()
                                if term_s and term_s not in xling_terms:
                                    xling_terms.append(term_s)
                        elif isinstance(raw_xling, str) and raw_xling.strip():
                            xling_terms = [raw_xling.strip()]
                        memory_probe = MemoryProbe(
                            entity=ent_str,
                            topic=str(probe_data.get("topic") or "").strip() or None,
                            time_anchor=str(probe_data.get("time_anchor") or "").strip() or None,
                            time_window_hours=time_window_hours,
                            recency_bias=bool(probe_data.get("recency_bias", False)),
                            query_text=str(
                                probe_data.get("query_text")
                                or " ".join(entities + semantic_concepts)
                                or data.get("intent_summary_en")
                                or ""
                            ).strip(),
                            cross_lingual_terms=xling_terms,
                        )
                    elif needs_memory or entities:
                        model_fallback_query = " ".join(entities + semantic_concepts) or str(data.get("intent_summary_en") or "").strip()
                        memory_probe = MemoryProbe(
                            entity=entities[0] if entities else None,
                            topic=model_fallback_query[:100] or None,
                            query_text=model_fallback_query
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
                        truncation_detected=is_truncated,
                        error="error_truncated" if is_truncated else model_error,
                        confidence=0.95
                    )
                except Exception:
                    pass

        # Fail-closed default when the model endpoint is unavailable, timed out, or returned invalid JSON:
        # User intent classification belongs exclusively to the model; never guess intent via regex heuristics.
        reason = (
            f"Model output truncated by max_tokens limit ({self.max_tokens} tokens) before structured JSON completed; failed closed to DIRECT_REPLY."
            if is_truncated
            else (f"{model_error}; failed closed to DIRECT_REPLY." if model_error else "invalid_model_response; failed closed to DIRECT_REPLY.")
        )
        return TriageAnalysis(
            route=CognitiveRoute.DIRECT_REPLY,
            memory_probe=None,
            semantic_concepts=[],
            suggested_missing_skill=None,
            is_actionable_task=False,
            needs_skills=False,
            needs_memory=False,
            intent_summary_en="",
            search_query="",
            required_capabilities=[],
            extracted_entities=[],
            domain=domain_info,
            reasoning=reason,
            truncation_detected=is_truncated,
            error="error_truncated" if is_truncated else (model_error or "invalid_model_response"),
            confidence=0.0,
        )
