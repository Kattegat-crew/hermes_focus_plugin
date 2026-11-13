from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class CognitiveRoute(str, Enum):
    """
    Explicit 4-quadrant cognitive routing policy:
    - DIRECT_REPLY: Casual chat, greetings, simple conceptual Q&A (0 tokens injected).
    - SKILL_ONLY: Procedural tasks requiring tools or execution skills, without historical memory dependency.
    - MEMORY_ONLY: Historical recall, agreements, facts or past session context, without procedural tools.
    - ENRICHED_TASK: Complex domain tasks tied to history/clients (two-phase: memory facts -> skill routing).
    """
    DIRECT_REPLY = "DIRECT_REPLY"
    SKILL_ONLY = "SKILL_ONLY"
    MEMORY_ONLY = "MEMORY_ONLY"
    ENRICHED_TASK = "ENRICHED_TASK"


class MemoryProbe(BaseModel):
    """
    Targeted probe for selective factual memory retrieval across any Hermes-supported language.
    Preserves interaction-language terms while optionally bridging cross-lingual terms and temporal bounds.
    """
    entity: Optional[str] = Field(
        default=None,
        description="Named entity, brand, client or project if mentioned."
    )
    topic: Optional[str] = Field(
        default=None,
        description="Core subject, agreement or topic in the native interaction language."
    )
    time_anchor: Optional[str] = Field(
        default=None,
        description="Temporal reference in the native language (e.g. yesterday, last sprint, previous turn)."
    )
    time_window_hours: Optional[int] = Field(
        default=None,
        description="Language-agnostic lookback window in hours inferred from time_anchor (e.g. 48 for yesterday, 168 for last week), or null."
    )
    recency_bias: bool = Field(
        default=False,
        description="True when the request asks for the latest/most recent facts or history even without an exact hour bound."
    )
    query_text: str = Field(
        default="",
        description="Search phrase preserved in the interaction language for lexical/semantic memory matching."
    )
    cross_lingual_terms: List[str] = Field(
        default_factory=list,
        description="Equivalent high-entropy search terms in canonical English and active Hermes locale when different from the interaction language."
    )
    detected_language: str = Field(
        default="auto",
        description="BCP-47/ISO language code detected from the interaction, or 'auto'."
    )


class DomainClassification(BaseModel):
    is_software_task: bool = Field(
        ...,
        description="True if the user request involves software development, debugging, git, devops, or codebase analysis."
    )
    detected_stack: List[str] = Field(
        default_factory=list,
        description="Detected technologies or stack descriptors (e.g. ['python', 'fastapi', 'docker'])."
    )
    governance_repo_root: Optional[str] = Field(
        default=None,
        description="Path to git root if workspace contains AGENTS.md or local .skills/ directory."
    )


class TriageAnalysis(BaseModel):
    route: CognitiveRoute = Field(
        default=CognitiveRoute.SKILL_ONLY,
        description="Explicit 4-quadrant cognitive route decided by the Arbiter."
    )
    detected_language: str = Field(
        default="auto",
        description="Detected ISO language code of the interaction."
    )
    memory_probe: Optional[MemoryProbe] = Field(
        default=None,
        description="Structured probe for selective memory search in user's native language."
    )
    semantic_concepts: List[str] = Field(
        default_factory=list,
        description="Broad, dynamic capability and domain concepts inferred by the model across any field of knowledge."
    )
    suggested_missing_skill: Optional[str] = Field(
        default=None,
        description="Recommended skill capability to install when no catalog skill matches the task."
    )
    is_actionable_task: bool = Field(
        default=True,
        description="False for trivial prompts, greetings, or acknowledgements; True for actionable user tasks."
    )
    needs_skills: bool = Field(
        default=True,
        description="True if external procedural skills should be retrieved to assist execution."
    )
    needs_memory: bool = Field(
        default=False,
        description="True if factual memory should be retrieved."
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_route_and_flags(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "route" not in data or data.get("route") is None:
                is_act = data.get("is_actionable_task", True)
                n_skills = data.get("needs_skills", is_act)
                n_mem = data.get("needs_memory", False)
                if not is_act or (not n_skills and not n_mem):
                    data["route"] = CognitiveRoute.DIRECT_REPLY
                elif n_skills and n_mem:
                    data["route"] = CognitiveRoute.ENRICHED_TASK
                elif n_mem:
                    data["route"] = CognitiveRoute.MEMORY_ONLY
                else:
                    data["route"] = CognitiveRoute.SKILL_ONLY
            else:
                r = data["route"]
                if isinstance(r, str) and r in CognitiveRoute.__members__:
                    r = CognitiveRoute(r)
                if r == CognitiveRoute.DIRECT_REPLY:
                    data["is_actionable_task"] = False
                    data["needs_skills"] = False
                    data["needs_memory"] = False
                elif r == CognitiveRoute.SKILL_ONLY:
                    data["is_actionable_task"] = True
                    data["needs_skills"] = True
                    data["needs_memory"] = False
                elif r == CognitiveRoute.MEMORY_ONLY:
                    data["is_actionable_task"] = False
                    data["needs_skills"] = False
                    data["needs_memory"] = True
                elif r == CognitiveRoute.ENRICHED_TASK:
                    data["is_actionable_task"] = True
                    data["needs_skills"] = True
                    data["needs_memory"] = True
        return data
    intent_summary_en: str = Field(
        default="",
        max_length=280,
        description="Standardized 1-sentence imperative description in English of the target task."
    )
    search_query: str = Field(
        default="",
        description="Optimized retrieval query in English formatted with imperative verbs and situations."
    )
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="3 to 6 technical capability nouns in English."
    )
    extracted_entities: List[str] = Field(
        default_factory=list,
        description="Named entities for selective memory retrieval."
    )
    domain: DomainClassification = Field(
        default_factory=lambda: DomainClassification(is_software_task=False)
    )
    reasoning: str = Field(
        default="",
        description="Brief cognitive explanation for the intent and triage decision."
    )
    truncation_detected: bool = Field(
        default=False,
        description="True if model response was truncated due to max_tokens limit exhaustion."
    )
    model_latency_ms: float = Field(
        default=0.0,
        description="Elapsed time spent in the cognitive model call, measured by this plugin.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if cognitive interpretation failed or degraded."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the cognitive interpretation."
    )


class SkillCandidate(BaseModel):
    name: str
    path: str
    description: str = ""
    lead_paragraph: str = ""
    sha256: str = ""
    is_project_local: bool = False
    score: float = 0.0


class MemoryItem(BaseModel):
    entity: str
    fact: str
    source_file: str = ""
    token_estimate: int = 0


class RouteResult(BaseModel):
    selected_skill: Optional[SkillCandidate] = None
    selected_skills: List[SkillCandidate] = Field(default_factory=list)
    injected_skill_content: Optional[str] = None
    selected_memories: List[MemoryItem] = Field(default_factory=list)
    combined_context_block: Optional[str] = None
    suggested_missing_skill: Optional[str] = None
    triage_analysis: Optional[TriageAnalysis] = None
    tokens_saved_estimate: int = 0
    triage_latency_ms: float = 0.0
    coordination_latency_ms: float = 0.0
    router_backend: Optional[str] = None
    configured_router_backend: str = "host_llm"
    status: str = "success"
    message: str = ""


class ReceiptRecord(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn_id: str = ""
    session_id: str = ""
    platform: str = "cli"
    model: str = ""
    route: str = "SKILL_ONLY"
    is_task: bool = True
    domain: str = "general"
    entities_detected: List[str] = Field(default_factory=list)
    memory_injected_tokens: int = 0
    selected_skill: Optional[str] = None
    suggested_missing_skill: Optional[str] = None
    skill_is_project_local: bool = False
    skill_injected_tokens: int = 0
    router_backend: Optional[str] = None
    configured_router_backend: str = "host_llm"
    triage_latency_ms: float = 0.0
    coordination_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    rendered_context_tokens: int = 0
    context_overhead_tokens: int = 0
    tokens_saved_estimate: int = 0
    status: str = "success"
    error: Optional[str] = None
