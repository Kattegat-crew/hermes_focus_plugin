from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
    is_actionable_task: bool = Field(
        ...,
        description="False for trivial prompts, greetings, or acknowledgements; True for actionable user tasks."
    )
    needs_skills: bool = Field(
        default=True,
        description="True if external procedural skills should be retrieved to assist execution."
    )
    needs_memory: bool = Field(
        default=False,
        description="True if factual memory should be retrieved for detected entities."
    )
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
        description="3 to 6 technical capability nouns in English (e.g., ['campaign-planning', 'copywriting'])."
    )
    extracted_entities: List[str] = Field(
        default_factory=list,
        description="Named entities for selective memory retrieval (e.g. client names, campaigns, system IDs)."
    )
    domain: DomainClassification = Field(
        default_factory=lambda: DomainClassification(is_software_task=False)
    )
    reasoning: str = Field(
        default="",
        description="Brief cognitive explanation for the intent and triage decision."
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
    injected_skill_content: Optional[str] = None
    selected_memories: List[MemoryItem] = Field(default_factory=list)
    combined_context_block: Optional[str] = None
    triage_analysis: Optional[TriageAnalysis] = None
    tokens_saved_estimate: int = 0
    triage_latency_ms: float = 0.0
    router_backend: str = "host_llm"
    status: str = "success"
    message: str = ""


class ReceiptRecord(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn_id: str = ""
    session_id: str = ""
    is_task: bool = True
    domain: str = "general"
    entities_detected: List[str] = Field(default_factory=list)
    memory_injected_tokens: int = 0
    selected_skill: Optional[str] = None
    skill_is_project_local: bool = False
    skill_injected_tokens: int = 0
    router_backend: str = "host_llm"
    triage_latency_ms: float = 0.0
    tokens_saved_estimate: int = 0
    status: str = "success"
    error: Optional[str] = None
