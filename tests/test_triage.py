import json
import pytest
from neural_context.triage import CognitiveTriage, inspect_workspace_governance


def test_triage_model_cognitive_interpretation():
    # Simulate a cognitive LLM response for an ambiguous real-world user query with typos
    def mock_model(prompt: str) -> str:
        assert "USER MESSAGE" in prompt
        return json.dumps({
            "is_actionable_task": True,
            "needs_skills": True,
            "needs_memory": True,
            "intent_summary_en": "Audit conversion tracking issues and lead drops for AcmeCorp client",
            "search_query": "When the user needs to audit conversion funnels — audit conversion tracking and capi leads",
            "required_capabilities": ["meta-pixel", "conversions-api", "event-deduplication"],
            "extracted_entities": ["AcmeCorp", "Meta"],
            "is_software_task": False,
            "detected_stack": [],
            "reasoning": "User noticed a lead drop for AcmeCorp and needs conversion audit procedures and past campaign memory."
        })

    triage = CognitiveTriage(model_fn=mock_model)
    # Ambiguous prompt in Spanish with typos
    result = triage.analyze("mira q se cayeron los leds de la campana de acmecorp en meta revisate eso")

    assert result.is_actionable_task is True
    assert result.needs_skills is True
    assert result.needs_memory is True
    assert "conversions-api" in result.required_capabilities
    assert "AcmeCorp" in result.extracted_entities
    assert "When the user needs to" in result.search_query


def test_triage_conversational_casual_no_skills_needed():
    # Simulate LLM detecting conversational chit-chat / greeting
    def mock_model(prompt: str) -> str:
        return json.dumps({
            "is_actionable_task": False,
            "needs_skills": False,
            "needs_memory": False,
            "intent_summary_en": "Greeting and inquiry about general status.",
            "search_query": "",
            "required_capabilities": [],
            "extracted_entities": [],
            "is_software_task": False,
            "detected_stack": [],
            "reasoning": "Purely conversational message without actionable task or tool needs."
        })

    triage = CognitiveTriage(model_fn=mock_model)
    result = triage.analyze("Hola que tal como te va hoy")

    assert result.is_actionable_task is False
    assert result.needs_skills is False
    assert result.needs_memory is False
    assert result.search_query == ""


def test_triage_software_task_with_governance_detection():
    def mock_model(prompt: str) -> str:
        return json.dumps({
            "is_actionable_task": True,
            "needs_skills": True,
            "needs_memory": False,
            "intent_summary_en": "Write unit tests for FastAPI router and dockerize the container",
            "search_query": "When the user needs to write unit tests — test fastapi and dockerize service",
            "required_capabilities": ["pytest", "fastapi", "docker-compose"],
            "extracted_entities": [],
            "is_software_task": True,
            "detected_stack": ["python", "docker"],
            "reasoning": "Development task requiring testing and containerization skills."
        })

    triage = CognitiveTriage(model_fn=mock_model)
    result = triage.analyze("armate los tests del router y dejalo listo en docker")

    assert result.is_actionable_task is True
    assert result.domain.is_software_task is True
    assert "pytest" in result.required_capabilities
    assert result.needs_skills is True


def test_triage_offline_grammatical_induction():
    # When no model is provided (e.g. offline fallback), tests grammatical induction
    triage = CognitiveTriage()
    result = triage.analyze("Escribe un informe de auditoría para el cliente 'AcmeCorp'")

    assert result.is_actionable_task is True
    assert "AcmeCorp" in result.extracted_entities
    assert result.needs_skills is True
