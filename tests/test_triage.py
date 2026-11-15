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


def test_unavailable_model_reports_cause_and_elapsed_time(tmp_path):
    triage = CognitiveTriage()

    result = triage.analyze("Please inspect the deployment plan", cwd=str(tmp_path))

    assert result.error == "model_unavailable"
    assert result.model_latency_ms >= 0
    assert result.reasoning == "model_unavailable; failed closed to DIRECT_REPLY."


def test_workspace_inspection_can_be_disabled(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    domain = inspect_workspace_governance(str(repo), enabled=False)

    assert domain.governance_repo_root is None
    assert domain.is_software_task is False


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


def test_triage_offline_fails_closed_to_direct_reply():
    # When no model is provided (e.g. offline), triage must never guess intent via regexes;
    # it must fail closed to DIRECT_REPLY with zero context injected.
    from neural_context.models import CognitiveRoute

    triage = CognitiveTriage()
    result = triage.analyze("Escribe un informe de auditoría para el cliente 'AcmeCorp'")

    assert result.route == CognitiveRoute.DIRECT_REPLY
    assert result.is_actionable_task is False
    assert result.needs_skills is False
    assert result.needs_memory is False


def test_triage_multidimensional_memory_framework():
    """
    Tests that the Cognitive Arbiter decomposes memory inquiries multidimensionally:
    - Normalizes colloquial phrasing ("lo que quedamos") to documentation vocabulary ("acuerdos").
    - Preserves native interaction language ('es').
    - Dispatches MEMORY_ONLY route with exact topic and temporal anchor.
    """
    import json
    from neural_context.models import CognitiveRoute

    def mock_model(prompt: str) -> str:
        return json.dumps({
            "route": "MEMORY_ONLY",
            "detected_language": "es",
            "memory_probe": {
                "entity": None,
                "topic": "acuerdos entrega api backend",
                "time_anchor": "ayer",
                "query_text": "acuerdos entrega api backend viernes"
            },
            "semantic_concepts": [],
            "suggested_missing_skill": None,
            "intent_summary_en": "Recall API delivery agreements made yesterday",
            "search_query": "",
            "is_software_task": False,
            "detected_stack": [],
            "reasoning": "Memory recall inquiry regarding yesterday agreements without tool execution."
        })

    triage = CognitiveTriage(model_fn=mock_model)
    result = triage.analyze("¿Qué fue lo que quedamos ayer sobre la entrega de la API?")

    assert result.route == CognitiveRoute.MEMORY_ONLY
    assert result.detected_language == "es"
    assert result.needs_memory is True
    assert result.needs_skills is False
    assert result.memory_probe is not None
    assert result.memory_probe.detected_language == "es"
    assert "acuerdos" in result.memory_probe.topic
    assert result.memory_probe.time_anchor == "ayer"


def test_sanitize_entity_name_generic():
    from neural_context.workspace import sanitize_entity_name

    assert sanitize_entity_name("payment-api") == "payment-api"
    assert sanitize_entity_name("data_worker") == "data_worker"
    assert sanitize_entity_name("   web-app   ") == "web-app"

    # Path traversal and injection defense
    assert sanitize_entity_name(".") is None
    assert sanitize_entity_name(".git") is None
    assert sanitize_entity_name("a") is None
    assert sanitize_entity_name("../../../etc/shadow") is None
    assert sanitize_entity_name("sub/path") is None
    assert sanitize_entity_name("sub\\path") is None
    assert sanitize_entity_name("null\x00byte") is None
    assert sanitize_entity_name("") is None
    assert sanitize_entity_name("x" * 150) is None


def test_resolve_workspace_multi_project_generic(tmp_path):
    from neural_context.workspace import resolve_workspace_dir

    projects_root = tmp_path / "projects"
    projects_root.mkdir()

    repo_a = projects_root / "payment-api"
    repo_a.mkdir()
    (repo_a / "package.json").write_text(json.dumps({"name": "payment-api"}))
    (repo_a / "AGENTS.md").write_text("# Payment API Governance\n")

    repo_b = projects_root / "data_worker"
    repo_b.mkdir()
    (repo_b / "pyproject.toml").write_text("[project]\nname = 'data-worker'\n")

    bot_cwd = tmp_path / "bot_home"
    bot_cwd.mkdir()

    # 1. Exact entity resolution
    res_a = resolve_workspace_dir(
        cwd=bot_cwd,
        entities=["payment-api"],
        project_scan_paths=[projects_root],
    )
    assert res_a == repo_a.resolve()

    # 2. Slugified entity resolution (data-worker vs data_worker)
    res_b = resolve_workspace_dir(
        cwd=bot_cwd,
        entities=["data-worker"],
        project_scan_paths=[projects_root],
    )
    assert res_b == repo_b.resolve()

    # 3. Path traversal attack falls back cleanly
    res_evil = resolve_workspace_dir(
        cwd=bot_cwd,
        entities=["../../../../etc/passwd"],
        project_scan_paths=[projects_root],
    )
    assert res_evil == bot_cwd.resolve()

    # 4. Unknown entity falls back to cwd
    res_unknown = resolve_workspace_dir(
        cwd=bot_cwd,
        entities=["unknown-service"],
        project_scan_paths=[projects_root],
    )
    assert res_unknown == bot_cwd.resolve()


def test_triage_entity_workspace_grounding(tmp_path):
    from neural_context.models import CognitiveRoute

    projects_root = tmp_path / "workspace"
    projects_root.mkdir()

    target_repo = projects_root / "reporting-api"
    target_repo.mkdir()
    (target_repo / "package.json").write_text(json.dumps({"name": "reporting"}))
    (target_repo / "AGENTS.md").write_text("# Reporting API\n")

    bot_cwd = tmp_path / "bot_home"
    bot_cwd.mkdir()

    # Case A: Actionable task naming an entity upgrades to ENRICHED_TASK with resolved stack
    def mock_model_task(prompt: str) -> str:
        return json.dumps({
            "route": "SKILL_ONLY",
            "is_actionable_task": True,
            "needs_skills": True,
            "needs_memory": False,
            "intent_summary_en": "Fix database timeout in reporting-api",
            "search_query": "database timeout query optimize",
            "semantic_concepts": ["database", "sql"],
            "required_capabilities": ["database-query"],
            "extracted_entities": ["reporting-api"],
            "is_software_task": False,
            "detected_stack": [],
            "reasoning": "Database fix for reporting-api",
        })

    triage_task = CognitiveTriage(model_fn=mock_model_task)
    analysis = triage_task.analyze(
        "Corrige el timeout de base de datos en reporting-api",
        cwd=bot_cwd,
        software_inspection_enabled=True,
        project_scan_paths=[projects_root],
    )

    assert analysis.route == CognitiveRoute.ENRICHED_TASK
    assert analysis.needs_memory is True
    assert analysis.domain.is_software_task is True
    assert "node" in analysis.domain.detected_stack
    assert analysis.domain.resolved_workspace_path == str(target_repo.resolve())

    # Case B: Pure greeting mentioning an entity MUST NOT be upgraded (preserves DIRECT_REPLY)
    def mock_model_greeting(prompt: str) -> str:
        return json.dumps({
            "route": "DIRECT_REPLY",
            "is_actionable_task": False,
            "needs_skills": False,
            "needs_memory": False,
            "intent_summary_en": "Casual greeting mentioning reporting-api",
            "search_query": "",
            "semantic_concepts": [],
            "required_capabilities": [],
            "extracted_entities": ["reporting-api"],
            "is_software_task": False,
            "detected_stack": [],
            "reasoning": "Purely conversational message without actionable task.",
        })

    triage_greeting = CognitiveTriage(model_fn=mock_model_greeting)
    greeting_res = triage_greeting.analyze(
        "Hola gente de reporting-api como van",
        cwd=bot_cwd,
        software_inspection_enabled=True,
        project_scan_paths=[projects_root],
    )

    assert greeting_res.route == CognitiveRoute.DIRECT_REPLY
    assert greeting_res.needs_skills is False
    assert greeting_res.needs_memory is False
