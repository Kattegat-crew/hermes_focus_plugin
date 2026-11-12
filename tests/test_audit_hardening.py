from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neural_context.cli import _coerce_and_validate, build_parser, handle_cli
from neural_context.config import PluginConfig
from neural_context.memory import SelectiveMemoryRetriever, is_safe_memory_file
from neural_context.models import CognitiveRoute, DomainClassification, MemoryProbe
from neural_context.router import ContextCoordinator, GraphRouter
from neural_context.sidecar import SidecarIndex
from neural_context.triage import CognitiveTriage
from __init__ import discover_all_skill_paths


def test_triage_empty_model_response_fails_closed_to_direct_reply():
    """Ensure an empty JSON response '{}' or missing route defaults safely to DIRECT_REPLY."""
    triage = CognitiveTriage()
    domain = DomainClassification(is_software_task=False)

    # Empty JSON object from model
    result_empty = triage._parse_structured_output("{}", "Hola, ¿cómo estás?", domain)
    assert result_empty.route == CognitiveRoute.DIRECT_REPLY
    assert result_empty.needs_skills is False
    assert result_empty.needs_memory is False

    # Malformed text
    result_malformed = triage._parse_structured_output("not json", "Hola amigo", domain)
    assert result_malformed.route == CognitiveRoute.DIRECT_REPLY
    assert result_malformed.needs_skills is False


def test_conversational_text_structural_fallback_defaults_to_direct_reply():
    """When model output is absent, even inside a software workspace, triage fails closed to DIRECT_REPLY."""
    triage = CognitiveTriage()
    domain = DomainClassification(is_software_task=True)

    res = triage._parse_structured_output(None, "Hola: ¿cómo estás hoy?", domain)
    assert res.route == CognitiveRoute.DIRECT_REPLY
    assert res.is_actionable_task is False
    assert res.needs_skills is False


def test_memory_discriminative_matching_prevents_spurious_single_word_matches(tmp_path: Path, monkeypatch):
    """Ensure multi-term query requires cooccurrence of discriminative terms, preventing spurious single-word matches without language stopword lists."""
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: empty_home)

    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text(
        "- Decidimos usar arquitectura hexagonal en el core del sistema\n"
        "- Las credenciales de produccion fueron rotadas ayer\n",
        encoding="utf-8"
    )

    retriever = SelectiveMemoryRetriever(token_budget=0)
    # Query with preposition "del" and "sobre" - should not match unless discriminative terms co-occur
    probe = MemoryProbe(
        query_text="cual es el estado del servidor",
        topic="del"
    )

    results = retriever.retrieve(probe=probe, cwd=str(tmp_path))
    # Cooccurrence requires min(2, len(topic_matchers)) matches; isolated match for 'del' does not trigger
    assert len(results) == 0


def test_sqlite_memory_escapes_like_wildcards(tmp_path: Path, monkeypatch):
    """Ensure SQLite memory inspection escapes % and _ wildcards."""
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: empty_home)

    db_path = tmp_path / "memories.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE memories (content TEXT)")
        conn.execute("INSERT INTO memories VALUES ('100% discount promotion campaign')")
        conn.execute("INSERT INTO memories VALUES ('Standard database configuration')")
        conn.commit()

    retriever = SelectiveMemoryRetriever(token_budget=0)
    # Search for entity with wildcard '%'; should match literal '100%' without broad match
    items = retriever.retrieve(entities=["100%"], cwd=str(tmp_path))
    assert len(items) == 1
    assert "100% discount" in items[0].fact


def test_cli_config_type_validation_rejects_invalid_values(tmp_path: Path, capsys):
    """Ensure hermes focus config --set rejects unknown keys or invalid types."""
    parser = build_parser()
    config = PluginConfig()

    # Test unknown key
    args_bad_key = parser.parse_args(["config", "--set", "unknown_key=123"])
    ret1 = handle_cli(args_bad_key, config, None, None)
    assert ret1 == 1
    out1 = capsys.readouterr().out
    assert "Unknown configuration setting" in out1

    # Test invalid int (bananas)
    args_bad_int = parser.parse_args(["config", "--set", "timeout_ms=bananas"])
    ret2 = handle_cli(args_bad_int, config, None, None)
    assert ret2 == 1
    out2 = capsys.readouterr().out
    assert "expected integer, got 'bananas'" in out2

    # Test invalid bool
    args_bad_bool = parser.parse_args(["config", "--set", "enabled=maybe"])
    ret3 = handle_cli(args_bad_bool, config, None, None)
    assert ret3 == 1
    out3 = capsys.readouterr().out
    assert "expected boolean" in out3

    # Test removed 'jev' backend is rejected
    args_bad_backend = parser.parse_args(["config", "--set", "router_backend=jev"])
    ret4 = handle_cli(args_bad_backend, config, None, None)
    assert ret4 == 1
    out4 = capsys.readouterr().out
    assert "Invalid backend 'jev'" in out4


def test_multi_tier_skill_discovery(tmp_path: Path):
    """Ensure discover_all_skill_paths finds profile skills, shared skills, and local workspace skills."""
    hermes_home = tmp_path / "hermes_root" / "profiles" / "agent_1"
    hermes_home.mkdir(parents=True)
    profile_skills = hermes_home / "skills" / "profile_tool"
    profile_skills.mkdir(parents=True)
    (profile_skills / "SKILL.md").write_text("# Profile Tool", encoding="utf-8")

    shared_skills = tmp_path / "hermes_root" / "skills" / "shared_tool"
    shared_skills.mkdir(parents=True)
    (shared_skills / "SKILL.md").write_text("# Shared Tool", encoding="utf-8")

    workspace = tmp_path / "workspace"
    local_skills = workspace / ".skills" / "local_tool"
    local_skills.mkdir(parents=True)
    (local_skills / "SKILL.md").write_text("# Local Tool", encoding="utf-8")

    globals_found, locals_found = discover_all_skill_paths(hermes_home, cwd=workspace)

    global_names = [p.parent.name for p in globals_found]
    local_names = [p.parent.name for p in locals_found]

    assert "profile_tool" in global_names
    assert "shared_tool" in global_names
    assert "local_tool" in local_names


def test_triage_uses_sufficient_max_tokens_budget_for_reasoning_models():
    """Ensure host LLM complete() receives max_tokens >= 2500 to prevent reasoning token truncation."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"route": "DIRECT_REPLY", "reasoning": "greeting"}'

    mock_ctx = MagicMock()
    mock_ctx.llm = mock_llm

    triage = CognitiveTriage(ctx=mock_ctx)
    triage.analyze("Hola")

    assert mock_llm.complete.called
    kwargs = mock_llm.complete.call_args[1]
    assert kwargs.get("max_tokens", 0) >= 2500
    assert kwargs.get("max_tokens") == 3000


def test_triage_truncation_detection_and_degradation_receipt(tmp_path):
    """Verifies that truncation in reasoning models triggers retry, explicit degradation, and error receipt."""
    from neural_context.models import RouteResult
    from neural_context.receipt import ReceiptEmitter
    from neural_context.router import ContextCoordinator
    from neural_context.sidecar import SidecarIndex
    from neural_context.config import PluginConfig

    # Simulate a reasoning model exhausting tokens on internal CoT: returns empty content
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_llm.complete.return_value = mock_resp

    mock_ctx = MagicMock()
    mock_ctx.llm = mock_llm

    triage = CognitiveTriage(ctx=mock_ctx, max_tokens=3000)
    analysis = triage.analyze("Implementa el pixel CAPI de Meta para AcmeCorp")

    # Verify adaptive retry occurred with doubled budget
    assert mock_llm.complete.call_count == 2
    retry_kwargs = mock_llm.complete.call_args_list[1][1]
    assert retry_kwargs["max_tokens"] == 6000

    # Verify explicit truncation detection on analysis
    assert analysis.truncation_detected is True
    assert analysis.error == "error_truncated"

    # Verify coordinator emits degraded status
    db_file = tmp_path / "sidecar.db"
    index = SidecarIndex(db_path=str(db_file))
    config = PluginConfig(sidecar_db_path=str(db_file), router_backend="heuristic")
    coordinator = ContextCoordinator(config=config, sidecar=index)

    result = coordinator.execute(analysis, user_message="Implementa el pixel CAPI de Meta para AcmeCorp")
    assert result.status == "error_truncated"

    # Verify receipt record explicitly logs error_truncated instead of silent fake green
    receipts_file = tmp_path / "receipts.jsonl"
    emitter = ReceiptEmitter(receipts_path=str(receiptss := str(receipts_file)))
    rec = emitter.record_turn(
        turn_id="turn_trunc_1",
        session_id="sess_1",
        analysis=analysis,
        route_result=result,
        platform="cli",
        model="deepseek-v4-flash"
    )
    assert rec.status == "error_truncated"
    assert "truncated" in rec.error.lower()


def test_host_llm_router_uses_sufficient_token_budget():
    """Ensure HostLLMRouter allocates max_tokens >= 1000 so reasoning models have thought headroom."""
    from neural_context.router import HostLLMRouter
    from neural_context.models import SkillCandidate, TriageAnalysis

    mock_ctx = MagicMock()
    mock_res = MagicMock()
    mock_res.text = "meta-pixel-capi"
    mock_ctx.llm.complete.return_value = mock_res

    router = HostLLMRouter(ctx=mock_ctx)
    analysis = TriageAnalysis(
        is_actionable_task=True,
        intent_summary_en="Setup Meta CAPI tracking",
        required_capabilities=["capi"]
    )
    candidates = [
        (SkillCandidate(name="meta-pixel-capi", description="Meta CAPI", lead_paragraph="", path="/tmp/1", source_type="global"), 0.9),
        (SkillCandidate(name="general-python", description="General Python", lead_paragraph="", path="/tmp/2", source_type="global"), 0.4),
    ]

    selected = router.route(analysis, candidates)
    assert selected is not None
    assert mock_ctx.llm.complete.called
    kwargs = mock_ctx.llm.complete.call_args.kwargs
    assert kwargs.get("max_tokens", 0) >= 1000


def test_enriched_task_routing_with_named_entity():
    """Ensure actionable task referencing a specific client entity triggers ENRICHED_TASK route."""
    triage = CognitiveTriage()
    domain = DomainClassification(is_software_task=True)

    json_reply = json.dumps({
        "route": "ENRICHED_TASK",
        "memory_probe": {
            "entity": "AcmeCorp",
            "topic": "pixel CAPI",
            "query_text": "AcmeCorp"
        },
        "semantic_concepts": ["meta-pixel", "capi"],
        "extracted_entities": ["AcmeCorp"],
        "intent_summary_en": "Implement Meta CAPI for client AcmeCorp",
        "search_query": "When the user needs to implement meta pixel capi",
        "reasoning": "Technical task referencing client AcmeCorp requires past configuration facts."
    })

    result = triage._parse_structured_output(json_reply, "Implementa el pixel CAPI de Meta para AcmeCorp", domain)
    assert result.route == CognitiveRoute.ENRICHED_TASK
    assert result.needs_skills is True
    assert result.needs_memory is True
    assert result.memory_probe is not None
    assert result.memory_probe.entity == "AcmeCorp"


def test_markdown_section_hierarchy_and_anti_contamination(tmp_path: Path):
    """
    Ensure Markdown memory probing captures bullets under `## Entity` sections
    without requiring entity repetition on every bullet, and prevents cross-section
    contamination when other sections share topic keywords.
    """
    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text(
        "# Memory\n\n"
        "## AcmeCorp\n"
        "- Client: Retail e-commerce platform\n"
        "- Stack: Uses Meta Conversions API (CAPI) and FastAPI backend\n\n"
        "## Golden Game\n"
        "- Golden Game runs WordPress and uses TikTok Pixel without any CAPI integration.\n",
        encoding="utf-8"
    )

    retriever = SelectiveMemoryRetriever(token_budget=0)
    probe = MemoryProbe(
        entity="AcmeCorp",
        topic="pixel CAPI de Meta",
        query_text="AcmeCorp pixel CAPI"
    )
    items = retriever.retrieve(entities=["AcmeCorp"], probe=probe, cwd=str(tmp_path))

    facts_joined = " ".join(i.fact for i in items)
    assert "Meta Conversions API (CAPI)" in facts_joined
    assert "FastAPI" in facts_joined
    assert "Golden Game" not in facts_joined
    assert "TikTok" not in facts_joined


def test_two_phase_shorthand_review_selects_domain_grounded_skill(tmp_path: Path):
    """
    Ensure 'Revisa lo de AcmeCorp' (where initial triage only sees generic 'review' concepts)
    grounds Phase 2 routing in discovered Phase 1 memory facts and selects 'meta-pixel-capi'
    over generic 'sdlc-review' / 'general-audit' distractors.
    """
    from neural_context.config import PluginConfig
    from neural_context.models import TriageAnalysis
    from neural_context.router import ContextCoordinator
    from neural_context.sidecar import SidecarIndex

    skills_dir = tmp_path / "skills"
    for name, desc, body in [
        ("sdlc-review", "Review SDLC project status and code", "General review checklist for software projects and audits."),
        ("general-audit", "Audit project configuration and status", "Inspect and review client project configuration."),
        ("meta-pixel-capi", "Implement and diagnose hybrid Meta Pixel and Conversions API (CAPI)", "Deduplicate and inspect Meta Pixel and Conversions API server-side events."),
    ]:
        sdir = skills_dir / name
        sdir.mkdir(parents=True)
        (sdir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n{body}\n", encoding="utf-8")

    cwd_dir = tmp_path / "workspace"
    cwd_dir.mkdir(parents=True)
    (cwd_dir / "MEMORY.md").write_text(
        "# Memory\n\n## AcmeCorp\n- Client: Retail\n- Stack: AcmeCorp uses Meta Conversions API (CAPI) and FastAPI.\n",
        encoding="utf-8"
    )

    db_file = tmp_path / "sidecar.db"
    sidecar = SidecarIndex(db_path=str(db_file))
    sidecar.sync_skills(list(skills_dir.glob("*/SKILL.md")))

    config = PluginConfig(sidecar_db_path=str(db_file), router_backend="graph", selective_memory_enabled=True)
    coordinator = ContextCoordinator(config=config, sidecar=sidecar)

    analysis = TriageAnalysis(
        route=CognitiveRoute.ENRICHED_TASK,
        memory_probe=MemoryProbe(entity="AcmeCorp", topic="project review", query_text="AcmeCorp"),
        semantic_concepts=["review", "inspection", "audit", "project"],
        extracted_entities=["AcmeCorp"],
        intent_summary_en="Review AcmeCorp project configuration and status",
        search_query="When the user needs to review a client project — review project status",
        domain=DomainClassification(is_software_task=True)
    )

    result = coordinator.execute(analysis, user_message="Revisa lo de AcmeCorp", cwd=str(cwd_dir))
    assert result.selected_skill is not None
    assert result.selected_skill.name == "meta-pixel-capi"


def test_single_unrelated_candidate_is_rejected():
    """Ensure a single candidate matching only generic stopwords is rejected rather than blindly injected."""
    from neural_context.models import SkillCandidate, TriageAnalysis
    from neural_context.router import GraphRouter

    router = GraphRouter()
    analysis = TriageAnalysis(
        route=CognitiveRoute.SKILL_ONLY,
        semantic_concepts=["quantum-teleportation", "wormhole-stabilizer"],
        intent_summary_en="Calibrate quantum wormhole stabilizer",
        search_query="When the user needs to calibrate wormhole — calibrate stabilizer"
    )
    candidates = [
        (SkillCandidate(name="cooking-pasta", description="When the user needs to boil water", lead_paragraph="How to cook pasta", path="/tmp/1"), 0.28)
    ]
    assert router.route(analysis, candidates) is None


def test_triage_enforces_timeout_ms_on_llm_and_model_fn():
    """Ensure timeout_ms is forwarded to ctx.llm.complete(timeout=...) and enforced on slow model_fn."""
    import time

    # 1. Verify ctx.llm.complete receives timeout in seconds
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"route": "DIRECT_REPLY", "reasoning": "ok"}'
    mock_ctx = MagicMock()
    mock_ctx.llm = mock_llm

    triage_llm = CognitiveTriage(ctx=mock_ctx, timeout_ms=5000)
    triage_llm.analyze("Hello")
    assert mock_llm.complete.called
    assert mock_llm.complete.call_args.kwargs.get("timeout") == 5.0

    # 2. Verify slow model_fn times out cleanly and falls back without hanging
    def slow_model(_prompt: str) -> str:
        time.sleep(0.5)
        return '{"route": "SKILL_ONLY"}'

    t0 = time.perf_counter()
    triage_slow = CognitiveTriage(model_fn=slow_model, timeout_ms=50)
    analysis = triage_slow.analyze("Hello there")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.35
    assert analysis.route == CognitiveRoute.DIRECT_REPLY


def test_graph_router_default_and_multi_hop_personalized_pagerank():
    """Verify 'graph' is the default router_backend and GraphRouter executes multi-hop Personalized PageRank."""
    from neural_context.config import PluginConfig
    from neural_context.models import SkillCandidate, TriageAnalysis
    from neural_context.router import GraphRouter

    assert PluginConfig().router_backend == "graph"

    # Candidate A shares seed 'capi' and bridge concepts 'deduplication' + 'event_id'.
    # Candidate B shares seed 'capi' only once alongside unrelated noise, while Candidate C
    # shares the dense domain cluster ('deduplication', 'event_id', 'conversions') reinforced on hop 2-3.
    cand_a = SkillCandidate(
        name="meta-pixel-capi",
        description="Meta CAPI conversions deduplication event_id tracking",
        lead_paragraph="Server-side conversions deduplication with event_id and pixel",
        path="/tmp/a",
    )
    cand_b = SkillCandidate(
        name="generic-webhook",
        description="Generic webhook capi endpoint",
        lead_paragraph="Send HTTP POST payload to external server endpoint",
        path="/tmp/b",
    )
    candidates = [(cand_b, 0.50), (cand_a, 0.50)]

    analysis = TriageAnalysis(
        route=CognitiveRoute.SKILL_ONLY,
        semantic_concepts=["capi", "deduplication"],
        intent_summary_en="Fix CAPI deduplication",
    )

    router_1hop = GraphRouter(damping=0.85, max_iter=1)
    router_3hop = GraphRouter(damping=0.85, max_iter=3)

    winner_1 = router_1hop.route(analysis, candidates)
    winner_3 = router_3hop.route(analysis, candidates)
    assert winner_1 is not None and winner_1.name == "meta-pixel-capi"
    assert winner_3 is not None and winner_3.name == "meta-pixel-capi"


def test_information_weight_formula_rejects_multiple_cooccurring_stopwords(tmp_path: Path, monkeypatch):
    """
    Ensure the IDF + morphological density + coverage formula rejects memory lines that match
    TWO OR MORE ubiquitous functional particles ('para', 'del') when the substantive domain
    terms ('credenciales', 'produccion') are absent, without any hardcoded stopword list.
    """
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: empty_home)

    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text(
        "# Memory\n"
        "- Notas para el equipo del proyecto de diseno visual\n"
        "- Flujo para revision del calendario trimestral\n"
        "- Proceso para entrega del reporte administrativo\n"
        "- guia para configuracion del entorno local\n",
        encoding="utf-8",
    )

    retriever = SelectiveMemoryRetriever(token_budget=0)
    # Even though 'para' and 'del' BOTH appear in every line (2 matches!), their corpus ubiquity
    # and low morphological density yield tiny IDF coverage compared to 'credenciales' and 'produccion'.
    probe = MemoryProbe(
        query_text="para del credenciales produccion",
        topic="credenciales produccion",
    )
    results = retriever.retrieve(probe=probe, cwd=str(tmp_path))
    assert len(results) == 0


def test_symbol_and_diacritic_normalization_enables_clean_skill_and_memory_matches(tmp_path: Path, monkeypatch):
    """
    Verify that Markdown symbols (**bold**, `code`, em-dashes —), snake_case/hyphenated compounds,
    and diacritics (configuración vs configuracion) are normalized before memory & graph matching.
    """
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: empty_home)

    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text(
        "# Memory\n\n"
        "## **`AcmeCorp`**\n"
        "- Configuración de `meta—pixel—capi` con `event_id` y deduplicación activa.\n",
        encoding="utf-8",
    )

    retriever = SelectiveMemoryRetriever(token_budget=0)
    # Query without accents ('configuracion', 'deduplicacion') and without Markdown backticks
    probe = MemoryProbe(
        entity="AcmeCorp",
        topic="configuracion deduplicacion",
        query_text="configuracion deduplicacion event_id",
    )
    items = retriever.retrieve(entities=["AcmeCorp"], probe=probe, cwd=str(tmp_path))
    assert len(items) == 1
    assert "deduplicación" in items[0].fact


def test_multilingual_tokenization_and_memory_retrieval_across_scripts(tmp_path: Path, monkeypatch):
    """
    Ensure normalization and IDF retrieval work across non-Latin scripts (Russian Cyrillic,
    Chinese/Japanese CJK) and European ligatures (German ß) without losing characters.
    """
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: empty_home)

    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text(
        "# Memory\n\n"
        "- Безопасность: продакшн сервер использует ключи шифрования Ed25519\n"
        "- 架构说明：生产环境服务器配置使用 Kubernetes 集群\n"
        "- Infrastruktur: Straßenverzeichnis und Größenbeschränkung auf dem Server\n",
        encoding="utf-8",
    )

    retriever = SelectiveMemoryRetriever(token_budget=0)

    # 1. Russian (Cyrillic)
    res_ru = retriever.retrieve(
        probe=MemoryProbe(topic="ключи шифрования", query_text="сервер ключи шифрования"),
        cwd=str(tmp_path),
    )
    assert len(res_ru) == 1
    assert "Ed25519" in res_ru[0].fact

    # 2. Chinese (CJK unspaced compound -> matches 2-char bigram '配置' / '服务器')
    res_zh = retriever.retrieve(
        probe=MemoryProbe(topic="服务器配置", query_text="服务器配置"),
        cwd=str(tmp_path),
    )
    assert len(res_zh) == 1
    assert "Kubernetes" in res_zh[0].fact

    # 3. German (ß -> ss folding)
    res_de = retriever.retrieve(
        probe=MemoryProbe(topic="grossenbeschrankung", query_text="strassenverzeichnis grossenbeschrankung"),
        cwd=str(tmp_path),
    )
    assert len(res_de) == 1
    assert "Größenbeschränkung" in res_de[0].fact


def test_large_scale_120_skills_multi_skill_routing_and_bipartite_memory_graph(tmp_path: Path, monkeypatch):
    """
    Verify that with >100 skills (120 skills) and a large 40-line entity section in MEMORY.md:
    1. Bipartite Memory Graph isolates only the topic-relevant facts from the 40-line AcmeCorp section.
    2. GraphRouter evaluates the entire catalog without a top_k=5 cap and injects ALL matching skills
       requested by the model's semantic concepts ('meta-pixel-capi' AND 'fastapi-testing').
    """
    from neural_context.models import TriageAnalysis

    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: empty_home)

    # 1. Create a large MEMORY.md where ## AcmeCorp has 30 unrelated administrative lines
    # and 2 specific technical lines at the very bottom.
    noise_lines = [
        f"- Administrative invoice record #{i} for quarterly office supplies and catering"
        for i in range(1, 31)
    ]
    target_lines = [
        "- Architecture: AcmeCorp backend runs FastAPI with pytest async integration tests",
        "- Tracking: AcmeCorp funnel uses Meta Conversions API (CAPI) with event_id deduplication",
    ]
    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text(
        "# Memory\n\n## AcmeCorp\n" + "\n".join(noise_lines + target_lines) + "\n",
        encoding="utf-8",
    )

    # 2. Create 120 skills (118 distractors + 2 target skills that sit beyond rank 5 in naive order)
    skills_dir = tmp_path / "skills"
    skill_paths = []
    for i in range(118):
        sdir = skills_dir / f"generic-workflow-{i:03d}"
        sdir.mkdir(parents=True)
        sf = sdir / "SKILL.md"
        sf.write_text(
            f"---\nname: generic-workflow-{i:03d}\ndescription: Generic workflow helper {i}\n---\n"
            f"# Generic Workflow {i}\nStandard operational checklist for routine task {i}.\n",
            encoding="utf-8",
        )
        skill_paths.append(sf)

    for name, desc, body in [
        (
            "meta-pixel-capi",
            "Hybrid browser-server Meta Pixel and Conversions API (CAPI) deduplication",
            "Configure Meta Conversions API (CAPI) event_id deduplication and pixel tracking.",
        ),
        (
            "fastapi-pytest",
            "FastAPI async endpoint testing with pytest fixtures and httpx AsyncClient",
            "Write deterministic pytest integration tests for FastAPI routers and endpoints.",
        ),
    ]:
        sdir = skills_dir / name
        sdir.mkdir(parents=True)
        sf = sdir / "SKILL.md"
        sf.write_text(f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n{body}\n", encoding="utf-8")
        skill_paths.append(sf)

    db_file = tmp_path / "sidecar.db"
    sidecar = SidecarIndex(db_path=str(db_file))
    sidecar.sync_skills(skill_paths)
    assert sidecar.n_docs == 120

    config = PluginConfig(sidecar_db_path=str(db_file), router_backend="graph", selective_memory_enabled=True)
    coordinator = ContextCoordinator(config=config, sidecar=sidecar)

    analysis = TriageAnalysis(
        route=CognitiveRoute.ENRICHED_TASK,
        memory_probe=MemoryProbe(
            entity="AcmeCorp",
            topic="FastAPI pytest Meta CAPI deduplication",
            query_text="AcmeCorp FastAPI pytest Meta CAPI deduplication",
        ),
        semantic_concepts=["meta-pixel-capi", "conversions-api", "fastapi", "pytest"],
        extracted_entities=["AcmeCorp"],
        intent_summary_en="Write FastAPI pytest integration tests for AcmeCorp Meta CAPI endpoint",
        search_query="meta-pixel-capi conversions deduplication fastapi pytest asyncclient",
        domain=DomainClassification(is_software_task=True),
    )

    result = coordinator.execute(analysis, user_message="Crea los tests de FastAPI para el CAPI de AcmeCorp", cwd=str(tmp_path))

    # Memory graph must isolate the 2 relevant technical facts and reject the 30 invoice noise lines
    assert len(result.selected_memories) == 2
    mem_text = " ".join(m.fact for m in result.selected_memories)
    assert "FastAPI" in mem_text and "Meta Conversions API" in mem_text
    assert "catering" not in mem_text

    # Skill graph must select BOTH matching skills ('meta-pixel-capi' and 'fastapi-pytest')
    selected_names = [s.name for s in result.selected_skills]
    assert "meta-pixel-capi" in selected_names
    assert "fastapi-pytest" in selected_names
    assert "## Active Procedural Guidance (Skill: meta-pixel-capi)" in result.combined_context_block
    assert "## Active Procedural Guidance (Skill: fastapi-pytest)" in result.combined_context_block


def test_default_hermes_prompt_files_skipped_while_state_db_and_custom_db_routed_via_graph(tmp_path: Path, monkeypatch):
    """
    Ensure:
    1. Default Hermes system-prompt files (<hermes_home>/MEMORY.md, USER.md, SOUL.md, AGENTS.md)
       are NOT re-injected by SelectiveMemoryRetriever (no duplicate system prompt tokens).
    2. Native Hermes `state.db` (`sessions` + `messages` + `messages_fts`) is queried excluding
       the active `session_id` and filtered via the Bipartite Graph.
    3. Custom user ecosystem SQLite tables (`client_notes` with arbitrary column names) are
       dynamically introspected and ranked via the Bipartite Graph.
    """
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)

    # 1. Default system-prompt files in hermes_home must be skipped
    (hermes_home / "MEMORY.md").write_text("## AcmeCorp\n- Default system prompt note about AcmeCorp.", encoding="utf-8")
    (hermes_home / "USER.md").write_text("## AcmeCorp\n- Default user profile note about AcmeCorp.", encoding="utf-8")
    (hermes_home / "SOUL.md").write_text("## AcmeCorp\n- Default soul note about AcmeCorp.", encoding="utf-8")
    (hermes_home / "AGENTS.md").write_text("## AcmeCorp\n- Default agents note about AcmeCorp.", encoding="utf-8")

    # 2. Native Hermes state.db with past session + active current session
    state_db = hermes_home / "state.db"
    with sqlite3.connect(state_db) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, active INTEGER DEFAULT 1)"
        )
        conn.execute("INSERT INTO sessions VALUES ('sess_old', 'AcmeCorp Infrastructure Setup')")
        conn.execute("INSERT INTO sessions VALUES ('sess_live', 'Current Live Session')")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, active) VALUES (?, ?, ?, 1)",
            ("sess_old", "assistant", "AcmeCorp production deploy uses nginx reverse proxy on port 8443."),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, active) VALUES (?, ?, ?, 1)",
            ("sess_old", "user", "Remember we ate pizza during the AcmeCorp kickoff."),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, active) VALUES (?, ?, ?, 1)",
            ("sess_live", "user", "AcmeCorp live turn message that must not be echoed from history."),
        )
        conn.commit()

    # 3. Custom ecosystem SQLite DB with non-standard table/column names
    custom_db = tmp_path / "ecosystem_crm.sqlite"
    with sqlite3.connect(custom_db) as conn:
        conn.execute("CREATE TABLE client_notes (note_id INTEGER PRIMARY KEY, client_name TEXT, summary TEXT)")
        conn.execute(
            "INSERT INTO client_notes (client_name, summary) VALUES (?, ?)",
            ("AcmeCorp", "AcmeCorp nginx TLS certificate rotates via certbot every 60 days."),
        )
        conn.commit()

    retriever = SelectiveMemoryRetriever(token_budget=0)
    probe = MemoryProbe(
        entity="AcmeCorp",
        topic="nginx reverse proxy TLS",
        query_text="AcmeCorp nginx reverse proxy TLS",
    )
    items = retriever.retrieve(
        entities=["AcmeCorp"],
        probe=probe,
        cwd=str(tmp_path),
        session_id="sess_live",
    )

    facts_joined = " ".join(it.fact for it in items)
    sources = {it.source_file for it in items}

    # Default .md files in hermes_home must NOT be duplicated
    assert "Default system prompt note" not in facts_joined
    assert "Default user profile note" not in facts_joined
    assert "Default soul note" not in facts_joined
    assert "Default agents note" not in facts_joined
    assert "MEMORY.md" not in sources
    assert "USER.md" not in sources

    # Live session messages must be excluded; old relevant nginx messages & custom DB rows must be retrieved
    assert "must not be echoed" not in facts_joined
    assert "pizza" not in facts_joined  # filtered out by bipartite topic graph
    assert "nginx reverse proxy on port 8443" in facts_joined
    assert "certbot every 60 days" in facts_joined
    assert "state.db:messages" in sources
    assert "ecosystem_crm.sqlite:client_notes" in sources


def test_statistical_column_profiling_handles_arbitrary_language_db_schemas(tmp_path: Path, monkeypatch):
    """
    Verify that a custom SQLite table with non-English column names ('bitacora_proyectos',
    'empresa_objetivo', 'nota_tecnica', 'huella_sha256', 'fecha_registro') is profiled
    directly from actual data distributions without static English column/table lists.
    """
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)

    custom_db = tmp_path / "negocio.db"
    with sqlite3.connect(custom_db) as conn:
        conn.execute(
            "CREATE TABLE bitacora_proyectos ("
            "registro_id INTEGER PRIMARY KEY, "
            "fecha_registro TEXT, "
            "huella_sha256 TEXT, "
            "empresa_objetivo TEXT, "
            "nota_tecnica TEXT)"
        )
        conn.execute(
            "INSERT INTO bitacora_proyectos (fecha_registro, huella_sha256, empresa_objetivo, nota_tecnica) "
            "VALUES (?, ?, ?, ?)",
            (
                "2026-11-12T10:00:00Z",
                "a3f1e9c2b8d7041234567890abcdef1234567890abcdef1234567890abcdef12",
                "AcmeCorp",
                "AcmeCorp migra sus colas de Celery hacia RabbitMQ con TLS mutuo.",
            ),
        )
        conn.commit()

    retriever = SelectiveMemoryRetriever(token_budget=0)
    probe = MemoryProbe(
        entity="AcmeCorp",
        topic="Celery RabbitMQ TLS",
        query_text="AcmeCorp Celery RabbitMQ TLS",
    )
    items = retriever.retrieve(entities=["AcmeCorp"], probe=probe, cwd=str(tmp_path))
    assert len(items) == 1
    assert "RabbitMQ" in items[0].fact
    assert "a3f1e9c2b8d704" not in items[0].fact
    assert items[0].source_file == "negocio.db:bitacora_proyectos"


def test_universal_tokenizer_supports_all_hermes_script_families():
    """
    Verify pure Unicode category tokenization preserves Thai combining marks + unspaced fragments,
    folds Turkish dotless 'ı'/'İ', strips optional Arabic/Hebrew vowel marks & Greek tonos,
    and preserves Cyrillic, Ukrainian, CJK/Kana/Hangul across all Hermes locales.
    """
    from neural_context.sidecar import tokenize

    # Thai unspaced sentence with tone/vowel combining marks must stay coherent and emit sub-fragments
    thai_toks = tokenize("การปรับใช้เซิร์ฟเวอร์ FastAPI")
    assert "fastapi" in thai_toks
    assert "การปรับใช้เซิร์ฟเวอร์" in thai_toks
    assert any("ปรับ" in t for t in thai_toks)

    # Turkish İ/ı folding matches both UTF-8 and ASCII-folded spellings
    tr_utf8 = set(tokenize("İstanbul yapılandırması şifre"))
    tr_ascii = set(tokenize("istanbul yapilandirmasi sifre"))
    assert tr_utf8 == tr_ascii

    # Arabic with optional harakat matches unvocalized text
    ar_harakat = set(tokenize("إِعْدَاد قَاعِدَة البَيَانَات"))
    ar_plain = set(tokenize("إعداد قاعدة البيانات"))
    assert ar_harakat == ar_plain

    # Greek tonos and final sigma fold cleanly
    el_accent = set(tokenize("Ρύθμιση διακομιστής"))
    el_plain = set(tokenize("ρυθμιση διακομιστησ"))
    assert el_accent == el_plain

    # Ukrainian, Korean, Japanese, Traditional Chinese, Czech, Hungarian, Vietnamese
    assert {"налаштування", "сервера"}.issubset(set(tokenize("Налаштування сервера")))
    assert {"데이터베이스", "설정"}.issubset(set(tokenize("데이터베이스 설정")))
    assert {"設定", "データベース"}.issubset(set(tokenize("データベースの設定確認")))
    assert {"資料庫", "設定"}.issubset(set(tokenize("資料庫伺服器設定")))
    assert set(tokenize("Cấu hình máy chủ")) == {"cau", "hinh", "may", "chu"}


def test_triage_parses_open_bcp47_cross_lingual_and_temporal_fields():
    """
    Ensure TRIAGE_SYSTEM_PROMPT contains no closed language enumeration list and CognitiveTriage
    cleanly extracts open BCP-47 codes, cross_lingual_terms, time_window_hours, and recency_bias.
    """
    import json
    from neural_context.triage import CognitiveTriage, TRIAGE_SYSTEM_PROMPT

    assert '"en" | "es" | "fr"' not in TRIAGE_SYSTEM_PROMPT
    assert "MEMORY.md, USER.md" not in TRIAGE_SYSTEM_PROMPT

    payload = json.dumps({
        "route": "ENRICHED_TASK",
        "detected_language": "pt-br",
        "memory_probe": {
            "entity": "AcmeCorp",
            "topic": "servidor webhook",
            "time_anchor": "ontem",
            "time_window_hours": 48,
            "recency_bias": True,
            "query_text": "AcmeCorp servidor webhook",
            "cross_lingual_terms": ["server", "webhook", "reverse proxy"],
        },
        "extracted_entities": ["AcmeCorp"],
        "semantic_concepts": ["webhook", "reverse-proxy"],
        "intent_summary_en": "Fix AcmeCorp webhook server using historical setup.",
        "search_query": "fastapi caddy reverse proxy webhook",
        "reasoning": "Cross-lingual and temporal memory retrieval with skill routing.",
    })
    triage = CognitiveTriage(model_fn=lambda _p: payload)
    res = triage.analyze("Arruma o servdor webhok da AcmeCorp q mexemos ontem")
    assert res.detected_language == "pt-br"
    assert res.memory_probe is not None
    assert res.memory_probe.detected_language == "pt-br"
    assert res.memory_probe.time_window_hours == 48
    assert res.memory_probe.recency_bias is True
    assert "reverse proxy" in res.memory_probe.cross_lingual_terms


def test_sqlite_multihop_pagerank_and_cross_session_bridge_recall(tmp_path: Path, monkeypatch):
    """
    Verify true 2-hop Personalized PageRank over Hermes state.db:
    - Session 1 links entity 'AcmeCorp' + 'webhook' + bridge concept 'FastAPI 8443'.
    - Session 2 (different session, NO mention of 'AcmeCorp' or 'webhook') connects 'FastAPI 8443' -> 'Caddy TLS 1.3'.
    - Admin noise under 'AcmeCorp' ('tacos') shares no topic/bridge concepts and must be suppressed.
    """
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)

    state_db = hermes_home / "state.db"
    with sqlite3.connect(state_db) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, active INTEGER DEFAULT 1)"
        )
        conn.execute("INSERT INTO sessions VALUES ('s1', 'AcmeCorp Webhook Rollout')")
        conn.execute("INSERT INTO sessions VALUES ('s2', 'Reverse Proxy Architecture')")
        conn.execute("INSERT INTO sessions VALUES ('s_noise', 'AcmeCorp Kickoff')")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, active) VALUES (?, ?, ?, 1)",
            ("s1", "assistant", "AcmeCorp webhook endpoint is hosted on uvicorn-bridge-8443 runtime."),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, active) VALUES (?, ?, ?, 1)",
            ("s2", "assistant", "The uvicorn-bridge-8443 runtime terminates Caddy TLS 1.3 with zero downtime."),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, active) VALUES (?, ?, ?, 1)",
            ("s_noise", "user", "AcmeCorp team ordered tacos during the morning standup."),
        )
        conn.commit()

    retriever = SelectiveMemoryRetriever(token_budget=0)
    probe = MemoryProbe(
        entity="AcmeCorp",
        topic="webhook",
        query_text="AcmeCorp webhook",
    )
    items = retriever.retrieve(entities=["AcmeCorp"], probe=probe, cwd=str(tmp_path), session_id="live_sess")
    joined = " ".join(it.fact for it in items)

    assert "uvicorn-bridge-8443" in joined
    assert "Caddy TLS 1.3" in joined  # Hop-2 bridge recovered across sessions!
    assert "tacos" not in joined      # Unrelated entity noise filtered out!


def test_cross_lingual_memory_retrieval_and_temporal_windowing(tmp_path: Path, monkeypatch):
    """
    Verify:
    1. Cross-lingual probe ('contraseñas base de datos' + cross_lingual_terms=['PostgreSQL', 'credentials'])
       recovers English facts stored in DB without failing coverage math.
    2. Language-agnostic `time_window_hours` isolates recent records inside the window over stale ones,
       and `recency_bias` ranks fresher facts higher in custom tables with arbitrary temporal column names.
    """
    import time

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setattr("neural_context.memory.get_hermes_home", lambda: hermes_home)

    now = time.time()
    recent_ts = now - (6 * 3600)          # 6 hours ago (inside 48h window)
    stale_ts = now - (45 * 24 * 3600)     # 45 days ago (outside 48h window)

    state_db = hermes_home / "state.db"
    with sqlite3.connect(state_db) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
            "content TEXT, timestamp REAL NOT NULL, active INTEGER DEFAULT 1)"
        )
        conn.execute("INSERT INTO sessions VALUES ('s_old', 'Database Rotation')")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, 1)",
            ("s_old", "assistant", "PostgreSQL production credentials rotated to vault-v2-secret.", recent_ts),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, 1)",
            ("s_old", "assistant", "PostgreSQL legacy credentials used deprecated static env var.", stale_ts),
        )
        conn.commit()

    retriever = SelectiveMemoryRetriever(token_budget=0)

    # Query in Spanish asking about "ayer" (48h window) targeting English facts via cross_lingual_terms
    probe_window = MemoryProbe(
        topic="contraseñas base datos",
        time_anchor="ayer",
        time_window_hours=48,
        recency_bias=True,
        query_text="contraseñas base de datos ayer",
        cross_lingual_terms=["PostgreSQL", "credentials"],
        detected_language="es",
    )
    items_recent = retriever.retrieve(probe=probe_window, cwd=str(tmp_path), session_id="live")
    assert len(items_recent) == 1
    assert "vault-v2-secret" in items_recent[0].fact
    assert "deprecated static" not in items_recent[0].fact

    # Without strict window, recency_bias still orders the newer fact #1 over the older one
    probe_bias = MemoryProbe(
        topic="contraseñas",
        recency_bias=True,
        query_text="contraseñas base datos",
        cross_lingual_terms=["PostgreSQL", "credentials"],
        detected_language="es",
    )
    items_all = retriever.retrieve(probe=probe_bias, cwd=str(tmp_path), session_id="live")
    assert len(items_all) == 2
    assert "vault-v2-secret" in items_all[0].fact











