from pathlib import Path
from neural_context.config import PluginConfig
from neural_context.models import CognitiveRoute, DomainClassification, MemoryProbe, TriageAnalysis
from neural_context.router import ContextCoordinator
from neural_context.sidecar import SidecarIndex
from neural_context.triage import CognitiveTriage


def test_arbiter_route_q1_direct_reply(tmp_path):
    """
    Route 1: DIRECT_REPLY
    Casual greeting or conceptual conversational turn.
    Guarantees fast exit: 0 tokens injected, no skills, no memory lookup.
    """
    db_file = tmp_path / "sidecar.db"
    index = SidecarIndex(str(db_file))
    config = PluginConfig(sidecar_db_path=str(db_file))
    coordinator = ContextCoordinator(config=config, sidecar=index)

    analysis = TriageAnalysis(
        route=CognitiveRoute.DIRECT_REPLY,
        reasoning="Casual conversation requires no external context."
    )

    result = coordinator.execute(analysis, user_message="Buenos días, ¿cómo estás?")
    assert result.status == "skipped_no_context_needed"
    assert result.combined_context_block is None
    assert result.selected_skill is None
    assert len(result.selected_memories) == 0


def test_arbiter_route_q2_skill_only(tmp_path):
    """
    Route 2: SKILL_ONLY
    Procedural execution task that does not depend on past history.
    Broad semantic concepts are used to route the skill via graph activation.
    """
    db_file = tmp_path / "sidecar.db"
    index = SidecarIndex(str(db_file))

    # Create dummy marketing skill
    skill_dir = tmp_path / "skills" / "marketing-funnel"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("""---
name: marketing-funnel
description: Build conversion funnels and lead acquisition strategies.
---
# Marketing Funnel Strategy
Procedure for lead acquisition, ad copy, and conversion funnels.
""")
    index.sync_skills([skill_dir / "SKILL.md"])

    config = PluginConfig(sidecar_db_path=str(db_file), router_backend="graph")
    coordinator = ContextCoordinator(config=config, sidecar=index)

    analysis = TriageAnalysis(
        route=CognitiveRoute.SKILL_ONLY,
        semantic_concepts=["marketing", "leads", "funnel"],
        search_query="When the user needs to build funnels — build marketing funnel",
        domain=DomainClassification(is_software_task=False)
    )

    result = coordinator.execute(analysis, user_message="Quiero armar un embudo de captación de leads")
    assert result.status == "success"
    assert result.selected_skill is not None
    assert result.selected_skill.name == "marketing-funnel"
    assert len(result.selected_memories) == 0
    assert "Active Procedural Guidance (Skill: marketing-funnel)" in result.combined_context_block


def test_arbiter_route_q3_memory_only_multilingual(tmp_path):
    """
    Route 3: MEMORY_ONLY (Multilingual & No-Entity Recall)
    User asks about a past agreement or decision without mentioning a capitalized proper entity.
    Search probe runs in the native interaction language (Spanish) and matches memory facts directly.
    Zero procedural skills are loaded.
    """
    db_file = tmp_path / "sidecar.db"
    index = SidecarIndex(str(db_file))

    # Mock workspace with Spanish MEMORY.md
    cwd_dir = tmp_path / "workspace"
    cwd_dir.mkdir(parents=True)
    (cwd_dir / "MEMORY.md").write_text("""
# Acuerdos y Memoria del Proyecto
- Acordamos realizar la entrega de los endpoints de autenticacion el proximo viernes.
- El despliegue de infraestructura se realiza en servidores Ubuntu con Docker.
""")

    config = PluginConfig(sidecar_db_path=str(db_file), selective_memory_enabled=True)
    coordinator = ContextCoordinator(config=config, sidecar=index)

    # Native language memory probe without proper entity
    analysis = TriageAnalysis(
        route=CognitiveRoute.MEMORY_ONLY,
        memory_probe=MemoryProbe(
            topic="entrega autenticacion",
            time_anchor="ayer",
            query_text="acordamos entrega autenticacion viernes"
        ),
        domain=DomainClassification(is_software_task=False)
    )

    result = coordinator.execute(
        analysis,
        user_message="¿Qué fue lo que acordamos ayer sobre la entrega de la autenticación?",
        cwd=str(cwd_dir)
    )

    assert result.status == "success"
    assert result.selected_skill is None  # Zero skills injected for pure memory query
    assert len(result.selected_memories) >= 1
    assert any("entrega de los endpoints de autenticacion" in m.fact for m in result.selected_memories)
    assert "<neural_context>" in result.combined_context_block
    assert "Active Memory Facts" in result.combined_context_block


def test_arbiter_route_q4_enriched_task(tmp_path):
    """
    Route 4: ENRICHED_TASK (Two-Phase Context Synthesis)
    1. Memory probe retrieves customer stack fact (AcmeCorp uses Docker and Rust).
    2. Fact seeds the graph and enriches the skill search.
    3. The skill matching both the task and historical stack is selected.
    """
    db_file = tmp_path / "sidecar.db"
    index = SidecarIndex(str(db_file))

    # Create skills: generic python vs specialized docker-rust
    s1 = tmp_path / "skills" / "python-dev"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text("""---
name: python-dev
description: General python development.
---
# Python Dev
""")

    s2 = tmp_path / "skills" / "docker-rust-pipeline"
    s2.mkdir(parents=True)
    (s2 / "SKILL.md").write_text("""---
name: docker-rust-pipeline
description: Build containerized CI/CD pipelines for Rust services in Docker.
---
# Docker Rust Pipeline
Build and deploy rust microservices in docker containers.
""")
    index.sync_skills([s1 / "SKILL.md", s2 / "SKILL.md"])

    cwd_dir = tmp_path / "workspace"
    cwd_dir.mkdir(parents=True)
    (cwd_dir / "MEMORY.md").write_text("""
# Project Memory
- AcmeCorp runs all core backend microservices in Rust packaged inside Docker containers.
""")

    config = PluginConfig(
        sidecar_db_path=str(db_file),
        router_backend="graph",
        selective_memory_enabled=True
    )
    coordinator = ContextCoordinator(config=config, sidecar=index)

    # User simply asks: "Despliega el servicio de AcmeCorp" (ambiguous without memory)
    analysis = TriageAnalysis(
        route=CognitiveRoute.ENRICHED_TASK,
        memory_probe=MemoryProbe(
            entity="AcmeCorp",
            topic="despliegue servicio",
            query_text="despliega servicio AcmeCorp"
        ),
        semantic_concepts=["deploy", "pipeline", "containers"],
        extracted_entities=["AcmeCorp"],
        domain=DomainClassification(is_software_task=True)
    )

    result = coordinator.execute(
        analysis,
        user_message="Despliega el servicio de AcmeCorp",
        cwd=str(cwd_dir)
    )

    assert result.status == "success"
    # Grounded by AcmeCorp memory (Rust + Docker), it selects docker-rust-pipeline over generic python-dev!
    assert result.selected_skill is not None
    assert result.selected_skill.name == "docker-rust-pipeline"
    assert len(result.selected_memories) >= 1
    assert "<neural_context>" in result.combined_context_block


def test_skill_gap_suggestion_on_unmatched_task(tmp_path):
    """
    Skill Gap Suggestion:
    When a task genuinely requires a capability that is missing in the local catalog,
    the plugin does NOT inject an incorrect skill (fail-closed); instead, it provides
    a structured recommendation notice.
    """
    db_file = tmp_path / "sidecar.db"
    index = SidecarIndex(str(db_file))

    # Catalog has only one unrelated skill
    skill_dir = tmp_path / "skills" / "text-editor"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("""---
name: text-editor
description: Formatting plain text notes.
---
# Text Editor
""")
    index.sync_skills([skill_dir / "SKILL.md"])

    config = PluginConfig(sidecar_db_path=str(db_file), router_backend="graph")
    coordinator = ContextCoordinator(config=config, sidecar=index)

    # Task for specialized quantum circuit simulation
    analysis = TriageAnalysis(
        route=CognitiveRoute.SKILL_ONLY,
        semantic_concepts=["quantum-circuit", "qiskit", "qubits"],
        suggested_missing_skill="quantum-computing-simulation",
        search_query="When the user needs to simulate quantum circuits — simulate qiskit",
        domain=DomainClassification(is_software_task=True)
    )

    result = coordinator.execute(analysis, user_message="Simula un circuito cuántico de 5 qubits en Qiskit")

    assert result.suggested_missing_skill == "quantum-computing-simulation"
    assert "Recommended capability to install: quantum-computing-simulation" in (result.combined_context_block or "")
