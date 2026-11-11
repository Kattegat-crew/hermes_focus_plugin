import sqlite3
import tempfile
from pathlib import Path
import pytest

from neural_context.sidecar import (
    SidecarIndex,
    compute_sha256,
    extract_frontmatter_and_lead,
)


def test_extract_frontmatter_and_lead():
    content = """---
name: meta-pixel-capi
description: >-
  Implement and deduplicate Meta Pixel and Conversions API events.
---

# Meta Pixel CAPI

Use this skill when integrating Meta Conversions API with server-side events.
It guarantees deduplication via event_id.

## Additional sections
Some extra details here.
"""
    desc, lead = extract_frontmatter_and_lead(content)
    assert "Implement and deduplicate" in desc
    assert "Use this skill when integrating" in lead


def test_sidecar_sync_and_bm25_search(tmp_path):
    db_file = tmp_path / "test_sidecar.db"
    index = SidecarIndex(str(db_file))

    # Create dummy skill files
    skill1_dir = tmp_path / "skills" / "meta-pixel-capi"
    skill1_dir.mkdir(parents=True)
    skill1_file = skill1_dir / "SKILL.md"
    skill1_file.write_text("""---
name: meta-pixel-capi
description: Server-side tracking and Conversions API deduplication.
---
# Meta CAPI
Comprehensive lead tracking and Meta advertising conversions.
""")

    skill2_dir = tmp_path / "skills" / "docker-deploy"
    skill2_dir.mkdir(parents=True)
    skill2_file = skill2_dir / "SKILL.md"
    skill2_file.write_text("""---
name: docker-deploy
description: Docker container deployment and compose orchestration.
---
# Docker Deploy
Container deployment patterns and healthcheck monitoring.
""")

    # Sync
    updated = index.sync_skills([skill1_file, skill2_file])
    assert updated == 2
    assert index.n_docs == 2

    # Second sync without changes should update 0 records
    updated2 = index.sync_skills([skill1_file, skill2_file])
    assert updated2 == 0

    # Search for meta tracking
    results = index.search_bm25("meta conversions tracking", top_k=2)
    assert len(results) > 0
    winner, score = results[0]
    assert winner.name == "meta-pixel-capi"
    assert score > 0.0

    # Search for docker container
    results_docker = index.search_bm25("deploy container with docker", top_k=2)
    assert len(results_docker) > 0
    winner_d, _ = results_docker[0]
    assert winner_d.name == "docker-deploy"
