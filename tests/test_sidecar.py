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


def test_sync_reconciles_only_managed_sources_and_restores_global_fallback(tmp_path):
    index = SidecarIndex(str(tmp_path / "sidecar.db"))
    global_root = tmp_path / "profile" / "skills"
    project_root = tmp_path / "project" / ".agents" / "skills"
    global_skill = global_root / "focus" / "SKILL.md"
    project_skill = project_root / "focus" / "SKILL.md"
    global_skill.parent.mkdir(parents=True)
    project_skill.parent.mkdir(parents=True)
    global_skill.write_text("---\nname: focus\ndescription: global\n---\n")
    project_skill.write_text("---\nname: focus\ndescription: local\n---\n")

    roots = {global_skill: (1, global_root), project_skill: (0, project_root)}
    index.sync_skills([global_skill, project_skill], source_info=roots, replace_tiers={0, 1})
    assert index._candidates["focus"].path == str(project_skill.resolve())

    project_skill.unlink()
    index.sync_skills([global_skill], source_info={global_skill: roots[global_skill]}, replace_tiers={0, 1})
    assert index._candidates["focus"].path == str(global_skill.resolve())
    assert index._candidates["focus"].description == "global"


def test_sync_removes_deleted_and_tracks_same_content_move_without_dropping_other_tiers(tmp_path):
    index = SidecarIndex(str(tmp_path / "sidecar.db"))
    profile_root = tmp_path / "profile" / "skills"
    external_root = tmp_path / "shared" / "skills"
    profile_skill = profile_root / "keep" / "SKILL.md"
    external_skill = external_root / "move" / "SKILL.md"
    profile_skill.parent.mkdir(parents=True)
    external_skill.parent.mkdir(parents=True)
    profile_skill.write_text("---\nname: keep\ndescription: profile\n---\n")
    external_skill.write_text("---\nname: move\ndescription: moved unchanged\n---\n")
    source_info = {profile_skill: (1, profile_root), external_skill: (3, external_root)}
    index.sync_skills([profile_skill, external_skill], source_info=source_info, replace_tiers={1, 3})

    moved_skill = external_root / "category" / "move" / "SKILL.md"
    moved_skill.parent.mkdir(parents=True)
    moved_skill.write_text(external_skill.read_text())
    external_skill.unlink()
    index.sync_skills(
        [profile_skill, moved_skill],
        source_info={profile_skill: source_info[profile_skill], moved_skill: (3, external_root)},
        replace_tiers={1, 3},
    )
    assert index._candidates["move"].path == str(moved_skill.resolve())
    assert index._candidates["keep"].path == str(profile_skill.resolve())

    moved_skill.unlink()
    index.sync_skills([profile_skill], source_info={profile_skill: source_info[profile_skill]}, replace_tiers={1, 3})
    assert set(index._candidates) == {"keep"}


def test_discovery_uses_native_trusted_project_iterator_and_tiers(tmp_path, monkeypatch):
    import sys
    import types

    import neural_context.sidecar as sidecar

    project_root = tmp_path / "project" / ".agents" / "skills"
    profile_root = tmp_path / "hermes" / "skills"
    unsafe = project_root / "untrusted" / "SKILL.md"
    trusted = project_root / "trusted" / "SKILL.md"
    profile = profile_root / "profile-skill" / "SKILL.md"
    disabled = profile_root / "disabled" / "SKILL.md"
    unavailable = profile_root / "unavailable" / "SKILL.md"
    for path in (unsafe, trusted, profile, disabled, unavailable):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {path.parent.name}\n---\n")

    fake = types.ModuleType("tools.skills_tool")
    fake.skill_search_dirs = lambda: ([(0, project_root), (1, profile_root)], profile_root)
    fake.find_all_skills = lambda: [{"name": "trusted"}, {"name": "profile-skill"}]

    def _locate(name, cat, roots):
        if name == "trusted" and any(r == project_root for _, r in roots):
            return None, trusted.parent, trusted
        if name == "profile-skill" and any(r == profile_root for _, r in roots):
            return None, profile.parent, profile
        return "not found", None, None

    fake.locate_skill = _locate
    tools = types.ModuleType("tools")
    tools.skills_tool = fake
    monkeypatch.setitem(sys.modules, "tools", tools)
    monkeypatch.setitem(sys.modules, "tools.skills_tool", fake)

    paths, project_paths, sources, names, tiers = sidecar.discover_skill_inventory()
    assert paths == [trusted, profile]
    assert project_paths == [trusted]
    assert sources[trusted] == (0, project_root)
    assert names[trusted] == "trusted"
    assert tiers == {0, 1}

    paths_without_project, project_paths, *_ = sidecar.discover_skill_inventory(include_project=False)
    assert paths_without_project == [profile]
    assert project_paths == []


def test_partial_subtree_sync_removes_only_deleted_entries_in_scope(tmp_path):
    index = SidecarIndex(str(tmp_path / "sidecar.db"))
    root = tmp_path / "skills"
    scoped = root / "scoped"
    outside = root / "outside"
    removed = scoped / "removed" / "SKILL.md"
    kept = outside / "keep" / "SKILL.md"
    for path in (removed, kept):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {path.parent.name}\n---\n")
    index.sync_skills([removed, kept], replace_tiers={1})

    removed.unlink()
    index.sync_skills([kept], replace_subtrees=[scoped])

    assert set(index._candidates) == {"keep"}
