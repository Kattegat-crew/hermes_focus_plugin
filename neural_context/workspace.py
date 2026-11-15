from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from .memory import _FORBIDDEN_PREFIXES, is_safe_memory_file
from .models import DomainClassification


def is_safe_workspace_dir(path: Path) -> bool:
    """Validate directory safety against symlink traversal and sensitive operating system roots."""
    try:
        resolved = path.resolve()
        for forb in _FORBIDDEN_PREFIXES:
            try:
                if resolved == forb or forb in resolved.parents:
                    return False
            except Exception:
                pass
        return resolved.is_dir()
    except Exception:
        return False


def sanitize_entity_name(entity: str) -> Optional[str]:
    """Sanitizes an extracted entity name to prevent path traversal or separator attacks."""
    if not entity or not isinstance(entity, str):
        return None
    cleaned = entity.strip()
    if len(cleaned) < 2 or len(cleaned) > 100 or cleaned.startswith("."):
        return None
    # Reject path separators, directory traversal tokens, and control characters
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned or "\x00" in cleaned:
        return None
    return cleaned


def resolve_workspace_dir(
    cwd: Optional[Union[str, Path]] = None,
    entities: Optional[List[str]] = None,
    project_scan_paths: Optional[List[Union[str, Path]]] = None,
) -> Optional[Path]:
    """
    Resolves the physical directory of a target software workspace.

    Evaluates configured *project_scan_paths* (e.g. ['~/projects', '~/workspace']) against
    *entities* extracted from conversational turns, falling back cleanly to *cwd*.
    """
    clean_entities: List[str] = []
    if entities:
        for ent in entities:
            s = sanitize_entity_name(ent)
            if s and s not in clean_entities:
                clean_entities.append(s)

    scan_roots: List[Path] = []
    if project_scan_paths:
        for p in project_scan_paths:
            try:
                expanded = Path(os.path.expanduser(str(p))).resolve()
                if is_safe_workspace_dir(expanded):
                    scan_roots.append(expanded)
            except Exception:
                continue

    if clean_entities and scan_roots:
        for ent in clean_entities:
            cand_names = [ent]
            if "_" in ent:
                cand_names.append(ent.replace("_", "-"))
            if "-" in ent:
                cand_names.append(ent.replace("-", "_"))

            for root in scan_roots:
                for cname in cand_names:
                    try:
                        candidate = (root / cname).resolve()
                        if candidate.is_relative_to(root) and is_safe_workspace_dir(candidate):
                            return candidate
                    except Exception:
                        continue

                # Case-insensitive direct child directory scan
                try:
                    target_lowers = {c.lower() for c in cand_names}
                    for child in root.iterdir():
                        if child.is_dir() and child.name.lower() in target_lowers:
                            c_res = child.resolve()
                            if c_res.is_relative_to(root) and is_safe_workspace_dir(c_res):
                                return c_res
                except Exception:
                    continue

    # 2. Fall back to cwd
    if cwd:
        try:
            cwd_p = Path(os.path.expanduser(str(cwd))).resolve()
            if is_safe_workspace_dir(cwd_p):
                return cwd_p
        except Exception:
            pass

    return None


def inspect_workspace_governance(
    cwd: Optional[Union[str, Path]] = None,
    enabled: bool = True,
    entities: Optional[List[str]] = None,
    project_scan_paths: Optional[List[Union[str, Path]]] = None,
) -> DomainClassification:
    """
    Inspects whether a workspace is a documented software repository with sovereign
    governance (AGENTS.md, .skills/, .agents/skills/) and inspects its technical stack.

    Supports resolving multi-repo directories in chat bots via *entities* and *project_scan_paths*.
    """
    if not enabled:
        return DomainClassification(is_software_task=False)

    target_dir = resolve_workspace_dir(
        cwd=cwd,
        entities=entities,
        project_scan_paths=project_scan_paths
    )
    if not target_dir:
        return DomainClassification(is_software_task=False)

    root = target_dir
    current = root
    governance_root: Optional[str] = None
    stack: List[str] = []

    # Ascend to find the sovereign governance boundary
    try:
        while current != current.parent:
            has_agents = (current / "AGENTS.md").is_file() or (current / "CLAUDE.md").is_file()
            has_skills = (current / ".skills").is_dir() or (current / ".agents" / "skills").is_dir()
            has_git = (current / ".git").is_dir()

            if has_agents or has_skills or has_git:
                governance_root = str(current)
                break
            current = current.parent
    except Exception:
        governance_root = None

    check_dir = Path(governance_root) if governance_root else root
    try:
        if (
            (check_dir / "pyproject.toml").is_file()
            or (check_dir / "setup.py").is_file()
            or (check_dir / "Pipfile").is_file()
            or (check_dir / "requirements.txt").is_file()
        ):
            stack.append("python")
        if (check_dir / "package.json").is_file():
            stack.append("node")
        if (check_dir / "Cargo.toml").is_file():
            stack.append("rust")
        if (check_dir / "go.mod").is_file():
            stack.append("go")
        if (check_dir / "Dockerfile").is_file() or (check_dir / "docker-compose.yml").is_file():
            stack.append("docker")
    except Exception:
        pass

    is_sw = bool(governance_root or stack)
    return DomainClassification(
        is_software_task=is_sw,
        detected_stack=stack,
        governance_repo_root=governance_root,
        resolved_workspace_path=str(target_dir) if (governance_root or stack) else None
    )
