---
name: python-packaging
description: >-
  Standard packaging workflow for Hermes Agent plugins using pyproject.toml,
  UV, isolated virtual environments, and catalog compliance.
---

# Python Packaging for Hermes Agent Plugins

Use this skill when scaffolding, building, dependency-locking, and testing Python packages for Hermes Agent plugins.

## Principles

1. **Single Source of Truth (`pyproject.toml`)**:
   - All metadata, entrypoints, and dependencies live in `pyproject.toml`.
   - Never use `setup.py` or `requirements.txt` as primary sources.
2. **Minimal External Dependencies**:
   - Lean towards standard library where feasible.
   - Any external dependency adds installation overhead and potential conflicts with the host Hermes environment.
3. **Reproducible Builds**:
   - Use `uv` for lightning-fast, reproducible dependency management and locking.
4. **Hermes Catalog Compliance**:
   - Package name must match the plugin name (`neural-context-plugin`).
   - Clean top-level namespace: `neural_context`.
   - No post-install execution scripts or auto-update mechanisms (violates catalog rules).

## Standard `pyproject.toml` Template

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "neural-context-plugin"
version = "0.1.0"
description = "Cognitive triage, selective memory ingestion, and dynamic skill routing for Hermes Agent."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
    { name = "NeuralCrew Labs", email = "engineering@neuralcrew.com" }
]
keywords = ["hermes", "hermes-agent", "plugin", "context", "skills", "memory"]
dependencies = [
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
]
jev = [
    "requests>=2.28.0",
]

[tool.hatch.build.targets.wheel]
packages = ["neural_context"]

[tool.ruff]
line-length = 100
target-version = "py310"
```

## Verification Checklist

- [ ] `uv pip compile pyproject.toml` succeeds without dependency conflicts.
- [ ] `pytest` runs in under 2 seconds.
- [ ] No undeclared imports in `neural_context`.
- [ ] Package wheel builds cleanly: `uv build`.
