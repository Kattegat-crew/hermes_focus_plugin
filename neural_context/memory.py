from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from .config import get_hermes_home
from .models import MemoryItem


def estimate_tokens(text: str) -> int:
    return len(text.split()) + len(text) // 4


_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?prev[i]ous\s+instructions|<system>|system:|assistant:|human:|<script|override\s+instructions)",
    re.IGNORECASE
)


class SelectiveMemoryRetriever:
    """
    Surgical entity-driven memory extractor.
    Instead of dumping full memory files, it retrieves only facts,
    agreements, and constraints matching the extracted entities.
    """

    def __init__(self, token_budget: int = 400):
        self.token_budget = token_budget

    def retrieve(self, entities: List[str], cwd: Optional[str] = None) -> List[MemoryItem]:
        if not entities:
            return []

        clean_entities = [e.strip() for e in entities if len(e.strip()) >= 2]
        if not clean_entities:
            return []

        entity_matchers = [(e, re.compile(re.escape(e), re.IGNORECASE)) for e in clean_entities]

        # Standard Hermes candidate memory files (strictly bounded paths)
        candidate_files: List[Path] = []
        if cwd:
            cwd_p = Path(cwd)
            if cwd_p.is_dir():
                for fname in ["PROJECT_MEMORY.md", "MEMORY.md"]:
                    fp = cwd_p / fname
                    if fp.is_file():
                        candidate_files.append(fp)

        hermes_home = get_hermes_home()
        for fname in ["MEMORY.md", "USER.md"]:
            fp = hermes_home / fname
            if fp.is_file() and fp not in candidate_files:
                candidate_files.append(fp)

        memories_dir = hermes_home / "memories"
        if memories_dir.is_dir():
            for fp in memories_dir.glob("*.md"):
                if fp.is_file() and fp not in candidate_files:
                    candidate_files.append(fp)

        results: List[MemoryItem] = []
        current_tokens = 0
        seen_facts = set()

        for fpath in candidate_files:
            if current_tokens >= self.token_budget:
                break
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for line in content.splitlines():
                trimmed = line.strip()
                if not trimmed or len(trimmed) < 10 or trimmed.startswith("#"):
                    continue

                # Prompt injection defense
                if _INJECTION_PATTERNS.search(trimmed):
                    continue

                for ent_name, pat in entity_matchers:
                    if pat.search(trimmed):
                        if trimmed in seen_facts:
                            continue
                        seen_facts.add(trimmed)
                        tok = estimate_tokens(trimmed)
                        if current_tokens + tok > self.token_budget:
                            break
                        results.append(MemoryItem(
                            entity=ent_name,
                            fact=trimmed,
                            source_file=fpath.name,
                            token_estimate=tok
                        ))
                        current_tokens += tok

        return results
