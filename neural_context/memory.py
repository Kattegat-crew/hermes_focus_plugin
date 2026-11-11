from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from .config import get_hermes_home
from .models import MemoryItem


def estimate_tokens(text: str) -> int:
    return len(text.split()) + len(text) // 4


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

        search_roots: List[Path] = []
        if cwd:
            search_roots.append(Path(cwd))

        hermes_home = get_hermes_home()
        search_roots.extend([
            hermes_home,
            hermes_home / "memories",
            hermes_home / "shared",
        ])

        results: List[MemoryItem] = []
        current_tokens = 0
        seen_facts = set()

        # Collect candidate markdown memory files
        candidate_files: List[Path] = []
        for root in search_roots:
            if not root.is_dir():
                continue
            for fname in ["MEMORY.md", "USER.md", "PROJECT_MEMORY.md"]:
                fpath = root / fname
                if fpath.is_file():
                    candidate_files.append(fpath)

        entity_patterns = [re.compile(re.escape(e), re.IGNORECASE) for e in entities if len(e) >= 3]
        if not entity_patterns:
            return []

        for fpath in candidate_files:
            if current_tokens >= self.token_budget:
                break
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Split into semantic chunks / bullet points / lines
            lines = content.splitlines()
            for line in lines:
                trimmed = line.strip()
                if not trimmed or len(trimmed) < 10 or trimmed.startswith("#"):
                    continue

                for pat in entity_patterns:
                    if pat.search(trimmed):
                        # Match found
                        if trimmed in seen_facts:
                            continue
                        seen_facts.add(trimmed)
                        tok = estimate_tokens(trimmed)
                        if current_tokens + tok > self.token_budget:
                            break
                        results.append(MemoryItem(
                            entity=pat.pattern,
                            fact=trimmed,
                            source_file=fpath.name,
                            token_estimate=tok
                        ))
                        current_tokens += tok

        return results
