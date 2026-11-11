from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from .config import get_hermes_home
from .models import MemoryItem, MemoryProbe


def estimate_tokens(text: str) -> int:
    return len(text.split()) + len(text) // 4


_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?prev[i]ous\s+instructions|<system>|system:|assistant:|human:|<script|override\s+instructions)",
    re.IGNORECASE
)


class SelectiveMemoryRetriever:
    """
    Surgical factual memory extractor.
    Retrieves targeted facts, agreements, and decisions matching
    entities, topics, or interaction phrases without loading full memory files.
    """

    def __init__(self, token_budget: int = 400):
        self.token_budget = token_budget

    def retrieve(
        self,
        entities: Optional[List[str]] = None,
        probe: Optional[MemoryProbe] = None,
        cwd: Optional[str] = None
    ) -> List[MemoryItem]:
        clean_entities = [e.strip() for e in (entities or []) if len(e.strip()) >= 2]
        if probe and probe.entity and probe.entity.strip() not in clean_entities:
            clean_entities.append(probe.entity.strip())

        # Collect multilingual topic keywords from probe
        topic_terms: List[str] = []
        if probe:
            for text_source in [probe.topic, probe.query_text]:
                if text_source:
                    tokens = re.findall(r"[\w]{3,}", text_source, re.UNICODE)
                    for tok in tokens:
                        t_low = tok.lower()
                        if t_low not in topic_terms and len(t_low) >= 3:
                            topic_terms.append(t_low)

        if not clean_entities and not topic_terms:
            return []

        entity_matchers = [(e, re.compile(re.escape(e), re.IGNORECASE)) for e in clean_entities]
        topic_matchers = [re.compile(re.escape(t), re.IGNORECASE) for t in topic_terms]

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

                matched_entity = None
                for ent_name, pat in entity_matchers:
                    if pat.search(trimmed):
                        matched_entity = ent_name
                        break

                if not matched_entity and topic_matchers:
                    # Match if line contains any topic keyword
                    matches = sum(1 for pat in topic_matchers if pat.search(trimmed))
                    if matches > 0:
                        matched_entity = (probe.topic if probe and probe.topic else "Context")

                if matched_entity:
                    if trimmed in seen_facts:
                        continue
                    seen_facts.add(trimmed)
                    tok = estimate_tokens(trimmed)
                    if current_tokens + tok > self.token_budget:
                        break
                    results.append(MemoryItem(
                        entity=matched_entity,
                        fact=trimmed,
                        source_file=fpath.name,
                        token_estimate=tok
                    ))
                    current_tokens += tok

        return results
