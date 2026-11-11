from __future__ import annotations

import os
import re
import sqlite3
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

_FORBIDDEN_PREFIXES = (
    Path("/etc"),
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/var/run"),
    Path("/root/.ssh"),
)


def is_safe_memory_file(path: Path) -> bool:
    """Validate path safety against symlink traversal and denial of service."""
    try:
        resolved = path.resolve()
        for forb in _FORBIDDEN_PREFIXES:
            try:
                if resolved == forb or forb in resolved.parents:
                    return False
            except Exception:
                pass
        # Bound file size to 2 MB to prevent memory bloat
        if resolved.is_file() and resolved.stat().st_size > 2_000_000:
            return False
        return resolved.is_file()
    except Exception:
        return False


class SelectiveMemoryRetriever:
    """
    Universal surgical factual memory extractor.
    Retrieves targeted facts, agreements, and decisions matching
    entities, topics, or interaction phrases without loading full memory files.

    Universal Adaptability:
    - Vanilla Hermes Markdown (cwd PROJECT_MEMORY.md, MEMORY.md,
      profile MEMORY.md, USER.md, memories/**/*.md, memory/**/*.md)
    - Custom directory configurations (HERMES_MEMORY_DIR / HERMES_MEMORY_PATH)
    - Pluggable SQLite memory stores (memories.db, memory.db, state.db)
    - Strict boundary protection (rejection of sensitive system files and oversized files)
    """

    def __init__(self, token_budget: int = 0):
        self.token_budget = token_budget  # 0 = unlimited by default, preserves full context

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

        # -------------------------------------------------------------
        # 1. Discover Candidate Files (Markdown & Custom Stores)
        # -------------------------------------------------------------
        candidate_files: List[Path] = []
        sqlite_candidates: List[Path] = []

        # Tier 1: Project / Workspace Local Memory
        if cwd:
            cwd_p = Path(cwd)
            if cwd_p.is_dir():
                for fname in [
                    "PROJECT_MEMORY.md", "MEMORY.md",
                    ".agents/MEMORY.md", ".agents/PROJECT_MEMORY.md"
                ]:
                    fp = cwd_p / fname
                    if fp.is_file() and is_safe_memory_file(fp):
                        candidate_files.append(fp)

                for sub in ["memories", "memory"]:
                    sub_dir = cwd_p / sub
                    if sub_dir.is_dir():
                        for fp in sub_dir.glob("*.md"):
                            if fp.is_file() and is_safe_memory_file(fp) and fp not in candidate_files:
                                candidate_files.append(fp)

                for db_name in ["memories.db", "memory.db"]:
                    dbp = cwd_p / db_name
                    if dbp.is_file() and is_safe_memory_file(dbp):
                        sqlite_candidates.append(dbp)

        # Tier 2: Standard Hermes Global Profile Memory
        hermes_home = get_hermes_home()
        if hermes_home.is_dir():
            for fname in ["MEMORY.md", "USER.md"]:
                fp = hermes_home / fname
                if fp.is_file() and is_safe_memory_file(fp) and fp not in candidate_files:
                    candidate_files.append(fp)

            for sub in ["memories", "memory"]:
                mem_dir = hermes_home / sub
                if mem_dir.is_dir():
                    for fp in mem_dir.glob("**/*.md"):
                        if fp.is_file() and is_safe_memory_file(fp) and fp not in candidate_files:
                            candidate_files.append(fp)
                            if len(candidate_files) >= 50:
                                break

            for db_name in ["memories.db", "memory.db"]:
                dbp = hermes_home / db_name
                if dbp.is_file() and is_safe_memory_file(dbp) and dbp not in sqlite_candidates:
                    sqlite_candidates.append(dbp)

        # Tier 3: Custom Environment Memory Directory
        custom_mem = os.environ.get("HERMES_MEMORY_DIR") or os.environ.get("HERMES_MEMORY_PATH")
        if custom_mem:
            cmd_p = Path(custom_mem)
            if cmd_p.is_dir():
                for fp in cmd_p.glob("**/*.md"):
                    if fp.is_file() and is_safe_memory_file(fp) and fp not in candidate_files:
                        candidate_files.append(fp)
                        if len(candidate_files) >= 50:
                            break

        results: List[MemoryItem] = []
        current_tokens = 0
        seen_facts = set()

        # -------------------------------------------------------------
        # 2. Extract Facts from Markdown Stores
        # -------------------------------------------------------------
        for fpath in candidate_files:
            if self.token_budget > 0 and current_tokens >= self.token_budget:
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
                    matches = sum(1 for pat in topic_matchers if pat.search(trimmed))
                    if matches > 0:
                        matched_entity = (probe.topic if probe and probe.topic else "Context")

                if matched_entity:
                    if trimmed in seen_facts:
                        continue
                    seen_facts.add(trimmed)
                    tok = estimate_tokens(trimmed)
                    if self.token_budget > 0 and current_tokens + tok > self.token_budget:
                        break
                    results.append(MemoryItem(
                        entity=matched_entity,
                        fact=trimmed,
                        source_file=fpath.name,
                        token_estimate=tok
                    ))
                    current_tokens += tok

        # -------------------------------------------------------------
        # 3. Extract Facts from Pluggable / Custom SQLite Stores
        # -------------------------------------------------------------
        for db_path in sqlite_candidates:
            if self.token_budget > 0 and current_tokens >= self.token_budget:
                break
            sqlite_items = self._retrieve_from_sqlite(db_path, entity_matchers, topic_matchers)
            for item in sqlite_items:
                if item.fact in seen_facts:
                    continue
                seen_facts.add(item.fact)
                if self.token_budget > 0 and current_tokens + item.token_estimate > self.token_budget:
                    break
                results.append(item)
                current_tokens += item.token_estimate

        return results

    def _retrieve_from_sqlite(
        self,
        db_path: Path,
        entity_matchers: List,
        topic_matchers: List
    ) -> List[MemoryItem]:
        """
        Safely inspects a candidate SQLite database in read-only mode.
        Fails closed on any schema mismatch, lock, or error.
        """
        items: List[MemoryItem] = []
        try:
            uri = f"file:{db_path.resolve()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=0.1) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in cursor.fetchall()]

                target_table = None
                for candidate in ["memories", "memory", "facts", "user_memories"]:
                    if candidate in tables:
                        target_table = candidate
                        break

                if not target_table:
                    return []

                cursor.execute(f"PRAGMA table_info({target_table})")
                cols = [r[1].lower() for r in cursor.fetchall()]
                text_col = next((c for c in ["fact", "content", "text", "body", "value"] if c in cols), None)
                if not text_col:
                    return []

                for ent_name, _ in entity_matchers:
                    cursor.execute(
                        f"SELECT {text_col} FROM {target_table} WHERE {text_col} LIKE ? LIMIT 5",
                        (f"%{ent_name}%",)
                    )
                    for row in cursor.fetchall():
                        val = str(row[0]).strip()
                        if val and len(val) >= 10 and not _INJECTION_PATTERNS.search(val):
                            items.append(MemoryItem(
                                entity=ent_name,
                                fact=val,
                                source_file=f"{db_path.name}:{target_table}",
                                token_estimate=estimate_tokens(val)
                            ))
        except Exception:
            pass  # Read-only probe must never crash or block
        return items
