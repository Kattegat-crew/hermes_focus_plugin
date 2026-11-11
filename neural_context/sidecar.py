from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import SkillCandidate


def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text)]


def extract_frontmatter_and_lead(content: str) -> Tuple[str, str]:
    """
    Extracts the frontmatter description and first clean lead paragraph
    (up to 400 chars) from a SKILL.md file. Never mutates the source file.
    """
    description = ""
    body = content

    # Parse YAML frontmatter if present
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            # Extract description
            desc_match = re.search(r"^description:\s*(?:>-\s*|\|-\s*)?(.*)$", fm_text, re.MULTILINE)
            if desc_match:
                desc_val = desc_match.group(1).strip()
                # If multi-line, collect subsequent indented lines
                lines = fm_text.splitlines()
                start_idx = -1
                for i, line in enumerate(lines):
                    if line.startswith("description:"):
                        start_idx = i
                        break
                if start_idx != -1:
                    collected = [desc_val]
                    for next_line in lines[start_idx + 1:]:
                        if next_line.startswith("  ") or next_line.startswith("\t"):
                            collected.append(next_line.strip())
                        else:
                            break
                    description = " ".join([c for c in collected if c]).strip()

    # Clean body: strip headings, markdown comments, empty lines
    clean_lines = []
    for line in body.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#") or trimmed.startswith("<!--"):
            continue
        clean_lines.append(trimmed)

    lead_paragraph = " ".join(clean_lines)[:400].strip()
    return description, lead_paragraph


class SidecarIndex:
    """
    Lightweight SQLite-backed sidecar index and in-memory BM25 ranker.
    Stores metadata hashed by sha256 to avoid touching user skill files.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        self._candidates: Dict[str, SkillCandidate] = {}
        # BM25 internals
        self.k1 = 1.5
        self.b = 0.75
        self.doc_len: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}
        self.tf: Dict[str, Dict[str, int]] = {}
        self.n_docs: int = 0

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills_sidecar (
                    skill_name TEXT PRIMARY KEY,
                    skill_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    description TEXT,
                    lead_paragraph TEXT,
                    is_project_local INTEGER DEFAULT 0,
                    indexed_tokens INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_sha256 ON skills_sidecar(sha256)")
            conn.commit()

    def sync_skills(self, skill_paths: List[Path], project_local_paths: Optional[List[Path]] = None) -> int:
        """
        Synchronizes skill metadata into SQLite index only when sha256 changes.
        Builds the in-memory BM25 index in sub-20ms.
        """
        project_paths_set = {str(p.resolve()) for p in (project_local_paths or [])}
        updated_count = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            existing = {
                row[0]: (row[1], row[2])  # name -> (sha256, is_local)
                for row in cursor.execute("SELECT skill_name, sha256, is_project_local FROM skills_sidecar")
            }

            for p in skill_paths:
                if not p.is_file():
                    continue
                skill_name = p.parent.name if p.name == "SKILL.md" else p.stem
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                sha = compute_sha256(content)
                is_local = 1 if str(p.resolve()) in project_paths_set else 0

                if skill_name in existing and existing[skill_name][0] == sha and existing[skill_name][1] == is_local:
                    continue  # Unchanged, skip disk parsing

                desc, lead = extract_frontmatter_and_lead(content)
                tokens = len(tokenize(f"{desc} {lead}"))

                cursor.execute("""
                    INSERT INTO skills_sidecar 
                    (skill_name, skill_path, sha256, description, lead_paragraph, is_project_local, indexed_tokens, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(skill_name) DO UPDATE SET
                        skill_path=excluded.skill_path,
                        sha256=excluded.sha256,
                        description=excluded.description,
                        lead_paragraph=excluded.lead_paragraph,
                        is_project_local=excluded.is_project_local,
                        indexed_tokens=excluded.indexed_tokens,
                        last_updated=CURRENT_TIMESTAMP
                """, (skill_name, str(p.resolve()), sha, desc, lead, is_local, tokens))
                updated_count += 1

            conn.commit()

        self.rebuild_bm25()
        return updated_count

    def rebuild_bm25(self) -> None:
        """Loads all indexed skill leads into memory for instant scoring."""
        self.doc_len.clear()
        self.df.clear()
        self.tf.clear()
        self._candidates.clear()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute("""
                SELECT skill_name, skill_path, sha256, description, lead_paragraph, is_project_local
                FROM skills_sidecar
            """).fetchall()

        self.n_docs = len(rows)
        for name, path, sha, desc, lead, is_local in rows:
            candidate = SkillCandidate(
                name=name,
                path=path,
                sha256=sha,
                description=desc or "",
                lead_paragraph=lead or "",
                is_project_local=bool(is_local),
            )
            self._candidates[name] = candidate

            # Index name, description, and lead
            text_to_index = f"{name} {desc or ''} {lead or ''}"
            tokens = tokenize(text_to_index)
            self.doc_len[name] = len(tokens)

            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.tf[name] = counts
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1

        total_tokens = sum(self.doc_len.values())
        self.avg_doc_len = total_tokens / max(1.0, self.n_docs)

    def search_bm25(self, query: str, top_k: int = 5, boost_project_local: float = 2.0) -> List[Tuple[SkillCandidate, float]]:
        """
        Calculates Robertson/BM25 scores over indexed skill leads.
        Applies a normative priority boost if the skill is project-local.
        """
        q_tokens = tokenize(query)
        scores: Dict[str, float] = {}

        for t in q_tokens:
            if t not in self.df:
                continue
            idf = math.log(1.0 + (self.n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for doc_id, counts in self.tf.items():
                if t not in counts:
                    continue
                tf = counts[t]
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_id] / max(1.0, self.avg_doc_len)))
                score_t = idf * (num / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + score_t

        results: List[Tuple[SkillCandidate, float]] = []
        for name, raw_score in scores.items():
            candidate = self._candidates.get(name)
            if not candidate:
                continue
            final_score = raw_score * (boost_project_local if candidate.is_project_local else 1.0)
            cand_copy = candidate.model_copy()
            cand_copy.score = final_score
            results.append((cand_copy, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
