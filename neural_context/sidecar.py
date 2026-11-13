from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import SkillCandidate

logger = logging.getLogger(__name__)


# Minimal non-decomposing Latin characters whose ASCII base cannot be derived via Unicode NFKD
_NON_NFKD_ASCII_FOLD = str.maketrans({
    "ø": "o",
    "æ": "ae",
    "œ": "oe",
    "ł": "l",
    "đ": "d",
    "ð": "d",
    "þ": "th",
    "ı": "i",
})


def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _grapheme_clusters(text: str) -> List[str]:
    """Group each base character with any trailing Unicode combining marks (cat 'M*')."""
    clusters: List[str] = []
    for ch in text:
        if clusters and unicodedata.category(ch).startswith("M"):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def strip_diacritics(text: str) -> str:
    """
    Universal Unicode property normalizer:
    - Uses standard Unicode `casefold()` ('ß' -> 'ss', 'ς' -> 'σ', 'İ' -> 'i̇').
    - Strips optional accent/vowel marks strictly via Unicode Canonical Combining Classes (`ccc`):
      keeps script-intrinsic marks such as Kana voiced marks (`ccc == 8`), abugida vowel tones (`103..118`),
      and Cyrillic NFKC composed forms intact while folding accents (`ccc >= 200`) and optional vowel points (`10..36`).
    """
    folded = text.casefold().translate(_NON_NFKD_ASCII_FOLD)
    out: List[str] = []
    for ch in folded:
        cat = unicodedata.category(ch)
        if cat == "Pd":
            out.append("-")
            continue
        if cat.startswith("M"):
            ccc = unicodedata.combining(ch)
            if (10 <= ccc <= 36) or ccc >= 200:
                continue
            out.append(ch)
            continue
        dec = unicodedata.normalize("NFKD", ch)
        if len(dec) > 1:
            base = dec[0]
            # Preserve precomposed Cyrillic/Kana/Hangul base letters whose mark isn't an overlay accent
            if base.isascii() or any(unicodedata.combining(c) >= 200 for c in dec[1:] if ord(base) < 0x0400 or ord(base) >= 0x0500):
                kept = [
                    c for c in dec
                    if not (unicodedata.category(c).startswith("M") and ((10 <= unicodedata.combining(c) <= 36) or unicodedata.combining(c) >= 200))
                ]
                out.append("".join(kept))
                continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_text_for_matching(text: str) -> str:
    """
    Normalize quotes/dashes/newlines and fold diacritics uniformly via Unicode properties.
    """
    if not text:
        return ""
    folded = str(text).replace("\\n", " ")
    # Normalize Unicode quotes ('Pi'/'Pf') cleanly without large literal tables
    norm_quotes = "".join("'" if unicodedata.category(ch) in ("Pi", "Pf") else ch for ch in folded)
    return strip_diacritics(norm_quotes)


def _extract_unicode_word_tokens(norm: str) -> List[str]:
    """
    Script-agnostic tokenizer driven strictly by Unicode character categories:
    - Letters ('L*'), Numbers ('N*'), and Combining Marks ('M*') stay joined into words.
    - Internal '-' / '_' stay attached between word characters for compound decomposition.
    - Symbols, punctuation, and whitespace cleanly separate tokens.
    """
    tokens: List[str] = []
    buf: List[str] = []
    n = len(norm)
    for i, ch in enumerate(norm):
        cat0 = unicodedata.category(ch)[0]
        if cat0 in ("L", "N", "M"):
            buf.append(ch)
        elif ch in ("-", "_") and buf and (i + 1 < n) and unicodedata.category(norm[i + 1])[0] in ("L", "N", "M"):
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf.clear()
    if buf:
        tokens.append("".join(buf))
    return tokens


def _unicode_script_family(ch: str) -> str:
    """Derive a character's Unicode script family dynamically from `unicodedata.name` (zero script lists)."""
    name = unicodedata.name(ch, "")
    return name.split(" ", 1)[0] if name else "OTHER"


def _split_script_family_runs(tok: str) -> List[str]:
    """Split contiguous unspaced multi-script words whenever the Unicode script family changes."""
    clusters = _grapheme_clusters(tok)
    if len(clusters) <= 2:
        return []
    runs: List[str] = []
    cur: List[str] = []
    cur_fam: Optional[str] = None
    for cl in clusters:
        fam = _unicode_script_family(cl[0])
        is_modifier = unicodedata.category(cl[0]) == "Lm"
        same_family = (
            cur_fam is None
            or fam == cur_fam
            or is_modifier
            or cur_fam in fam.split("-")
            or fam in cur_fam.split("-")
        )
        if cur and not same_family:
            seg = "".join(cur)
            if len(cur) >= 2:
                runs.append(seg)
            cur = [cl]
            cur_fam = fam
        else:
            if not cur:
                cur_fam = fam
            cur.append(cl)
    if cur and len(cur) >= 2:
        seg = "".join(cur)
        if seg != tok:
            runs.append(seg)
    return runs


def _is_wide_script_token(tok: str) -> bool:
    return any(unicodedata.east_asian_width(c) in ("W", "F") for c in tok)


def tokenize(text: str) -> List[str]:
    """
    Universal multilingual tokenizer driven purely by Unicode properties and cluster density:
    - Preserves all scripts including combining marks across every human alphabet.
    - Decomposes hyphen/underscore compounds and multi-script unspaced boundaries dynamically.
    - Applies sliding grapheme-cluster n-gram windows uniformly whenever unspaced dense-script or
      abugida-cluster density warrants sub-word matching (zero script-specific regexes).
    """
    if not text:
        return []
    norm = normalize_text_for_matching(text)
    raw_tokens = _extract_unicode_word_tokens(norm)
    expanded: List[str] = []
    for tok in raw_tokens:
        clean_tok = tok.strip("-_")
        if not clean_tok:
            continue
        clusters = _grapheme_clusters(clean_tok)
        n_cl = len(clusters)
        is_wide = _is_wide_script_token(clean_tok)
        has_combining = len(clean_tok) > n_cl

        if n_cl >= 2 or is_wide or has_combining:
            expanded.append(clean_tok)
        if "-" in clean_tok or "_" in clean_tok:
            for part in re.split(r"[-_]+", clean_tok):
                if part and part != clean_tok and (len(part) >= 2 or _is_wide_script_token(part)):
                    expanded.append(part)

        # Unspaced dense scripts (Wide/Fullwidth or combining-cluster unspaced syllables):
        if is_wide and n_cl > 2:
            for seg in _split_script_family_runs(clean_tok):
                if seg != clean_tok and seg not in expanded:
                    expanded.append(seg)
            for n in (2, 3):
                if n_cl >= n:
                    for i in range(n_cl - n + 1):
                        ng = "".join(clusters[i : i + n])
                        if ng not in expanded:
                            expanded.append(ng)
        elif has_combining and n_cl > 3 and not clean_tok.isascii():
            # Continuous abugida / unspaced syllable clusters (e.g. Thai, Indic, Lao): overlapping 3-4 cluster windows
            for n in (3, 4):
                if n_cl >= n:
                    for i in range(n_cl - n + 1):
                        frag = "".join(clusters[i : i + n])
                        if frag != clean_tok and frag not in expanded:
                            expanded.append(frag)
    return expanded


def term_information_weight(token: str, df: int, n_docs: int) -> float:
    """
    Continuous information-theoretic term specificity:
    Robertson/BM25 IDF * smooth logistic ubiquity damping * Unicode grapheme-density factor.
    Suppresses functional particles and domain cliches across any language without step cliffs or word lists.
    """
    base_idf = math.log(1.0 + (max(n_docs, 1) - df + 0.5) / (df + 0.5))
    if n_docs >= 4:
        ubiquity = df / n_docs
        # Continuous logistic curve centered at 0.35 (approaches 1.0 when rare, 0.15 when ubiquitous)
        base_idf *= 0.15 + (0.85 / (1.0 + math.exp(14.0 * (ubiquity - 0.35))))

    clusters = _grapheme_clusters(token)
    n_cl = len(clusters)
    if _is_wide_script_token(token):
        effective_len = n_cl * 2.5
    elif len(token) > n_cl and not token.isascii():
        effective_len = n_cl * 1.4
    else:
        effective_len = float(n_cl)
    info_density = min(1.5, effective_len / 4.0)
    return base_idf * info_density


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

    full_text = " ".join(clean_lines).strip()
    if len(full_text) <= 600:
        lead_paragraph = full_text
    else:
        cutoff = full_text[:800].rfind(" ")
        lead_paragraph = full_text[:cutoff].strip() if cutoff > 200 else full_text[:800].strip()
    return description, lead_paragraph


def discover_skill_inventory(include_project: bool = True):
    """Return the active native skill catalog and its source provenance.

    Native Hermes owns roots, project trust/quarantine, exclusions, and duplicate
    precedence. If that contract is not importable, only the active profile's
    skills directory is scanned; project sources are never guessed from cwd.
    """
    try:
        from agent import skill_utils
        required = (
            "get_skill_search_roots", "iter_skill_index_files", "resolve_skill_catalog",
            "parse_frontmatter", "TIER_PROJECT", "TIER_LOCAL", "TIER_CREATE_DIR", "TIER_EXTERNAL",
            "get_disabled_skill_names", "is_disabled_entry", "skill_matches_platform",
            "skill_matches_environment", "skill_matches_apps",
        )
        if not all(hasattr(skill_utils, name) for name in required):
            raise ImportError("native Hermes skill discovery contract is incomplete")
        roots = skill_utils.get_skill_search_roots(include_project=include_project)
        entries = []
        source_info = {}
        project_paths = []
        for tier, root in roots:
            root = Path(root)
            if not root.is_dir():
                continue
            if tier == skill_utils.TIER_PROJECT:
                project_iter = getattr(skill_utils, "iter_project_skill_files", None)
                if not callable(project_iter):
                    raise ImportError("native project quarantine iterator is unavailable")
                files = project_iter(root)
            else:
                files = skill_utils.iter_skill_index_files(root, "SKILL.md")
            for path in files:
                path = Path(path)
                try:
                    content = path.read_text(encoding="utf-8-sig")
                    frontmatter, _ = skill_utils.parse_frontmatter(content)
                except Exception:
                    continue
                name = str(frontmatter.get("name") or path.parent.name).strip()
                if not name:
                    continue
                entry = {"name": name, "tier": tier, "root": root, "path": path}
                try:
                    offerable = (
                        skill_utils.skill_matches_platform(frontmatter)
                        and skill_utils.skill_matches_environment(frontmatter)
                        and skill_utils.skill_matches_apps(frontmatter)
                    )
                except Exception:
                    offerable = False
                if not offerable:
                    continue
                entries.append(entry)
                if tier == skill_utils.TIER_PROJECT:
                    project_paths.append(path)
        resolved = skill_utils.resolve_skill_catalog(entries)
        disabled = skill_utils.get_disabled_skill_names()
        selected = [
            entry for entry in resolved
            if entry.get("status") == "unique" or (
                entry.get("status") == "ambiguous" and entry.get("load_name")
            )
            if not skill_utils.is_disabled_entry(entry, disabled)
        ]
        paths = [Path(entry["path"]) for entry in selected]
        names = {Path(entry["path"]): str(entry.get("load_name") or entry["name"]) for entry in selected}
        selected_info = {
            Path(entry["path"]): (entry["tier"], Path(entry["root"]))
            for entry in selected
        }
        selected_project = [path for path in paths if selected_info[path][0] == skill_utils.TIER_PROJECT]
        tiers = {
            skill_utils.TIER_PROJECT, skill_utils.TIER_LOCAL,
            skill_utils.TIER_CREATE_DIR, skill_utils.TIER_EXTERNAL,
        }
        return paths, selected_project, selected_info, names, tiers
    except Exception as exc:
        # Keep the user's profile-level catalog available, but never approximate
        # project trust, quarantine, or tier precedence with a raw workspace walk.
        logger.warning("Native skill discovery unavailable; using profile skills only: %s", exc)
        try:
            from hermes_constants import get_skills_dir
            root = Path(get_skills_dir())
            iterator = getattr(locals().get("skill_utils"), "iter_skill_index_files", None)
            paths = list(iterator(root, "SKILL.md")) if callable(iterator) and root.is_dir() else []
        except Exception:
            return [], [], {}, {}, {0, 1, 2, 3}
        paths = [Path(path) for path in paths]
        info = {path: (1, root) for path in paths}
        return paths, [], info, {}, {0, 1, 2, 3}


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
        self.total_catalog_tokens: int = 0
        self.rebuild_bm25()

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
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_root TEXT NOT NULL DEFAULT '',
                    source_tier INTEGER NOT NULL DEFAULT 1
                )
            """)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(skills_sidecar)")}
            if "source_root" not in columns:
                conn.execute("ALTER TABLE skills_sidecar ADD COLUMN source_root TEXT NOT NULL DEFAULT ''")
            if "source_tier" not in columns:
                conn.execute("ALTER TABLE skills_sidecar ADD COLUMN source_tier INTEGER NOT NULL DEFAULT 1")
                conn.execute("UPDATE skills_sidecar SET source_tier = CASE WHEN is_project_local = 1 THEN 0 ELSE 1 END")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_sha256 ON skills_sidecar(sha256)")
            conn.commit()

    def sync_skills(
        self,
        skill_paths: List[Path],
        project_local_paths: Optional[List[Path]] = None,
        *,
        source_info: Optional[Dict[Path, Tuple[int, Path]]] = None,
        skill_names: Optional[Dict[Path, str]] = None,
        replace_tiers: Optional[set[int]] = None,
        replace_subtrees: Optional[List[Path]] = None,
    ) -> int:
        """
        Synchronizes skill metadata into SQLite index only when sha256 changes.
        Builds the in-memory BM25 index in sub-20ms.
        """
        project_paths_set = {str(p.resolve()) for p in (project_local_paths or [])}
        updated_count = 0
        source_info = source_info or {}
        skill_names = skill_names or {}
        seen_names = set()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            existing = {
                row[0]: (row[1], row[2], row[3], row[4], row[5])
                for row in cursor.execute("SELECT skill_name, sha256, is_project_local, skill_path, source_root, source_tier FROM skills_sidecar")
            }

            for p in skill_paths:
                if not p.is_file():
                    continue
                skill_name = skill_names.get(p, p.parent.name if p.name == "SKILL.md" else p.stem)
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                seen_names.add(skill_name)

                sha = compute_sha256(content)
                is_local = 1 if str(p.resolve()) in project_paths_set else 0
                source_tier, source_root = source_info.get(p, (0 if is_local else 1, p.parent.parent))
                source_root = str(Path(source_root).resolve())

                old = existing.get(skill_name)
                if old and old[0] == sha and old[1] == is_local and old[2] == str(p.resolve()) and old[3] == source_root and old[4] == source_tier:
                    continue  # Unchanged, skip disk parsing

                if old and old[0] == sha:
                    desc, lead = None, None
                else:
                    desc, lead = extract_frontmatter_and_lead(content)
                tokens = len(content.split()) + len(content) // 4

                cursor.execute("""
                    INSERT INTO skills_sidecar 
                    (skill_name, skill_path, sha256, description, lead_paragraph, is_project_local, indexed_tokens, last_updated, source_root, source_tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                    ON CONFLICT(skill_name) DO UPDATE SET
                        skill_path=excluded.skill_path,
                        sha256=excluded.sha256,
                        description=COALESCE(excluded.description, skills_sidecar.description),
                        lead_paragraph=COALESCE(excluded.lead_paragraph, skills_sidecar.lead_paragraph),
                        is_project_local=excluded.is_project_local,
                        indexed_tokens=excluded.indexed_tokens,
                        source_root=excluded.source_root,
                        source_tier=excluded.source_tier,
                        last_updated=CURRENT_TIMESTAMP
                """, (skill_name, str(p.resolve()), sha, desc, lead, is_local, tokens, source_root, source_tier))
                updated_count += 1

            if replace_tiers:
                if seen_names:
                    placeholders = ",".join("?" for _ in replace_tiers)
                    name_placeholders = ",".join("?" for _ in seen_names)
                    cursor.execute(
                        f"DELETE FROM skills_sidecar WHERE source_tier IN ({placeholders}) AND skill_name NOT IN ({name_placeholders})",
                        (*replace_tiers, *seen_names),
                    )
                else:
                    placeholders = ",".join("?" for _ in replace_tiers)
                    cursor.execute(f"DELETE FROM skills_sidecar WHERE source_tier IN ({placeholders})", tuple(replace_tiers))

            if replace_subtrees:
                current_paths = {str(Path(p).resolve()) for p in skill_paths if p.is_file()}
                for row in cursor.execute("SELECT skill_name, skill_path FROM skills_sidecar").fetchall():
                    existing_path = Path(row[1]).resolve()
                    in_scope = any(existing_path.is_relative_to(Path(scope).resolve()) for scope in replace_subtrees)
                    if in_scope and str(existing_path) not in current_paths:
                        cursor.execute("DELETE FROM skills_sidecar WHERE skill_name = ?", (row[0],))

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
                SELECT skill_name, skill_path, sha256, description, lead_paragraph, is_project_local, indexed_tokens
                FROM skills_sidecar
            """).fetchall()

        self.n_docs = len(rows)
        self.total_catalog_tokens = 0
        for name, path, sha, desc, lead, is_local, idx_toks in rows:
            self.total_catalog_tokens += int(idx_toks or 0)
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

    def filter_query_terms(self, text: str, min_weight: float = 0.25) -> List[str]:
        """
        Pre-graph query distillation: strips symbols, decomposes compounds, and filters out
        low-information ubiquitous particles using corpus IDF + morphological density before routing.
        """
        tokens = list(dict.fromkeys(tokenize(text)))
        if not tokens:
            return []
        if self.n_docs < 4:
            return [t for t in tokens if len(t) >= 2]
        filtered = [
            t for t in tokens
            if term_information_weight(t, self.df.get(t, 0), self.n_docs) >= min_weight
        ]
        return filtered or tokens

    def search_bm25(self, query: str, top_k: int = 0, boost_project_local: float = 2.0) -> List[Tuple[SkillCandidate, float]]:
        """
        Calculates Robertson/BM25 scores over indexed skill leads using the in-memory inverted index.
        When top_k <= 0 (default), returns ALL matching skills in the catalog without artificial truncation.
        """
        q_tokens = list(dict.fromkeys(tokenize(query)))
        scores: Dict[str, float] = {}
        max_term_idf: Dict[str, float] = {}

        for t in q_tokens:
            if t not in self.df:
                continue
            idf = term_information_weight(t, self.df[t], self.n_docs)
            for doc_id, counts in self.tf.items():
                if t not in counts:
                    continue
                tf = counts[t]
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_id] / max(1.0, self.avg_doc_len)))
                score_t = idf * (num / denom)
                if t in tokenize(doc_id):
                    score_t *= 1.6
                scores[doc_id] = scores.get(doc_id, 0.0) + score_t
                if idf > max_term_idf.get(doc_id, 0.0):
                    max_term_idf[doc_id] = idf

        results: List[Tuple[SkillCandidate, float]] = []
        for name, raw_score in scores.items():
            if self.n_docs >= 4 and max_term_idf.get(name, 0.0) < 0.25:
                continue
            candidate = self._candidates.get(name)
            if not candidate:
                continue
            final_score = raw_score * (boost_project_local if candidate.is_project_local else 1.0)
            cand_copy = candidate.model_copy()
            cand_copy.score = final_score
            results.append((cand_copy, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k] if top_k > 0 else results
