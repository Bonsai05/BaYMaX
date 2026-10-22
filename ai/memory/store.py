"""SQLite store for memory records: canonical rows, an FTS5 index and a vector table.

Canonical data is the `memories` table. `memories_fts` and `vectors` are derived
and can always be rebuilt from it (the "vector index corrupt" row of the v5
failure matrix). Spike item from the architecture docs: if SQLCipher + FTS5 or a
vector extension misbehaves, this class is the only place that has to change,
because vectors are kept in their own table and searched with numpy.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

import numpy as np

from ..contracts import MemoryKind, MemoryRecord, PrivacyClass, TrustTier
from ..text import STOPWORDS

_WORD = re.compile(r"\w+", re.UNICODE)

# A memory carrying any of these flags stays in canonical storage (history, audit) but is
# invisible to retrieval. One list, used by every read path.
FLAG_SUPERSEDED = "superseded"        # replaced by a newer confirmed fact
FLAG_REVOKED = "revoked"              # withdrawn on its own
FLAG_SOURCE_REVOKED = "source_revoked"  # its source (event or file) was revoked
INACTIVE_FLAGS = (FLAG_SUPERSEDED, FLAG_REVOKED, FLAG_SOURCE_REVOKED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    canonical_source TEXT NOT NULL,
    source_device TEXT NOT NULL,
    source_artifact TEXT NOT NULL,
    trust_tier TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    authority TEXT NOT NULL,
    privacy_class TEXT NOT NULL,
    project TEXT NOT NULL,
    quarantined INTEGER NOT NULL,
    flags TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    supersedes TEXT
);
CREATE INDEX IF NOT EXISTS idx_mem_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_mem_source ON memories(canonical_source);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(memory_id UNINDEXED, content, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS memory_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    detail TEXT NOT NULL,
    at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_mem ON memory_audit(memory_id);
CREATE TABLE IF NOT EXISTS vectors (
    memory_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
"""


class MemoryStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ---- writes ----------------------------------------------------------
    def add(self, rec: MemoryRecord) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.memory_id, rec.content_hash, rec.canonical_source, rec.source_device,
                    rec.source_artifact, rec.trust_tier.value, rec.kind.value, rec.content,
                    rec.confidence, rec.authority, rec.privacy_class.value, rec.project,
                    int(rec.quarantined), json.dumps(rec.flags), rec.created_at,
                    rec.updated_at, rec.supersedes,
                ),
            )
            self.db.execute(
                "INSERT INTO memories_fts(memory_id, content) VALUES (?,?)",
                (rec.memory_id, rec.content),
            )

    def put_vectors(self, ids: list[str], model_id: str, vecs: np.ndarray) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                [(i, model_id, int(vecs.shape[1]), vecs[n].astype(np.float32).tobytes())
                 for n, i in enumerate(ids)],
            )

    def clear_vectors(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM vectors")

    def delete(self, memory_id: str) -> None:
        """Hard delete (user-requested). Removes the record and both derived indexes."""
        with self.db:
            self.db.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))
            self.db.execute("DELETE FROM vectors WHERE memory_id=?", (memory_id,))
            self.db.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))


    # ---- lifecycle -------------------------------------------------------
    # Every change below is deterministic code (no LLM) and writes an audit row, so the
    # history of a memory (who confirmed what, when, and what it was before) is never lost.
    def _audit(self, memory_id: str, action: str, actor: str, **detail: Any) -> None:
        self.db.execute(
            "INSERT INTO memory_audit(memory_id, action, actor, detail, at) VALUES (?,?,?,?,?)",
            (memory_id, action, actor, json.dumps(detail, sort_keys=True), time.time()),
        )

    def audit_trail(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT action, actor, detail, at FROM memory_audit WHERE memory_id=? ORDER BY audit_id",
            (memory_id,),
        ).fetchall()
        return [{"action": r["action"], "actor": r["actor"], "detail": json.loads(r["detail"]),
                 "at": r["at"]} for r in rows]

    def _flags(self, memory_id: str) -> Optional[list[str]]:
        row = self.db.execute("SELECT flags FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        return None if row is None else json.loads(row["flags"])

    def inactive_reason(self, memory_id: str) -> Optional[str]:
        """The first inactive flag on a memory, or None if it is active (or does not exist)."""
        return next((f for f in (self._flags(memory_id) or []) if f in INACTIVE_FLAGS), None)

    def is_active(self, memory_id: str) -> bool:
        flags = self._flags(memory_id)
        return flags is not None and not any(f in INACTIVE_FLAGS for f in flags)

    def promote(self, memory_id: str, confirmed: bool = False, actor: str = "user",
                acknowledge_injection_flag: bool = False) -> None:
        """Promote one quarantined memory to trusted (T0) after explicit user confirmation.

        The earlier tier and flags are written to the audit trail, and the injection warning
        flag is kept on the record. A memory that admission flagged as a possible prompt
        injection needs a second, separate acknowledgement.
        """
        if not confirmed:
            raise ValueError("explicit_confirmation_required")
        with self.db:
            row = self.db.execute(
                "SELECT quarantined, flags, trust_tier, authority FROM memories WHERE memory_id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise KeyError(memory_id)
            if not row["quarantined"]:
                return
            flags = json.loads(row["flags"])
            if "possible_prompt_injection" in flags and not acknowledge_injection_flag:
                raise ValueError("injection_flag_requires_acknowledgement")
            if "promoted_by_user" not in flags:
                flags.append("promoted_by_user")
            self.db.execute(
                "UPDATE memories SET quarantined=0, trust_tier='T0', authority='user', flags=?, "
                "updated_at=? WHERE memory_id=?",
                (json.dumps(flags), time.time(), memory_id),
            )
            self._audit(memory_id, "promote", actor, previous_tier=row["trust_tier"],
                        previous_authority=row["authority"], previous_flags=json.loads(row["flags"]),
                        injection_flag_acknowledged=bool(acknowledge_injection_flag))

    def resolve_contradiction(self, new_memory_id: str, old_memory_id: str,
                              confirmed: bool = False, actor: str = "user") -> None:
        """Keep both facts, mark the old one superseded and link the new one to it."""
        if not confirmed:
            raise ValueError("explicit_confirmation_required")
        if new_memory_id == old_memory_id:
            raise ValueError("cannot_supersede_itself")
        with self.db:
            new = self.db.execute("SELECT quarantined FROM memories WHERE memory_id=?",
                                  (new_memory_id,)).fetchone()
            old_flags = self._flags(old_memory_id)
            if new is None or old_flags is None:
                raise KeyError(new_memory_id if new is None else old_memory_id)
            if new["quarantined"]:
                raise ValueError("new_memory_must_be_promoted_first")
            if not self.is_active(new_memory_id):
                raise ValueError("new_memory_is_not_active")
            if FLAG_SUPERSEDED not in old_flags:
                old_flags.append(FLAG_SUPERSEDED)
            now = time.time()
            self.db.execute("UPDATE memories SET flags=?, updated_at=? WHERE memory_id=?",
                            (json.dumps(old_flags), now, old_memory_id))
            self.db.execute("UPDATE memories SET supersedes=?, updated_at=? WHERE memory_id=?",
                            (old_memory_id, now, new_memory_id))
            self._audit(old_memory_id, "superseded", actor, by=new_memory_id)
            self._audit(new_memory_id, "supersedes", actor, replaced=old_memory_id)

    def _revoke_where(self, column: str, value: str, flag: str, actor: str) -> int:
        with self.db:
            rows = self.db.execute(f"SELECT memory_id, flags FROM memories WHERE {column}=?",
                                   (value,)).fetchall()
            changed, now = 0, time.time()
            for row in rows:
                flags = json.loads(row["flags"])
                if flag in flags:
                    continue
                flags.append(flag)
                changed += 1
                self.db.execute("UPDATE memories SET flags=?, updated_at=? WHERE memory_id=?",
                                (json.dumps(flags), now, row["memory_id"]))
                self._audit(row["memory_id"], flag, actor, **{column: value})
            return changed

    def revoke_source(self, canonical_source: str, actor: str = "user") -> int:
        """Revoke every memory made from one event (its canonical source). History is kept."""
        return self._revoke_where("canonical_source", canonical_source, FLAG_SOURCE_REVOKED, actor)

    def revoke_artifact(self, source_artifact: str, actor: str = "user") -> int:
        """Revoke every memory made from one file or URL, across all events that imported it."""
        if not source_artifact:
            raise ValueError("artifact_path_required")
        return self._revoke_where("source_artifact", source_artifact, FLAG_SOURCE_REVOKED, actor)

    def restore(self, memory_id: str, confirmed: bool = False, actor: str = "user") -> None:
        """Undo supersede/revoke on one memory after explicit confirmation."""
        if not confirmed:
            raise ValueError("explicit_confirmation_required")
        with self.db:
            flags = self._flags(memory_id)
            if flags is None:
                raise KeyError(memory_id)
            kept = [f for f in flags if f not in INACTIVE_FLAGS]
            if kept == flags:
                return
            self.db.execute("UPDATE memories SET flags=?, updated_at=? WHERE memory_id=?",
                            (json.dumps(kept), time.time(), memory_id))
            self._audit(memory_id, "restore", actor, removed=[f for f in flags if f in INACTIVE_FLAGS])

    @staticmethod
    def _active_sql() -> tuple[str, list[str]]:
        """SQL fragment that keeps only active memories (used by every retrieval path)."""
        return (" AND ".join("m.flags NOT LIKE ?" for _ in INACTIVE_FLAGS),
                [f'%"{f}"%' for f in INACTIVE_FLAGS])

    # ---- reads -----------------------------------------------------------
    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def get(self, memory_id: str) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()

    def id_for_hash(self, content_hash: str) -> Optional[str]:
        row = self.db.execute(
            "SELECT memory_id FROM memories WHERE content_hash=?", (content_hash,)
        ).fetchone()
        return row[0] if row else None

    def ids_without_vector(self, model_id: str) -> list[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT m.memory_id, m.content FROM memories m "
            "LEFT JOIN vectors v ON v.memory_id = m.memory_id AND v.model_id = ? "
            "WHERE v.memory_id IS NULL",
            (model_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def vector_models(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT DISTINCT model_id FROM vectors")}

    @staticmethod
    def _where(f: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if f.get("project"):
            clauses.append("m.project = ?"); params.append(f["project"])
        if f.get("source_device"):
            clauses.append("m.source_device = ?"); params.append(f["source_device"])
        if f.get("kinds"):
            clauses.append(f"m.kind IN ({','.join('?' * len(f['kinds']))})"); params += f["kinds"]
        if not f.get("include_quarantined", True):
            clauses.append("m.quarantined = 0")
        if f.get("for_cloud"):
            clauses.append("m.privacy_class = ?"); params.append(PrivacyClass.CLOUD_ALLOWED.value)
        active_sql, active_params = MemoryStore._active_sql()
        clauses.append(active_sql); params += active_params
        if f.get("since") is not None:
            clauses.append("m.created_at >= ?"); params.append(f["since"])
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def fts_query(text: str) -> str:
        toks = [t for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]
        return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))

    def lexical_search(self, query: str, limit: int, filters: dict[str, Any]) -> list[tuple[str, float]]:
        match = self.fts_query(query)
        if not match:
            return []
        where, params = self._where(filters)
        sql = (
            "SELECT m.memory_id, bm25(memories_fts) AS s FROM memories_fts "
            "JOIN memories m ON m.memory_id = memories_fts.memory_id "
            f"WHERE memories_fts MATCH ?{where} ORDER BY s LIMIT ?"
        )
        rows = self.db.execute(sql, [match, *params, limit]).fetchall()
        return [(r[0], -float(r[1])) for r in rows]  # higher = better

    def lexical_hit_count(self, query: str, cap: int = 50) -> int:
        match = self.fts_query(query)
        if not match:
            return 0
        active_sql, active_params = self._active_sql()
        row = self.db.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM memories_fts "
            "JOIN memories m ON m.memory_id = memories_fts.memory_id "
            f"WHERE memories_fts MATCH ? AND {active_sql} LIMIT ?)",
            (match, *active_params, cap),
        ).fetchone()
        return int(row[0])

    def vector_search(self, qvec: np.ndarray, model_id: str, limit: int,
                      filters: dict[str, Any]) -> list[tuple[str, float]]:
        where, params = self._where(filters)
        rows = self.db.execute(
            "SELECT v.memory_id, v.vec FROM vectors v JOIN memories m ON m.memory_id = v.memory_id "
            f"WHERE v.model_id = ?{where}",
            [model_id, *params],
        ).fetchall()
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        sims = mat @ qvec.astype(np.float32)
        top = np.argsort(-sims)[:limit]
        return [(rows[i][0], float(sims[i])) for i in top]

    def rows_by_ids(self, ids: Iterable[str]) -> dict[str, sqlite3.Row]:
        ids = list(ids)
        if not ids:
            return {}
        rows = self.db.execute(
            f"SELECT * FROM memories WHERE memory_id IN ({','.join('?' * len(ids))})", ids
        ).fetchall()
        return {r["memory_id"]: r for r in rows}

    def record(self, memory_id: str) -> Optional[MemoryRecord]:
        r = self.get(memory_id)
        if r is None:
            return None
        return MemoryRecord(
            memory_id=r["memory_id"], canonical_source=r["canonical_source"],
            source_device=r["source_device"], source_artifact=r["source_artifact"],
            trust_tier=TrustTier(r["trust_tier"]), kind=MemoryKind(r["kind"]),
            content=r["content"], confidence=r["confidence"], authority=r["authority"],
            privacy_class=PrivacyClass(r["privacy_class"]), project=r["project"],
            quarantined=bool(r["quarantined"]), flags=json.loads(r["flags"]),
            created_at=r["created_at"], updated_at=r["updated_at"],
            supersedes=r["supersedes"], content_hash=r["content_hash"],
        )
