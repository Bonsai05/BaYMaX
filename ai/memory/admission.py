"""Deterministic memory admission gate.

Rules from the v3/v5 architecture:
  * Trust tier and origin are decided here, from `event.ingest_path`, which the Core
    sets from the code path that wrote the event. Content can never raise its own tier
    and the LLM is never asked to decide admission.
  * Unknown ingest paths are treated as T2 (deny by default).
  * T2 content is stored as quarantined evidence: retrievable as quoted data, never
    promoted to profile memory, and it carries no authority.
  * Secrets are not stored in memory at all; the Core should route them to its
    encrypted store.
  * Privacy defaults to LOCAL_ONLY. Only the user (T0) can mark an item cloud_allowed.

The gate is plain code so it can be unit tested and measured (admission precision
and recall in the benchmark).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

from ..contracts import (
    INGEST_ASSISTANT_CONVERSATION, INGEST_USER_CONFIRMED, INGEST_USER_TYPED,
    AdmissionResult, Decision, Event, MemoryKind, MemoryRecord, PrivacyClass, TrustTier,
)
from .store import MemoryStore

TIER_BY_INGEST_PATH = {
    INGEST_USER_TYPED: TrustTier.T0,
    INGEST_USER_CONFIRMED: TrustTier.T0,
    INGEST_ASSISTANT_CONVERSATION: TrustTier.T1,
}

INJECTION_PATTERNS = [re.compile(p, re.I) for p in [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions?|prompts?|rules?)",
    r"disregard (the |all |your )?(previous|prior|above|system)",
    r"\byou are now\b",
    r"new (system )?instructions?\s*:",
    r"(reveal|show|print) (your|the) (system|hidden) (prompt|instructions?)",
    r"(run|execute) (this|the following) (command|code|script)",
    r"\brm\s+-rf\b",
    r"do not (tell|inform) the user",
    r"forget .{0,60}(told you|instructions?|rules?)",
    r"act as (an? )?(admin|root|system)",
    r"<\s*/?\s*system\s*>",
    r"send (all |the )?(files?|data|contents?) to\b",
]]

SECRET_PATTERNS = [
    ("password_or_key", re.compile(r"(?i)\b(password|passwd|pwd|api[_-]?key|secret|access[_-]?token)\s*[:=]\s*\S+")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("api_token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
]
CARD_LIKE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
AADHAAR_LIKE = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")

PROFILE_CUES = re.compile(
    r"\b(i (prefer|always|never|usually|like to)|my (name|email|birthday|favou?rite) is|remember that i)\b", re.I
)

MIN_CHARS = 3
CHUNK_CHARS = 800


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\r\n?", "\n", text)).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Pack paragraphs into chunks of at most max_chars (long paragraphs are split)."""
    chunks, cur = [], ""
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(para[:cut].strip()); para = para[cut:].strip()
        if len(cur) + len(para) + 1 > max_chars and cur:
            chunks.append(cur); cur = ""
        cur = f"{cur}\n{para}".strip() if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def find_secret(text: str) -> str | None:
    for name, pat in SECRET_PATTERNS:
        if pat.search(text):
            return name
    for m in CARD_LIKE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return "card_number"
    return None


def classify_kind(event: Event, tier: TrustTier, text: str) -> tuple[MemoryKind, float, list[str]]:
    flags: list[str] = []
    et = event.event_type
    if PROFILE_CUES.search(text):
        if tier == TrustTier.T0:
            return MemoryKind.PROFILE, 0.8, flags
        flags.append("profile_claim_not_promoted")  # only the user can write profile memory
    if et in ("file", "document", "artifact"):
        return MemoryKind.ARTIFACT, 0.9, flags
    if et in ("device", "capability"):
        return MemoryKind.DEVICE, 0.9, flags
    if et in ("message", "conversation", "task", "activity"):
        return MemoryKind.EPISODIC, 0.9, flags
    if et == "note":
        return MemoryKind.SEMANTIC, 0.8, flags
    return MemoryKind.EPISODIC, 0.5, flags


class AdmissionGate:
    def __init__(self, store: MemoryStore):
        self.store = store

    def admit(self, event: Event) -> AdmissionResult:
        text = normalize(str(event.payload.get("text", "")))
        if len(text) < MIN_CHARS:
            return AdmissionResult(Decision.DISCARD, "empty_or_too_short")

        secret = find_secret(text)
        if secret:
            # Never stored in memory. The Core should send it to the encrypted store.
            return AdmissionResult(Decision.DISCARD, f"secret_detected:{secret}", flags=[secret])

        tier = TIER_BY_INGEST_PATH.get(event.ingest_path, TrustTier.T2)
        flags: list[str] = []
        if event.ingest_path not in TIER_BY_INGEST_PATH:
            flags.append("untrusted_ingest_path")

        if any(p.search(text) for p in INJECTION_PATTERNS):
            flags.append("possible_prompt_injection")

        kind, conf, kind_flags = classify_kind(event, tier, text)
        flags += kind_flags
        if AADHAAR_LIKE.search(text):
            flags.append("pii_id_number")

        # Quarantine anything untrusted, and anything suspicious regardless of tier.
        quarantined = tier == TrustTier.T2 or "possible_prompt_injection" in flags
        if quarantined and kind == MemoryKind.PROFILE:
            kind = MemoryKind.SEMANTIC
        authority = "user" if tier == TrustTier.T0 and not quarantined else "none"

        privacy = PrivacyClass.LOCAL_ONLY
        if (tier == TrustTier.T0 and not quarantined
                and event.payload.get("privacy") == PrivacyClass.CLOUD_ALLOWED.value
                and "pii_id_number" not in flags and kind != MemoryKind.PROFILE):
            privacy = PrivacyClass.CLOUD_ALLOWED

        ids: list[str] = []
        inactive_dupes: list[str] = []
        for n, chunk in enumerate(chunk_text(text)):
            h = content_hash(chunk)
            existing = self.store.id_for_hash(h)
            if existing:
                # Already stored. If it was superseded or revoked it stays that way: a re-sync
                # of a revoked source must not bring it back. The Core can offer restore_memory.
                if self.store.inactive_reason(existing):
                    inactive_dupes.append(existing)
                continue
            mem_id = "mem_" + hashlib.sha1(f"{event.event_id}:{n}".encode()).hexdigest()[:14]
            self.store.add(MemoryRecord(
                memory_id=mem_id, canonical_source=event.event_id,
                source_device=event.source_device_id,
                source_artifact=str(event.payload.get("path", "")),
                trust_tier=tier, kind=kind, content=chunk, confidence=conf,
                authority=authority, privacy_class=privacy,
                project=str(event.payload.get("project", "")),
                quarantined=quarantined, flags=list(flags),
                created_at=event.created_at, updated_at=event.created_at,
                content_hash=h,
            ))
            ids.append(mem_id)

        if not ids:
            if inactive_dupes:
                why = self.store.inactive_reason(inactive_dupes[0])
                return AdmissionResult(Decision.DISCARD, f"duplicate_of_inactive:{why}:{inactive_dupes[0]}",
                                       memory_ids=inactive_dupes, trust_tier=tier, flags=flags)
            return AdmissionResult(Decision.DISCARD, "duplicate", trust_tier=tier, flags=flags)
        decision = Decision.QUARANTINE if quarantined else Decision.ADMIT
        reason = "quarantined:" + ",".join(flags or [tier.value]) if quarantined else f"admitted:{tier.value}"
        return AdmissionResult(decision, reason, memory_ids=ids, trust_tier=tier, flags=flags)
