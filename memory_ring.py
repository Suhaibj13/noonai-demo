# memory_ring.py
import os, time, hashlib
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

def _now_ms() -> int:
    return int(time.time() * 1000)

class MemoryManager:
    """
    In-process, per-session ring buffer.
    Each session_id maps to a deque(maxlen=5) holding dicts: {role, content, time, meta}
    Thread-safe enough for typical Flask + gunicorn threads; if you need multi-instance safety,
    swap to Redis (same interface).
    """
    def __init__(self, maxlen: int = 5):
        self._maxlen = maxlen
        self._store: Dict[str, deque] = {}

    def _ensure(self, session_id: str) -> deque:
        if session_id not in self._store:
            self._store[session_id] = deque(maxlen=self._maxlen)
        return self._store[session_id]

    def add(self, session_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None):
        buf = self._ensure(session_id)
        buf.append({
            "role": role,
            "content": content,
            "time": _now_ms(),
            "meta": meta or {}
        })

    def history(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._ensure(session_id))

    def last_answer(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the most recent assistant message, if any.
        """
        for msg in reversed(self._ensure(session_id)):
            if msg.get("role") == "assistant":
                return msg
        return None

    def size(self, session_id: str) -> int:
        return len(self._ensure(session_id))


def derive_session_id(remote_addr: str, user_agent: str) -> str:
    """
    Stateless derivation so we don't need to modify front-end cookies right now.
    Good enough for dev; for prod, consider a real session cookie.
    """
    raw = f"{remote_addr}|{user_agent}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
