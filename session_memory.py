# session_memory.py
from collections import deque
from typing import Deque, Dict, Optional, Tuple, Callable, List
import re, time

# A single buffer entry: (timestamp_seconds, query, reply)
Entry = Tuple[float, str, str]

class ConversationHistory:
    def __init__(self, maxlen: int = 5):
        self.maxlen = maxlen
        self._buf: Dict[str, Deque[Entry]] = {
            "web": deque(maxlen=maxlen),
            "analysis": deque(maxlen=maxlen),
            "data": deque(maxlen=maxlen),
        }

    def log(self, mode: str, query: str, reply: str) -> None:
        if mode not in self._buf:
            self._buf[mode] = deque(maxlen=self.maxlen)
        self._buf[mode].append((time.time(), query or "", reply or ""))

    def last_reply(self, mode: str) -> Optional[str]:
        dq = self._buf.get(mode)
        if not dq:
            return None
        return dq[-1][2] if dq else None

    def latest_reply_any(self, prefer_order: Optional[List[str]] = None) -> Optional[str]:
        prefer_order = prefer_order or ["web", "analysis", "data"]
        latest: Optional[Entry] = None
        latest_mode: Optional[str] = None
        for m, dq in self._buf.items():
            if dq:
                t, q, r = dq[-1]
                if (latest is None) or (t > latest[0]) or (t == latest[0] and prefer_order.index(m) < prefer_order.index(latest_mode or m)):
                    latest, latest_mode = (t, q, r), m
        return latest[2] if latest else None

# ---- intent detection for observation ----
_OBS_PATTERNS = [
    r"\bcreate (an )?observation\b",
    r"\bmake (an )?observation\b",
    r"\bwrite (an )?observation\b",
    r"\bobservation on (this|it|that|above)\b",
    r"\bturn this into an observation\b",
    r"\bgive me (the )?observation\b",
]

def wants_observation(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _OBS_PATTERNS)

# ---- observation builder using a provided llm() callable ----
def build_observation(llm: Callable[[str, str, float], str], source_text: str) -> str:
    system = (
        "You are AURA, an Internal Audit assistant. Write concise, board-ready text. "
        "Be specific, factual, and action-oriented. Avoid fluff."
    )
    prompt = f"""
Convert the SOURCE text into an Internal Audit observation with EXACTLY these H2 sections
(in this order, no extra sections):

## Background
## Observation
## Risk
## Recommendation

Rules:
- Derive a brief Background from context implied in SOURCE.
- Keep wording tight and factual.
- If numbers exist, cite simply (e.g., SAR 125.6M; top-3 ≈ 31%).
- Risk: state financial/operational exposure.
- Recommendation: practical, control-focused, testable.
- Output only Markdown, no preamble.

SOURCE:
{source_text.strip()}
"""
    return llm(prompt, system, 0.2)

# one shared instance
history = ConversationHistory(maxlen=5)
