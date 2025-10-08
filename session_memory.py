# session_memory.py
from collections import deque
from typing import Deque, Dict, Optional, Tuple, Callable
import re

class ConversationHistory:
    """Rolling store of last N (query, reply) per mode."""
    def __init__(self, maxlen: int = 5):
        self.maxlen = maxlen
        self._buf: Dict[str, Deque[Tuple[str, str]]] = {
            "web": deque(maxlen=maxlen),
            "analysis": deque(maxlen=maxlen),
            "data": deque(maxlen=maxlen),
        }
        self.awaiting_background: Dict[str, bool] = {"web": False}

    def log(self, mode: str, query: str, reply: str) -> None:
        if mode not in self._buf:
            self._buf[mode] = deque(maxlen=self.maxlen)
        self._buf[mode].append((query or "", reply or ""))

    def last_reply(self, mode: str) -> Optional[str]:
        dq = self._buf.get(mode)
        if not dq:
            return None
        return dq[-1][1] if dq else None

    def last(self, mode: str) -> Optional[Tuple[str, str]]:
        dq = self._buf.get(mode)
        if not dq:
            return None
        return dq[-1] if dq else None


# ---- intent detection for observation ----
_OBS_PATTERNS = [
    r"\bcreate (an )?observation\b",
    r"\bmake (an )?observation\b",
    r"\bwrite (an )?observation\b",
    r"\bobservation on (this|it|that|above)\b",
    r"\bturn this into an observation\b",
]

def wants_observation(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _OBS_PATTERNS)


# ---- observation builder using a provided llm() callable ----
def build_observation(llm: Callable[[str, str, float], str],
                      background: str,
                      source_text: str) -> str:
    system = (
        "You are AURA, an Internal Audit assistant. Write concise, board-ready text. "
        "Be specific, factual, and action-oriented. Avoid fluff."
    )
    prompt = f"""
Rewrite the BACKGROUND + SOURCE (web reply) into an Internal Audit observation with EXACTLY these H2 sections:

## Background
## Observation
## Risk
## Recommendation

Rules:
- Tighten wording, keep evidence.
- If numbers exist, cite simply (e.g., SAR 125.6M; top-3 ≈ 31%).
- Risk: state financial/operational exposure.
- Recommendation: practical, control-focused, testable.

BACKGROUND:
{background.strip()}

SOURCE (prior web reply):
{source_text.strip()}
"""
    return llm(prompt, system, 0.2)


# a single shared instance
history = ConversationHistory(maxlen=5)
