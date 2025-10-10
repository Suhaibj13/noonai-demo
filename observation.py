# observation.py
import re
from typing import Callable

OBS_KEYWORDS = [
    r"\bcreate (an )?observation\b",
    r"\bmake (an )?observation\b",
    r"\bwrite (an )?observation\b",
    r"\bobservation on (this|it)\b",
    r"\bturn this into an observation\b",
]

def detect_observation_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in OBS_KEYWORDS)

def generate_observation(llm: Callable[[str, str, float], str],
                         background: str,
                         analysis_text: str) -> str:
    system = (
        "You are AURA, an Internal Audit assistant. Produce concise, "
        "board-ready writing in clear business English. Be specific, use evidence, "
        "avoid waffle, and keep each section scannable."
    )
    prompt = f"""
Using the BACKGROUND and ANALYSIS below, rewrite into a polished Internal Audit observation.

Rules:
- Output Markdown with the following H2 sections only (in this exact order):
  ## Background
  ## Observation
  ## Risk
  ## Recommendation
- Rewrite and tighten wording; keep it factual and action-oriented.
- If numbers exist in the analysis, cite them simply (e.g., SAR 125.6M, top-3 ≈ 31%).
- Risk should mention financial and/or operational impact.
- Recommendation should be practical, control-focused, and testable.

BACKGROUND:
{background.strip()}

ANALYSIS (source for facts and findings):
{analysis_text.strip()}
"""
    return llm(prompt, system, 0.2)

# ---- ADD THIS AT THE END OF observation.py ----
def build_observation_from_text(source_text: str, extra_meta: dict | None = None) -> dict:
    """
    Safe wrapper that builds an observation purely from provided text.
    Keeps existing module API intact and avoids any dependency on shared/global state.
    """
    extra_meta = extra_meta or {}
    if not source_text or not source_text.strip():
        return {
            "ok": False,
            "reply": "No source content found to create an observation.",
            "observation": None
        }

    # If you already have a core builder function, re-use it here.
    # Example: return build_observation(source_text=source_text, meta=extra_meta)
    # Fallback minimal structure:
    observation = {
        "title": "Observation from previous analysis",
        "summary": source_text.strip()[:1000],
        "details": source_text.strip(),
        "meta": extra_meta
    }
    return {
        "ok": True,
        "reply": "Created observation from the previous answer.",
        "observation": observation
    }
