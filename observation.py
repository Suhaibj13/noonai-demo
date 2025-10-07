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
