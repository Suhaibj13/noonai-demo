# shared_state.py
from dataclasses import dataclass

@dataclass
class ObservationState:
    awaiting_background: bool = False
    last_analysis_text: str = ""
    last_analysis_title: str = ""

obs_state = ObservationState()
