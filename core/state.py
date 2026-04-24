from typing import TypedDict, Dict

class AgentState(TypedDict):
    topic: str
    research_data: Dict[str, str]
    draft: str
    feedback: str
    iterations: int
