from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from core.logger import get_logger

logger = get_logger(__name__)

class EditorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = (
            "You are a strict technical editor. Review the draft against standard blog guidelines. "
            "If it is excellent, respond with exactly: 'APPROVED'. "
            "If it is flawed, provide specific, constructive feedback on what must be rewritten."
        )

    def __call__(self, state: AgentState):
        logger.info("--- [EDITOR] Reviewing Draft ---")
        draft = state.get("draft", "")
        
        # If the human already manually approved it during HITL, skip AI review
        if state.get("feedback") == "APPROVED_BY_HUMAN":
            logger.info("Draft was manually approved. Bypassing AI evaluation.")
            return {"feedback": "APPROVED"}
            
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Please review this draft:\n\n{draft}")
        ]
        
        response = self.llm.invoke(messages)
        feedback = response.content.strip()
        
        if "APPROVED" in feedback.upper():
            logger.info("Draft APPROVED by AI Editor.")
            return {"feedback": "APPROVED"}
        else:
            logger.info("Draft REJECTED by AI Editor. Routing back to Writer.")
            return {"feedback": feedback}
