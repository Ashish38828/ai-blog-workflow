from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from core.logger import get_logger

logger = get_logger(__name__)

class WriterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = (
            "You are an expert technical writer. Synthesize the provided research "
            "into a cohesive, engaging blog draft. Incorporate any feedback provided."
        )

    def __call__(self, state: AgentState):
        iteration = state.get('iterations', 0) + 1
        logger.info(f"--- [WRITER] Drafting Content (Iteration {iteration}) ---")
        
        research = state.get("research_data", {})
        feedback = state.get("feedback", "")
        
        content = f"Topic: {state['topic']}\n\nResearch Aggregated:\n{research}"
        
        if feedback and feedback != "APPROVED":
            logger.info("Applying previous rejection feedback to new draft.")
            content += f"\n\nCRITICAL FEEDBACK TO ADDRESS:\n{feedback}"
            
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=content)
        ]
        
        response = self.llm.invoke(messages)
        logger.info("Draft successfully synthesized.")
        
        return {"draft": response.content.strip(), "iterations": iteration}
