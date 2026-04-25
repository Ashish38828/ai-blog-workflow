from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from core.logger import get_logger

# Initialize the logger for this specific file
logger = get_logger(__name__)

class ResearchAgent:
    def __init__(self, openai_llm, claude_llm, gemini_llm, local_deepseek, local_llama):
        self.llm_map = {
            "OpenAI": openai_llm,
            "Anthropic Claude": claude_llm,
            "Google Gemini": gemini_llm,
            "Local DeepSeek-R1": local_deepseek,
            "Local Llama 3.1": local_llama
        }
        self.system_prompt = (
            "You are an expert AI researcher. Provide a short, factual summary "
            "about the latest capabilities regarding the requested topic, "
            "specifically focusing on {model_name}."
        )

    def __call__(self, state: AgentState):
        logger.info("--- [RESEARCHERS] Gathering Data (3 Cloud APIs, 2 Local Models) ---")
        topic = state.get("topic", "the requested topic")
        
        research_data = {}
        
        for model_name, target_llm in self.llm_map.items():
            logger.info(f"Sending prompt to: {model_name}...")
            
            messages = [
                SystemMessage(content=self.system_prompt.format(model_name=model_name)),
                HumanMessage(content=f"Topic: {topic}")
            ]
            
            response = target_llm.invoke(messages)
            research_data[model_name] = response.content.strip()
            
        logger.info("All 5 models successfully queried.")
        return {"research_data": research_data, "iterations": state.get("iterations", 0)}
