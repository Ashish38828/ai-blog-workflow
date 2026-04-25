from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

class ResearchAgent:
    def __init__(self, openai_llm, claude_llm, gemini_llm, deepseek_llm, ollama_llm):
        # Map the model names to their actual LangChain API clients
        self.llm_map = {
            "OpenAI": openai_llm,
            "Anthropic Claude": claude_llm,
            "Google Gemini": gemini_llm,
            "DeepSeek": deepseek_llm,
            "Local Ollama (Llama 3)": ollama_llm
        }
        self.system_prompt = (
            "You are an expert AI researcher. Provide a short, factual summary "
            "about the latest capabilities regarding the requested topic, "
            "specifically focusing on {model_name}."
        )

    def __call__(self, state: AgentState):
        print("\n--- [RESEARCHERS] Gathering Data (Hitting 5 Distinct Providers) ---")
        topic = state.get("topic", "the requested topic")
        
        research_data = {}
        
        # Iterate through the map, triggering a different API/Local instance on each loop
        for model_name, target_llm in self.llm_map.items():
            print(f"  -> Sending prompt to: {model_name}...")
            
            messages = [
                SystemMessage(content=self.system_prompt.format(model_name=model_name)),
                HumanMessage(content=f"Topic: {topic}")
            ]
            
            # This handles OpenAI, Anthropic, Google, DeepSeek, AND Local Llama3 natively!
            response = target_llm.invoke(messages)
            research_data[model_name] = response.content.strip()
            
        print("  -> All 5 models successfully queried.")
        return {"research_data": research_data, "iterations": state.get("iterations", 0)}
