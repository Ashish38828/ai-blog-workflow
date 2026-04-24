from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

class ResearchAgent:
    def __init__(self, llm):
        # Accept the LLM from the orchestrator
        self.llm = llm
        self.system_prompt = (
            "You are an expert AI researcher. Provide a short, factual summary (3-4 sentences) "
            "about the latest capabilities or approaches regarding the requested topic, "
            "specifically focusing on {model_name}. If you don't know the absolute latest, "
            "provide the most accurate recent information you have."
        )

    def __call__(self, state: AgentState):
        print("\n--- [RESEARCHERS] Gathering Data (Executing Real LLM Calls) ---")
        topic = state.get("topic", "the requested topic")
        
        research_data = {}
        models_to_research = ["OpenAI", "Anthropic Claude", "Google Gemini"]
        
        # Loop through each model and ask the LLM to research it
        for model_name in models_to_research:
            print(f"  -> Researching perspective for: {model_name}...")
            
            messages = [
                SystemMessage(content=self.system_prompt.format(model_name=model_name)),
                HumanMessage(content=f"Topic: {topic}")
            ]
            
            # The actual real LLM call
            response = self.llm.invoke(messages)
            research_data[model_name] = response.content.strip()
            
        print("  -> Research complete.")
        return {"research_data": research_data, "iterations": state.get("iterations", 0)}
