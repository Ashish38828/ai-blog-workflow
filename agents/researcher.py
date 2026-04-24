from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

class ResearchAgent:
    def __init__(self, openai_llm, claude_llm, gemini_llm):
        # Map the model names to their actual LangChain API clients
        self.llm_map = {
            "OpenAI": openai_llm,
            "Anthropic Claude": claude_llm,
            "Google Gemini": gemini_llm
        }
        self.system_prompt = (
            "You are an expert AI researcher. Provide a short, factual summary "
            "about the latest capabilities regarding the requested topic, "
            "specifically focusing on {model_name}."
        )

    def __call__(self, state: AgentState):
        print("\n--- [RESEARCHERS] Gathering Data (Hitting Multiple APIs) ---")
        topic = state.get("topic", "the requested topic")
        
        research_data = {}
        
        # Iterate through our map, triggering a different API on each loop
        for model_name, target_llm in self.llm_map.items():
            print(f"  -> Sending API request to: {model_name}...")
            
            messages = [
                SystemMessage(content=self.system_prompt.format(model_name=model_name)),
                HumanMessage(content=f"Topic: {topic}")
            ]
            
            # This now invokes OpenAI, then Anthropic, then Google
            response = target_llm.invoke(messages)
            research_data[model_name] = response.content.strip()
            
        print("  -> All APIs successfully queried.")
        return {"research_data": research_data, "iterations": state.get("iterations", 0)}
