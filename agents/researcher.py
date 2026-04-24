# [LANGCHAIN] We import message types to properly format our requests to OpenAI.
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

class ResearchAgent:
    def __init__(self, llm):
        # [LANGCHAIN] We receive the initialized LLM client (e.g., GPT-4o) from the main workflow.
        self.llm = llm
        self.system_prompt = (
            "You are an expert AI researcher. Provide a short, factual summary "
            "about the latest capabilities regarding the requested topic, "
            "specifically focusing on {model_name}."
        )

    def __call__(self, state: AgentState):
        print("\n--- [RESEARCHERS] Gathering Data ---")
        topic = state.get("topic", "the requested topic")
        
        research_data = {}
        models_to_research = ["OpenAI", "Anthropic Claude", "Google Gemini"]
        
        for model_name in models_to_research:
            print(f"  -> Researching perspective for: {model_name}...")
            
            # [LANGCHAIN] We structure the prompt. SystemMessage tells the AI its role, 
            # HumanMessage provides the user's specific request.
            messages = [
                SystemMessage(content=self.system_prompt.format(model_name=model_name)),
                HumanMessage(content=f"Topic: {topic}")
            ]
            
            # [LANGCHAIN] .invoke() triggers the actual HTTP call to the OpenAI API.
            response = self.llm.invoke(messages)
            
            # We extract the text from the response and store it.
            research_data[model_name] = response.content.strip()
            
        print("  -> Research complete.")
        
        # [LANGGRAPH] We return a dictionary. LangGraph automatically takes this 
        # and updates the global 'AgentState' with our new 'research_data'.
        return {"research_data": research_data, "iterations": state.get("iterations", 0)}
