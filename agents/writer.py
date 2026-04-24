from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

class WriterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = "You are an expert tech blogger. Synthesize the research into a short 2-paragraph blog post."

    def __call__(self, state: AgentState):
        current_iteration = state.get("iterations", 0) + 1
        print(f"\n--- [WRITER] Drafting (Iteration {current_iteration}) ---")
        
        # We format the research data from the global state into a readable string.
        context = "\n".join([f"{k}: {v}" for k, v in state.get("research_data", {}).items()])
        
        # If the human or editor provided feedback in a previous loop, we inject it here.
        feedback_prompt = ""
        if state.get("feedback"):
            feedback_prompt = f"\n\nCRITICAL EDITOR FEEDBACK TO ADDRESS:\n{state['feedback']}"

        # [LANGCHAIN] Formatting the prompt with the injected research and feedback.
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Topic: {state['topic']}\n\nResearch:\n{context}{feedback_prompt}")
        ]
        
        # [LANGCHAIN] Asking OpenAI to write the draft.
        response = self.llm.invoke(messages)
        
        # [LANGGRAPH] Returning the newly written draft so LangGraph can update the state.
        return {"draft": response.content, "iterations": current_iteration}
