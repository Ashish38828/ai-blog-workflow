from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

class EditorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = (
            "You are a strict Managing Editor. Review the blog draft. "
            "If it mentions all three models clearly, reply with ONLY the word 'APPROVED'. "
            "If not, provide specific, harsh feedback on what to fix. Do not write the draft yourself."
        )

    def __call__(self, state: AgentState):
        print("\n--- [EDITOR] Reviewing Draft ---")
        
        # [LANGCHAIN] Asking OpenAI to evaluate the draft written by the WriterAgent.
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Draft to review:\n\n{state.get('draft', '')}")
        ]
        
        response = self.llm.invoke(messages)
        feedback = response.content.strip()
        
        # We determine if the LLM approved it or not.
        if "APPROVED" in feedback.upper():
            print("  -> Editor Decision: APPROVED")
            return {"feedback": "APPROVED"}
        else:
            print(f"  -> Editor Decision: REJECTED. Feedback: {feedback}")
            return {"feedback": feedback}
