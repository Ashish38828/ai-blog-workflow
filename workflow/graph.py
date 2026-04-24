# [LANGGRAPH] Importing the tools needed to build the workflow graph and memory.
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import all three model providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from core.state import AgentState
from agents.researcher import ResearchAgent
from agents.writer import WriterAgent
from agents.editor import EditorAgent

class BlogWorkflow:
    def __init__(self):
        # Initialize the actual connections to all three providers
        self.openai_llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
        self.claude_llm = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0.5)
        self.gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.5)
        
        # Pass all three to the Researcher
        self.researcher = ResearchAgent(self.openai_llm, self.claude_llm, self.gemini_llm)
        
        # The Writer and Editor can still use GPT-4o as their reasoning engine
        self.writer = WriterAgent(self.openai_llm)
        self.editor = EditorAgent(self.openai_llm)
        
        self.app = self._build_graph()

    def _routing_logic(self, state: AgentState):
        # [LANGGRAPH] This is a conditional routing function. LangGraph uses this 
        # to decide which node to run next based on the current state variables.
        if state.get("feedback") == "APPROVED":
            return "continue_to_end"
        if state.get("iterations", 0) >= 3:
            return "continue_to_end"
        return "revise_draft"

    def _build_graph(self):
        # [LANGGRAPH] Initialize the graph structure using our custom memory dictionary.
        workflow = StateGraph(AgentState)
        
        # [LANGGRAPH] Register our Python classes as official "Nodes" (steps in the workflow).
        workflow.add_node("researchers", self.researcher)
        workflow.add_node("writer", self.writer)
        workflow.add_node("editor", self.editor)
        
        # [LANGGRAPH] Draw the standard, one-way lines connecting the nodes.
        workflow.add_edge(START, "researchers")
        workflow.add_edge("researchers", "writer")
        workflow.add_edge("writer", "editor")
        
        # [LANGGRAPH] Draw the logic-based routing line. This allows the graph to loop backwards.
        workflow.add_conditional_edges(
            "editor",
            self._routing_logic,
            {
                "revise_draft": "writer",      # Loop back to writer
                "continue_to_end": END         # Finish the workflow
            }
        )

        # [LANGGRAPH] MemorySaver allows the graph to pause execution without losing data.
        memory = MemorySaver()
        
        # [LANGGRAPH] We compile the graph, telling it to explicitly pause right before 
        # it executes the "editor" node to allow for Human-in-the-Loop review.
        return workflow.compile(checkpointer=memory, interrupt_before=["editor"])

    def run(self, topic: str):
        # [LANGGRAPH] Thread ID tracks this specific execution's memory so we can pause/resume it.
        config = {"configurable": {"thread_id": "blog_run_001"}}
        initial_state = {"topic": topic, "iterations": 0}
        
        # 1. Run the graph from the start until it hits the breakpoint (before the Editor).
        for event in self.app.stream(initial_state, config):
            pass 
        
        # 2. Retrieve the current variables from LangGraph's paused memory.
        current_state = self.app.get_state(config).values
        
        print("\n===========================================")
        print("=== [HUMAN IN THE LOOP: DRAFT REVIEW] ===")
        print("===========================================")
        print(current_state.get("draft"))
        print("===========================================\n")
        
        user_input = input("Provide manual feedback to force a rewrite, or press Enter to let the AI Editor review it: ")
        
        if user_input.strip():
            print("\nInjecting human feedback and routing back to Writer...")
            # [LANGGRAPH] Manually insert the human's feedback into the state, 
            # tricking the routing logic into thinking the AI Editor rejected it.
            self.app.update_state(config, {"feedback": user_input}, as_node="editor")
        else:
            print("\nNo human input. Resuming and handing over to AI Editor...")

        # 3. Resume the graph execution from exactly where it paused.
        for event in self.app.stream(None, config):
            pass
            
        return self.app.get_state(config).values
