from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.state import AgentState
from agents.researcher import ResearchAgent
from agents.writer import WriterAgent
from agents.editor import EditorAgent
from core.logger import get_logger

logger = get_logger(__name__)

class BlogWorkflow:
    def __init__(self):
        logger.info("Initializing Enterprise Multi-Model APIs...")
        
        # Cloud APIs
        self.openai_llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
        self.claude_llm = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0.5)
        self.gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.5)
        
        # Local Offline Models
        self.local_deepseek = ChatOllama(model="deepseek-r1", temperature=0.5)
        self.local_llama = ChatOllama(model="llama3.1", temperature=0.5)
        
        # Initialize Agents
        self.researcher = ResearchAgent(
            self.openai_llm, self.claude_llm, self.gemini_llm,
            self.local_deepseek, self.local_llama
        )
        self.writer = WriterAgent(self.openai_llm)
        self.editor = EditorAgent(self.openai_llm)
        
        self.app = self._build_graph()

    def _build_graph(self):
        logger.info("Building LangGraph state machine...")
        builder = StateGraph(AgentState)
        
        # Add Nodes
        builder.add_node("researcher", self.researcher)
        builder.add_node("writer", self.writer)
        builder.add_node("editor", self.editor)
        
        # Define the Standard Path
        builder.add_edge(START, "researcher")
        builder.add_edge("researcher", "writer")
        builder.add_edge("writer", "editor")
        
        # Define the Cyclic/Conditional Logic
        def route_editor(state: AgentState):
            if state.get("feedback") == "APPROVED":
                return END
            if state.get("iterations", 0) >= 3:
                logger.warning("Max iteration cap (3) reached. Forcing loop termination.")
                return END
            return "writer"
            
        builder.add_conditional_edges("editor", route_editor)
        
        # Compile with Memory to enable Human-in-the-Loop breakpoint
        memory = MemorySaver()
        return builder.compile(checkpointer=memory, interrupt_before=["editor"])
