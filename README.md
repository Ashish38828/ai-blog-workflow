# Agentic Blog Workflow with LangGraph & LangChain

This repository demonstrates a stateful, multi-agent AI workflow using **LangGraph** and **LangChain**. It coordinates three distinct AI personas to autonomously research, draft, and edit a technical blog post, featuring a Human-in-the-Loop (HITL) breakpoint for manual review.

## 🧠 How It Works

This system is built as a **Cyclic Graph**, meaning it can loop and self-correct rather than just moving in a straight line. 

1. **Researcher Agent (Multi-Model Router):** Takes a topic and makes simultaneous, live API calls to **OpenAI (GPT-4o)**, **Anthropic (Claude 3.7 Sonnet)**, and **Google (Gemini 2.5 Pro)** to gather distinct, cutting-edge context.
2. **Writer Agent:** Ingests the aggregated research from all three platforms and synthesizes it into a cohesive blog draft using GPT-4o.
3. **Human-in-the-Loop (HITL):** LangGraph explicitly pauses execution. A human can read the draft and provide manual feedback to force a rewrite.
4. **Editor Agent:** If the human skips manual review, the strict AI Editor evaluates the draft. 
5. **Cyclic Routing:** If either the human or the AI Editor rejects the draft, the workflow loops back to the Writer Agent, passing along the critical feedback so the Writer can improve the next iteration.

## 📂 Project Structure

To maintain modularity and separation of concerns, the project is structured as follows:

```text
ai-blog-workflow/
│
├── core/
│   └── state.py            # Global memory dictionary (AgentState)
│
├── agents/
│   ├── researcher.py       # Multi-LLM API routing logic
│   ├── writer.py           # LangChain drafting logic
│   └── editor.py           # LangChain evaluation logic
│
├── workflow/
│   └── graph.py            # LangGraph orchestration, memory, and routing
│
├── .env.example            # Placeholder for required environment variables
├── requirements.txt        # Python dependencies
├── main.py                 # Execution script
└── README.md               # Project documentation
```

* **LangChain** is used inside the `agents/` directory to format prompts and connect to the OpenAI, Anthropic, and Google APIs.
* **LangGraph** is used in the `workflow/` directory to manage the global state, track memory, and route the data between the agents.

## 🚀 Setup & Execution

### Prerequisites
* Python 3.9+
* API Keys for OpenAI, Anthropic, and Google Gemini

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-blog-workflow.git
cd ai-blog-workflow
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a file named `.env` in the root directory (you can copy the `.env.example` file) and add your API keys:
```env
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-your-anthropic-key-here
GEMINI_API_KEY=AIzaSy-your-google-key-here
```

### 4. Run the Workflow
Execute the main script to start the agentic loop:
```bash
python main.py
```

Watch the terminal as the Researcher Agent pings the different APIs. The console will pause and prompt you when the draft is ready for Human-in-the-Loop review!

## 🛠️ Built With
* [LangGraph](https://python.langchain.com/docs/langgraph/) - For stateful multi-agent orchestration.
* [LangChain](https://python.langchain.com/) - For LLM communication and prompt structuring.
* [OpenAI / Anthropic / Google] - Powering the underlying reasoning and research engines.
