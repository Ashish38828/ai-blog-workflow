import os
from dotenv import load_dotenv
from workflow.graph import BlogWorkflow

# Load environment variables (OPENAI_API_KEY) from the .env file
load_dotenv()

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Please create a .env file.")
        exit(1)

    # Initialize our entire LangGraph/LangChain architecture.
    workflow_manager = BlogWorkflow()
    
    # Start the workflow with a specific topic.
    final_state = workflow_manager.run("Latest multi-modal capabilities in Q4")
    
    print("\n===========================================")
    print("======== FINAL APPROVED BLOG POST =========")
    print("===========================================\n")
    
    # Print the final draft saved in the state after all loops are complete.
    print(final_state.get("draft"))
    print("\n===========================================")
