import os
from dotenv import load_dotenv
from workflow.graph import BlogWorkflow
from core.logger import get_logger

# Load API keys from .env
load_dotenv()
logger = get_logger(__name__)

def main():
    logger.info("Starting Multi-Agent Workflow Execution...")
    workflow = BlogWorkflow()
    
    # LangGraph requires a thread_id to track memory and state pauses
    config = {"configurable": {"thread_id": "blog_run_001"}}
    
    initial_state = {
        "topic": "The Enterprise Shift: Moving from Linear AI to Multi-Agent Workflows",
        "iterations": 0,
        "feedback": ""
    }
    
    logger.info(f"Invoking graph for topic: '{initial_state['topic']}'")
    
    # 1. Run the workflow until it hits the `interrupt_before=["editor"]` breakpoint
    for event in workflow.app.stream(initial_state, config, stream_mode="values"):
        pass
        
    logger.info("======== BREAKPOINT REACHED: HUMAN-IN-THE-LOOP ========")
    current_state = workflow.app.get_state(config).values
    draft = current_state.get("draft", "")
    
    print("\n" + "="*50)
    print("CURRENT DRAFT PREVIEW:")
    print("="*50)
    print(draft[:800] + "...\n[Draft Truncated for Preview]")
    print("="*50 + "\n")
    
    # 2. Capture manual human input
    print("ACTION REQUIRED:")
    print(" - Type 'approve' to accept the draft as-is.")
    print(" - Type feedback to force a rewrite.")
    print(" - Press ENTER to skip and let the AI Editor decide.")
    user_input = input("\nInput > ")
    
    if user_input.strip().lower() == "approve":
        logger.info("Human manually approved the draft.")
        workflow.app.update_state(config, {"feedback": "APPROVED_BY_HUMAN"})
    elif user_input.strip():
        logger.info("Human provided manual revision feedback.")
        workflow.app.update_state(config, {"feedback": user_input})
    else:
        logger.info("Human deferred to AI Editor.")
        
    # 3. Resume the workflow from the breakpoint
    logger.info("Resuming workflow execution...")
    for event in workflow.app.stream(None, config, stream_mode="values"):
        pass
        
    final_state = workflow.app.get_state(config).values
    logger.info("Workflow Successfully Completed!")
    
    print("\n\n" + "*"*50)
    print("FINAL PUBLISHED DRAFT")
    print("*"*50)
    print(final_state.get("draft", ""))

if __name__ == "__main__":
    main()
