import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

from src.research_agent import researcher_agent

def main():
    # Ensure experiments directory exists
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)

    # Read the problem description
    task_file = Path("task.json")
    if not task_file.exists():
        print(f"Error: {task_file} not found.")
        return

    with open(task_file, "r") as f:
        task_data = json.load(f)
        task_description = task_data.get("task_description", "")

    print("Starting RCA analysis...")
    print(f"Task: {task_description[:100]}...")

    # Initialize state
    initial_state = {
        "researcher_messages": [HumanMessage(content=task_description)]
    }

    # Run the agent
    # The agent returns the final state
    # Increase recursion limit for complex RCA tasks
    print("Running agent... (this may take a while)")
    
    final_state = initial_state
    # Use stream to show progress and debug
    try:
        for event in researcher_agent.stream(initial_state, config={"recursion_limit": 100}):
            for key, value in event.items():
                if key == "llm_call":
                    print("Agent decided to call tools:")
                    if "researcher_messages" in value:
                        last_msg = value["researcher_messages"][-1]
                        if hasattr(last_msg, "tool_calls"):
                            for tc in last_msg.tool_calls:
                                print(f"  - {tc['name']}: {tc['args']}")
                elif key == "tool_node":
                    print("Tools executed.")
                    if "researcher_messages" in value:
                        for msg in value["researcher_messages"]:
                            if hasattr(msg, "name"):
                                content_preview = str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
                                print(f"  - {msg.name} output: {content_preview}")
                elif key == "compress_research":
                    print("Research compressed.")
                    final_state.update(value)
                
                # Update final state with the latest state components
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k == "researcher_messages":
                            if "researcher_messages" not in final_state:
                                final_state["researcher_messages"] = []
                            final_state["researcher_messages"].extend(v)
                        else:
                            final_state[k] = v
    except BaseException as e:
        print(f"\nAn error occurred or execution was interrupted: {e}")
        print("Saving partial results...")

    # Extract relevant information for output
    # The user wants "record conversation history"
    messages = final_state.get("researcher_messages", [])
    
    output_data = []
    for msg in messages:
        msg_data = {
            "type": msg.type,
            "content": msg.content,
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_data["tool_calls"] = msg.tool_calls
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            msg_data["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "name") and msg.name:
            msg_data["name"] = msg.name
            
        output_data.append(msg_data)

    # Also include the compressed research if available
    compressed_research = final_state.get("compressed_research")
    if compressed_research:
        output_data.append({
            "type": "compressed_research",
            "content": compressed_research
        })

    output_file = experiments_dir / "output.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Analysis complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()
