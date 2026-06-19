import asyncio
from dotenv import load_dotenv
from band.config import load_agent_config
from band import Agent

load_dotenv()

async def main():
    print("Starting Multi-Agent System...\n")

    # Load agents
    my_agent_id, my_api = load_agent_config("my_agent")
    viva_id, viva_api = load_agent_config("viva_coach")
    lab_id, lab_api = load_agent_config("lab_reviewer")

    print("Loaded my_agent")
    print("Agent ID:", my_agent_id)

    print("\nLoaded viva_coach")
    print("Agent ID:", viva_id)

    print("\nLoaded lab_reviewer")
    print("Agent ID:", lab_id)

    print("\nAll agents initialized successfully.")
    print("System is running...\n")

    # Create ONLY entry agent (controller)
    analyzer = Agent.create(
        agent_id=my_agent_id,
        api_key=my_api
    )

    # 🔥 CRITICAL PART: send first task (this triggers whole system)
    task = """
You are Experiment Analyzer.

Task:
Generate a complete laboratory report for:
'Verification of Thevenin Theorem'

Then:
1. Ask Viva Coach to generate viva questions
2. Ask Lab Reviewer to verify formatting and correctness
3. Combine outputs into final submission report

Be structured and strict.
"""

    print("STEP 1: Sending task to Band system...")

    await analyzer.send_message(task)

    print("Task sent. Waiting for agent execution...")

    await analyzer.run()


if __name__ == "__main__":
    asyncio.run(main())