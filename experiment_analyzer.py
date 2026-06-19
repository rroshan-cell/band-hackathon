import os
import asyncio
import logging
import traceback

from dotenv import load_dotenv
from band import Agent
from band.adapters import LangGraphAdapter
from band.config import load_agent_config

from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.WARNING)


# -----------------------------
# LOAD LLM (OpenRouter)
# -----------------------------
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )


# -----------------------------
# CREATE AGENT WRAPPER
# -----------------------------
def create_band_agent(config_name: str, llm):
    agent_id, api_key = load_agent_config(config_name)

    print(f"\nLoaded {config_name}")
    print("Agent ID:", agent_id)

    adapter = LangGraphAdapter(
        llm=llm,
        checkpointer=InMemorySaver(),
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )

    return agent


# -----------------------------
# MAIN
# -----------------------------
async def main():
    load_dotenv()

    print("Starting Multi-Agent System...\n")

    llm = get_llm()

    # Load ALL agents from YAML
    experiment_agent = create_band_agent("my_agent", llm)
    viva_agent = create_band_agent("viva_coach", llm)
    reviewer_agent = create_band_agent("lab_reviewer", llm)

    print("\nAll agents initialized successfully.")
    print("System is running...\n")

    # Run all agents concurrently
    try:
        await asyncio.gather(
            experiment_agent.run(),
            viva_agent.run(),
            reviewer_agent.run()
        )

    except Exception as e:
        print("ERROR OCCURRED:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())