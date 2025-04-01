# test_planning_agent.py
import asyncio

from openhands.agenthub.planning_agent import PlanningAgent
from openhands.core.config import AgentConfig, LLMConfig
from openhands.llm.llm import LLM


async def test_planning_agent():
    # Initialize components
    llm = LLM(LLMConfig())  # Use appropriate model
    config = AgentConfig()
    planning_agent = PlanningAgent(llm=llm, config=config)

    # User prompt to test
    user_prompt = 'Create a simple weather app'

    print(f'Running Planning Agent with prompt: {user_prompt}')
    print('-' * 80)

    # Execute and display results
    async for update in planning_agent.run(user_prompt):
        print(f"Update Type: {update['mtype']}")
        print(f"Content:\n{update['content']}")
        print('-' * 80)

        # For planning updates, we want to track when steps are completed
        if update['mtype'] == 'planning':
            # Simple parsing to get step statuses
            content_lines = update['content'].split('\n')
            print(content_lines)
            step_lines = content_lines[content_lines.index('Steps:') + 1 :]

            for line in step_lines:
                if '[✓]' in line:  # Completed step
                    print(f'Completed step: {line.strip()}')


if __name__ == '__main__':
    asyncio.run(test_planning_agent())
