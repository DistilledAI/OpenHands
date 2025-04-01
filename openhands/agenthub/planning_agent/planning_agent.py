import time
from typing import AsyncGenerator, Optional

from openhands.agenthub.general_agent.general_agent import GeneralAgent
from openhands.agenthub.planning_agent.planning_tool import PlanningTool, PlanStepStatus
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import (
    Action,
)
from openhands.llm.llm import LLM
from openhands.memory.conversation_memory import ConversationMemory


class PlanningAgent(GeneralAgent):
    VERSION = '1.0'
    """
    PlanningAgent is an agent specialized in planning and execution.

    This agent extends GeneralAgent and adds the ability to plan.
    The main functions include:
    1. Creating an initial plan based on user request
    2. Tracking the execution status of each step in the plan
    3. Executing steps sequentially
    4. Providing a summary after the plan is completed

    Agent uses PlanningTool to store and manage plans.
    """

    def __init__(
        self, llm: LLM, config: AgentConfig, mcp_tools: list[dict] | None = None
    ) -> None:
        """Initialize a new instance of PlanningAgent.

        Parameters:
        - llm (LLM): The LLM used by this agent
        - config (AgentConfig): Configuration for this agent
        - mcp_tools (list[dict] | None, optional): The list of MCP tools used by this agent. Defaults to None.
        """
        super().__init__(llm, config, mcp_tools)

        # Initialize planning tool
        self.planning_tool = PlanningTool()

        # Add planning_tool to tools list
        self.tools.append(self.planning_tool.to_param())

        # Current plan information
        self.active_plan_id: str = f'plan_{int(time.time())}'
        self.current_step_index: Optional[int] = None
        self.current_user_request: Optional[str] = None
        self.is_planning_mode: bool = False

        # Ensure prompt_manager is initialized before creating conversation_memory
        if self.prompt_manager is None:
            from openhands.utils.prompt import PromptManager
            self.prompt_manager = PromptManager()

        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        logger.info(f'Initialized PlanningAgent with plan_id: {self.active_plan_id}')

    async def run(self, input_text: str) -> AsyncGenerator[dict, None]:
        """Run PlanningAgent with specific input text.

        This method processes the complete planning and execution process:
        1. Create initial plan
        2. Execute steps sequentially
        3. Summarize results

        Tham số:
        - input_text (str): The input text from the user

        Returns:
        - AsyncGenerator with information about the progress and results
        """
        # Save the current user request
        self.current_user_request = input_text
        self.is_planning_mode = True

        # Create initial plan
        plan_result = await self._create_initial_plan(input_text)
        yield {'content': plan_result, 'mtype': 'planning'}

        # Check if the plan was created successfully
        if self.active_plan_id not in self.planning_tool.plans:
            error_msg = f'Cannot create plan for: {input_text}'
            logger.error(error_msg)
            yield {'content': error_msg, 'mtype': 'error'}
            return

        # Execute each step in the plan
        while True:
            # Get the current step information to execute
            self.current_step_index, step_info = await self._get_current_step_info()

            # Get the current plan and send it back to the user
            current_plan = await self.planning_tool.execute(
                command='get', plan_id=self.active_plan_id
            )
            yield {'content': current_plan['output'], 'mtype': 'planning'}

            # Exit if there are no more steps or the plan is completed
            if self.current_step_index is None:
                final_answer = await self._finalize_plan()
                yield {'content': final_answer, 'mtype': 'final_answer'}
                break

            logger.info(f'Executing step {self.current_step_index}')

            # Execute the current step with a safe step_info dictionary
            step_info_safe = step_info if step_info is not None else {'text': f'Step {self.current_step_index}'}
            step_result = await self._execute_step(step_info_safe)

            # Mark the step as completed and add the result
            await self._mark_step_completed(step_result)

    async def _create_initial_plan(self, request: str) -> str:
        """Create an initial plan based on the request using LLM and PlanningTool."""
        logger.info(f'Creating initial plan with ID: {self.active_plan_id}')

        # Prepare the message for plan creation
        system_message = Message(
            role='system',
            content=[
                {
                    'type': 'text',
                    'text': 'You are a planning assistant. Create a short and feasible plan with general tasks. '
                    '(usually under 5 tasks per plan). Optimize for clarity and efficiency.',
                }
            ],
        )

        user_message = Message(
            role='user',
            content=[
                {
                    'type': 'text',
                    'text': f'Create a reasonable plan with clear tasks to complete the request: {request}',
                }
            ],
        )

        # Create messages directly instead of adding to State
        messages = [system_message, user_message]

        # Call LLM to create a plan
        self.is_planning_mode = True

        # Create params for LLM
        params = {
            'messages': self.llm.format_messages_for_llm(messages),
            'tools': self.tools,
        }

        # Call LLM to create a plan
        try:
            # Execute LLM call but don't need to use the response directly
            # as the planning tool should be invoked through function calling
            self.llm.completion(**params)

            # Check if we have a plan created after LLM call
            if self.active_plan_id not in self.planning_tool.plans:
                logger.warning('Creating default plan')

                try:
                    result = await self.planning_tool.execute(
                        command='create',
                        plan_id=self.active_plan_id,
                        title=f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                        steps=[
                            'Analyze the request',
                            'Perform tasks',
                            'Check the result',
                        ],
                    )
                    logger.info(f'Default plan creation result: {result}')
                except Exception as e:
                    logger.error(f'Error creating default plan: {e}')
                    return f'Error creating initial plan: {str(e)}'
        except Exception as e:
            logger.error(f'Error calling LLM to create plan: {e}')
            return f'Error creating plan: {str(e)}'

        # Get the created plan
        try:
            plan_result = await self.planning_tool.execute(
                command='get', plan_id=self.active_plan_id
            )
            return plan_result['output']
        except Exception as e:
            logger.error(f'Error getting plan: {e}')
            return f'Error getting plan: {str(e)}'

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        Analyze the current plan to determine the index and information of the first unfinished step.
        Returns (None, None) if no active step is found.
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f'Cannot find plan with ID {self.active_plan_id}')
            return None, None

        try:
            # Access the plan data directly from the planning_tool memory
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get('steps', [])
            step_statuses = plan_data.get('step_statuses', [])

            # Find the first unfinished step
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # Extract the step type/category if present
                    step_info = {'text': step}

                    # Try to extract the step type/category from the text (e.g. [SEARCH] or [CODE])
                    import re

                    type_match = re.search(r'\[([A-Z_]+)\]', step)
                    if type_match:
                        step_info['type'] = type_match.group(1).lower()

                    # Mark the current step as in progress
                    try:
                        await self.planning_tool.execute(
                            command='mark_step',
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f'Error marking step as in progress: {e}')
                        # Update the step status directly if needed
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data['step_statuses'] = step_statuses

                    return i, step_info

            return None, None  # No active step found

        except Exception as e:
            logger.warning(f'Error finding current step index: {e}')
            return None, None

    async def _execute_step(self, step_info: dict) -> str:
        """Execute the current step with the specified agent using agent.run()."""
        # Prepare the context for the agent with the current plan status
        plan_status = await self.planning_tool.execute(
            command='get', plan_id=self.active_plan_id
        )
        plan_status_text = plan_status['output']
        step_text = step_info.get('text', f'Step {self.current_step_index}')

        # Create the prompt for the agent to execute the current step
        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status_text}

        CURRENT TASK FOR YOU:
        You are working on step {self.current_step_index}: "{step_text}"

        Please complete this step using the appropriate tools. When finished, provide the final answer about what you have completed.
        """

        # Use the fake run feature to execute the step
        result = ''

        # Ensure we have a system prompt
        system_prompt = "You are an AI assistant, capable of supporting all user needs."
        if self.prompt_manager is not None and hasattr(self.prompt_manager, 'get_system_message'):
            try:
                system_prompt = self.prompt_manager.get_system_message()
            except Exception as e:
                logger.warning(f"Error getting system prompt: {e}")
                # Use fallback prompt from above

        messages = [
            Message(role='system', content=[{'type': 'text', 'text': system_prompt}]),
            Message(role='user', content=[{'type': 'text', 'text': step_prompt}]),
        ]

        # Reset the planning mode to avoid recursion
        old_planning_mode = self.is_planning_mode
        self.is_planning_mode = False

        # Run the agent with the current step
        try:
            # Create params for LLM
            params = {
                'messages': self.llm.format_messages_for_llm(messages),
                'tools': self.tools,
            }

            # Call LLM to execute the step
            response = self.llm.completion(**params)

            # Process the result from LLM
            if hasattr(response, 'content') and response.content:
                result = response.content
            elif hasattr(response, 'choices') and response.choices:
                result = response.choices[0].message.content or ''
            else:
                result = str(response)

        except Exception as e:
            logger.error(f'Error executing step {self.current_step_index}: {e}')
            result = f'Error executing step {self.current_step_index}: {str(e)}'

        # Reset the planning mode
        self.is_planning_mode = old_planning_mode

        return result.strip()

    async def _mark_step_completed(self, step_result: str) -> None:
        """Mark the current step as completed."""
        if self.current_step_index is None:
            return

        try:
            await self.planning_tool.execute(
                command='mark_step',
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f'Marked step {self.current_step_index} as completed in plan {self.active_plan_id}'
            )

            await self.planning_tool.execute(
                command='add_result',
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_result=step_result,
            )

            logger.info(
                f'Added result for step {self.current_step_index} in plan {self.active_plan_id}'
            )

        except Exception as e:
            logger.warning(f'Cannot update plan status: {e}')
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get('step_statuses', [])

                # Ensure the step_statuses list is long enough
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # Update the status
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data['step_statuses'] = step_statuses

                # Update the step result
                step_results = plan_data.get('step_results', [])
                while len(step_results) <= self.current_step_index:
                    step_results.append(None)
                step_results[self.current_step_index] = step_result
                plan_data['step_results'] = step_results

    async def _finalize_plan(self) -> str:
        """Finalize the plan and provide a summary using the LLM of the flow."""
        plan_text = await self.planning_tool.execute(
            command='get', plan_id=self.active_plan_id
        )
        plan_text = plan_text['output']
        try:
            system_message = Message(
                role='system',
                content=[
                    {
                        'type': 'text',
                        'text': 'You are an AI assistant, capable of supporting all user needs.',
                    }
                ],
            )

            user_message = Message(
                role='user',
                content=[
                    {
                        'type': 'text',
                        'text': f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\nPlease provide the final answer for the user's request: {self.current_user_request}",
                    }
                ],
            )

            messages = [system_message, user_message]

            response = ''

            old_planning_mode = self.is_planning_mode
            self.is_planning_mode = False

            params = {'messages': self.llm.format_messages_for_llm(messages)}

            try:
                llm_response = self.llm.completion(**params)
                if hasattr(llm_response, 'content') and llm_response.content:
                    response = llm_response.content
                elif hasattr(llm_response, 'choices') and llm_response.choices:
                    response = llm_response.choices[0].message.content or ''
                else:
                    response = str(llm_response)
            except Exception as e:
                logger.error(f'Error finalizing plan with LLM: {e}')
                response = f'The plan has been completed, but an error occurred while creating the summary: {str(e)}'

            # Reset the planning mode
            self.is_planning_mode = old_planning_mode

            return response

        except Exception as e:
            logger.error(f'Error finalizing plan with LLM: {e}')
            return 'The plan has been completed. Error creating summary.'

    def step(self, state: State) -> Action:
        """Perform a step using PlanningAgent.
        Override the method of Agent to add planning processing.

        Parameters:
        - state (State): used to get update information

        Returns:
        - Action: The next action to perform
        """
        # If in planning mode, ensure planning_tool is used
        if self.is_planning_mode:
            # Ensure planning_tool is in the list of tools
            planning_tool_exists = any(
                tool.get('function', {}).get('name') == 'planning'
                for tool in self.tools
            )

            if not planning_tool_exists:
                self.tools.append(self.planning_tool.to_param())

        # Continue with the actions to be processed if any
        return super().step(state)
