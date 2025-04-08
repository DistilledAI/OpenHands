import os
from collections import deque

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.config.functionhub_config import FunctionHubConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import (
    Action,
    AgentFinishAction,
)
from openhands.llm.llm import LLM
from openhands.memory.condenser import Condenser
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.runtime.run_functionhub import FunctionHubRunner
from openhands.utils.prompt import PromptManager


class CodeActAgent(Agent):
    VERSION = '2.2'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
        mcp_tools: list[dict] | None = None,
        function_hub_config: FunctionHubConfig | None = None,
    ) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        - config (AgentConfig): The configuration for this agent
        - mcp_tools (list[dict] | None, optional): List of MCP tools to be used by this agent. Defaults to None.
        - function_hub_config (FunctionHubConfig | None, optional): The configuration for the FunctionHub to be used by this agent. Defaults to None.
        """
        super().__init__(llm, config, mcp_tools)
        self.pending_actions: deque[Action] = deque()
        self.reset()

        built_in_tools = codeact_function_calling.get_tools(
            codeact_enable_browsing=self.config.codeact_enable_browsing,
            codeact_enable_jupyter=self.config.codeact_enable_jupyter,
            codeact_enable_llm_editor=self.config.codeact_enable_llm_editor,
            llm=self.llm,
        )

        # self.tools = built_in_tools + (mcp_tools if mcp_tools is not None else [])
        self.tools = built_in_tools

        # Retrieve the enabled tools
        logger.info(
            f"TOOLS loaded for CodeActAgent: {', '.join([tool.get('function').get('name') for tool in self.tools])}"
        )
        self.prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
        )

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser)
        self.functionhub_runner = FunctionHubRunner(function_hub_config)
        logger.debug(f'Using condenser: {type(self.condenser)}')

    def reset(self) -> None:
        """Resets the CodeAct Agent."""
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> Action:
        """Performs one step using the CodeAct Agent.
        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages = self._get_messages(state)

        # Extract the current plan state and step from the messages and state
        current_plan_state = self._extract_plan_state(state, messages)
        current_step = self._extract_current_step(state)
        logger.info(f'Current plan state: {current_plan_state}')
        logger.info(f'Current step: {current_step}')

        logger.info('Searching for additional tools from Function Hub for current task')
        # Use search_tool to get additional tools based on current plan and step
        function_hub_tools = self.functionhub_runner.search_with_rerank(
            current_plan_state, current_step
        )

        logger.info(
            f'Found {len(function_hub_tools)} additional tools from Function Hub for current task'
        )

        # Combine base tools with the dynamically found tools
        combined_tools = self.tools + function_hub_tools

        # Handle the case where tools have the same name
        # Find duplicate tool names
        tool_names = [tool['function']['name'] for tool in combined_tools]
        duplicates = [name for name in tool_names if tool_names.count(name) > 1]

        if duplicates:
            logger.warning(f'Duplicate tool names found: {duplicates}')
            combined_tools_unique = []
            combined_tools_unique_names = []
            for tool in combined_tools:
                if tool['function']['name'] not in combined_tools_unique_names:
                    combined_tools_unique.append(tool)
                    combined_tools_unique_names.append(tool['function']['name'])
                else:
                    logger.warning(
                        f"Duplicate tool name: {tool['function']['name']}, using the first one"
                    )
            combined_tools = combined_tools_unique
            logger.info(
                f'Combined tools after removing duplicates: {combined_tools_unique_names}'
            )

        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
            'tools': combined_tools,  # Use combined tools instead of just self.tools
        }

        function_hub_tools_names_to_tool_id = {
            tool['function']['name']: tool['id_functionhub']
            for tool in function_hub_tools
        }
        # log to litellm proxy if possible
        params['extra_body'] = {'metadata': state.to_llm_metadata(agent_name=self.name)}
        response = self.llm.completion(**params)
        logger.debug(f'Response from LLM: {response}')
        actions = codeact_function_calling.response_to_actions(
            response, function_hub_tools_names_to_tool_id
        )
        logger.debug(f'Actions after response_to_actions: {actions}')
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _extract_plan_state(self, state: State, messages: list[Message]) -> str:
        """Extract the current state of the plan from the state object and messages.

        Parameters:
        - state (State): The current state
        - messages (list[Message]): Processed messages for the LLM

        Returns:
        - str: A string describing the current state of the plan
        """
        # Try to get plan state from extra_data if it exists
        if 'plan_state' in state.extra_data:
            return state.extra_data['plan_state']

        # Or try to infer it from recent history
        current_intent, _ = state.get_current_user_intent()
        last_agent_message = state.get_last_agent_message()

        plan_state = f"User intent: {current_intent if current_intent else 'Unknown'}"

        if last_agent_message:
            plan_state += (
                f'\nLast agent response: {last_agent_message.content[:200]}...'
            )

        # Add task information if available
        if state.root_task and hasattr(state.root_task, 'description'):
            plan_state += f'\nTask: {state.root_task.description}'

        return plan_state

    def _extract_current_step(self, state: State) -> str:
        """Extract the current step from the state object.

        Parameters:
        - state (State): The current state

        Returns:
        - str: A string describing the current step
        """
        # Try to get current step from extra_data if it exists
        if 'current_step' in state.extra_data:
            return state.extra_data['current_step']

        # Or infer it from the current iteration
        return f'Step {state.local_iteration} of task'

    def _get_messages(self, state: State) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Initializes with system prompt and optional initial user message
        2. Processes events (Actions and Observations) into messages
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            state (State): The current state object containing conversation history and other metadata

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt
                - Initial user message (if configured)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
            - For Anthropic models, specific messages are cached according to their documentation
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Use ConversationMemory to process initial messages
        messages = self.conversation_memory.process_initial_messages(
            with_caching=self.llm.is_caching_prompt_active()
        )

        # Condense the events from the state.
        events = self.condenser.condensed_history(state)

        logger.debug(
            f'Processing {len(events)} events from a total of {len(state.history)} events'
        )

        # Use ConversationMemory to process events
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_messages=messages,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        messages = self._enhance_messages(messages)

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def _enhance_messages(self, messages: list[Message]) -> list[Message]:
        """Enhances the user message with additional context based on keywords matched.

        Args:
            messages (list[Message]): The list of messages to enhance

        Returns:
            list[Message]: The enhanced list of messages
        """
        assert self.prompt_manager, 'Prompt Manager not instantiated.'

        results: list[Message] = []
        is_first_message_handled = False
        prev_role = None

        for msg in messages:
            if msg.role == 'user' and not is_first_message_handled:
                is_first_message_handled = True
                # compose the first user message with examples
                self.prompt_manager.add_examples_to_initial_message(msg)

            elif msg.role == 'user':
                # Add double newline between consecutive user messages
                if prev_role == 'user' and len(msg.content) > 0:
                    # Find the first TextContent in the message to add newlines
                    for content_item in msg.content:
                        if isinstance(content_item, TextContent):
                            # If the previous message was also from a user, prepend two newlines to ensure separation
                            content_item.text = '\n\n' + content_item.text
                            break

            results.append(msg)
            prev_role = msg.role

        return results
