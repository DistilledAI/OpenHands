import asyncio
import copy
import os
import traceback
from datetime import datetime
from typing import Callable, ClassVar, Dict

import litellm  # noqa
from litellm.exceptions import (  # noqa
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.replay import ReplayManager
from openhands.controller.state.plan import Plan
from openhands.controller.state.state import State, TrafficControlState
from openhands.controller.stuck import StuckDetector
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.exceptions import (
    AgentStuckInLoopError,
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMContextWindowExceedError,
    LLMMalformedActionError,
    LLMNoActionError,
    LLMResponseError,
)
from openhands.core.logger import LOG_ALL_EVENTS
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events import (
    EventSource,
    EventStream,
    EventStreamSubscriber,
    RecallType,
)
from openhands.events.action import (
    Action,
    ActionConfirmationStatus,
    AgentFinishAction,
    AgentRejectAction,
    AssignTaskAction,
    ChangeAgentStateAction,
    CmdRunAction,
    CreatePlanAction,
    IPythonRunCellAction,
    MarkTaskAction,
    MessageAction,
    NullAction,
    TaskStatus,
)
from openhands.events.action.agent import RecallAction
from openhands.events.event import Event
from openhands.events.observation import (
    AgentCondensationObservation,
    AgentStateChangedObservation,
    ErrorObservation,
    NullObservation,
    Observation,
    PlanStatusObservation,
)
from openhands.events.serialization.event import event_to_trajectory, truncate_content
from openhands.llm.metrics import Metrics, TokenUsage

# note: RESUME is only available on web GUI
TRAFFIC_CONTROL_REMINDER = (
    "Please click on resume button if you'd like to continue, or start a new task."
)


class PlanController:
    id: str
    agent: Agent
    planning_agent: Agent
    max_iterations: int
    event_stream: EventStream
    state: State
    confirmation_mode: bool
    agent_to_llm_config: dict[str, LLMConfig]
    agent_configs: dict[str, AgentConfig]
    _pending_action: Action | None = None
    _closed: bool = False
    filter_out: ClassVar[tuple[type[Event], ...]] = (
        NullAction,
        NullObservation,
        ChangeAgentStateAction,
        AgentStateChangedObservation,
        PlanStatusObservation,
        MarkTaskAction,
    )
    # pass type of events that should be passed to the agent when delegate agents are resolving tasks
    pass_type: ClassVar[tuple[type[Event], ...]] = (AgentFinishAction, AssignTaskAction)
    _cached_first_user_message: MessageAction | None = None

    # task_controllers
    task_controllers: Dict[
        str, Dict[int, AgentController]
    ] = {}  # plan_id -> task_id -> agent_controller

    def __init__(
        self,
        agent: Agent,
        planning_agent: Agent,
        event_stream: EventStream,
        max_iterations: int,
        max_budget_per_task: float | None = None,
        agent_to_llm_config: dict[str, LLMConfig] | None = None,
        agent_configs: dict[str, AgentConfig] | None = None,
        sid: str | None = None,
        confirmation_mode: bool = False,
        initial_state: State | None = None,
        headless_mode: bool = True,
        status_callback: Callable | None = None,
        replay_events: list[Event] | None = None,
    ):
        """Initializes a new instance of the PlanController: class.

        Args:
            agent: The agent instance to control.
            event_stream: The event stream to publish events to.
            max_iterations: The maximum number of iterations the agent can run.
            max_budget_per_task: The maximum budget (in USD) allowed per task, beyond which the agent will stop.
            agent_to_llm_config: A dictionary mapping agent names to LLM configurations in the case that
                we delegate to a different agent.
            agent_configs: A dictionary mapping agent names to agent configurations in the case that
                we delegate to a different agent.
            sid: The session ID of the agent.
            confirmation_mode: Whether to enable confirmation mode for agent actions.
            initial_state: The initial state of the controller.
            headless_mode: Whether the agent is run in headless mode.
            status_callback: Optional callback function to handle status updates.
            replay_events: A list of logs to replay.
        """
        self.id = sid or event_stream.sid
        self.agent = agent
        self.planning_agent = planning_agent
        self.headless_mode = headless_mode

        # the event stream must be set before maybe subscribing to it
        self.event_stream = event_stream

        # subscribe to the event stream
        self.event_stream.subscribe(
            EventStreamSubscriber.PLANNING_CONTROLLER, self.on_event, self.id
        )

        # state from the previous session, state from a parent agent, or a fresh state
        self.set_initial_state(
            state=initial_state,
            max_iterations=max_iterations,
            confirmation_mode=confirmation_mode,
        )
        self.max_budget_per_task = max_budget_per_task
        self.agent_to_llm_config = agent_to_llm_config if agent_to_llm_config else {}
        self.agent_configs = agent_configs if agent_configs else {}
        self._initial_max_iterations = max_iterations
        self._initial_max_budget_per_task = max_budget_per_task

        # stuck helper
        self._stuck_detector = StuckDetector(self.state)
        self.status_callback = status_callback

        # replay-related
        self._replay_manager = ReplayManager(replay_events)

    async def close(self, set_stop_state=True) -> None:
        """Closes the agent controller, canceling any ongoing tasks and unsubscribing from the event stream.

        Note that it's fairly important that this closes properly, otherwise the state is incomplete.
        """
        if set_stop_state:
            await self.set_agent_state_to(AgentState.STOPPED)

        # we made history, now is the time to rewrite it!
        # the final state.history will be used by external scripts like evals, tests, etc.
        # like the regular agent history, it does not include:
        # - 'hidden' events, events with hidden=True
        # - backend events (the default 'filtered out' types, types in self.filter_out)
        start_id = self.state.start_id if self.state.start_id >= 0 else 0
        end_id = (
            self.state.end_id
            if self.state.end_id >= 0
            else self.event_stream.get_latest_event_id()
        )
        self.state.history = list(
            self.event_stream.get_events(
                start_id=start_id,
                end_id=end_id,
                reverse=False,
                filter_out_type=self.filter_out,
                filter_hidden=True,
            )
        )

        # unsubscribe from the event stream
        self.event_stream.unsubscribe(
            EventStreamSubscriber.PLANNING_CONTROLLER, self.id
        )

        self._closed = True

    def log(self, level: str, message: str, extra: dict | None = None) -> None:
        """Logs a message to the agent controller's logger.

        Args:
            level (str): The logging level to use (e.g., 'info', 'debug', 'error').
            message (str): The message to log.
            extra (dict | None, optional): Additional fields to log. Includes session_id by default.
        """
        message = f'[Agent Controller {self.id}] {message}'
        if extra is None:
            extra = {}
        extra_merged = {'session_id': self.id, **extra}
        getattr(logger, level)(message, extra=extra_merged, stacklevel=2)

    def update_state_before_step(self):
        self.state.iteration += 1
        self.state.local_iteration += 1

    async def update_state_after_step(self):
        # update metrics especially for cost. Use deepcopy to avoid it being modified by agent._reset()
        self.state.local_metrics = copy.deepcopy(self.planning_agent.llm.metrics)

    async def _react_to_exception(
        self,
        e: Exception,
    ):
        """React to an exception by setting the agent state to error and sending a status message."""
        await self.set_agent_state_to(AgentState.ERROR)
        if self.status_callback is not None:
            err_id = ''
            if isinstance(e, AuthenticationError):
                err_id = 'STATUS$ERROR_LLM_AUTHENTICATION'
            elif isinstance(
                e,
                (
                    ServiceUnavailableError,
                    APIConnectionError,
                    APIError,
                ),
            ):
                err_id = 'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE'
            elif isinstance(e, InternalServerError):
                err_id = 'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR'
            elif isinstance(e, BadRequestError) and 'ExceededBudget' in str(e):
                err_id = 'STATUS$ERROR_LLM_OUT_OF_CREDITS'
            elif isinstance(e, RateLimitError):
                await self.set_agent_state_to(AgentState.RATE_LIMITED)
                return
            self.status_callback('error', err_id, type(e).__name__ + ': ' + str(e))

    def step(self):
        asyncio.create_task(self._step_with_exception_handling())

    async def _step_with_exception_handling(self):
        try:
            await self._step()
        except Exception as e:
            self.log(
                'error',
                f'Error while running the agent (session ID: {self.id}): {e}. '
                f'Traceback: {traceback.format_exc()}',
            )
            reported = RuntimeError(
                f'There was an unexpected error while running the agent: {e.__class__.__name__}. You can refresh the page or ask the agent to try again.'
            )
            if (
                isinstance(e, Timeout)
                or isinstance(e, APIError)
                or isinstance(e, BadRequestError)
                or isinstance(e, NotFoundError)
                or isinstance(e, InternalServerError)
                or isinstance(e, AuthenticationError)
                or isinstance(e, RateLimitError)
                or isinstance(e, LLMContextWindowExceedError)
            ):
                reported = e
            else:
                self.log(
                    'warning',
                    f'Unknown exception type while running the agent: {type(e).__name__}.',
                )
            await self._react_to_exception(reported)

    async def _step(self) -> None:
        """Executes a single step of the parent or delegate agent. Detects stuck agents and limits on the number of iterations and the task budget."""

        if self._is_awaiting_for_task_resolving():
            return

        if self.get_agent_state() != AgentState.RUNNING:
            return

        if self._pending_action:
            return

        self.log(
            'info',
            f'LEVEL {self.state.delegate_level} LOCAL STEP {self.state.local_iteration} GLOBAL STEP {self.state.iteration}',
            extra={'msg_type': 'STEP'},
        )

        stop_step = False
        if self.state.iteration >= self.state.max_iterations:
            stop_step = await self._handle_traffic_control(
                'iteration', self.state.iteration, self.state.max_iterations
            )
        if self.max_budget_per_task is not None:
            current_cost = self.state.metrics.accumulated_cost
            if current_cost > self.max_budget_per_task:
                stop_step = await self._handle_traffic_control(
                    'budget', current_cost, self.max_budget_per_task
                )
        if stop_step:
            logger.warning('Stopping agent due to traffic control')
            return

        if self._is_stuck():
            await self._react_to_exception(
                AgentStuckInLoopError('Agent got stuck in a loop')
            )
            return

        self.update_state_before_step()
        action: Action = NullAction()

        if self._replay_manager.should_replay():
            # in replay mode, we don't let the agent to proceed
            # instead, we replay the action from the replay trajectory
            action = self._replay_manager.step()
        else:
            try:
                action = self.planning_agent.step(self.state)
                if action is None:
                    raise LLMNoActionError('No action was returned')
                action._source = EventSource.AGENT  # type: ignore [attr-defined]
            except (
                LLMMalformedActionError,
                LLMNoActionError,
                LLMResponseError,
                FunctionCallValidationError,
                FunctionCallNotExistsError,
            ) as e:
                self.event_stream.add_event(
                    ErrorObservation(
                        content=str(e),
                    ),
                    EventSource.AGENT,
                )
                return
            except (ContextWindowExceededError, BadRequestError, OpenAIError) as e:
                # FIXME: this is a hack until a litellm fix is confirmed
                # Check if this is a nested context window error
                # We have to rely on string-matching because LiteLLM doesn't consistently
                # wrap the failure in a ContextWindowExceededError
                error_str = str(e).lower()
                if (
                    'contextwindowexceedederror' in error_str
                    or 'prompt is too long' in error_str
                    or 'input length and `max_tokens` exceed context limit' in error_str
                    or isinstance(e, ContextWindowExceededError)
                ):
                    if self.planning_agent.config.enable_history_truncation:
                        self._handle_long_context_error()
                        return
                    else:
                        raise LLMContextWindowExceedError()
                else:
                    raise e

        if action.runnable:
            if self.state.confirmation_mode and (
                type(action) is CmdRunAction or type(action) is IPythonRunCellAction
            ):
                action.confirmation_state = (
                    ActionConfirmationStatus.AWAITING_CONFIRMATION
                )
            self._pending_action = action

        if not isinstance(action, NullAction):
            if (
                hasattr(action, 'confirmation_state')
                and action.confirmation_state
                == ActionConfirmationStatus.AWAITING_CONFIRMATION
            ):
                await self.set_agent_state_to(AgentState.AWAITING_USER_CONFIRMATION)

            # Create and log metrics for frontend display
            self._prepare_metrics_for_frontend(action)

            self.event_stream.add_event(action, action._source)  # type: ignore [attr-defined]

        await self.update_state_after_step()

        log_level = 'info' if LOG_ALL_EVENTS else 'debug'
        self.log(log_level, str(action), extra={'msg_type': 'ACTION'})

    def should_step(self, event: Event) -> bool:
        """Whether the agent should take a step based on an event.

        In general, the agent should take a step if it receives a message from the user,
        or observes something in the environment (after acting).
        """
        # it might be the delegate's day in the sun
        # if self.delegate is not None:
        #     return False

        if isinstance(event, Action):
            if isinstance(event, CreatePlanAction) or isinstance(event, MarkTaskAction):
                return False

            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return True
            if (
                isinstance(event, MessageAction)
                and self.get_agent_state() != AgentState.AWAITING_USER_INPUT
            ):
                # TODO: this is fragile, but how else to check if eligible?
                return True

            return False

        if isinstance(event, Observation):
            if isinstance(event, PlanStatusObservation):
                return False
            if (
                isinstance(event, NullObservation)
                and event.cause is not None
                and event.cause
                > 0  # NullObservation has cause > 0 (RecallAction), not 0 (user message)
            ):
                return True
            if isinstance(event, AgentStateChangedObservation) or isinstance(
                event, NullObservation
            ):
                return False
            return True
        return False

    def on_event(self, event: Event) -> None:
        """Callback from the event stream. Notifies the controller of incoming events.

        Args:
            event (Event): The incoming event to process.
        """

        # continue parent processing
        asyncio.get_event_loop().run_until_complete(self._on_event(event))

    async def _on_event(self, event: Event) -> None:
        if hasattr(event, 'hidden') and event.hidden:
            return

        # Give others a little chance
        await asyncio.sleep(0.01)

        # if the event is not filtered out and tasks are not resolved by delegate agent, add it to the history
        if not any(isinstance(event, filter_type) for filter_type in self.filter_out):
            if not self._is_awaiting_for_task_resolving():
                self.state.history.append(event)
            elif any(isinstance(event, pass_type) for pass_type in self.pass_type):
                self.state.history.append(event)
            else:
                pass

        if isinstance(event, Action):
            await self._handle_action(event)
        elif isinstance(event, Observation):
            await self._handle_observation(event)

        if self.should_step(event):
            self.step()

    async def _handle_action(self, action: Action) -> None:
        """Handles an Action from the agent or delegate."""
        if isinstance(action, ChangeAgentStateAction):
            await self.set_agent_state_to(action.agent_state)  # type: ignore
        elif isinstance(action, MessageAction):
            await self._handle_message_action(action)
        elif isinstance(action, MarkTaskAction):
            if action.task_status == TaskStatus.IN_PROGRESS:
                self.state.current_task_index = action.task_index
                plan: Plan = self.state.plans[self.state.active_plan_id]
                # assign the task to the agent
                self.event_stream.add_event(
                    AssignTaskAction(
                        plan_id=self.state.active_plan_id,
                        task_index=action.task_index,
                        task_content=plan.tasks[action.task_index].content,
                        delegate_id=self.id + f'_{action.task_index}',
                    ),
                    EventSource.USER,
                )
        elif isinstance(action, AssignTaskAction):
            await self._assign_task_to_the_delegate(action)

        elif isinstance(action, AgentFinishAction):
            # AgentFinishAction from current planning agent
            if self._is_all_task_resolved():
                self.state.outputs = action.outputs
                self.state.metrics.merge(self.state.local_metrics)
                await self.set_agent_state_to(AgentState.FINISHED)
            # AgentFinishAction from delegate agent
            else:
                # mark the task as completed
                active_plan_obj: Plan = self.state.plans[self.state.active_plan_id]
                current_task = active_plan_obj.tasks[self.state.current_task_index]
                current_task.status = TaskStatus.COMPLETED
                self.event_stream.add_event(
                    MarkTaskAction(
                        plan_id=self.state.active_plan_id,
                        task_index=self.state.current_task_index,
                        task_content=current_task.content,
                        task_status=TaskStatus.COMPLETED,
                    ),
                    EventSource.AGENT,
                )

                # update result to the active plan
                active_plan_obj.tasks[
                    self.state.current_task_index
                ].result = action.final_thought

                # delete the controller corresponding to the task
                if self.state.active_plan_id in self.task_controllers:
                    del self.task_controllers[self.state.active_plan_id][
                        self.state.current_task_index
                    ]

                # move to the next task if plan is not finished
                if self.state.current_task_index + 1 < len(active_plan_obj.tasks):
                    self.state.current_task_index += 1
                    current_task = active_plan_obj.tasks[self.state.current_task_index]
                    current_task.status = TaskStatus.IN_PROGRESS
                    self.event_stream.add_event(
                        MarkTaskAction(
                            plan_id=self.state.active_plan_id,
                            task_index=self.state.current_task_index,
                            task_content=current_task.content,
                            task_status=TaskStatus.IN_PROGRESS,
                        ),
                        EventSource.AGENT,
                    )
                # if plan is finished, add user message to trigger planner finalize the plan
                else:
                    self.event_stream.add_event(
                        MessageAction(
                            content='All tasks are completed. Please accomplish the plan and send it to the user.',
                        ),
                        EventSource.USER,
                    )

        elif isinstance(action, AgentRejectAction):
            self.state.outputs = action.outputs
            self.state.metrics.merge(self.state.local_metrics)
            await self.set_agent_state_to(AgentState.REJECTED)
        elif isinstance(action, CreatePlanAction):
            # Create a plan
            self._create_plan(action)

            # Add the plan status to the event stream
            # self.event_stream.add_event(
            #     self._get_active_plan_status(), EventSource.ENVIRONMENT #PlanStatusObservation
            # )

            # mark the first task as in progress
            active_plan: Plan = self.state.plans[self.state.active_plan_id]
            self.state.current_task_index = 0
            active_plan.tasks[0].status = TaskStatus.IN_PROGRESS
            self.event_stream.add_event(
                MarkTaskAction(
                    plan_id=self.state.active_plan_id,
                    task_index=self.state.current_task_index,
                    task_content=active_plan.tasks[0].content,
                    task_status=TaskStatus.IN_PROGRESS,
                ),
                EventSource.AGENT,
            )

    async def _handle_observation(self, observation: Observation) -> None:
        """Handles observation from the event stream.

        Args:
            observation (observation): The observation to handle.
        """
        observation_to_print = copy.deepcopy(observation)
        if (
            len(observation_to_print.content)
            > self.planning_agent.llm.config.max_message_chars
        ):
            observation_to_print.content = truncate_content(
                observation_to_print.content,
                self.planning_agent.llm.config.max_message_chars,
            )
        # Use info level if LOG_ALL_EVENTS is set
        log_level = 'info' if os.getenv('LOG_ALL_EVENTS') in ('true', '1') else 'debug'
        self.log(
            log_level, str(observation_to_print), extra={'msg_type': 'OBSERVATION'}
        )

        if observation.llm_metrics is not None:
            self.planning_agent.llm.metrics.merge(observation.llm_metrics)

        # this happens for runnable actions and microagent actions
        if self._pending_action and self._pending_action.id == observation.cause:
            if self.state.agent_state == AgentState.AWAITING_USER_CONFIRMATION:
                return
            self._pending_action = None
            if self.state.agent_state == AgentState.USER_CONFIRMED:
                await self.set_agent_state_to(AgentState.RUNNING)
            if self.state.agent_state == AgentState.USER_REJECTED:
                await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
            return
        elif isinstance(observation, ErrorObservation):
            if self.state.agent_state == AgentState.ERROR:
                self.state.metrics.merge(self.state.local_metrics)

    async def _handle_message_action(self, action: MessageAction) -> None:
        """Handles message actions from the event stream.

        Args:
            action (MessageAction): The message action to handle.
        """
        if action.source == EventSource.USER:
            # Use info level if LOG_ALL_EVENTS is set
            log_level = (
                'info' if os.getenv('LOG_ALL_EVENTS') in ('true', '1') else 'debug'
            )
            self.log(
                log_level,
                str(action),
                extra={'msg_type': 'ACTION', 'event_source': EventSource.USER},
            )
            # Extend max iterations when the user sends a message (only in non-headless mode)
            if self._initial_max_iterations is not None and not self.headless_mode:
                self.state.max_iterations = (
                    self.state.iteration + self._initial_max_iterations
                )
                if (
                    self.state.traffic_control_state == TrafficControlState.THROTTLING
                    or self.state.traffic_control_state == TrafficControlState.PAUSED
                ):
                    self.state.traffic_control_state = TrafficControlState.NORMAL
                self.log(
                    'debug',
                    f'Extended max iterations to {self.state.max_iterations} after user message',
                )
            # try to retrieve microagents relevant to the user message
            # set pending_action while we search for information

            # if this is the first user message for this agent, matters for the microagent info type
            first_user_message = self._first_user_message()
            is_first_user_message = (
                action.id == first_user_message.id if first_user_message else False
            )
            recall_type = (
                RecallType.WORKSPACE_CONTEXT
                if is_first_user_message
                else RecallType.KNOWLEDGE
            )

            recall_action = RecallAction(query=action.content, recall_type=recall_type)
            self._pending_action = recall_action
            # this is source=USER because the user message is the trigger for the microagent retrieval
            self.event_stream.add_event(recall_action, EventSource.USER)

            if self.get_agent_state() != AgentState.RUNNING:
                await self.set_agent_state_to(AgentState.RUNNING)
        elif action.source == EventSource.AGENT and action.wait_for_response:
            await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)

    def _reset(self) -> None:
        """Resets the agent controller."""
        # Runnable actions need an Observation
        # make sure there is an Observation with the tool call metadata to be recognized by the agent
        # otherwise the pending action is found in history, but it's incomplete without an obs with tool result
        if self._pending_action and hasattr(self._pending_action, 'tool_call_metadata'):
            # find out if there already is an observation with the same tool call metadata
            found_observation = False
            for event in self.state.history:
                if (
                    isinstance(event, Observation)
                    and event.tool_call_metadata
                    == self._pending_action.tool_call_metadata
                ):
                    found_observation = True
                    break

            # make a new ErrorObservation with the tool call metadata
            if not found_observation:
                obs = ErrorObservation(content='The action has not been executed.')
                obs.tool_call_metadata = self._pending_action.tool_call_metadata
                obs._cause = self._pending_action.id  # type: ignore[attr-defined]
                self.event_stream.add_event(obs, EventSource.AGENT)

        # NOTE: RecallActions don't need an ErrorObservation upon reset, as long as they have no tool calls

        # reset the pending action, this will be called when the agent is STOPPED or ERROR
        self._pending_action = None
        self.planning_agent.reset()

    async def set_agent_state_to(self, new_state: AgentState) -> None:
        """Updates the agent's state and handles side effects. Can emit events to the event stream.

        Args:
            new_state (AgentState): The new state to set for the agent.
        """
        self.log(
            'info',
            f'Setting agent({self.planning_agent.name}) state from {self.state.agent_state} to {new_state}',
        )

        if new_state == self.state.agent_state:
            return

        if new_state in (AgentState.STOPPED, AgentState.ERROR):
            # sync existing metrics BEFORE resetting the agent
            await self.update_state_after_step()
            self.state.metrics.merge(self.state.local_metrics)
            self._reset()
        elif (
            new_state == AgentState.RUNNING
            and self.state.agent_state == AgentState.PAUSED
            # TODO: do we really need both THROTTLING and PAUSED states, or can we clean up one of them completely?
            and self.state.traffic_control_state == TrafficControlState.THROTTLING
        ):
            # user intends to interrupt traffic control and let the task resume temporarily
            self.state.traffic_control_state = TrafficControlState.PAUSED
            # User has chosen to deliberately continue - lets double the max iterations
            if (
                self.state.iteration is not None
                and self.state.max_iterations is not None
                and self._initial_max_iterations is not None
                and not self.headless_mode
            ):
                if self.state.iteration >= self.state.max_iterations:
                    self.state.max_iterations += self._initial_max_iterations

            if (
                self.state.metrics.accumulated_cost is not None
                and self.max_budget_per_task is not None
                and self._initial_max_budget_per_task is not None
            ):
                if self.state.metrics.accumulated_cost >= self.max_budget_per_task:
                    self.max_budget_per_task += self._initial_max_budget_per_task
        elif self._pending_action is not None and (
            new_state in (AgentState.USER_CONFIRMED, AgentState.USER_REJECTED)
        ):
            if hasattr(self._pending_action, 'thought'):
                self._pending_action.thought = ''  # type: ignore[union-attr]
            if new_state == AgentState.USER_CONFIRMED:
                confirmation_state = ActionConfirmationStatus.CONFIRMED
            else:
                confirmation_state = ActionConfirmationStatus.REJECTED
            self._pending_action.confirmation_state = confirmation_state  # type: ignore[attr-defined]
            self._pending_action._id = None  # type: ignore[attr-defined]
            self.event_stream.add_event(self._pending_action, EventSource.AGENT)

        self.state.agent_state = new_state
        self.event_stream.add_event(
            AgentStateChangedObservation('', self.state.agent_state),
            EventSource.ENVIRONMENT,
        )

    def get_agent_state(self) -> AgentState:
        """Returns the current state of the agent.

        Returns:
            AgentState: The current state of the agent.
        """
        return self.state.agent_state

    def _notify_on_llm_retry(self, retries: int, max: int) -> None:
        if self.status_callback is not None:
            msg_id = 'STATUS$LLM_RETRY'
            self.status_callback(
                'info', msg_id, f'Retrying LLM request, {retries} / {max}'
            )

    async def _handle_traffic_control(
        self, limit_type: str, current_value: float, max_value: float
    ) -> bool:
        """Handles agent state after hitting the traffic control limit.

        Args:
            limit_type (str): The type of limit that was hit.
            current_value (float): The current value of the limit.
            max_value (float): The maximum value of the limit.
        """
        stop_step = False
        if self.state.traffic_control_state == TrafficControlState.PAUSED:
            self.log(
                'debug', 'Hitting traffic control, temporarily resume upon user request'
            )
            self.state.traffic_control_state = TrafficControlState.NORMAL
        else:
            self.state.traffic_control_state = TrafficControlState.THROTTLING
            # Format values as integers for iterations, keep decimals for budget
            if limit_type == 'iteration':
                current_str = str(int(current_value))
                max_str = str(int(max_value))
            else:
                current_str = f'{current_value:.2f}'
                max_str = f'{max_value:.2f}'

            if self.headless_mode:
                e = RuntimeError(
                    f'Agent reached maximum {limit_type} in headless mode. '
                    f'Current {limit_type}: {current_str}, max {limit_type}: {max_str}'
                )
                await self._react_to_exception(e)
            else:
                e = RuntimeError(
                    f'Agent reached maximum {limit_type}. '
                    f'Current {limit_type}: {current_str}, max {limit_type}: {max_str}. '
                )
                # FIXME: this isn't really an exception--we should have a different path
                await self._react_to_exception(e)
            stop_step = True
        return stop_step

    def get_state(self) -> State:
        """Returns the current running state object.

        Returns:
            State: The current state object.
        """
        return self.state

    def set_initial_state(
        self,
        state: State | None,
        max_iterations: int,
        confirmation_mode: bool = False,
    ) -> None:
        """Sets the initial state for the agent, either from the previous session, or from a parent agent, or by creating a new one.

        Args:
            state: The state to initialize with, or None to create a new state.
            max_iterations: The maximum number of iterations allowed for the task.
            confirmation_mode: Whether to enable confirmation mode.
        """
        # state can come from:
        # - the previous session, in which case it has history
        # - from a parent agent, in which case it has no history
        # - None / a new state

        # If state is None, we create a brand new state and still load the event stream so we can restore the history
        if state is None:
            self.state = State(
                session_id=self.id.removesuffix('-delegate'),
                inputs={},
                max_iterations=max_iterations,
                confirmation_mode=confirmation_mode,
            )
            self.state.start_id = 0

            self.log(
                'debug',
                f'PlanController: {self.id} - created new state. start_id: {self.state.start_id}',
            )
        else:
            self.state = state

            if self.state.start_id <= -1:
                self.state.start_id = 0

            self.log(
                'debug',
                f'PlanController: {self.id} initializing history from event {self.state.start_id}',
            )

        # Always load from the event stream to avoid losing history
        self._init_history()

    def get_trajectory(self, include_screenshots: bool = False) -> list[dict]:
        # state history could be partially hidden/truncated before controller is closed
        assert self._closed
        return [
            event_to_trajectory(event, include_screenshots)
            for event in self.state.history
        ]

    def _init_history(self) -> None:
        """Initializes the agent's history from the event stream.

        The history is a list of events that:
        - Excludes events of types listed in self.filter_out
        - Excludes events with hidden=True attribute
        - For delegate events (between AgentDelegateAction and AgentDelegateObservation):
            - Excludes all events between the action and observation
            - Includes the delegate action and observation themselves

        The history is loaded in two parts if truncation_id is set:
        1. First user message from start_id onwards
        2. Rest of history from truncation_id to the end

        Otherwise loads normally from start_id.
        """
        # define range of events to fetch
        # delegates start with a start_id and initially won't find any events
        # otherwise we're restoring a previous session
        start_id = self.state.start_id if self.state.start_id >= 0 else 0
        end_id = (
            self.state.end_id
            if self.state.end_id >= 0
            else self.event_stream.get_latest_event_id()
        )

        # sanity check
        if start_id > end_id + 1:
            self.log(
                'warning',
                f'start_id {start_id} is greater than end_id + 1 ({end_id + 1}). History will be empty.',
            )
            self.state.history = []
            return

        events: list[Event] = []

        # If we have a truncation point, get first user message and then rest of history
        if hasattr(self.state, 'truncation_id') and self.state.truncation_id > 0:
            # Find first user message from stream
            first_user_msg = next(
                (
                    e
                    for e in self.event_stream.get_events(
                        start_id=start_id,
                        end_id=end_id,
                        reverse=False,
                        filter_out_type=self.filter_out,
                        filter_hidden=True,
                    )
                    if isinstance(e, MessageAction) and e.source == EventSource.USER
                ),
                None,
            )
            if first_user_msg:
                events.append(first_user_msg)

            # the rest of the events are from the truncation point
            start_id = self.state.truncation_id

        # Get rest of history
        events_to_add = list(
            self.event_stream.get_events(
                start_id=start_id,
                end_id=end_id,
                reverse=False,
                filter_out_type=self.filter_out,
                filter_hidden=True,
            )
        )
        events.extend(events_to_add)

        self.state.history = events

        # make sure history is in sync
        self.state.start_id = start_id

    def _handle_long_context_error(self) -> None:
        # When context window is exceeded, keep roughly half of agent interactions
        self.state.history = self._apply_conversation_window(self.state.history)

        # Save the ID of the first event in our truncated history for future reloading
        if self.state.history:
            self.state.start_id = self.state.history[0].id

        # Add an error event to trigger another step by the agent
        self.event_stream.add_event(
            AgentCondensationObservation(
                content='Trimming prompt to meet context window limitations'
            ),
            EventSource.AGENT,
        )

    def _apply_conversation_window(self, events: list[Event]) -> list[Event]:
        """Cuts history roughly in half when context window is exceeded.

        It preserves action-observation pairs and ensures that the first user message is always included.

        The algorithm:
        1. Cut history in half
        2. Check first event in new history:
           - If Observation: find and include its Action
           - If MessageAction: ensure its related Action-Observation pair isn't split
        3. Always include the first user message

        Args:
            events: List of events to filter

        Returns:
            Filtered list of events keeping newest half while preserving pairs
        """
        if not events:
            return events

        # Find first user message - we'll need to ensure it's included
        first_user_msg = next(
            (
                e
                for e in events
                if isinstance(e, MessageAction) and e.source == EventSource.USER
            ),
            None,
        )

        # cut in half
        mid_point = max(1, len(events) // 2)
        kept_events = events[mid_point:]

        # Handle first event in truncated history
        if kept_events:
            i = 0
            while i < len(kept_events):
                first_event = kept_events[i]
                if isinstance(first_event, Observation) and first_event.cause:
                    # Find its action and include it
                    matching_action = next(
                        (
                            e
                            for e in reversed(events[:mid_point])
                            if isinstance(e, Action) and e.id == first_event.cause
                        ),
                        None,
                    )
                    if matching_action:
                        kept_events = [matching_action] + kept_events
                    else:
                        self.log(
                            'warning',
                            f'Found Observation without matching Action at id={first_event.id}',
                        )
                        # drop this observation
                        kept_events = kept_events[1:]
                    break

                elif isinstance(first_event, MessageAction) or (
                    isinstance(first_event, Action)
                    and first_event.source == EventSource.USER
                ):
                    # if it's a message action or a user action, keep it and continue to find the next event
                    i += 1
                    continue

                else:
                    # if it's an action with source == EventSource.AGENT, we're good
                    break

        # Save where to continue from in next reload
        if kept_events:
            self.state.truncation_id = kept_events[0].id

        # Ensure first user message is included
        if first_user_msg and first_user_msg not in kept_events:
            kept_events = [first_user_msg] + kept_events

        # start_id points to first user message
        if first_user_msg:
            self.state.start_id = first_user_msg.id

        return kept_events

    def _is_stuck(self) -> bool:
        """Checks if the agent or its delegate is stuck in a loop.

        Returns:
            bool: True if the agent is stuck, False otherwise.
        """
        # check if delegate stuck
        # if self.delegate and self.delegate._is_stuck():
        #     return True

        return self._stuck_detector.is_stuck(self.headless_mode)

    def _prepare_metrics_for_frontend(self, action: Action) -> None:
        """Create a minimal metrics object for frontend display and log it.

        To avoid performance issues with long conversations, we only keep:
        - accumulated_cost: The current total cost
        - latest token_usage: Token statistics from the most recent API call

        Args:
            action: The action to attach metrics to
        """
        metrics = Metrics(model_name=self.planning_agent.llm.metrics.model_name)
        metrics.accumulated_cost = self.planning_agent.llm.metrics.accumulated_cost
        if self.planning_agent.llm.metrics.token_usages:
            latest_usage = self.planning_agent.llm.metrics.token_usages[-1]
            metrics.add_token_usage(
                prompt_tokens=latest_usage.prompt_tokens,
                completion_tokens=latest_usage.completion_tokens,
                cache_read_tokens=latest_usage.cache_read_tokens,
                cache_write_tokens=latest_usage.cache_write_tokens,
                response_id=latest_usage.response_id,
            )
        action.llm_metrics = metrics

        # Log the metrics information for frontend display
        log_usage: TokenUsage | None = (
            metrics.token_usages[-1] if metrics.token_usages else None
        )
        self.log(
            'debug',
            f'Action metrics - accumulated_cost: {metrics.accumulated_cost}, '
            f'tokens (prompt/completion/cache_read/cache_write): '
            f'{log_usage.prompt_tokens if log_usage else 0}/'
            f'{log_usage.completion_tokens if log_usage else 0}/'
            f'{log_usage.cache_read_tokens if log_usage else 0}/'
            f'{log_usage.cache_write_tokens if log_usage else 0}',
            extra={'msg_type': 'METRICS'},
        )

    def __repr__(self):
        return (
            f'PlanController:(id={getattr(self, "id", "<uninitialized>")}, '
            f'agent={getattr(self, "agent", "<uninitialized>")!r}, '
            f'event_stream={getattr(self, "event_stream", "<uninitialized>")!r}, '
            f'state={getattr(self, "state", "<uninitialized>")!r}, '
            f'delegate={getattr(self, "delegate", "<uninitialized>")!r}, '
            f'_pending_action={getattr(self, "_pending_action", "<uninitialized>")!r})'
        )

    def _is_awaiting_observation(self):
        events = self.event_stream.get_events(reverse=True)
        for event in events:
            if isinstance(event, AgentStateChangedObservation):
                result = event.agent_state == AgentState.RUNNING
                return result
        return False

    def _first_user_message(self) -> MessageAction | None:
        """Get the first user message for this agent.

        For regular agents, this is the first user message from the beginning (start_id=0).
        For delegate agents, this is the first user message after the delegate's start_id.

        Returns:
            MessageAction | None: The first user message, or None if no user message found
        """
        # Return cached message if any
        if self._cached_first_user_message is not None:
            return self._cached_first_user_message

        # Find the first user message
        self._cached_first_user_message = next(
            (
                e
                for e in self.event_stream.get_events(
                    start_id=self.state.start_id,
                )
                if isinstance(e, MessageAction) and e.source == EventSource.USER
            ),
            None,
        )
        return self._cached_first_user_message

    def _create_plan(self, action: CreatePlanAction) -> None:
        """Creates a plan for the agent.

        Args:
            action: The CreatePlanAction to process.
        """
        self.state.plans[action.plan_id] = Plan.from_create_plan_action(action)

        self.state.active_plan_id = action.plan_id
        # self.state.current_task_index = 0

    def _get_active_plan_status(self, w_result: bool = False) -> PlanStatusObservation:
        """Returns the status of the active plan.

        Returns:
            PlanStatusObservation: The status of the active plan.
        """
        active_plan: Plan = self.state.plans[self.state.active_plan_id]
        return PlanStatusObservation(
            status=active_plan.to_dict(),
            content=active_plan._format_plan(w_result=w_result),
        )

    async def _assign_task_to_the_delegate(self, action: AssignTaskAction) -> None:
        """Assign a task to the delegate.

        Args:
            action: The AssignTaskAction to process.
        """

        # init the task controllers if not already done
        if action.plan_id not in self.task_controllers:
            self.task_controllers[action.plan_id] = {}

        # init controller for the task
        controller = AgentController(
            sid=action.delegate_id,
            event_stream=self.event_stream,
            agent=self.agent,
            max_iterations=self.state.max_iterations // 2,
            max_budget_per_task=self.max_budget_per_task,
            agent_to_llm_config=self.agent_to_llm_config,
            agent_configs=self.agent_configs,
            confirmation_mode=self.state.confirmation_mode,
            headless_mode=False,
            status_callback=self.status_callback,
            initial_state=None,
            replay_events=None,
        )

        self.task_controllers[action.plan_id][action.task_index] = controller

        # assign the task to the agent controller
        assign_plan: Plan = self.state.plans[action.plan_id]

        assign_task_prompt = f"""
        CURRENT PLAN STATUS:
        {assign_plan._format_plan(w_result=True)}

        YOUR CURRENT TASK:
        You are now working on task {action.task_index}: "{assign_plan.tasks[action.task_index].content}".
        Please make it done as less steps as possible (preferably in max 5 steps).
        Know that current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
        """

        self.event_stream.add_event(
            MessageAction(content=assign_task_prompt, displayable=False),
            EventSource.USER,
        )

    def _is_awaiting_for_task_resolving(self) -> bool:
        for plan_id, task_controllers in self.task_controllers.items():
            for task_index, controller in task_controllers.items():
                if controller.get_agent_state() == AgentState.RUNNING:
                    return True
        return False

    def _is_all_task_resolved(self) -> bool:
        # if not self.task_controllers:
        #     return False

        active_plan: Plan = self.state.plans[self.state.active_plan_id]

        for task in active_plan.tasks:
            if (
                task.status != TaskStatus.COMPLETED
                and task.status != TaskStatus.BLOCKED
            ):
                return False

        return True
