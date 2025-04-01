from enum import Enum
from typing import Dict, List, Optional


class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a step in a plan"""

    NOT_STARTED = 'not_started'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    BLOCKED = 'blocked'

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """Return a list of all possible step status values"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """Return a list of values representing active statuses (not started or in progress)"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """Return a mapping of statuses to their marker symbols"""
        return {
            cls.COMPLETED.value: '[✓]',
            cls.IN_PROGRESS.value: '[→]',
            cls.BLOCKED.value: '[!]',
            cls.NOT_STARTED.value: '[ ]',
        }


class PlanningTool:
    """
    The planning tool allows agents to create and manage plans to solve complex tasks.
    The tool provides functionality to create plans, update plan steps, and track progress.
    """

    def __init__(self):
        self.plans = {}  # Dictionary to store plans by plan_id
        self._current_plan_id = None  # Track the current active plan

    def to_param(self) -> dict:
        """Return the tool definition as a parameter for function calling to LLM."""
        return {
            'type': 'function',
            'function': {
                'name': 'planning',
                'description': 'The planning tool allows agents to create and manage plans to solve complex tasks.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {
                            'description': 'The command to execute. The available commands are: create, update, list, get, set_active, mark_step, delete, add_result.',
                            'enum': [
                                'create',
                                'update',
                                'list',
                                'get',
                                'set_active',
                                'mark_step',
                                'delete',
                                'add_result',
                            ],
                            'type': 'string',
                        },
                        'plan_id': {
                            'description': 'The unique identifier for the plan. Required for commands: create, update, set_active, and delete. Optional for commands: get and mark_step (use the active plan if not specified).',
                            'type': 'string',
                        },
                        'title': {
                            'description': 'The title for the plan. Required for command: create, optional for command: update.',
                            'type': 'string',
                        },
                        'steps': {
                            'description': 'The list of steps for the plan. Required for command: create, optional for command: update.',
                            'type': 'array',
                            'items': {'type': 'string'},
                        },
                        'step_index': {
                            'description': 'The index of the step to update (starting from 0). Required for commands: mark_step and add_result.',
                            'type': 'integer',
                        },
                        'step_status': {
                            'description': 'The status to set for a step. Used with command: mark_step.',
                            'enum': [
                                'not_started',
                                'in_progress',
                                'completed',
                                'blocked',
                            ],
                            'type': 'string',
                        },
                        'step_notes': {
                            'description': 'Additional notes for a step. Optional for command: mark_step.',
                            'type': 'string',
                        },
                        'step_result': {
                            'description': 'The result of a step. Used with command: add_result.',
                            'type': 'string',
                        },
                    },
                    'required': ['command'],
                },
            },
        }

    async def execute(self, **args):
        """
        Execute the planning tool with the given command and parameters.

        Parameters:
        - command: The action to perform
        - plan_id: The unique identifier for the plan
        - title: The title for the plan (used with command: create)
        - steps: The list of steps for the plan (used with command: create)
        - step_index: The index of the step to update (used with command: mark_step)
        - step_status: The status to set for a step (used with command: mark_step)
        - step_notes: Additional notes for a step (used with command: mark_step)
        - step_result: The result of a step (used with command: add_result)
        """
        command = args.get('command')
        plan_id = args.get('plan_id')
        title = args.get('title')
        steps = args.get('steps')
        step_index = args.get('step_index')
        step_status = args.get('step_status')
        step_notes = args.get('step_notes')
        step_result = args.get('step_result')

        if command == 'create':
            return self._create_plan(plan_id, title, steps)
        elif command == 'update':
            return self._update_plan(plan_id, title, steps)
        elif command == 'list':
            return self._list_plans()
        elif command == 'get':
            return self._get_plan(plan_id)
        elif command == 'set_active':
            return self._set_active_plan(plan_id)
        elif command == 'mark_step':
            return self._mark_step(plan_id, step_index, step_status, step_notes)
        elif command == 'delete':
            return self._delete_plan(plan_id)
        elif command == 'add_result':
            return self._add_result(plan_id, step_index, step_result)
        else:
            raise ValueError(
                f'Command not recognized: {command}. The allowed commands are: create, update, list, get, set_active, mark_step, delete, add_result'
            )

    def _create_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> dict:
        """Create a new plan with the given ID, title, and steps."""
        if not plan_id:
            raise ValueError('The `plan_id` parameter is required for command: create')

        if plan_id in self.plans:
            raise ValueError(
                f"Plan with ID '{plan_id}' already exists. Use 'update' to modify the existing plan."
            )

        if not title:
            raise ValueError('The `title` parameter is required for command: create')

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            raise ValueError(
                'The `steps` parameter must be a non-empty list of strings for command: create'
            )

        # Create a new plan with step statuses initialized
        plan = {
            'plan_id': plan_id,
            'title': title,
            'steps': steps,
            'step_statuses': ['not_started'] * len(steps),
            'step_notes': [''] * len(steps),
            'step_results': [None] * len(steps),
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id

        return {
            'output': f'Plan created successfully with ID: {plan_id}\n\n{self._format_plan(plan)}'
        }

    def _add_result(self, plan_id: str, step_index: int, result: Optional[str]) -> dict:
        """Add the result for a specific step in the plan."""
        if not plan_id:
            plan_id = self._current_plan_id

        if not plan_id or plan_id not in self.plans:
            raise ValueError(f'Plan not found with ID: {plan_id}')

        plan = self.plans[plan_id]

        # Ensure step_index is an integer
        try:
            step_index_int = int(step_index) if step_index is not None else 0
        except (ValueError, TypeError):
            raise ValueError(f"Invalid step_index: {step_index}. Must be an integer.")

        if step_index_int < 0 or step_index_int >= len(plan['steps']):
            raise ValueError(
                f"Invalid step_index: {step_index_int}. Valid indices are 0 to {len(plan['steps']) - 1}."
            )

        # Ensure result is a string
        result_str = str(result) if result is not None else ""

        plan['step_results'][step_index_int] = result_str

        return {
            'output': f"Result added to step {step_index_int} in plan '{plan_id}'.\n\n{self._format_plan(plan)}"
        }

    def _update_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> dict:
        """Update the existing plan with a new title or steps."""
        if not plan_id:
            plan_id = self._current_plan_id

        if not plan_id or plan_id not in self.plans:
            raise ValueError(f'Plan not found with ID: {plan_id}')

        plan = self.plans[plan_id]

        if title:
            plan['title'] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                raise ValueError(
                    'The `steps` parameter must be a list of strings for command: update'
                )

            # Keep the existing step statuses for unchanged steps
            old_steps = plan['steps']
            old_statuses = plan['step_statuses']
            old_notes = plan['step_notes']
            old_results = plan['step_results']

            # Create new step statuses and notes
            new_statuses = []
            new_notes = []
            new_results = []

            for i, step in enumerate(steps):
                # If the step exists at the same position in old_steps, keep the status and notes
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                    new_results.append(old_results[i] if i < len(old_results) else None)
                else:
                    new_statuses.append('not_started')
                    new_notes.append('')
                    new_results.append(None)

            plan['steps'] = steps
            plan['step_statuses'] = new_statuses
            plan['step_notes'] = new_notes
            plan['step_results'] = new_results

        return {
            'output': f'Plan updated successfully: {plan_id}\n\n{self._format_plan(plan)}'
        }

    def _list_plans(self) -> dict:
        """List all available plans."""
        if not self.plans:
            return {
                'output': "No plans found. Create a plan using the 'create' command."
            }

        output = 'Available plans:\n'
        for plan_id, plan in self.plans.items():
            current_marker = ' (active)' if plan_id == self._current_plan_id else ''
            completed = sum(
                1 for status in plan['step_statuses'] if status == 'completed'
            )
            total = len(plan['steps'])
            progress = f'{completed}/{total} steps completed'
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        return {'output': output}

    def _get_plan(self, plan_id: Optional[str]) -> dict:
        """Get the details of a specific plan."""
        if not plan_id:
            plan_id = self._current_plan_id

        if not plan_id or plan_id not in self.plans:
            raise ValueError(f'Plan not found with ID: {plan_id}')

        plan = self.plans[plan_id]
        return {'output': self._format_plan(plan)}

    def _set_active_plan(self, plan_id: Optional[str]) -> dict:
        """Set the current active plan."""
        if not plan_id:
            raise ValueError(
                'The `plan_id` parameter is required for command: set_active'
            )

        if plan_id not in self.plans:
            raise ValueError(f'Plan not found with ID: {plan_id}')

        self._current_plan_id = plan_id
        return {'output': f'The active plan is now set to: {plan_id}'}

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> dict:
        """Mark a step in the plan with a new status and optional notes."""
        if not plan_id:
            plan_id = self._current_plan_id

        if not plan_id or plan_id not in self.plans:
            raise ValueError(f'Plan not found with ID: {plan_id}')

        if step_index is None:
            raise ValueError(
                'The `step_index` parameter is required for command: mark_step'
            )

        plan = self.plans[plan_id]

        if step_index < 0 or step_index >= len(plan['steps']):
            raise ValueError(
                f"Invalid step_index: {step_index}. Valid indices are 0 to {len(plan['steps']) - 1}."
            )

        if step_status:
            if step_status not in PlanStepStatus.get_all_statuses():
                raise ValueError(
                    f"Invalid step_status: {step_status}. Valid statuses: {', '.join(PlanStepStatus.get_all_statuses())}"
                )
            plan['step_statuses'][step_index] = step_status

        # Update the notes if provided
        if step_notes:
            plan['step_notes'][step_index] = step_notes

        return {
            'output': f"Step {step_index} updated in plan '{plan_id}'.\n\n{self._format_plan(plan)}"
        }

    def _delete_plan(self, plan_id: Optional[str]) -> dict:
        """Delete a plan from memory."""
        if not plan_id:
            raise ValueError('The `plan_id` parameter is required for command: delete')

        if plan_id not in self.plans:
            raise ValueError(f'Plan not found with ID: {plan_id}')

        # Remove the plan from storage
        self.plans.pop(plan_id)

        if self._current_plan_id == plan_id:
            self._current_plan_id = next(iter(self.plans)) if self.plans else None

        return {'output': f"Plan with ID '{plan_id}' has been deleted successfully."}

    def _format_plan(self, plan: Dict) -> str:
        """Format the plan into a structured text for display."""
        title = plan.get('title', 'Plan without title')
        steps = plan.get('steps', [])
        step_statuses = plan.get('step_statuses', [])
        step_notes = plan.get('step_notes', [])
        step_results = plan.get('step_results', [])

        while len(step_statuses) < len(steps):
            step_statuses.append(PlanStepStatus.NOT_STARTED.value)
        while len(step_notes) < len(steps):
            step_notes.append('')
        while len(step_results) < len(steps):
            step_results.append(None)

        status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

        for status in step_statuses:
            if status in status_counts:
                status_counts[status] += 1

        completed = status_counts[PlanStepStatus.COMPLETED.value]
        total = len(steps)
        progress = (completed / total) * 100 if total > 0 else 0

        plan_text = f"Plan: {title} (ID: {plan['plan_id']})\n"
        plan_text += '=' * len(plan_text) + '\n\n'

        plan_text += (
            f'Progress: {completed}/{total} steps completed ({progress:.1f}%)\n'
        )
        plan_text += f'Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, '
        plan_text += f'{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n'
        plan_text += 'Steps:\n'

        status_marks = PlanStepStatus.get_status_marks()

        for i, (step, status, notes, result) in enumerate(
            zip(steps, step_statuses, step_notes, step_results)
        ):
            status_mark = status_marks.get(
                status, status_marks[PlanStepStatus.NOT_STARTED.value]
            )

            plan_text += f'{i}. {status_mark} {step}\n'
            if notes:
                plan_text += f'   Notes: {notes}\n'
            if result:
                plan_text += f'   Result: {result}\n'

        return plan_text

    def _format_plan_wo_result(self, plan: Dict) -> str:
        """Format the plan into a structured text for display, excluding step results."""
        title = plan.get('title', 'Plan without title')
        steps = plan.get('steps', [])
        step_statuses = plan.get('step_statuses', [])
        step_notes = plan.get('step_notes', [])

        # Ensure step_statuses and step_notes match the number of steps
        while len(step_statuses) < len(steps):
            step_statuses.append(PlanStepStatus.NOT_STARTED.value)
        while len(step_notes) < len(steps):
            step_notes.append('')

        status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

        for status in step_statuses:
            if status in status_counts:
                status_counts[status] += 1

        completed = status_counts[PlanStepStatus.COMPLETED.value]
        total = len(steps)
        progress = (completed / total) * 100 if total > 0 else 0

        plan_text = f"Plan: {title} (ID: {plan['plan_id']})\n"
        plan_text += '=' * len(plan_text) + '\n\n'

        plan_text += (
            f'Progress: {completed}/{total} steps completed ({progress:.1f}%)\n'
        )
        plan_text += f'Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, '
        plan_text += f'{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n'
        plan_text += 'Steps:\n'

        status_marks = PlanStepStatus.get_status_marks()

        for i, (step, status, notes) in enumerate(
            zip(steps, step_statuses, step_notes)
        ):
            # Use the status mark to indicate the step status
            status_mark = status_marks.get(
                status, status_marks[PlanStepStatus.NOT_STARTED.value]
            )

            plan_text += f'{i}. {status_mark} {step}\n'
            if notes:
                plan_text += f'   Ghi chú: {notes}\n'

        return plan_text
