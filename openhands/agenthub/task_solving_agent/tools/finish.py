from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_FINISH_DESCRIPTION = """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task and saved the final answer to file
- You maynot proceed further due to technical limitations or missing information

The message should concise and include:
- The absolute path to the file where the final answer is saved. Eg: `/workspace/36eedc34afb34d84ba1a1bfdb13e0e97/result.md or /workspace/result.md`. If there's a session id, it should be included in the path, e.g. `/workspace/36eedc34afb34d84ba1a1bfdb13e0e97/result.md`."

The task_completed field should be set to True if you believe you have successfully completed the task, and False otherwise.
"""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish',
        description=_FINISH_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['message', 'task_completed'],
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'The final message to the user, including the absolute path to the file where the final answer is saved.',
                },
                'task_completed': {
                    'type': 'boolean',
                    'description': "Whether you believe you have successfully completed the user's task",
                },
            },
            'additionalProperties': False,
        },
    ),
)
