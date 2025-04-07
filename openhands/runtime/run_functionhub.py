#!/usr/bin/env python
import json
from enum import Enum
from typing import Any, Dict, List

import aiohttp
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk
from pydantic import BaseModel, Field

from openhands.core.config.functionhub_config import FunctionHubConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.observation.functionhub import FunctionHubObservation


class ResponseType(str, Enum):
    TEXT: str = 'text'
    IMAGE_URL: str = 'image_url'
    VIDEO_URL: str = 'video_url'
    AUDIO_URL: str = 'audio_url'
    IMAGE: str = 'image'  # base64 encoded image
    VIDEO: str = 'video'  # base64 encoded video
    AUDIO: str = 'audio'  # base64 encoded
    BLOB: str = 'blob'  # base64 encoded file
    ERROR: str = 'error'


class BaseFunctionHubResponse(BaseModel):
    response_type: ResponseType = Field(default=ResponseType.TEXT)
    content: str = Field(default='')
    description: str = Field(default='')


class FunctionHubChatCompletionToolParam(ChatCompletionToolParam):
    id_functionhub: str = Field(default='')


class FunctionHubRunner:
    def __init__(self, config: FunctionHubConfig | None = None):
        self.config = config

        # Initialize default values
        self.function_hub_url = None
        self.function_hub_wallet_address = None
        self.function_hub_api_key = None

        # Only access config attributes if config is not None
        if self.config is not None:
            self.function_hub_url = self.config.function_hub_url
            self.function_hub_wallet_address = self.config.function_hub_wallet_address
            self.function_hub_api_key = self.config.function_hub_api_key

        logger.info(
            f'FunctionHubRunner config: {self.function_hub_url}, {self.function_hub_wallet_address}, {self.function_hub_api_key}'
        )

        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'X-API-KEY': self.function_hub_api_key,
        }

    async def run(self, id_functionhub: str, arguments: dict) -> FunctionHubObservation:
        """Run a FunctionHub function and return an observation."""
        logger.info(f'Running FunctionHub function: {id_functionhub}, {arguments}')
        responses = await self.execute_function(id_functionhub, arguments)
        logger.info(f'Responses: {responses}')

        # Extract function name if available in the first response description
        function_name = ''
        if (
            responses
            and hasattr(responses[0], 'description')
            and responses[0].description
        ):
            function_name = responses[0].description

        # Initialize lists for different response types
        text_contents = []
        image_urls = []
        video_urls = []
        audio_urls = []
        blob = ''
        error = ''

        # Process each response based on its type
        for response in responses:
            if response.response_type == ResponseType.ERROR:
                if error:
                    error += '\n'
                error += response.content
                continue

            if response.response_type == ResponseType.TEXT:
                text_contents.append(response.content)

            elif response.response_type == ResponseType.IMAGE_URL:
                image_urls.append(response.content)
                # Also add a reference to the image in text content
                text_contents.append(
                    f"[Image: {response.description or 'Generated image'}]"
                )

            elif response.response_type == ResponseType.VIDEO_URL:
                video_urls.append(response.content)
                text_contents.append(
                    f"[Video: {response.description or 'Generated video'}]"
                )

            elif response.response_type == ResponseType.AUDIO_URL:
                audio_urls.append(response.content)
                text_contents.append(
                    f"[Audio: {response.description or 'Generated audio'}]"
                )

            elif response.response_type == ResponseType.BLOB:
                blob = response.content
                text_contents.append(
                    f"[File: {response.description or 'Generated file'}]"
                )

            # Handle other response types (IMAGE, VIDEO, AUDIO)
            elif response.response_type in [
                ResponseType.IMAGE,
                ResponseType.VIDEO,
                ResponseType.AUDIO,
            ]:
                if not blob:
                    blob = response.content
                text_contents.append(
                    f"[{response.response_type.value.capitalize()}: {response.description or f'Generated {response.type.value}'}]"
                )

        # Join all text content with newlines
        combined_text_content = '\n'.join(text_contents)
        logger.info(f'Combined text content: {combined_text_content}')
        logger.info(f'Image urls: {image_urls}')
        logger.info(f'Video urls: {video_urls}')
        logger.info(f'Audio urls: {audio_urls}')
        logger.info(f'Blob: {blob}')
        logger.info(f'Error: {error}')
        # Create and return the observation
        observation = FunctionHubObservation(
            function_name=function_name,
            id_functionhub=id_functionhub,
            text_content=combined_text_content,
            content=combined_text_content,
            image_urls=image_urls,
            video_urls=video_urls,
            audio_urls=audio_urls,
            blob=blob,
            error=error,
        )
        return observation

    async def execute_function(
        self, id_functionhub: str, arguments: dict
    ) -> List[BaseFunctionHubResponse]:
        """Execute a function in FunctionHub and return the responses."""
        logger.info(f'Executing FunctionHub function: {id_functionhub}, {arguments}')
        url = f'{self.function_hub_url}/v1/functions/execute-function'

        payload = {
            'wallet': self.function_hub_wallet_address,
            'function_id': id_functionhub,
            'arguments': arguments,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=self.headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f'Error executing function: {error_text}')
                        return [
                            BaseFunctionHubResponse(
                                response_type=ResponseType.ERROR,
                                content=f'Error {response.status}: {error_text}',
                                description='Failed to execute function',
                            )
                        ]

                    result = await response.json()

                    if 'result' in result:
                        result = result['result']
                    else:
                        logger.error(f'Unknown response format: {result}')

                    # Process and convert the result to BaseFunctionHubResponse objects
                    responses = []
                    if isinstance(result, dict):
                        # Handle single response
                        responses.append(self._process_response(result))
                    elif isinstance(result, list):
                        # Handle multiple responses
                        for item in result:
                            responses.append(self._process_response(item))
                    else:
                        # Handle unexpected response format
                        responses.append(
                            BaseFunctionHubResponse(
                                response_type=ResponseType.TEXT,
                                content=str(result),
                                description='Unknown response format',
                            )
                        )

                    return responses

        except Exception as e:
            logger.error(f'Exception during function execution: {str(e)}')
            return [
                BaseFunctionHubResponse(
                    response_type=ResponseType.ERROR,
                    content=str(e),
                    description='Exception during function execution',
                )
            ]

    def search_with_rerank(
        self,
        search_query: str,
        reranker_query: str,
        top_k_search: int = 20,
        top_k_reranked: int = 5,
        timeout: int = 5,
    ) -> List[FunctionHubChatCompletionToolParam]:
        """Search for functions and rerank results."""
        import requests

        url = f'{self.function_hub_url}/v1/functions/search-function-and-rerank'

        payload = {
            'wallet': self.function_hub_wallet_address,
            'search_query': search_query,
            'reranker_query': reranker_query,
            'top_k_search': top_k_search,
            'top_k_reranked': top_k_reranked,
        }

        try:
            response = requests.post(
                url, headers=self.headers, json=payload, timeout=timeout
            )

            if response.status_code != 200:
                logger.error(f'Error in search with rerank: {response.text}')
                return []

            result = response.json()

            # Process and convert the search results to ChatCompletionToolParam objects
            tools = []

            if 'results' in result and isinstance(result['results'], list):
                for item in result['results']:
                    if 'entity' in item and 'function_metadata' in item['entity']:
                        metadata = item['entity']['function_metadata']
                        function_id = item['entity'].get('function_id', '')

                        if 'function' in metadata:
                            func_info = metadata['function']

                            # Create a ChatCompletionToolParam object
                            tool = FunctionHubChatCompletionToolParam(
                                type='function',
                                id_functionhub=function_id,
                                function=ChatCompletionToolParamFunctionChunk(
                                    name=func_info.get(
                                        'name', f'function_{function_id}'
                                    ),
                                    description=func_info.get('description', ''),
                                    parameters=func_info.get('parameters', {}),
                                ),
                            )

                            # Add to the list of tools
                            tools.append(tool)

            return tools

        except Exception as e:
            logger.error(f'Exception during search with rerank: {str(e)}')
            return []

    async def asearch_with_rerank(
        self,
        search_query: str,
        reranker_query: str,
        top_k_search: int = 20,
        top_k_reranked: int = 5,
    ) -> List[FunctionHubChatCompletionToolParam]:
        """Search for functions and rerank results."""
        url = f'{self.function_hub_url}/v1/functions/search-function-and-rerank'

        payload = {
            'wallet': self.function_hub_wallet_address,
            'search_query': search_query,
            'reranker_query': reranker_query,
            'top_k_search': top_k_search,
            'top_k_reranked': top_k_reranked,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=self.headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f'Error in search with rerank: {error_text}')
                        return []

                    result = await response.json()

                    # Process and convert the search results to ChatCompletionToolParam objects
                    tools = []

                    if 'results' in result and isinstance(result['results'], list):
                        for item in result['results']:
                            if (
                                'entity' in item
                                and 'function_metadata' in item['entity']
                            ):
                                metadata = item['entity']['function_metadata']
                                function_id = item['entity'].get('function_id', '')

                                if 'function' in metadata:
                                    func_info = metadata['function']

                                    # Create a ChatCompletionToolParam object
                                    tool = FunctionHubChatCompletionToolParam(
                                        type='function',
                                        id_functionhub=function_id,
                                        function=ChatCompletionToolParamFunctionChunk(
                                            name=func_info.get(
                                                'name', f'function_{function_id}'
                                            ),
                                            description=func_info.get(
                                                'description', ''
                                            ),
                                            parameters=func_info.get('parameters', {}),
                                        ),
                                    )

                                    # Add to the list of tools
                                    tools.append(tool)

                    return tools

        except Exception as e:
            logger.error(f'Exception during search with rerank: {str(e)}')
            return []

    def _process_response(
        self, response_data: Dict[str, Any]
    ) -> BaseFunctionHubResponse:
        """Process and convert a single response to BaseFunctionHubResponse format."""
        # Determine response type based on content
        response_type = ResponseType.TEXT
        content = ''
        description = ''

        if (
            'type' in response_data
            and response_data['type'] in ResponseType.__members__.values()
        ):
            response_type = response_data['type']

        if 'content' in response_data:
            content = response_data['content']
        if 'description' in response_data:
            description = response_data['description']

        return BaseFunctionHubResponse(
            response_type=response_type, content=content, description=description
        )

    def _process_search_result(
        self, result_item: Dict[str, Any]
    ) -> BaseFunctionHubResponse:
        """Process and convert a search result item to BaseFunctionHubResponse format."""
        description = result_item.get('function_name', '') or result_item.get(
            'name', ''
        )

        # Get content from the result item
        if 'description' in result_item:
            content = result_item['description']
        elif 'summary' in result_item:
            content = result_item['summary']
        else:
            content = json.dumps(result_item)

        # Additional metadata
        if 'function_id' in result_item:
            content += f"\nFunction ID: {result_item['function_id']}"

        return BaseFunctionHubResponse(
            response_type=ResponseType.TEXT, content=content, description=description
        )
