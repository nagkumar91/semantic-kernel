"""Azure OpenAI Reasoning Service for o3/o1 models."""

import logging
import os
from typing import Any, List
import httpx
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.exceptions import ServiceResponseException

logger = logging.getLogger(__name__)


class AzureReasoningCompletion(AzureChatCompletion):
    """Azure OpenAI Reasoning service for o3/o1 models that use the reasoning API."""
    
    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings,
        **kwargs: Any,
    ) -> List[ChatMessageContent]:
        """Override to use reasoning API endpoint with Chain of Thought enhancement."""
        try:
            # Convert chat history to reasoning API format
            messages = []
            for message in chat_history.messages:
                # Skip system messages as they're not supported by reasoning models
                if message.role == AuthorRole.SYSTEM:
                    continue
                    
                # Enhance user messages with chain-of-thought prompting
                content = message.content
                if message.role == AuthorRole.USER and content:
                    # Check if this is a planning/complex task that would benefit from CoT
                    cot_keywords = ["plan", "book", "find", "search", "recommend", "budget", "analyze", "compare"]
                    if any(keyword in content.lower() for keyword in cot_keywords):
                        content = f"Think step by step about this request. {content}"
                
                messages.append({
                    "role": str(message.role),
                    "content": content
                })
            
            # Prepare reasoning API request
            request_data = {
                "messages": messages,
                "max_completion_tokens": getattr(settings, "max_tokens", 4000),
                "temperature": getattr(settings, "temperature", 0.7),
            }
            
            # Get configuration from environment variables
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
            api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
            deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "o3-mini")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            
            # Use reasoning endpoint instead of chat/completions
            url = f"{endpoint}/openai/deployments/{deployment_name}/reasoning/completions"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key,
            }
            
            logger.info(f"Making reasoning API call to: {url}")
            logger.info(f"Deployment: {deployment_name}, API Version: {api_version}")
            
            # Make the API call
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=url,
                    headers=headers,
                    json=request_data,
                    params={"api-version": api_version},
                    timeout=120.0  # Reasoning models can take longer
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Azure Reasoning API error: {response.status_code} - {error_text}")
                    logger.error(f"Request URL: {url}")
                    logger.error(f"API Version: {api_version}")
                    
                    raise ServiceResponseException(
                        f"Azure Reasoning API failed with status {response.status_code}: {error_text}"
                    )
                
                response_data = response.json()
                logger.info(f"Reasoning API response received successfully")
                
                # Extract the response content
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0]["message"]["content"]
                    return [ChatMessageContent(
                        role=AuthorRole.ASSISTANT,
                        content=content,
                        model_id=deployment_name
                    )]
                else:
                    raise ServiceResponseException("No valid response from Azure Reasoning API")
                    
        except httpx.RequestError as e:
            logger.error(f"Request error calling Azure Reasoning API: {e}")
            raise ServiceResponseException(f"Request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Azure Reasoning API: {e}")
            raise ServiceResponseException(f"Azure Reasoning service error: {e}")
    
    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings,
        **kwargs: Any,
    ):
        """Reasoning models don't typically support streaming, fall back to non-streaming."""
        results = await self.get_chat_message_contents(chat_history, settings, **kwargs)
        # Simulate streaming by yielding the full response
        for result in results:
            yield result
