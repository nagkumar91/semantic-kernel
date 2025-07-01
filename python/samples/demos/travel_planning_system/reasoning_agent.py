"""Reasoning-compatible agent that can work with both standard and reasoning models."""

import sys
import os
import logging
from typing import List, Optional, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add current directory to path for importing azure_reasoning_service
sys.path.append(os.path.dirname(__file__))

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole

try:
    from azure_reasoning_service import AzureReasoningCompletion
except ImportError:
    # If import fails, we'll handle it gracefully in the class
    AzureReasoningCompletion = None

logger = logging.getLogger(__name__)

# Known reasoning models that require the reasoning API endpoint
REASONING_MODELS = {
    "o1-mini", "o1-preview", "o1-pro",
    "o3-mini", "o3-pro",
    # Add more as they become available
}


class ReasoningCompatibleAgent(ChatCompletionAgent):
    """
    Agent that automatically detects reasoning models and uses the appropriate API.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        plugins: Optional[List[Any]] = None,
        **kwargs
    ):
        # Determine which service to use based on the model
        deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
        
        if self._is_reasoning_model(deployment_name):
            if AzureReasoningCompletion is None:
                logger.warning("AzureReasoningCompletion not available, falling back to standard service")
                service = AzureChatCompletion()
                modified_instructions = instructions
                self._is_fallback = True
            else:
                logger.info(f"Using reasoning model '{deployment_name}' with AzureReasoningCompletion")
                service = AzureReasoningCompletion()
                modified_instructions = self._adapt_instructions_for_reasoning(instructions)
                self._is_fallback = False
        else:
            logger.info(f"Using standard model '{deployment_name}' with AzureChatCompletion")
            service = AzureChatCompletion()
            modified_instructions = instructions
            self._is_fallback = False
        
        # Initialize the parent ChatCompletionAgent
        super().__init__(
            name=name,
            description=description,
            instructions=modified_instructions,
            service=service,
            plugins=plugins or [],
            **kwargs
        )
        
        self._deployment_name = deployment_name
        self._is_reasoning = self._is_reasoning_model(deployment_name)
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if the given model name is a reasoning model."""
        return any(reasoning_model in model_name.lower() for reasoning_model in REASONING_MODELS)
    
    def _adapt_instructions_for_reasoning(self, instructions: str) -> str:
        """
        Adapt instructions for reasoning models with Chain of Thought prompting.
        """
        cot_prompt = """
When solving problems, please think step by step using this approach:

1. **Understanding**: First, carefully read and understand what is being asked
2. **Analysis**: Break down the problem into smaller components
3. **Planning**: Consider different approaches and choose the best one
4. **Reasoning**: Work through your solution step by step, showing your thinking
5. **Verification**: Check if your solution makes sense and addresses all requirements

For travel planning tasks specifically:
- Consider budget constraints carefully
- Think about practical logistics (travel times, connections)
- Account for dietary restrictions and preferences
- Verify availability and realistic pricing
- Provide specific, actionable recommendations

Original instructions: """ + instructions + """

Please follow both the chain-of-thought approach above and the specific instructions for your role."""
        
        return cot_prompt
    
    async def invoke(self, chat_history: ChatHistory, **kwargs) -> List[ChatMessageContent]:
        """
        Override invoke to handle reasoning model specifics.
        """
        if self._is_reasoning:
            # For reasoning models, we need to handle system messages differently
            modified_history = self._adapt_chat_history_for_reasoning(chat_history)
            return await super().invoke(modified_history, **kwargs)
        else:
            return await super().invoke(chat_history, **kwargs)
    
    def _adapt_chat_history_for_reasoning(self, chat_history: ChatHistory) -> ChatHistory:
        """
        Adapt chat history for reasoning models by handling system messages.
        """
        adapted_history = ChatHistory()
        
        # Process each message
        for message in chat_history.messages:
            if message.role == AuthorRole.SYSTEM:
                # Convert system message to user message with a prefix
                user_content = f"[System Instructions]: {message.content}\n\nPlease follow these instructions in your responses."
                adapted_history.add_user_message(user_content)
            else:
                # Keep other messages as-is
                adapted_history.add_message(message)
        
        return adapted_history
    
    def get_model_info(self) -> dict:
        """Get information about the current model configuration."""
        return {
            "deployment_name": self._deployment_name,
            "is_reasoning_model": self._is_reasoning,
            "service_type": "AzureReasoningCompletion" if self._is_reasoning else "AzureChatCompletion"
        }


def create_reasoning_compatible_agent(
    name: str,
    description: str,
    instructions: str,
    plugins: Optional[List[Any]] = None,
    **kwargs
) -> ReasoningCompatibleAgent:
    """
    Factory function to create a reasoning-compatible agent.
    """
    return ReasoningCompatibleAgent(
        name=name,
        description=description,
        instructions=instructions,
        plugins=plugins,
        **kwargs
    )
