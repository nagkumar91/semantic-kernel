# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import logging
import sys
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar, Any
from opentelemetry import trace
DefaultMessageType = "default"

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.orchestration.agent_actor_base import ActorBase, AgentActorBase
from semantic_kernel.agents.orchestration.orchestration_base import DefaultTypeAlias, OrchestrationBase, TIn, TOut
from semantic_kernel.agents.runtime.core.cancellation_token import CancellationToken
from semantic_kernel.agents.runtime.core.core_runtime import CoreRuntime
from semantic_kernel.agents.runtime.core.message_context import MessageContext
from semantic_kernel.agents.runtime.core.routed_agent import message_handler
from semantic_kernel.agents.runtime.core.topic import TopicId
from semantic_kernel.agents.runtime.in_process.type_subscription import TypeSubscription
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.utils.feature_stage_decorator import experimental

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


# region Messages and Types


@experimental
class GroupChatStartMessage(KernelBaseModel):
    """A message type to start a group chat."""

    body: DefaultTypeAlias


@experimental
class GroupChatRequestMessage(KernelBaseModel):
    """A request message type for agents in a group chat."""

    agent_name: str


@experimental
class GroupChatResponseMessage(KernelBaseModel):
    """A response message type from agents in a group chat."""

    body: ChatMessageContent


_TGroupChatManagerResult = TypeVar("_TGroupChatManagerResult", ChatMessageContent, str, bool)


@experimental
class GroupChatManagerResult(KernelBaseModel, Generic[_TGroupChatManagerResult]):
    """A result message type from the group chat manager."""

    result: _TGroupChatManagerResult
    reason: str


# Subclassing GroupChatManagerResult to create specific result types because
# we need to change the names of the classes to remove the generic type parameters.
# Many model services (e.g. OpenAI) do not support generic type parameters in the
# class name (e.g. "GroupChatManagerResult[bool]").
@experimental
class BooleanResult(GroupChatManagerResult[bool]):
    """A result message type from the group chat manager with a boolean result."""

    pass


@experimental
class StringResult(GroupChatManagerResult[str]):
    """A result message type from the group chat manager with a string result."""

    pass


@experimental
class MessageResult(GroupChatManagerResult[ChatMessageContent]):
    """A result message type from the group chat manager with a message result."""

    pass


# endregion Messages and Types

# region GroupChatAgentActor



@experimental
class GroupChatAgentActor(AgentActorBase):
    """An agent actor that process messages in a group chat."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version="1.0.0"
        )

    @message_handler
    async def _handle_start_message(self, message: GroupChatStartMessage, ctx: MessageContext) -> None:
        """Handle the start message for the group chat."""
        logger.debug(f"{self.id}: Received group chat start message.")
        if isinstance(message.body, ChatMessageContent):
            if self._agent_thread:
                await self._agent_thread.on_new_message(message.body)
            else:
                self._chat_history.add_message(message.body)
        elif isinstance(message.body, list) and all(isinstance(m, ChatMessageContent) for m in message.body):
            if self._agent_thread:
                for m in message.body:
                    await self._agent_thread.on_new_message(m)
            else:
                for m in message.body:
                    self._chat_history.add_message(m)
        else:
            raise ValueError(f"Invalid message body type: {type(message.body)}. Expected {DefaultTypeAlias}.")

    @message_handler
    async def _handle_response_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        """Handle response messages."""
        logger.debug(f"{self.id}: Received group chat response message.")
        if self._agent_thread is not None:
            await self._agent_thread.on_new_message(message.body)
        else:
            self._chat_history.add_message(message.body)

    @message_handler
    async def _handle_request_message(self, message: GroupChatRequestMessage, ctx: MessageContext) -> None:
        if message.agent_name != self._agent.name:
            return

        logger.debug(f"{self.id}: Received group chat request message.")

        response = await self._invoke_agent()

        logger.debug(f"{self.id} responded with {response}.")

        await self.publish_message(
            GroupChatResponseMessage(body=response),
            TopicId(self._internal_topic_type, self.id.key),
            cancellation_token=ctx.cancellation_token,
        )


# endregion GroupChatAgentActor


# region GroupChatManager


@experimental
class GroupChatManager(KernelBaseModel, ABC):
    """A group chat manager that manages the flow of a group chat."""

    current_round: int = 0
    max_rounds: int | None = None

    human_response_function: Callable[[ChatHistory], Awaitable[ChatMessageContent] | ChatMessageContent] | None = None

    @abstractmethod
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """Check if the group chat should request user input.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
        """
        ...

    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """Check if the group chat should terminate.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
        """
        self.current_round += 1

        if self.max_rounds is not None:
            return BooleanResult(
                result=self.current_round > self.max_rounds,
                reason="Maximum rounds reached."
                if self.current_round > self.max_rounds
                else "Not reached maximum rounds.",
            )
        return BooleanResult(result=False, reason="No maximum rounds set.")

    @abstractmethod
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringResult:
        """Select the next agent to speak.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
            participant_descriptions (dict[str, str]): The descriptions of the participants in the group chat.
        """
        ...

    @abstractmethod
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> MessageResult:
        """Filter the results of the group chat.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
            participant_descriptions (dict[str, str]): The descriptions of the participants in the group chat.
        """
        ...


@experimental
class RoundRobinGroupChatManager(GroupChatManager):
    """A round-robin group chat manager with comprehensive OpenTelemetry instrumentation."""

    current_index: int = 0
    _tracer: trace.Tracer = None

    def __init__(self, **kwargs):
        """Initialize with tracer support."""
        super().__init__(**kwargs)
        self._tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version="1.0.0"
        )

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """Check if the group chat should request user input with tracing."""
        with self._tracer.start_as_current_span("plan_task") as span:
            # Check if the last message contains a question
            should_ask = False
            reason = "Round-robin manager checking for questions"
            
            if chat_history.messages and self.human_response_function:
                last_message = chat_history.messages[-1]
                if last_message.content and "?" in last_message.content:
                    should_ask = True
                    reason = "Agent asked a question requiring user input"
            
            # Set required attributes
            span.set_attributes({
                # Required gen_ai attributes
                "gen_ai.operation.name": "plan_task",
                "gen_ai.system": "semantic_kernel",
                
                # Required planning attributes
                "gen_ai.planning.type": "decision",
                "gen_ai.planning.complexity": "simple",
                
                # Conditionally required for decision
                "gen_ai.decision.category": "flow_control",
                "gen_ai.decision.outcome": "user_input_needed" if should_ask else "no_user_input_needed",
                
                # Recommended
                "gen_ai.planning.reasoning": reason,
                "gen_ai.planning.iteration": self.current_round,
                
                # Context
                "chat.history_length": len(chat_history.messages),
                "manager.type": "round_robin",
                "has_question": should_ask,
            })
            
            # Add decision event
            span.add_event(
                "decision_made",
                {
                    "decision_type": "user_input_request",
                    "result": should_ask,
                    "reason": reason,
                }
            )
            
            return BooleanResult(
                result=should_ask,
                reason=reason,
            )


    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """Check if the group chat should terminate with comprehensive tracing."""
        with self._tracer.start_as_current_span("plan_task") as span:
            # Increment round counter
            self.current_round += 1
            
            # Determine if we should terminate
            should_terminate = self.max_rounds is not None and self.current_round > self.max_rounds
            termination_reason = (
                "Maximum rounds reached." if should_terminate 
                else "Not reached maximum rounds." if self.max_rounds 
                else "No maximum rounds set."
            )
            
            # Set span attributes
            span.set_attributes({
                # Required gen_ai attributes
                "gen_ai.operation.name": "plan_task", 
                "gen_ai.system": "semantic_kernel",
                
                # Required planning attributes
                "gen_ai.planning.type": "decision",
                "gen_ai.planning.complexity": "simple",
                
                # Conditionally required for decision
                "gen_ai.decision.category": "termination",
                "gen_ai.decision.outcome": str(should_terminate),
                
                # Recommended
                "gen_ai.planning.reasoning": termination_reason,
                "gen_ai.planning.iteration": self.current_round,
                
                # Round-robin specific
                "round_robin.current_round": self.current_round,
                "round_robin.max_rounds": self.max_rounds or -1,
                "round_robin.should_terminate": should_terminate,
                "chat.message_count": len(chat_history.messages),
            })
            
            # Add termination check event
            span.add_event(
                "termination_check",
                {
                    "current_round": self.current_round,
                    "max_rounds": self.max_rounds,
                    "should_terminate": should_terminate,
                    "reason": termination_reason,
                }
            )
            
            return BooleanResult(
                result=should_terminate,
                reason=termination_reason,
            )

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringResult:
        """Select the next agent to speak with comprehensive tracing."""
        with self._tracer.start_as_current_span("plan_task") as span:
            # Get the next agent in round-robin order
            agents = list(participant_descriptions.keys())
            next_agent = agents[self.current_index]
            previous_index = self.current_index
            self.current_index = (self.current_index + 1) % len(agents)
            
            # Set span attributes
            span.set_attributes({
                # Required gen_ai attributes
                "gen_ai.operation.name": "plan_task",
                "gen_ai.system": "semantic_kernel",
                
                # Required planning attributes  
                "gen_ai.planning.type": "decision",
                "gen_ai.planning.complexity": "simple",
                
                # Conditionally required for decision
                "gen_ai.decision.category": "agent_delegation",
                "gen_ai.decision.outcome": next_agent,
                
                # Recommended
                "gen_ai.planning.reasoning": f"Round-robin selection: position {previous_index} -> {next_agent}",
                "gen_ai.planning.iteration": self.current_round,
                
                # Round-robin specific
                "round_robin.current_index": previous_index,
                "round_robin.next_index": self.current_index,
                "round_robin.selected_agent": next_agent,
                "round_robin.total_agents": len(agents),
                "round_robin.available_agents": json.dumps(agents),
                "chat.history_length": len(chat_history.messages),
            })
            
            # Add agent selection event
            span.add_event(
                "agent_selected",
                {
                    "selection_method": "round_robin",
                    "selected_agent": next_agent,
                    "agent_position": previous_index,
                    "next_position": self.current_index,
                    "total_agents": len(agents),
                }
            )
            
            # Log the selection
            logger.debug(
                f"Round-robin selected agent: {next_agent} (position {previous_index}/{len(agents)-1})",
                extra={
                    "gen_ai.interaction.source_agent_id": "round_robin_manager",
                    "gen_ai.interaction.target_agent_id": next_agent,
                    "gen_ai.interaction.message_type": "request",
                    "gen_ai.interaction.pattern": "round_robin",
                }
            )
            
            return StringResult(
                result=next_agent, 
                reason=f"Round-robin selection at position {previous_index}."
            )

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> MessageResult:
        """Filter the results of the group chat with memory operation tracing."""
        with self._tracer.start_as_current_span("memory_operation") as span:
            # Get the last message as the result
            last_message = chat_history.messages[-1] if chat_history.messages else None
            
            # Set span attributes
            span.set_attributes({
                # Required gen_ai attributes
                "gen_ai.operation.name": "memory_operation",
                "gen_ai.system": "semantic_kernel",
                
                # Required memory attributes
                "gen_ai.memory.operation_type": "read",
                "gen_ai.memory.agent_id": "round_robin_manager",
                
                # Recommended memory attributes
                "gen_ai.memory.source_type": "conversation",
                "gen_ai.memory.memory_type": "working",
                "gen_ai.memory.size_bytes": len(str(last_message)) if last_message else 0,
                
                # Additional context
                "chat.total_messages": len(chat_history.messages),
                "result.has_content": bool(last_message),
                "result.agent_name": last_message.name if last_message else "none",
            })
            
            if last_message:
                # Add memory snapshot event
                span.add_event(
                    "gen_ai.memory.snapshot",
                    {
                        "gen_ai.snapshot.id": f"round_robin_result_{self.current_round}",
                        "gen_ai.snapshot.size_bytes": len(str(last_message)),
                        "gen_ai.snapshot.fields": json.dumps(["content", "role", "name"]),
                    }
                )
            
            return MessageResult(
                result=last_message or ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content="No messages in chat history.",
                ),
                reason="The last message in the chat history is the result in the default round-robin group chat manager.",
            )

# endregion GroupChatManager

# region GroupChatManagerActor

@experimental
class GroupChatManagerActor(ActorBase):
    """A group chat manager actor that manages the flow of a group chat."""

    def __init__(
        self,
        manager: GroupChatManager,
        participant_descriptions: dict[str, str],
        internal_topic_type: type | None = None, 
        result_callback: Callable[[ChatMessageContent], Awaitable[Any]] | None = None,
    ):
        """Initialize the GroupChatManagerActor."""
        super().__init__(description="Group chat manager that orchestrates the conversation flow")
        self._manager = manager
        self._participant_descriptions = participant_descriptions
        self._internal_topic_type = internal_topic_type or DefaultMessageType
        self._result_callback = result_callback
        self._chat_history = ChatHistory()
        self._tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version="1.0.0"
        )

    @message_handler
    async def _handle_start_message(self, message: GroupChatStartMessage, ctx: MessageContext) -> None:
        """Handle the start message for the group chat with comprehensive tracing."""
        with self._tracer.start_as_current_span("agent_interaction") as span:
            span.set_attributes({
                # Required attributes
                "gen_ai.operation.name": "agent_interaction",
                "gen_ai.system": "semantic_kernel",
                
                # Required interaction attributes
                "gen_ai.interaction.type": "initiation",
                "gen_ai.interaction.source_agent_id": "external",
                "gen_ai.interaction.target_agent_id": "group_chat_manager",
                
                # Recommended attributes
                "gen_ai.interaction.message_type": "start",
                "gen_ai.interaction.pattern": "orchestration",
                "gen_ai.interaction.is_async": True,
                
                # Context
                "manager.actor_id": str(self.id),
                "internal_topic": self._internal_topic_type,
            })
            
            logger.debug(f"{self.id}: Received group chat start message.")
            
            # Memory operation for storing initial messages
            with self._tracer.start_as_current_span("memory_operation") as memory_span:
                memory_span.set_attributes({
                    "gen_ai.operation.name": "memory_operation",
                    "gen_ai.system": "semantic_kernel",
                    "gen_ai.memory.operation_type": "write",
                    "gen_ai.memory.agent_id": "group_chat_manager",
                    "gen_ai.memory.source_type": "initialization",
                    "gen_ai.memory.memory_type": "working",
                })
                
                if isinstance(message.body, ChatMessageContent):
                    self._chat_history.add_message(message.body)
                    memory_span.set_attribute("gen_ai.memory.size_bytes", len(str(message.body)))
                elif isinstance(message.body, list) and all(isinstance(m, ChatMessageContent) for m in message.body):
                    for m in message.body:
                        self._chat_history.add_message(m)
                    memory_span.set_attribute("gen_ai.memory.size_bytes", sum(len(str(m)) for m in message.body))
                else:
                    raise ValueError(f"Invalid message body type: {type(message.body)}. Expected {DefaultTypeAlias}.")
                
                memory_span.add_event(
                    "memory_initialized",
                    {
                        "message_count": len(self._chat_history.messages),
                        "initial_content_length": memory_span.attributes.get("gen_ai.memory.size_bytes", 0),
                    }
                )

            await self._determine_state_and_take_action(ctx.cancellation_token)
    
    @message_handler
    async def _handle_response_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        """Handle response messages with comprehensive tracing."""
        
        with self._tracer.start_as_current_span("agent_interaction") as span:
            # Fix the token usage extraction
            usage = message.body.metadata.get("usage") if message.body.metadata else None
            prompt_tokens = 0
            completion_tokens = 0
            
            if usage:
                # Handle both dict and object cases
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                else:
                    # It's a CompletionUsage object
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)
            
            span.set_attributes({
                # Required attributes
                "gen_ai.operation.name": "agent_interaction",
                "gen_ai.system": "semantic_kernel",
                
                # Required interaction attributes
                "gen_ai.interaction.type": "response",
                "gen_ai.interaction.source_agent_id": message.body.name or ("human" if message.body.role == AuthorRole.USER else "unknown"),
                "gen_ai.interaction.target_agent_id": "group_chat_manager",
                
                # Recommended attributes
                "gen_ai.interaction.message_type": "user_response" if message.body.role == AuthorRole.USER else "agent_response",
                "gen_ai.interaction.pattern": "orchestration",
                "gen_ai.interaction.is_async": True,
                "gen_ai.interaction.duration_ms": 0,  # Would be set by actual timing
                
                # Optional attributes
                "gen_ai.interaction.tokens.input": prompt_tokens,
                "gen_ai.interaction.tokens.output": completion_tokens,
                
                # Context
                "response.role": str(message.body.role),
                "response.has_content": bool(message.body.content),
            })
            
            # Memory operation for storing response
            with self._tracer.start_as_current_span("memory_operation") as memory_span:
                memory_span.set_attributes({
                    "gen_ai.operation.name": "memory_operation",
                    "gen_ai.system": "semantic_kernel",
                    "gen_ai.memory.operation_type": "write",
                    "gen_ai.memory.agent_id": "group_chat_manager",
                    "gen_ai.memory.source_type": "user_response" if message.body.role == AuthorRole.USER else "agent_response",
                    "gen_ai.memory.memory_type": "working",
                    "gen_ai.memory.size_bytes": len(str(message.body)),
                })
                
                if message.body.role != AuthorRole.USER:
                    transfer_message = ChatMessageContent(
                        role=AuthorRole.USER,
                        content=f"Transferred to {message.body.name}",
                    )
                    self._chat_history.add_message(transfer_message)
                    memory_span.add_event(
                        "transfer_message_added",
                        {
                            "transfer_to": message.body.name,
                            "message_content": transfer_message.content,
                        }
                    )
                
                self._chat_history.add_message(message.body)
                
                memory_span.add_event(
                    "gen_ai.memory.snapshot",
                    {
                        "gen_ai.snapshot.id": f"response_{len(self._chat_history.messages)}",
                        "gen_ai.snapshot.size_bytes": len(str(message.body)),
                        "gen_ai.snapshot.fields": json.dumps(["content", "role", "name"]),
                    }
                )

            await self._determine_state_and_take_action(ctx.cancellation_token)

    async def _determine_state_and_take_action(self, cancellation_token: CancellationToken) -> None:
        """Determine the state of the group chat and take action accordingly with comprehensive tracing."""
        
        with self._tracer.start_as_current_span("plan_task") as span:
            span.set_attributes({
                # Required attributes
                "gen_ai.operation.name": "plan_task",
                "gen_ai.system": "semantic_kernel",
                "gen_ai.planning.type": "orchestration",
                "gen_ai.planning.complexity": "moderate",
                
                # Conditionally required for orchestration
                "gen_ai.orchestration.pattern": "state_machine",
                "gen_ai.orchestration.participants": json.dumps(list(self._participant_descriptions.keys())),
                
                # Recommended
                "gen_ai.planning.reasoning": "Determining next action in group chat flow",
                "gen_ai.planning.iteration": self._manager.current_round,
                
                # Context
                "chat.history_length": len(self._chat_history.messages),
                "manager.current_round": self._manager.current_round,
            })
            
            # Check for user input FIRST (before any other actions)
            should_request_user_input = await self._manager.should_request_user_input(
                self._chat_history.model_copy(deep=True)
            )
            
            if should_request_user_input.result and self._manager.human_response_function:
                span.add_event(
                    "user_input_requested",
                    {
                        "reason": should_request_user_input.reason,
                        "round": self._manager.current_round,
                    }
                )
                
                logger.debug(f"Group chat manager requested user input. Reason: {should_request_user_input.reason}")
                
                # Get user input with interaction span
                with self._tracer.start_as_current_span("agent_interaction") as user_span:
                    user_span.set_attributes({
                        "gen_ai.operation.name": "agent_interaction",
                        "gen_ai.system": "semantic_kernel",
                        "gen_ai.interaction.type": "human_input",
                        "gen_ai.interaction.source_agent_id": "human",
                        "gen_ai.interaction.target_agent_id": "group_chat_manager",
                        "gen_ai.interaction.message_type": "user_input",
                        "gen_ai.interaction.pattern": "synchronous",
                        "gen_ai.interaction.is_async": False,
                    })
                    
                    user_input_message = await self._call_human_response_function()
                    
                    user_span.set_attribute("gen_ai.interaction.duration_ms", 0)  # Would be set by actual timing
                
                # Store user input
                with self._tracer.start_as_current_span("memory_operation") as memory_span:
                    memory_span.set_attributes({
                        "gen_ai.operation.name": "memory_operation",
                        "gen_ai.system": "semantic_kernel",
                        "gen_ai.memory.operation_type": "write",
                        "gen_ai.memory.agent_id": "group_chat_manager",
                        "gen_ai.memory.source_type": "user_input",
                        "gen_ai.memory.memory_type": "working",
                        "gen_ai.memory.size_bytes": len(str(user_input_message)),
                    })
                    
                    self._chat_history.add_message(user_input_message)
                
                # Publish user response
                await self.publish_message(
                    GroupChatResponseMessage(body=user_input_message),
                    TopicId(self._internal_topic_type, self.id.key),
                    cancellation_token=cancellation_token,
                )
                logger.debug("User input received and added to chat history.")
                # Return here so we don't select next agent immediately after user input
                # return

            # Determine if the group chat should terminate
            should_terminate = await self._manager.should_terminate(self._chat_history.model_copy(deep=True))
            
            if should_terminate.result:
                span.add_event(
                    "termination_decided",
                    {
                        "reason": should_terminate.reason,
                        "final_round": self._manager.current_round,
                        "total_messages": len(self._chat_history.messages),
                    }
                )
                
                logger.debug(f"Group chat manager decided to terminate the group chat. Reason: {should_terminate.reason}")
                
                if self._result_callback:
                    # Filter results with memory operation
                    result = await self._manager.filter_results(self._chat_history.model_copy(deep=True))
                    result.result.metadata["termination_reason"] = should_terminate.reason
                    result.result.metadata["filter_result_reason"] = result.reason
                    
                    # Final result callback with span
                    with self._tracer.start_as_current_span("agent_interaction") as result_span:
                        result_span.set_attributes({
                            "gen_ai.operation.name": "agent_interaction",
                            "gen_ai.system": "semantic_kernel",
                            "gen_ai.interaction.type": "result_delivery",
                            "gen_ai.interaction.source_agent_id": "group_chat_manager",
                            "gen_ai.interaction.target_agent_id": "external",
                            "gen_ai.interaction.message_type": "final_result",
                            "gen_ai.interaction.pattern": "callback",
                            "gen_ai.interaction.is_async": True,
                        })
                        
                        await self._result_callback(result.result)
                        
                        result_span.add_event(
                            "result_delivered",
                            {
                                "termination_reason": should_terminate.reason,
                                "filter_reason": result.reason,
                            }
                        )
                
                span.set_attribute("gen_ai.orchestration.outcome", "terminated")
                return

            # Select the next agent to speak if the group chat is not terminating
            next_agent = await self._manager.select_next_agent(
                self._chat_history.model_copy(deep=True),
                self._participant_descriptions,
            )
            
            span.add_event(
                "next_agent_selected",
                {
                    "selected_agent": next_agent.result,
                    "selection_reason": next_agent.reason,
                    "round": self._manager.current_round,
                }
            )
            
            logger.debug(
                f"Group chat manager selected agent: {next_agent.result} on round {self._manager.current_round}. "
                f"Reason: {next_agent.reason}"
            )

            # Publish request to next agent with interaction span
            with self._tracer.start_as_current_span("agent_interaction") as request_span:
                request_span.set_attributes({
                    "gen_ai.operation.name": "agent_interaction",
                    "gen_ai.system": "semantic_kernel",
                    "gen_ai.interaction.type": "delegation",
                    "gen_ai.interaction.source_agent_id": "group_chat_manager",
                    "gen_ai.interaction.target_agent_id": next_agent.result,
                    "gen_ai.interaction.message_type": "request",
                    "gen_ai.interaction.pattern": "orchestration",
                    "gen_ai.interaction.is_async": True,
                    "gen_ai.interaction.sequence_number": self._manager.current_round,
                })
                
                await self.publish_message(
                    GroupChatRequestMessage(agent_name=next_agent.result),
                    TopicId(self._internal_topic_type, self.id.key),
                    cancellation_token=cancellation_token,
                )
                
                request_span.add_event(
                    "request_published",
                    {
                        "target_agent": next_agent.result,
                        "topic": self._internal_topic_type,
                    }
                )
            
            span.set_attribute("gen_ai.orchestration.outcome", "continue")

    async def _call_human_response_function(self) -> ChatMessageContent:
        """Call the human response function if it is set."""
        assert self._manager.human_response_function  # nosec B101
        if inspect.iscoroutinefunction(self._manager.human_response_function):
            return await self._manager.human_response_function(self._chat_history.model_copy(deep=True))
        return self._manager.human_response_function(self._chat_history.model_copy(deep=True))  # type: ignore[return-value]

# endregion GroupChatManagerActor

# region GroupChatOrchestration


@experimental
class GroupChatOrchestration(OrchestrationBase[TIn, TOut]):
    """A group chat multi-agent pattern orchestration with comprehensive tracing."""

    def __init__(
        self,
        members: list[Agent],
        manager: GroupChatManager,
        name: str | None = None,
        description: str | None = None,
        input_transform: Callable[[TIn], Awaitable[DefaultTypeAlias] | DefaultTypeAlias] | None = None,
        output_transform: Callable[[DefaultTypeAlias], Awaitable[TOut] | TOut] | None = None,
        agent_response_callback: Callable[[DefaultTypeAlias], Awaitable[None] | None] | None = None,
        streaming_agent_response_callback: Callable[[StreamingChatMessageContent, bool], Awaitable[None] | None]
        | None = None,
    ) -> None:
        """Initialize the group chat orchestration.

        Args:
            members (list[Agent | OrchestrationBase]): A list of agents or orchestrations that are part of the
                handoff group. This first agent in the list will be the one that receives the first message.
            manager (GroupChatManager): The group chat manager that manages the flow of the group chat.
            name (str | None): The name of the orchestration.
            description (str | None): The description of the orchestration.
            input_transform (Callable | None): A function that transforms the external input message.
            output_transform (Callable | None): A function that transforms the internal output message.
            agent_response_callback (Callable | None): A function that is called when a full response is produced
                by the agents.
            streaming_agent_response_callback (Callable | None): A function that is called when a streaming response
                is produced by the agents.
        """
        self._manager = manager
        self._tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version="1.0.0"
        )

        for member in members:
            if member.description is None:
                raise ValueError("All members must have a description.")

        super().__init__(
            members=members,
            name=name,
            description=description,
            input_transform=input_transform,
            output_transform=output_transform,
            agent_response_callback=agent_response_callback,
            streaming_agent_response_callback=streaming_agent_response_callback,
        )

    @override
    async def _start(
        self,
        task: DefaultTypeAlias,
        runtime: CoreRuntime,
        internal_topic_type: str,
        cancellation_token: CancellationToken,
    ) -> None:
        """Start the group chat process with comprehensive tracing.

        This ensures that all initial messages are sent to the individual actors
        and processed before the group chat begins. It's important because if the
        manager actor processes its start message too quickly (or other actors are
        too slow), it might send a request to the next agent before the other actors
        have the necessary context.
        """
        with self._tracer.start_as_current_span("plan_task") as span:
            span.set_attributes({
                # Required attributes
                "gen_ai.operation.name": "plan_task",
                "gen_ai.system": "semantic_kernel",
                "gen_ai.planning.type": "orchestration",
                "gen_ai.planning.complexity": "complex",
                
                # Conditionally required for orchestration
                "gen_ai.orchestration.pattern": "group_chat",
                "gen_ai.orchestration.participants": json.dumps([agent.name for agent in self._members]),
                
                # Recommended
                "gen_ai.planning.reasoning": "Initializing group chat with all participants",
                "gen_ai.planning.iteration": 0,
                
                # Context
                "orchestration.name": self.name or "group_chat",
                "orchestration.member_count": len(self._members),
                "manager.type": type(self._manager).__name__,
            })
            
            # Send start messages to all agents
            async def send_start_message(agent: Agent) -> None:
                with self._tracer.start_as_current_span("agent_interaction") as agent_span:
                    agent_span.set_attributes({
                        "gen_ai.operation.name": "agent_interaction",
                        "gen_ai.system": "semantic_kernel",
                        "gen_ai.interaction.type": "broadcast",
                        "gen_ai.interaction.source_agent_id": "orchestrator",
                        "gen_ai.interaction.target_agent_id": agent.name,
                        "gen_ai.interaction.message_type": "initialization",
                        "gen_ai.interaction.pattern": "group_chat",
                        "gen_ai.interaction.is_async": True,
                    })
                    
                    target_actor_id = await runtime.get(self._get_agent_actor_type(agent, internal_topic_type))
                    await runtime.send_message(
                        GroupChatStartMessage(body=task),
                        target_actor_id,
                        cancellation_token=cancellation_token,
                    )
                    
                    agent_span.add_event(
                        "agent_initialized",
                        {
                            "agent": agent.name,
                            "actor_id": str(target_actor_id),
                        }
                    )

            await asyncio.gather(*[send_start_message(agent) for agent in self._members])
            
            span.add_event(
                "all_agents_initialized",
                {
                    "agent_count": len(self._members),
                    "agents": json.dumps([agent.name for agent in self._members]),
                }
            )

            # Send the start message to the manager actor
            with self._tracer.start_as_current_span("agent_interaction") as manager_span:
                manager_span.set_attributes({
                    "gen_ai.operation.name": "agent_interaction",
                    "gen_ai.system": "semantic_kernel",
                    "gen_ai.interaction.type": "initiation",
                    "gen_ai.interaction.source_agent_id": "orchestrator",
                    "gen_ai.interaction.target_agent_id": "group_chat_manager",
                    "gen_ai.interaction.message_type": "start_orchestration",
                    "gen_ai.interaction.pattern": "group_chat",
                    "gen_ai.interaction.is_async": True,
                })
                
                target_actor_id = await runtime.get(self._get_manager_actor_type(internal_topic_type))
                await runtime.send_message(
                    GroupChatStartMessage(body=task),
                    target_actor_id,
                    cancellation_token=cancellation_token,
                )
                
                manager_span.add_event(
                    "manager_started",
                    {
                        "manager_type": type(self._manager).__name__,
                        "max_rounds": self._manager.max_rounds or -1,
                    }
                )

    @override
    async def _prepare(
        self,
        runtime: CoreRuntime,
        internal_topic_type: str,
        result_callback: Callable[[DefaultTypeAlias], Awaitable[None]],
    ) -> None:
        """Register the actors and orchestrations with the runtime and add the required subscriptions with tracing."""
        with self._tracer.start_as_current_span("plan_task") as span:
            span.set_attributes({
                # Required attributes
                "gen_ai.operation.name": "plan_task",
                "gen_ai.system": "semantic_kernel",
                "gen_ai.planning.type": "setup",
                "gen_ai.planning.complexity": "moderate",
                
                # Recommended
                "gen_ai.planning.reasoning": "Preparing group chat runtime infrastructure",
                "gen_ai.planning.iteration": 0,
                
                # Context
                "runtime.type": type(runtime).__name__,
                "topic_type": internal_topic_type,
            })
            
            await self._register_members(runtime, internal_topic_type)
            span.add_event(
                "members_registered",
                {
                    "member_count": len(self._members),
                }
            )
            
            await self._register_manager(runtime, internal_topic_type, result_callback=result_callback)
            span.add_event(
                "manager_registered",
                {
                    "manager_type": type(self._manager).__name__,
                }
            )
            
            await self._add_subscriptions(runtime, internal_topic_type)
            span.add_event(
                "subscriptions_added",
                {
                    "subscription_count": len(self._members) + 1,  # agents + manager
                }
            )

    async def _register_members(self, runtime: CoreRuntime, internal_topic_type: str) -> None:
        """Register the agents with tracing."""
        with self._tracer.start_as_current_span("memory_operation") as span:
            span.set_attributes({
                "gen_ai.operation.name": "memory_operation",
                "gen_ai.system": "semantic_kernel",
                "gen_ai.memory.operation_type": "write",
                "gen_ai.memory.agent_id": "orchestrator",
                "gen_ai.memory.source_type": "runtime_registration",
                "gen_ai.memory.memory_type": "system",
            })
            
            await asyncio.gather(*[
                GroupChatAgentActor.register(
                    runtime,
                    self._get_agent_actor_type(agent, internal_topic_type),
                    lambda agent=agent: GroupChatAgentActor(  # type: ignore[misc]
                        agent,
                        internal_topic_type,
                        agent_response_callback=self._agent_response_callback,
                        streaming_agent_response_callback=self._streaming_agent_response_callback,
                    ),
                )
                for agent in self._members
            ])
            
            span.set_attribute("gen_ai.memory.size_bytes", len(self._members) * 100)  # Approximate
            span.add_event(
                "agents_registered",
                {
                    "agents": json.dumps([agent.name for agent in self._members]),
                }
            )

    async def _register_manager(
        self,
        runtime: CoreRuntime,
        internal_topic_type: str,
        result_callback: Callable[[DefaultTypeAlias], Awaitable[None]] | None = None,
    ) -> None:
        """Register the group chat manager with tracing."""
        with self._tracer.start_as_current_span("memory_operation") as span:
            span.set_attributes({
                "gen_ai.operation.name": "memory_operation",
                "gen_ai.system": "semantic_kernel",
                "gen_ai.memory.operation_type": "write",
                "gen_ai.memory.agent_id": "orchestrator",
                "gen_ai.memory.source_type": "runtime_registration",
                "gen_ai.memory.memory_type": "system",
            })
            
            await GroupChatManagerActor.register(
                runtime,
                self._get_manager_actor_type(internal_topic_type),
                lambda: GroupChatManagerActor(
                    self._manager,
                    internal_topic_type=internal_topic_type,
                    participant_descriptions={agent.name: agent.description for agent in self._members},  # type: ignore[misc]
                    result_callback=result_callback,
                ),
            )
            
            span.add_event(
                "manager_registered",
                {
                    "manager_type": type(self._manager).__name__,
                    "participants": json.dumps([agent.name for agent in self._members]),
                }
            )

    async def _add_subscriptions(self, runtime: CoreRuntime, internal_topic_type: str) -> None:
        """Add subscriptions with tracing."""
        with self._tracer.start_as_current_span("memory_operation") as span:
            span.set_attributes({
                "gen_ai.operation.name": "memory_operation",
                "gen_ai.system": "semantic_kernel",
                "gen_ai.memory.operation_type": "write",
                "gen_ai.memory.agent_id": "orchestrator",
                "gen_ai.memory.source_type": "runtime_configuration",
                "gen_ai.memory.memory_type": "system",
            })
            
            subscriptions: list[TypeSubscription] = []
            for agent in self._members:
                subscriptions.append(
                    TypeSubscription(internal_topic_type, self._get_agent_actor_type(agent, internal_topic_type))
                )
            subscriptions.append(TypeSubscription(internal_topic_type, self._get_manager_actor_type(internal_topic_type)))

            await asyncio.gather(*[runtime.add_subscription(sub) for sub in subscriptions])
            
            span.set_attribute("gen_ai.memory.size_bytes", len(subscriptions) * 50)  # Approximate
            span.add_event(
                "subscriptions_configured",
                {
                    "subscription_count": len(subscriptions),
                    "topic": internal_topic_type,
                }
            )

    def _get_agent_actor_type(self, agent: Agent, internal_topic_type: str) -> str:
        """Get the actor type for an agent.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{agent.name}_{internal_topic_type}"

    def _get_manager_actor_type(self, internal_topic_type: str) -> str:
        """Get the actor type for the group chat manager.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{GroupChatManagerActor.__name__}_{internal_topic_type}"



# endregion GroupChatOrchestration
