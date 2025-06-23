# Copyright (c) Microsoft. All rights reserved.

import functools
import time
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from opentelemetry.trace import Status, StatusCode, get_tracer
from semantic_kernel.utils.feature_stage_decorator import experimental

P = ParamSpec("P")
T = TypeVar("T")

# Creates a tracer from the global tracer provider
tracer = get_tracer(__name__)


@experimental
def trace_group_chat_agent_message(operation_name: str):
    """Decorator to trace group chat agent message handling."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            agent_actor = args[0]  # GroupChatAgentActor instance
            
            span_name = f"group_chat.agent.{operation_name}"
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("agent.id", str(agent_actor.id))
                span.set_attribute("agent.name", agent_actor._agent.name)
                span.set_attribute("operation", operation_name)
                
                try:
                    if operation_name == "invoke":
                        # Add interaction span for receiving delegation from manager
                        message = args[1] if len(args) > 1 else None  # GroupChatRequestMessage
                        
                        with tracer.start_as_current_span("agent_receives_delegation") as interaction_span:
                            interaction_span.set_attributes({
                                "gen_ai.operation.name": "agent_to_agent_interaction",
                                "gen_ai.interaction.type": "receive_delegation",
                                "gen_ai.interaction.source_agent": "Manager",
                                "gen_ai.interaction.target_agent": agent_actor._agent.name,
                                "agent.role": getattr(agent_actor._agent, 'role', 'unknown'),
                                "requested_agent": getattr(message, 'agent_name', 'unknown') if message else 'unknown'
                            })
                        
                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        span.set_attribute("execution.duration_seconds", duration)
                        
                        # Add interaction span for agent responding to manager
                        with tracer.start_as_current_span("agent_responds_to_manager") as interaction_span:
                            interaction_span.set_attributes({
                                "gen_ai.operation.name": "agent_to_agent_interaction",
                                "gen_ai.interaction.type": "send_response",
                                "gen_ai.interaction.source_agent": agent_actor._agent.name,
                                "gen_ai.interaction.target_agent": "Manager",
                                "agent.role": getattr(agent_actor._agent, 'role', 'unknown'),
                                "execution.duration_seconds": duration
                            })
                        
                        return result
                    elif operation_name == "handle_response_message":
                        # Add interaction span for agent receiving messages from other agents
                        message = args[1] if len(args) > 1 else None  # GroupChatResponseMessage
                        
                        with tracer.start_as_current_span("agent_receives_peer_message") as interaction_span:
                            interaction_span.set_attributes({
                                "gen_ai.operation.name": "agent_to_agent_interaction",
                                "gen_ai.interaction.type": "receive_peer_message",
                                "gen_ai.interaction.source_agent": getattr(message.body, 'name', 'unknown') if message else 'unknown',
                                "gen_ai.interaction.target_agent": agent_actor._agent.name,
                                "message.role": str(message.body.role) if message else 'unknown'
                            })
                        
                        result = await func(*args, **kwargs)
                        return result
                    else:
                        result = await func(*args, **kwargs)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        wrapper.__group_chat_diagnostics__ = True  # type: ignore
        return wrapper
    return decorator


@experimental
def trace_group_chat_manager_message(operation_name: str):
    """Decorator to trace group chat manager message handling."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            manager_actor = args[0]  # GroupChatManagerActor instance
            
            span_name = f"group_chat.manager.{operation_name}"
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("operation", operation_name)
                manager_type = type(manager_actor._manager).__name__
                span.set_attribute("manager_type", manager_type)
                
                if hasattr(manager_actor._manager, 'current_round'):
                    current_round = manager_actor._manager.current_round
                    span.set_attribute("manager.current_round", current_round)
                
                if hasattr(manager_actor, '_chat_history'):
                    msg_count = len(manager_actor._chat_history.messages)
                    span.set_attribute("chat_history.message_count", msg_count)
                
                try:
                    # ENHANCED: Add agent-to-agent interaction detection for key operations
                    if operation_name == "determine_state_and_take_action":
                        # Capture state before execution to detect agent selection
                        chat_history_before = manager_actor._chat_history.model_copy(deep=True)
                        
                        result = await func(*args, **kwargs)
                        
                        # Add agent-to-agent interaction span for delegation
                        # We need to detect if an agent was selected by checking if a GroupChatRequestMessage was sent
                        # Since we can't easily intercept the message sending, we'll add a span that represents the delegation decision
                        with tracer.start_as_current_span("manager_delegation_decision") as interaction_span:
                            interaction_span.set_attributes({
                                "gen_ai.operation.name": "agent_to_agent_interaction",
                                "gen_ai.interaction.type": "delegation_decision",
                                "gen_ai.interaction.source_agent": "Manager",
                                "manager.current_round": current_round,
                                "chat_history.message_count_before": len(chat_history_before.messages),
                                "chat_history.message_count_after": len(manager_actor._chat_history.messages)
                            })
                        
                        return result
                    elif operation_name == "handle_response_message":
                        # Add agent-to-agent interaction span for receiving agent responses
                        message = args[1] if len(args) > 1 else None  # GroupChatResponseMessage
                        
                        # Create interaction span for receiving agent response
                        with tracer.start_as_current_span("manager_receives_agent_response") as interaction_span:
                            interaction_span.set_attributes({
                                "gen_ai.operation.name": "agent_to_agent_interaction", 
                                "gen_ai.interaction.type": "receive_response",
                                "gen_ai.interaction.source_agent": getattr(message.body, 'name', 'unknown') if message else 'unknown',
                                "gen_ai.interaction.target_agent": "Manager",
                                "message.role": str(message.body.role) if message else 'unknown'
                            })
                        
                        result = await func(*args, **kwargs)
                        return result
                    else:
                        result = await func(*args, **kwargs)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        wrapper.__group_chat_diagnostics__ = True  # type: ignore
        return wrapper
    return decorator

@experimental
def trace_group_chat_orchestration(operation_name: str):
    """Decorator to trace group chat orchestration operations."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            orchestration = args[0]  # GroupChatOrchestration instance
            
            span_name = f"group_chat.orchestration.{operation_name}"
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("operation", operation_name)
                span.set_attribute("participant_count", len(orchestration._members))
                manager_type = type(orchestration._manager).__name__
                span.set_attribute("manager_type", manager_type)
                
                # Add participant names
                participant_names = [agent.name for agent in orchestration._members]
                span.set_attribute("participants", ",".join(participant_names))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        wrapper.__group_chat_diagnostics__ = True  # type: ignore
        return wrapper
    return decorator
