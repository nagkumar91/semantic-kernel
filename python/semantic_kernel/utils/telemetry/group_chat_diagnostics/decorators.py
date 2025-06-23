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
                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        span.set_attribute("execution.duration_seconds", duration)
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
