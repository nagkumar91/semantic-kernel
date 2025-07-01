# Copyright (c) Microsoft. All rights reserved.

import functools
import json
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import ParamSpec, TypeVar, cast

from opentelemetry.trace import get_tracer

from semantic_kernel.utils.feature_stage_decorator import experimental
from semantic_kernel.utils.telemetry.agent_diagnostics import gen_ai_attributes

P = ParamSpec("P")
T = TypeVar("T")

# Creates a tracer from the global tracer provider
tracer = get_tracer(__name__)


@experimental
def trace_group_chat_agent_message(operation_name: str):
    """Decorator to trace group chat agent message operations with rich input/output."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            agent_actor = args[0] if args else None
            
            with tracer.start_as_current_span(f"group_chat_agent_message.{operation_name}") as span:
                # Set basic operation attributes
                span.set_attributes({
                    gen_ai_attributes.OPERATION: f"group_chat_agent_message.{operation_name}",
                    "group_chat.operation": operation_name,
                    "group_chat.component": "agent_message_handler"
                })
                
                # Capture agent information
                if agent_actor and hasattr(agent_actor, '_agent'):
                    agent = agent_actor._agent
                    span.set_attributes({
                        gen_ai_attributes.AGENT_ID: agent.id,
                        gen_ai_attributes.AGENT_NAME: agent.name,
                    })
                    if agent.description:
                        span.set_attribute(gen_ai_attributes.AGENT_DESCRIPTION, agent.description)
                
                # Capture input context based on operation
                try:
                    if operation_name == "handle_start_message":
                        # This is the start of a conversation
                        span.add_event("group_chat.conversation_start", {
                            "event.type": "conversation_initialization",
                            "operation.phase": "start"
                        })
                    elif operation_name == "handle_response_message":
                        # This is handling a response from another agent
                        message_context = args[1] if len(args) > 1 else kwargs.get('message_context')
                        if message_context and hasattr(message_context, 'message'):
                            msg = message_context.message
                            if hasattr(msg, 'content'):
                                input_content = str(msg.content)[:1000]
                                span.add_event("gen_ai.content.prompt", {
                                    "gen_ai.prompt": input_content,
                                    "input.message_type": "agent_response",
                                    "input.source": "other_agent",
                                    "group_chat.message_flow": "agent_to_agent"
                                })
                            
                            # Capture message metadata
                            if hasattr(msg, 'role'):
                                span.set_attribute("input.message_role", str(msg.role))
                            if hasattr(message_context, 'sender'):
                                span.set_attribute("input.sender", str(message_context.sender))
                    elif operation_name == "invoke":
                        # This is the main invoke operation
                        span.add_event("group_chat.agent_invoke", {
                            "event.type": "agent_invocation",
                            "operation.phase": "invoke"
                        })
                        
                except Exception as e:
                    span.set_attribute("input.capture_error", str(e))
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Capture output
                try:
                    if result:
                        if hasattr(result, 'content'):
                            output_content = str(result.content)[:1000]
                            span.add_event("gen_ai.content.completion", {
                                "gen_ai.completion": output_content,
                                "output.message_type": "agent_decision",
                                "group_chat.output_type": operation_name
                            })
                        elif isinstance(result, bool):
                            span.add_event("group_chat.decision", {
                                "decision.result": result,
                                "decision.operation": operation_name
                            })
                        
                        span.set_attribute("operation.success", True)
                    else:
                        span.set_attribute("operation.no_result", True)
                        
                except Exception as e:
                    span.set_attribute("output.capture_error", str(e))
                
                return result
        
        return wrapper
    return decorator


@experimental  
def trace_group_chat_manager_message(operation_name: str):
    """Enhanced decorator to trace group chat manager message operations with comprehensive input/output for Langfuse."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            manager = args[0] if args else None
            
            with tracer.start_as_current_span(f"group_chat_manager.{operation_name}") as span:
                # Set basic operation attributes
                span.set_attributes({
                    gen_ai_attributes.OPERATION: f"group_chat_manager.{operation_name}",
                    "group_chat.operation": operation_name,
                    "group_chat.component": "manager",
                    "langfuse.operation": operation_name,
                    "langfuse.component": "group_chat_manager"
                })
                
                # Capture manager information
                if manager and hasattr(manager, 'name'):
                    span.set_attribute("manager.name", manager.name)
                if manager and hasattr(manager, 'id'):
                    span.set_attribute("manager.id", str(manager.id))
                
                # ENHANCED: Comprehensive input context capture
                try:
                    if operation_name == "determine_state_and_take_action":
                        # This is the key planning/coordination operation
                        span.add_event("group_chat.planning", {
                            "event.type": "coordination_planning",
                            "operation.phase": "planning",
                            "planning.type": "state_determination",
                            "langfuse.planning": True
                        })
                        
                        # Capture complete conversation history for planning context
                        if hasattr(manager, '_chat_history') and manager._chat_history:
                            messages = manager._chat_history.messages
                            if messages:
                                # Capture recent conversation context (last 5 messages)
                                recent_messages = messages[-5:] if len(messages) > 5 else messages
                                conversation_context = "\n".join([
                                    f"{msg.role}: {str(msg.content)[:200]}" 
                                    for msg in recent_messages
                                    if hasattr(msg, 'content') and hasattr(msg, 'role')
                                ])[:1500]
                                
                                span.add_event("gen_ai.content.prompt", {
                                    "gen_ai.prompt": conversation_context,
                                    "input.message_type": "conversation_history",
                                    "input.source": "group_chat_history",
                                    "input.message_count": len(messages),
                                    "input.recent_message_count": len(recent_messages),
                                    "langfuse.input": conversation_context,
                                    "langfuse.conversation": conversation_context
                                })
                                
                                # Capture the very last message for immediate context
                                last_message = messages[-1]
                                if hasattr(last_message, 'content'):
                                    span.add_event("input.last_message", {
                                        "last.message": str(last_message.content)[:500],
                                        "last.role": str(last_message.role),
                                        "last.agent": getattr(last_message, 'name', 'unknown'),
                                        "context.type": "immediate_context"
                                    })
                                
                                # Capture participant information for agent selection context
                                if hasattr(manager, '_participant_descriptions') and manager._participant_descriptions:
                                    span.add_event("planning.participants", {
                                        "participants": json.dumps(manager._participant_descriptions),
                                        "participant.count": len(manager._participant_descriptions),
                                        "participant.names": list(manager._participant_descriptions.keys())
                                    })
                        
                        # Try to capture conversation history if available in args
                        if len(args) > 1:
                            conversation_data = args[1] if hasattr(args[1], '__iter__') else None
                            if conversation_data:
                                span.set_attribute("planning.has_conversation_context", True)
                                
                    elif operation_name in ["handle_start_message", "handle_response_message"]:
                        # Capture message context for start and response operations
                        message_context = args[1] if len(args) > 1 else kwargs.get('message_context')
                        if message_context and hasattr(message_context, 'message'):
                            msg = message_context.message
                            if hasattr(msg, 'content'):
                                input_content = str(msg.content)[:1000]
                                span.add_event("gen_ai.content.prompt", {
                                    "gen_ai.prompt": input_content,
                                    "input.message_type": "coordination_input",
                                    "input.source": "conversation",
                                    "group_chat.coordination_context": operation_name,
                                    "langfuse.input": input_content
                                })
                            
                            # Enhanced message metadata
                            if hasattr(msg, 'role'):
                                span.set_attribute("input.message_role", str(msg.role))
                            if hasattr(message_context, 'sender'):
                                span.set_attribute("input.sender", str(message_context.sender))
                            if hasattr(msg, 'name'):
                                span.set_attribute("input.agent_name", str(msg.name))
                
                except Exception as e:
                    span.set_attribute("input.capture_error", str(e))
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # ENHANCED: Comprehensive output capture - especially important for planning operations
                try:
                    if operation_name == "determine_state_and_take_action":
                        # This is the planning result - very important for observability
                        span.add_event("group_chat.coordination_decision", {
                            "event.type": "coordination_completed",
                            "coordination.operation": operation_name,
                            "coordination.success": True,
                            "planning.decision_made": True,
                            "langfuse.coordination": "Agent coordination decision completed"
                        })
                        
                        # Try to capture any planning decision details
                        if result:
                            if hasattr(result, '__dict__'):
                                # Try to capture planning decision details
                                result_dict = vars(result)
                                span.add_event("group_chat.planning_result", {
                                    "planning.decision_made": True,
                                    "planning.result_type": type(result).__name__,
                                    "planning.has_details": bool(result_dict),
                                    "langfuse.planning_result": True
                                })
                                
                                # Capture specific planning details if available
                                if 'next_agent' in result_dict:
                                    span.set_attribute("planning.next_agent", str(result_dict['next_agent']))
                                if 'action' in result_dict:
                                    span.set_attribute("planning.action", str(result_dict['action']))
                                    
                            elif isinstance(result, str):
                                # String result - might be agent selection or action
                                span.add_event("gen_ai.content.completion", {
                                    "gen_ai.completion": result[:500],
                                    "output.message_type": "planning_decision",
                                    "planning.decision": result[:200],
                                    "langfuse.output": result[:500],
                                    "langfuse.planning_decision": result[:200]
                                })
                        
                        # Always capture the planning completion
                        span.add_event("gen_ai.planning.decision", {
                            "planning.type": "overall_coordination",
                            "planning.component": "group_chat_manager",
                            "planning.operation": operation_name,
                            "planning.completed": True,
                            "langfuse.plan": f"Group chat coordination decision for {operation_name}"
                        })
                    
                    elif operation_name in ["handle_start_message", "handle_response_message"]:
                        # Capture message handling results
                        span.add_event("group_chat.message_handled", {
                            "event.type": "message_processing_complete",
                            "message.operation": operation_name,
                            "message.success": True,
                            "langfuse.message_handled": True
                        })
                        
                        if result and hasattr(result, 'content'):
                            output_content = str(result.content)[:1000]
                            span.add_event("gen_ai.content.completion", {
                                "gen_ai.completion": output_content,
                                "output.message_type": "manager_response",
                                "group_chat.manager_output": operation_name,
                                "langfuse.output": output_content
                            })
                    
                    # Always set success indicator
                    span.set_attribute("operation.success", True)
                    span.set_attribute("langfuse.success", True)
                        
                except Exception as e:
                    span.set_attribute("output.capture_error", str(e))
                    span.set_attribute("langfuse.error", str(e))
                
                return result
        
        return wrapper
    return decorator


@experimental
def trace_group_chat_orchestration(operation_name: str):
    """Decorator to trace group chat orchestration operations with rich input/output."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            orchestrator = args[0] if args else None
            
            with tracer.start_as_current_span(f"group_chat_orchestration.{operation_name}") as span:
                # Set basic operation attributes
                span.set_attributes({
                    gen_ai_attributes.OPERATION: f"group_chat_orchestration.{operation_name}",
                    "group_chat.operation": operation_name,
                    "group_chat.component": "orchestration"
                })
                
                # This is likely the root span for the whole group chat operation
                if operation_name == "start":
                    span.set_attribute("group_chat.is_root_operation", True)
                    span.add_event("group_chat.session_start", {
                        "event.type": "orchestration_start",
                        "session.phase": "initialization"
                    })
                
                # Capture orchestration input
                try:
                    # Look for input message or conversation starter
                    input_message = None
                    if len(args) > 1:
                        input_message = args[1]
                    elif 'message' in kwargs:
                        input_message = kwargs['message']
                    elif 'input' in kwargs:
                        input_message = kwargs['input']
                    
                    if input_message and hasattr(input_message, 'content'):
                        input_content = str(input_message.content)[:1000]
                        span.add_event("gen_ai.content.prompt", {
                            "gen_ai.prompt": input_content,
                            "input.message_type": "orchestration_input",
                            "input.source": "user_request",
                            "group_chat.session_input": True
                        })
                        
                        span.set_attribute("orchestration.has_input", True)
                        span.set_attribute("orchestration.input_length", len(input_message.content))
                    
                    # Capture orchestration configuration
                    if orchestrator:
                        if hasattr(orchestrator, '_members'):
                            agent_count = len(orchestrator._members) if orchestrator._members else 0
                            span.set_attribute("orchestration.agent_count", agent_count)
                            
                            # Capture agent names for better observability
                            if agent_count > 0:
                                agent_names = [agent.name for agent in orchestrator._members if hasattr(agent, 'name')]
                                span.add_event("orchestration.agents_configured", {
                                    "agents.count": agent_count,
                                    "agents.names": str(agent_names)[:500]
                                })
                
                except Exception as e:
                    span.set_attribute("input.capture_error", str(e))
                
                # Execute the function and capture result
                result = await func(*args, **kwargs)
                
                # Capture orchestration completion
                try:
                    span.set_attribute("orchestration.completed", True)
                    
                    if operation_name == "start":
                        span.add_event("group_chat.session_started", {
                            "event.type": "orchestration_started",
                            "session.success": True
                        })
                        
                except Exception as e:
                    span.set_attribute("output.capture_error", str(e))
                
                return result
        
        return wrapper
    return decorator

@experimental
def trace_agent_invocation(invoke_func: Callable[P, AsyncIterable[T]]) -> Callable[P, AsyncIterable[T]]:
    """Enhanced decorator to trace agent invocation with multi-agent context."""
    OPERATION_NAME = "invoke_agent"

    @functools.wraps(invoke_func)
    async def wrapper_decorator(*args: P.args, **kwargs: P.kwargs) -> AsyncIterable[T]:
        from semantic_kernel.agents.agent import Agent

        agent = cast(Agent, args[0])
        
        with tracer.start_as_current_span(f"{OPERATION_NAME} {agent.name}") as span:
            # Basic agent information
            span.set_attributes({
                gen_ai_attributes.OPERATION: OPERATION_NAME,
                gen_ai_attributes.AGENT_ID: agent.id,
                gen_ai_attributes.AGENT_NAME: agent.name,
            })

            if agent.description:
                span.set_attribute(gen_ai_attributes.AGENT_DESCRIPTION, agent.description)

            # ENHANCED: Capture multi-agent context and input/output
            try:
                # Try to get additional context from the agent actor
                agent_actor = args[0] if hasattr(args[0], '_agent') else None
                
                # Capture who is invoking this agent (calling context)
                if hasattr(agent_actor, 'id'):
                    span.set_attribute("invocation.source_actor", str(agent_actor.id))
                
                # Enhanced: Track calling context for multi-agent scenarios
                import inspect
                frame = inspect.currentframe()
                if frame and frame.f_back and frame.f_back.f_back:
                    caller_name = frame.f_back.f_back.f_code.co_name
                    span.set_attribute("invocation.caller_function", caller_name)
                    
                    # Detect if this is part of multi-agent coordination
                    if any(keyword in caller_name.lower() for keyword in ['select', 'should', 'filter', 'manage', 'plan']):
                        span.set_attribute("invocation.context", "multi_agent_coordination")
                        span.set_attribute("invocation.coordination_type", caller_name)
                        
                        # Add specific coordination context
                        if 'select' in caller_name.lower():
                            span.set_attribute("coordination.operation", "agent_selection")
                        elif 'should' in caller_name.lower():
                            span.set_attribute("coordination.operation", "decision_making")
                        elif 'filter' in caller_name.lower():
                            span.set_attribute("coordination.operation", "result_filtering")
                        elif 'plan' in caller_name.lower():
                            span.set_attribute("coordination.operation", "planning")

                # Capture input messages if available
                messages_arg = args[1] if len(args) > 1 else None
                if messages_arg:
                    if hasattr(messages_arg, '__iter__') and not isinstance(messages_arg, str):
                        # Collection of messages
                        message_list = list(messages_arg)
                        if message_list:
                            span.set_attribute("invocation.input_message_count", len(message_list))
                            
                            # Capture the latest message content
                            latest_msg = message_list[-1]
                            if hasattr(latest_msg, 'content'):
                                input_content = str(latest_msg.content)[:1000]
                                span.add_event("gen_ai.content.prompt", {
                                    "gen_ai.prompt": input_content,
                                    "input.message_type": "conversation_context",
                                    "input.total_messages": len(message_list),
                                    "input.source": "multi_agent_conversation"
                                })
                                
                                # Capture conversation context (last few messages)
                                if len(message_list) > 1:
                                    context_msgs = message_list[-3:] if len(message_list) > 3 else message_list
                                    context_content = "\n".join([
                                        f"{getattr(msg, 'role', 'unknown')}: {str(getattr(msg, 'content', ''))[:200]}" 
                                        for msg in context_msgs
                                    ])[:1000]
                                    
                                    span.add_event("invocation.conversation_context", {
                                        "context.content": context_content,
                                        "context.message_count": len(context_msgs),
                                        "context.type": "recent_conversation"
                                    })
                    
                    elif hasattr(messages_arg, 'content'):
                        # Single message
                        input_content = str(messages_arg.content)[:1000]
                        span.add_event("gen_ai.content.prompt", {
                            "gen_ai.prompt": input_content,
                            "input.message_type": "single_message",
                            "input.source": "direct_invocation"
                        })

            except Exception as e:
                span.set_attribute("invocation.context_capture_error", str(e))

            # Execute the async generator and capture output
            response_count = 0
            final_message = None
            total_content_length = 0

            async for message in invoke_func(*args, **kwargs):
                response_count += 1
                final_message = message
                
                # Track streaming content
                if hasattr(message, 'content') and message.content:
                    content_length = len(str(message.content))
                    total_content_length += content_length
                    
                    # Add streaming events for intermediate responses
                    if response_count <= 5:  # Limit events to avoid spam
                        span.add_event("invocation.streaming_response", {
                            "response.chunk": response_count,
                            "response.content": str(message.content)[:200],
                            "response.length": content_length
                        })

                yield message

            # ENHANCED: Capture final output with rich context
            if final_message:
                if hasattr(final_message, 'content'):
                    final_content = str(final_message.content)[:1000]
                    span.add_event("gen_ai.content.completion", {
                        "gen_ai.completion": final_content,
                        "gen_ai.assistant.message": final_content,
                        "output.message_type": "agent_response",
                        "output.chunk_count": response_count,
                        "output.agent": agent.name,
                        "output.total_length": total_content_length
                    })
                
                span.set_attributes({
                    "invocation.response_count": response_count,
                    "invocation.total_content_length": total_content_length,
                    "invocation.final_message_length": len(final_message.content) if final_message and hasattr(final_message, 'content') else 0,
                    "invocation.success": True
                })
            else:
                span.set_attribute("invocation.no_response", True)

    # Mark the wrapper decorator as an agent diagnostics decorator
    wrapper_decorator.__agent_diagnostics__ = True  # type: ignore

    return wrapper_decorator


@experimental
def trace_agent_get_response(get_response_func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Enhanced decorator to trace agent get_response with multi-agent context."""
    OPERATION_NAME = "invoke_agent"

    @functools.wraps(get_response_func)
    async def wrapper_decorator(*args: P.args, **kwargs: P.kwargs) -> T:
        from semantic_kernel.agents.agent import Agent

        agent = cast(Agent, args[0])

        with tracer.start_as_current_span(f"{OPERATION_NAME} {agent.name}") as span:
            # Basic agent information
            span.set_attributes({
                gen_ai_attributes.OPERATION: OPERATION_NAME,
                gen_ai_attributes.AGENT_ID: agent.id,
                gen_ai_attributes.AGENT_NAME: agent.name,
            })

            if agent.description:
                span.set_attribute(gen_ai_attributes.AGENT_DESCRIPTION, agent.description)

            # ENHANCED: Capture input message context
            try:
                # Capture input messages for get_response
                messages = args[1] if len(args) > 1 else kwargs.get('messages', None)
                if messages:
                    if isinstance(messages, list) and messages:
                        # Multiple messages - capture the conversation context
                        recent_messages = messages[-3:] if len(messages) > 3 else messages
                        context_content = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages if hasattr(msg, 'content') and hasattr(msg, 'role')])[:1000]
                        
                        span.add_event("gen_ai.content.prompt", {
                            "gen_ai.prompt": context_content,
                            "input.message_type": "conversation_context",
                            "input.total_messages": len(messages),
                            "input.context_messages": len(recent_messages),
                            "input.source": "conversation_history"
                        })
                        
                        span.set_attributes({
                            "invocation.input_message_count": len(messages),
                            "invocation.conversation_length": len(messages)
                        })
                        
                        # Capture the last user message specifically
                        last_user_msg = None
                        for msg in reversed(messages):
                            if hasattr(msg, 'role') and str(msg.role).lower() in ['user', 'human']:
                                last_user_msg = msg
                                break
                        
                        if last_user_msg and hasattr(last_user_msg, 'content'):
                            span.add_event("input.last_user_message", {
                                "user.message": str(last_user_msg.content)[:500],
                                "message.role": str(last_user_msg.role)
                            })
                            
                    elif hasattr(messages, 'content'):
                        # Single message
                        input_content = str(messages.content)[:1000]
                        span.add_event("gen_ai.content.prompt", {
                            "gen_ai.prompt": input_content,
                            "gen_ai.user.message": input_content,
                            "input.message_type": "single_message",
                            "input.source": "direct_call"
                        })

                # Capture execution settings if provided
                arguments = kwargs.get('arguments', None)
                if arguments and hasattr(arguments, 'execution_settings'):
                    span.set_attribute("invocation.has_execution_settings", True)
                    # Try to capture some execution settings details
                    try:
                        settings = arguments.execution_settings
                        if hasattr(settings, 'model_id'):
                            span.set_attribute("invocation.model_id", settings.model_id)
                        if hasattr(settings, 'temperature'):
                            span.set_attribute("invocation.temperature", settings.temperature)
                        if hasattr(settings, 'max_tokens'):
                            span.set_attribute("invocation.max_tokens", settings.max_tokens)
                    except:
                        pass

                # Enhanced: Track calling context for multi-agent scenarios
                import inspect
                frame = inspect.currentframe()
                if frame and frame.f_back and frame.f_back.f_back:
                    caller_name = frame.f_back.f_back.f_code.co_name
                    span.set_attribute("invocation.caller_function", caller_name)
                    
                    # Detect if this is part of multi-agent coordination
                    if any(keyword in caller_name.lower() for keyword in ['select', 'should', 'filter', 'manage', 'plan']):
                        span.set_attribute("invocation.context", "multi_agent_coordination")
                        span.set_attribute("invocation.coordination_type", caller_name)
                        
                        # Add specific coordination context
                        if 'select' in caller_name.lower():
                            span.set_attribute("coordination.operation", "agent_selection")
                        elif 'should' in caller_name.lower():
                            span.set_attribute("coordination.operation", "decision_making")
                        elif 'filter' in caller_name.lower():
                            span.set_attribute("coordination.operation", "result_filtering")
                        elif 'plan' in caller_name.lower():
                            span.set_attribute("coordination.operation", "planning")

            except Exception as e:
                span.set_attribute("invocation.context_capture_error", str(e))

            # Execute the function and capture output
            result = await get_response_func(*args, **kwargs)
            
            # ENHANCED: Capture output content
            try:
                if hasattr(result, 'message') and result.message:
                    if hasattr(result.message, 'content') and result.message.content:
                        output_content = str(result.message.content)[:1000]
                        span.add_event("gen_ai.content.completion", {
                            "gen_ai.completion": output_content,
                            "gen_ai.assistant.message": output_content,
                            "output.message_type": "agent_response",
                            "output.agent": agent.name
                        })
                        
                        span.set_attributes({
                            "invocation.output_length": len(result.message.content),
                            "invocation.success": True
                        })
                        
                        # If this looks like a structured response (JSON), try to parse it
                        try:
                            if result.message.content.strip().startswith('{') and result.message.content.strip().endswith('}'):
                                parsed = json.loads(result.message.content)
                                span.add_event("output.structured_response", {
                                    "response.type": "structured_json",
                                    "response.keys": list(parsed.keys()) if isinstance(parsed, dict) else "non_dict"
                                })
                        except:
                            pass
                            
                elif hasattr(result, 'content') and result.content:
                    # Direct content response
                    output_content = str(result.content)[:1000]
                    span.add_event("gen_ai.content.completion", {
                        "gen_ai.completion": output_content,
                        "output.message_type": "direct_response",
                        "output.agent": agent.name
                    })
                    
                    span.set_attribute("invocation.output_length", len(result.content))

            except Exception as e:
                span.set_attribute("invocation.output_capture_error", str(e))

            return result

    # Mark the wrapper decorator as an agent diagnostics decorator
    wrapper_decorator.__agent_diagnostics__ = True  # type: ignore

    return wrapper_decorator