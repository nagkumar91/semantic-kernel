# Copyright (c) Microsoft. All rights reserved.
"""
Round-robin group chat travel planning system with comprehensive OpenTelemetry tracing.
Demonstrates all span types from spec: task execution, planning, agent interactions, and memory operations.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional
os.environ['OTEL_SERVICE_NAME'] = "round_robin_travel_planning_system"
from langfuse import Langfuse
from opentelemetry import trace, baggage, context
from opentelemetry.trace import Status, StatusCode, Link
from pydantic import Field

from agents import get_agents
from observability import enable_observability
from semantic_kernel.agents import (
    RoundRobinGroupChatManager,
    GroupChatOrchestration,
    ChatCompletionAgent,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
    StreamingChatMessageContent,
)

# Initialize Langfuse
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://us.cloud.langfuse.com",
)

# Global flag for streaming messages
is_new_message = True

def create_traceparent() -> str:
    """Create a W3C traceparent header for context propagation."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    return f"00-{format(ctx.trace_id, '032x')}-{format(ctx.span_id, '016x')}-01"

def streaming_agent_response_callback(message: StreamingChatMessageContent, is_final: bool) -> None:
    """Enhanced callback with comprehensive tracing for streaming responses."""
    global is_new_message
    
    # Only create span for final messages to reduce duplicate spans (from Shipra's changes)
    if is_final and (message.content or message.items):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("streaming_message_final") as stream_span:
            stream_span.set_attributes({
                # Required OpenTelemetry attributes
                "span.kind": "INTERNAL",
                
                # Required gen_ai attributes
                "gen_ai.operation.name": "streaming_message_final",
                "gen_ai.system": "semantic_kernel",
                
                # Agent information
                "gen_ai.agent.name": message.name or "unknown",
                "gen_ai.agent.id": message.name or "unknown",
                
                # Message attributes
                "message.role": str(message.role),
                "message.content_length": len(str(message.content)) if message.content else 0,
                "message.has_function_calls": bool(any(isinstance(item, FunctionCallContent) for item in message.items)),
                "message.has_function_results": bool(any(isinstance(item, FunctionResultContent) for item in message.items)),
            })
            
            # Add events for message content
            if message.content:
                stream_span.add_event(
                    "gen_ai.assistant.message",
                    {
                        "gen_ai.system": "semantic_kernel",
                        "content": str(message.content)[:1000],
                        "role": "assistant",
                        "agent_name": message.name,
                    }
                )
            
            # Track function calls and results
            for item in message.items:
                if isinstance(item, FunctionCallContent):
                    stream_span.add_event(
                        "gen_ai.tool.message",
                        {
                            "gen_ai.system": "semantic_kernel",
                            "tool_name": item.name,
                            "tool_arguments": str(item.arguments)[:500],
                            "role": "tool",
                        }
                    )
                elif isinstance(item, FunctionResultContent):
                    stream_span.add_event(
                        "gen_ai.tool.message", 
                        {
                            "gen_ai.system": "semantic_kernel",
                            "tool_name": item.name,
                            "tool_result": str(item.result)[:500],
                            "role": "tool",
                        }
                    )
    
    # Console output
    if is_new_message:
        print(f"\n[{message.name}]: ", end="")
        is_new_message = False
    
    print(message.content, end="", flush=True)
    
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"\n  🔧 Calling '{item.name}' with arguments '{item.arguments}'", end="", flush=True)
        if isinstance(item, FunctionResultContent):
            print(f"\n  ✅ Result from '{item.name}': '{item.result}'", end="", flush=True)
    
    if is_final:
        print()
        is_new_message = True


def human_response_function(chat_history: ChatHistory) -> ChatMessageContent:
    """Observer function to handle user input with telemetry."""
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("human_in_the_loop") as span:
        span.set_attributes({
            "gen_ai.operation.name": "human_in_the_loop",
            "gen_ai.system": "semantic_kernel",
            "chat.history_length": len(chat_history.messages),
            "interaction.type": "manual",
        })
        
        user_input = input("User: ")
        
        span.add_event(
            "gen_ai.user.message",
            {
                "gen_ai.system": "semantic_kernel",
                "content": user_input,
                "role": "user",
            }
        )
        
        return ChatMessageContent(role=AuthorRole.USER, content=user_input)


def automated_response_function(chat_history: ChatHistory) -> ChatMessageContent:
    """Automated response function that doesn't require human input."""
    
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("automated_input") as span:
        span.set_attributes({
            "gen_ai.operation.name": "automated_input",
            "gen_ai.system": "semantic_kernel",
            "chat.history_length": len(chat_history.messages),
            "interaction.type": "automated",
        })
        
        # Generate an automated response based on the last message
        last_message = chat_history.messages[-1] if chat_history.messages else None
        
        # More comprehensive auto-responses
        response = "Please continue with the planning."
        
        if last_message and last_message.content:
            content_lower = last_message.content.lower()
            
            # Activity planning questions
            if "activity planning" in content_lower or "proceed with the activity" in content_lower:
                response = "Yes, please proceed with the activity planning. Include popular tourist attractions, cultural experiences, and dining recommendations."
            elif "proceed with that" in content_lower or "make any adjustments" in content_lower:
                response = "Yes, please proceed with the suggested plan. The budget allocation looks good."
            elif "do you prefer" in content_lower:
                response = "Please choose the cheapest flight options (FL123 at $200 each way) and Hotel Sunshine at $150/night for budget-friendly accommodation."
            elif "would you like" in content_lower:
                response = "Yes, please proceed with your recommendation."
            elif "shall i proceed" in content_lower or "should i proceed" in content_lower:
                response = "Yes, please proceed with the detailed planning."
            elif "concurrently search" in content_lower or "will search" in content_lower:
                response = "Yes, please go ahead and search for the flights and hotels."
            elif "any preferences" in content_lower or "do you have" in content_lower:
                response = "No specific preferences, please suggest the best options within the $4000 budget."
            elif "is this acceptable" in content_lower or "does this work" in content_lower:
                response = "Yes, that works well. Please continue."
            elif "?" in content_lower:  # Any question
                response = "Yes, please proceed with your recommendation. Choose the most cost-effective options."
        
        print(f"🤖 Auto-response: {response}")
        
        span.add_event(
            "gen_ai.user.message",
            {
                "gen_ai.system": "semantic_kernel",
                "content": response,
                "role": "user",
                "is_automated": True,
            }
        )
        
        return ChatMessageContent(role=AuthorRole.USER, content=response)


@enable_observability
async def main():
    """Main function demonstrating round-robin orchestration with comprehensive tracing."""
    tracer = trace.get_tracer(__name__)
    
    # Set up W3C baggage for cross-service propagation
    tokens = baggage.set_baggage("gen_ai.workflow.id", "travel_planning_workflow_001")
    tokens = baggage.set_baggage("gen_ai.user.session_id", "demo_session_001", context=tokens)
    tokens = baggage.set_baggage("gen_ai.conversation.id", "conv_roundrobin_001", context=tokens)
    context.attach(tokens)
    
    # Root span for the entire session
    with tracer.start_as_current_span("round_robin_travel_planning_session") as session_span:
        # Set resource attributes
        session_span.set_attributes({
            # Required OpenTelemetry attributes
            "service.name": "travel-planner-agent-system",
            "service.version": "1.0.0",
            "service.instance.id": "instance-001",
            "deployment.environment": "demo",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.45.0",
            
            # Session attributes
            "user.id": "demo_user_roundrobin",
            "conversation.id": "conv_roundrobin_001",
            "session.type": "round_robin_multi_agent_demo",
            "orchestration.type": "round_robin",
            "orchestration.pattern": "group_chat",
            
            # Langfuse integration
            "langfuse.trace.type": "multi_agent_orchestration",
            "langfuse.version": "1.0",
            "langfuse.session_type": "round_robin_group_chat",
        })
        
        # Get agents
        agents: dict[str, ChatCompletionAgent] = get_agents()
        
        # Create round-robin orchestration
        print("🔄 Creating Round-Robin Group Chat Orchestration")
        print(f"   Agents: {', '.join([a.name for a in [agents['planner'], agents['flight_agent'], agents['hotel_agent']]])}")
        print(f"   Max rounds: 12 (4 rounds per agent)")
        print("   Pattern: Each agent speaks in turn\n")
        
        group_chat_orchestration = GroupChatOrchestration(
            members=[
                agents["planner"],
                agents["flight_agent"], 
                agents["hotel_agent"],
            ],
            manager=RoundRobinGroupChatManager(
                max_rounds=12,  # 4 rounds per agent
                human_response_function=automated_response_function
            ),
            streaming_agent_response_callback=streaming_agent_response_callback,
        )
        
        # Task description
        task_description = (
            "Plan a trip to Tokyo for 5 days including flights, hotels, and "
            "activities for a couple. They live in San Francisco, CA, USA. "
            "Their vacation starts on August 15th 2025. They have a budget of $4000. "
            "The planner should outline the overall strategy, the flight agent should find flights, "
            "and the hotel agent should find accommodations. Work together in a round-robin fashion."
        )
        
        # Add task input event
        session_span.add_event(
            "gen_ai.user.message",
            {
                "gen_ai.system": "semantic_kernel",
                "content": task_description,
                "role": "user",
                "task.destination": "Tokyo",
                "task.origin": "San Francisco",
                "task.duration_days": 5,
                "task.budget": 4000,
                "task.travelers": 2,
            }
        )
        
        # Execute task with comprehensive task execution span
        with tracer.start_as_current_span("execute_task") as task_span:
            # Set traceparent for W3C context propagation
            traceparent = create_traceparent()
            
            task_span.set_attributes({
                # Required OpenTelemetry attributes
                "span.kind": "INTERNAL",
                
                # Required gen_ai attributes
                "gen_ai.operation.name": "execute_task",
                "gen_ai.system": "semantic_kernel",
                
                # Required task attributes
                "gen_ai.task.id": "tokyo_trip_planning_001",
                "gen_ai.task.description": task_description,
                
                # Recommended task attributes
                "gen_ai.task.expected_output": "Complete travel plan with flights and hotels",
                "gen_ai.task.status": "in_progress",
                "gen_ai.task.root_task_id": "tokyo_trip_planning_001",
                
                # Optional task attributes
                "gen_ai.task.constraints": json.dumps(["budget:4000", "duration:5_days", "travelers:2"]),
                "gen_ai.task.assigned_agents": json.dumps(["planner", "flight_agent", "hotel_agent"]),
                "gen_ai.task.completion_criteria": "All agents contribute their expertise",
                "gen_ai.task.timeout_ms": 300000,  # 5 minutes
                
                # W3C context propagation
                "traceparent": traceparent,
                
                # Additional context
                "orchestration.max_rounds": 12,
                "orchestration.agents_count": 3,
                "orchestration.expected_turns_per_agent": 4,
            })
            
            print(f"📋 Task: {task_description}\n")
            print("=" * 80)
            
            # Create runtime
            runtime = InProcessRuntime()
            runtime.start()
            
            try:
                # Planning span for initial task planning
                with tracer.start_as_current_span("plan_task") as planning_span:
                    planning_span.set_attributes({
                        # Required attributes
                        "gen_ai.operation.name": "plan_task",
                        "gen_ai.system": "semantic_kernel",
                        "gen_ai.planning.type": "planning",
                        "gen_ai.planning.complexity": "complex",
                        
                        # Conditionally required for planning
                        "gen_ai.planning.stage": "initial",
                        "gen_ai.planning.strategy": "task_decomposition",
                        
                        # Recommended
                        "gen_ai.planning.reasoning": "Decomposing travel planning into agent-specific subtasks",
                        "gen_ai.planning.iteration": 1,
                        
                        # Optional
                        "gen_ai.planning.sub_tasks": json.dumps([
                            "overall_strategy",
                            "flight_search", 
                            "hotel_search",
                            "coordination"
                        ]),
                    })
                    
                    planning_span.add_event(
                        "planning_decision",
                        {
                            "decision": "Use round-robin to ensure equal participation",
                            "agent_order": "planner -> flight_agent -> hotel_agent",
                            "expected_iterations": 4,
                        }
                    )
                
                # Execute with timeout
                try:
                    orchestration_result = await asyncio.wait_for(
                        group_chat_orchestration.invoke(
                            task=task_description,
                            runtime=runtime,
                        ),
                        timeout=120.0  # 120 second timeout
                    )
                except asyncio.TimeoutError:
                    print("\n❌ ERROR: Orchestration timed out after 120 seconds")
                    task_span.set_status(Status(StatusCode.ERROR, "Timeout"))
                    raise

                # Memory operation for storing final result
                with tracer.start_as_current_span("memory_operation") as result_span:
                    result_span.set_attributes({
                        "gen_ai.operation.name": "memory_operation",
                        "gen_ai.system": "semantic_kernel",
                        "gen_ai.memory.operation_type": "write",
                        "gen_ai.memory.agent_id": "orchestrator",
                        "gen_ai.memory.source_type": "task_completion",
                        "gen_ai.memory.memory_type": "long_term",
                    })
                    
                    # Get final result
                    final_result = await orchestration_result.get()
                    
                    result_span.set_attribute("gen_ai.memory.size_bytes", len(str(final_result)))
                    
                    # Add completion event
                    result_span.add_event(
                        "gen_ai.memory.snapshot",
                        {
                            "gen_ai.snapshot.id": "final_result_001",
                            "gen_ai.snapshot.size_bytes": len(str(final_result)),
                            "gen_ai.snapshot.fields": json.dumps(["travel_plan", "flights", "hotels"]),
                        }
                    )
                
                # Update task status
                task_span.set_attribute("gen_ai.task.status", "completed")
                task_span.set_status(Status(StatusCode.OK))
                
                # Final output
                print("\n" + "=" * 80)
                print("✅ FINAL TRAVEL PLAN:")
                print("=" * 80)
                print(final_result)
                
                # Add final result event
                session_span.add_event(
                    "gen_ai.assistant.message",
                    {
                        "gen_ai.system": "semantic_kernel",
                        "content": str(final_result)[:2000],
                        "role": "assistant",
                        "task.status": "completed",
                        "langfuse.output": str(final_result)[:2000],
                    }
                )
                
            except Exception as e:
                task_span.set_status(Status(StatusCode.ERROR, str(e)))
                task_span.record_exception(e)
                raise
                
            finally:
                # Stop runtime
                await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())