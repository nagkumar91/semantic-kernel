# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from opentelemetry import trace
from samples.demos.travel_planning_system.agents import get_agents
from samples.demos.travel_planning_system.observability import enable_observability, tracer
from semantic_kernel.agents import HandoffOrchestration
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.handoffs import OrchestrationHandoffs
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# Uncomment and configure if you want to use Langfuse
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://us.cloud.langfuse.com",
)

def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents."""
    if message.content:
        with tracer.start_as_current_span("agent_response_callback") as span:
            span.set_attributes({
                "gen_ai.operation.name": "agent_response_callback",
                "gen_ai.agent.name": message.name,
                "gen_ai.message.content": message.content,
            })
            span.add_event(
                name="agent_message",
                attributes={
                    "message.role": message.role.value,
                    "message.content": message.content,
                    "message.name": getattr(message, "name", ""),
                }
            )
        print(f"# {message.name}\n{message.content}")

def human_response_function() -> ChatMessageContent:
    """Observer function to print the messages from the agents."""
    tracer_local = trace.get_tracer(__name__)
    with tracer_local.start_as_current_span("human_in_the_loop") as span:
        user_input = input("User: ")
        span.add_event(
            name="user_message",
            attributes={
                "message.role": "user",
                "message.content": user_input,
            }
        )
        return ChatMessageContent(role=AuthorRole.USER, content=user_input)

def get_agents_and_handoffs():
    """Create agents and define handoffs for the travel planning system."""
    BASE_TRANSFER_DESCRIPTION = "Do not call this function in parallel with other functions."

    agents: dict[str, ChatCompletionAgent] = get_agents()

    conversation_manager = agents["conversation_manager"]
    planner = agents["planner"]
    router = agents["router"]
    destination_expert = agents["destination_expert"]
    flight_agent = agents["flight_agent"]
    hotel_agent = agents["hotel_agent"]

    handoffs = (
        OrchestrationHandoffs()
        .add_many(
            source_agent=conversation_manager,
            target_agents={
                planner.name: f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for trip planning.",
                router.name: (
                    f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for routing tasks to specialized agents."
                ),
                destination_expert.name: (
                    f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for destination expertise."
                ),
                flight_agent.name: f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for flight-related tasks.",
                hotel_agent.name: f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for hotel-related tasks.",
            },
        )
        .add(
            source_agent=planner,
            target_agent=router,
            description=f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for routing tasks to specialized agents.",
        )
        .add_many(
            source_agent=router,
            target_agents={
                destination_expert.name: (
                    f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for destination expertise."
                ),
                flight_agent.name: f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for flight-related tasks.",
                hotel_agent.name: f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for hotel-related tasks.",
            },
        )
        .add(
            source_agent=destination_expert,
            target_agent=conversation_manager,
            description=f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for non-destination related questions.",
        )
        .add(
            source_agent=flight_agent,
            target_agent=conversation_manager,
            description=f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for non-flight related questions.",
        )
        .add(
            source_agent=hotel_agent,
            target_agent=conversation_manager,
            description=f"{BASE_TRANSFER_DESCRIPTION} Transfer to this agent for non-hotel related questions.",
        )
    )

    return [
        conversation_manager,
        planner,
        router,
        destination_expert,
        flight_agent,
        hotel_agent,
    ], handoffs

@enable_observability
async def main():
    """Main function to run the agents."""
    agents, handoffs = get_agents_and_handoffs()
    handoff_orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        agent_response_callback=agent_response_callback,
        human_response_function=human_response_function,
    )

    runtime = InProcessRuntime()
    runtime.start()
    task_description = "Plan a trip to bali for 5 days including flights, hotels, and activities for a vegetarian family of 4 members."
    # Root execute_task span for the orchestration
    with tracer.start_as_current_span("execute_task") as execute_task_span:
        execute_task_span.set_attributes({
            "gen_ai.operation.name": "execute_task",
            "gen_ai.task.description": task_description,
            "gen_ai.orchestration.context": "handoff_orchestration",

        })

        # Optionally, add a plan_task span for the planner agent
        with tracer.start_as_current_span("plan_task") as plan_task_span:
            plan_task_span.set_attributes({
                "gen_ai.operation.name": "plan_task",
                "gen_ai.agent.name": "planner"
            })
            # (If you want to explicitly invoke the planner here, do so.)

        # Optionally, add an agent_to_agent_interaction span for the first handoff
        with tracer.start_as_current_span("agent_to_agent_interaction") as agent_interaction_span:
            agent_interaction_span.set_attributes({
                "gen_ai.operation.name": "agent_to_agent_interaction",
                "gen_ai.source_agent": "conversation_manager",
                "gen_ai.target_agent": "planner"
            })

            orchestration_result = await handoff_orchestration.invoke(
                task=task_description,
                runtime=runtime,
            )

        value = await orchestration_result.get()
        print(value)

    await runtime.stop_when_idle()

if __name__ == "__main__":
    # Run the main function with asyncio
    asyncio.run(main())