from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command, Durability, interrupt


def test_parent_command_from_nested_subgraph() -> None:
    class ParentState(TypedDict):
        jump_from_idx: int

    class ChildState(TypedDict):
        jump: bool

    child_builder: StateGraph[ChildState] = StateGraph(ChildState)

    def child_node(state: ChildState) -> Command | ChildState:
        if state["jump"]:
            return Command(graph=Command.PARENT, goto="parent_second")
        return state

    child_builder.add_node("node", child_node)
    child_builder.add_edge(START, "node")

    child_0 = child_builder.compile()
    child_1 = child_builder.compile()

    parent_builder: StateGraph[ParentState] = StateGraph(ParentState)

    def parent_first(state: ParentState) -> ParentState:
        child_0.invoke({"jump": state["jump_from_idx"] == 1})
        if state["jump_from_idx"] == 1:
            raise AssertionError("Shouldn't be here")

        child_1.invoke({"jump": state["jump_from_idx"] == 2})
        if state["jump_from_idx"] == 2:
            raise AssertionError("Shouldn't be here")

        return state

    def parent_second(state: ParentState) -> ParentState:
        return state

    parent_builder.add_node("parent_first", parent_first)
    parent_builder.add_node("parent_second", parent_second)
    parent_builder.add_edge(START, "parent_first")
    parent_builder.add_edge("parent_second", END)

    graph = parent_builder.compile()

    assert graph.invoke({"jump_from_idx": 1}) == {"jump_from_idx": 1}
    assert graph.invoke({"jump_from_idx": 2}) == {"jump_from_idx": 2}


@pytest.mark.anyio
@pytest.mark.parametrize("is_async", [False, True])
async def test_stateful_parent_command_repeated_calls_after_interrupt(
    durability: Durability, is_async: bool
) -> None:
    seen: list[list[str]] = []

    def model(state: MessagesState):
        seen.append([message.content for message in state["messages"]])
        return {"messages": [("ai", "call" if len(seen) == 1 else "answer")]}

    def approve(state: MessagesState):
        interrupt("approve?")
        return {"messages": [("human", "tool result")]}

    agent = (
        StateGraph(MessagesState)
        .add_node("model", model)
        .add_node("approve", approve)
        .add_edge(START, "model")
        .add_conditional_edges(
            "model",
            lambda state: "approve" if state["messages"][-1].content == "call" else END,
            ["approve", END],
        )
        .add_edge("approve", "model")
        .compile()
    )

    class ChildState(TypedDict):
        x: int

    def run_agent(state: ChildState, config: RunnableConfig):
        agent.invoke({"messages": [("human", f"prompt {state['x']}")]}, config)
        return Command(graph=Command.PARENT, goto="tick", update={"x": state["x"] + 1})

    async def arun_agent(state: ChildState, config: RunnableConfig):
        await agent.ainvoke({"messages": [("human", f"prompt {state['x']}")]}, config)
        return Command(graph=Command.PARENT, goto="tick", update={"x": state["x"] + 1})

    child = (
        StateGraph(ChildState)
        .add_node("agent", arun_agent if is_async else run_agent)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile(checkpointer=True)
    )

    class State(ChildState):
        rounds: int

    graph = (
        StateGraph(State)
        .add_node("child", child, destinations=("tick",))
        .add_node("tick", lambda state: {"rounds": state["rounds"] + 1})
        .add_edge(START, "child")
        .add_conditional_edges(
            "tick",
            lambda state: "child" if state["rounds"] < 2 else END,
            ["child", END],
        )
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "parent-command"}}
    next_input = {"x": 0, "rounds": 0}
    for _ in range(12):
        if is_async:
            result = await graph.ainvoke(
                next_input, config, interrupt_after="*", durability=durability
            )
        else:
            result = graph.invoke(
                next_input, config, interrupt_after="*", durability=durability
            )
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        next_input = (
            Command(resume="yes")
            if any(task.interrupts for task in snapshot.tasks)
            else None
        )

    assert result == {"x": 2, "rounds": 2}
    assert seen == [["prompt 0"], ["prompt 0", "call", "tool result"], ["prompt 1"]]
    assert snapshot.next == ()
