"""The LangGraph support workflow, with any DecisionModel plugged in as the router node.

    START -> router --(route / requires_human / low confidence)--> <team>_agent  -> END
                                                               \\-> human_handoff -> END

The team agents are stubs: they only write a reply. That keeps the comparison about the
routing step itself, which is the only thing that changes between experiments.
"""

import operator
import time
from typing import Annotated, TypedDict

import pandas as pd

from src.dataset import ROUTES
from src.interface import DecisionModel


class TicketState(TypedDict, total=False):
    ticket: dict                               # the incoming request (query + metadata)
    decision: dict                             # filled in by the router
    reply: str                                 # filled in by whichever agent handles it
    handled_by: str                            # name of the node that answered
    trace: Annotated[list[str], operator.add]  # every node appends a line


def expected_destination(row) -> str:
    """Where a ticket should end up according to its true labels."""
    return "human_handoff" if bool(row["requires_human"]) else row["route_target"] + "_agent"


def build_router_graph(router_model: DecisionModel, confidence_floor: float = 0.5):
    """Compile the graph with `router_model` deciding where each ticket goes."""
    from langgraph.graph import END, START, StateGraph

    def router_node(state: TicketState) -> dict:
        out = router_model.predict_one(state["ticket"])
        note = (f"router[{router_model.name}]: route={out.route} priority={out.priority} "
                f"human={out.requires_human} conf={out.confidence:.2f} ({out.latency_ms:.1f} ms)")
        return {"decision": out.to_dict(), "trace": [note]}

    def pick_next_node(state: TicketState) -> str:
        decision = state["decision"]
        if decision["requires_human"] or decision["confidence"] < confidence_floor:
            return "human_handoff"
        return decision["route"] + "_agent"

    def make_agent(team: str):
        def agent_node(state: TicketState) -> dict:
            priority = state["decision"]["priority"]
            reply = f"[{team} agent | {priority} priority] handling: {state['ticket']['query']}"
            return {"reply": reply, "handled_by": team + "_agent", "trace": [f"{team}_agent: replied"]}
        return agent_node

    def human_handoff(state: TicketState) -> dict:
        d = state["decision"]
        reply = f"[HUMAN QUEUE | {d['priority']}] for {d['route']} team: {state['ticket']['query']}"
        return {"reply": reply, "handled_by": "human_handoff", "trace": ["human_handoff: queued for a person"]}

    builder = StateGraph(TicketState)
    builder.add_node("router", router_node)
    builder.add_node("human_handoff", human_handoff)
    for route in ROUTES:
        builder.add_node(route + "_agent", make_agent(route))

    builder.add_edge(START, "router")
    destinations = {route + "_agent": route + "_agent" for route in ROUTES}
    destinations["human_handoff"] = "human_handoff"
    builder.add_conditional_edges("router", pick_next_node, destinations)
    for node in destinations:
        builder.add_edge(node, END)
    return builder.compile()


def run_tickets(graph, df: pd.DataFrame) -> pd.DataFrame:
    """Send every row of df through the graph once. One row of results per ticket."""
    records = []
    for row in df.to_dict(orient="records"):
        ticket = {key: row[key] for key in ["query", "account_tier", "channel", "urgency_score", "past_failures"]}
        start = time.perf_counter()
        final = graph.invoke({"ticket": ticket})
        total_ms = (time.perf_counter() - start) * 1000.0
        decision = final["decision"]
        records.append({
            "expected": expected_destination(row),
            "handled_by": final["handled_by"],
            "route": decision["route"],
            "priority": decision["priority"],
            "requires_human": decision["requires_human"],
            "exact": (decision["route"] == row["route_target"]
                      and decision["priority"] == row["priority_level"]
                      and bool(decision["requires_human"]) == bool(row["requires_human"])),
            "router_ms": decision["latency_ms"],
            "graph_ms": total_ms,
        })
    results = pd.DataFrame(records)
    results["correct_destination"] = results["expected"] == results["handled_by"]
    return results
