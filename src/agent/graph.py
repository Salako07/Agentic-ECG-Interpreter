"""
LangGraph agent: hybrid orchestration (D-015).

Deterministic core (always runs, fixed order, reproducible):
    classify node -> record loaded, features extracted, classifier run,
    plus one automatic seed retrieval for the leading predicted superclass.

LLM-driven follow-up (agentic):
    agent node -> the LLM examines predictions + seed context and either calls
    lookup_guidelines for additional retrieval, or writes the final grounded
    report. Every tool call is captured in the message trace, giving an
    auditable record of the agent's reasoning and evidence gathering.

Run:  python -m src.agent.graph <ecg_id>
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.tools import ECGClassifier, SUPERCLASSES
from src.rag.retriever import GuidelineRetriever

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

AGENT_SYSTEM_PROMPT = """You are a cardiology decision-support assistant that \
writes structured, evidence-grounded ECG interpretation reports.

You are given the output of a deterministic ECG classifier (per-class \
probabilities for five diagnostic superclasses) and baseline reference \
passages. Your job:

1. Reason about the classifier output. Note which findings are positive, which \
are borderline, and what they imply together.
2. If you need more clinical context than the baseline passages provide, call \
the lookup_guidelines tool with a specific query. You may call it more than \
once. Retrieve before you assert.
3. Write a structured report with these sections: Summary, Findings, Clinical \
Context, Limitations.
4. Ground every clinical claim in the retrieved reference passages and cite the \
source in brackets, e.g. [Myocardial Infarction — Lead territories]. Do not \
state clinical facts that are not supported by retrieved passages.
5. In Limitations, state plainly that this is an automated research prototype \
operating on a feature-based classifier, not a validated diagnostic device, \
and must not be used for clinical decisions.

Be precise and concise. Do not invent measurements the classifier did not \
provide."""


# --- lazy singletons (avoid loading models/store at import time) ---
_classifier = None
_retriever = None


def _get_classifier() -> ECGClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ECGClassifier()
    return _classifier


def _get_retriever() -> GuidelineRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GuidelineRetriever()
    return _retriever


@tool
def lookup_guidelines(query: str) -> str:
    """Retrieve cardiology reference passages relevant to a clinical query.

    Use this to gather evidence before making clinical statements — for example
    to look up the significance of a wide QRS, the lead territories of an
    infarct, or voltage criteria for hypertrophy.
    """
    passages = _get_retriever().retrieve(query, k=3)
    if not passages:
        return "No relevant reference passages found."
    return "\n\n".join(f"[{p.citation()}]\n{p.text}" for p in passages)


# --- graph state ---
class AgentState(TypedDict):
    ecg_id: int
    classification: dict
    messages: Annotated[list, add_messages]


def classify_node(state: AgentState) -> dict:
    """Deterministic core + seed retrieval. Always runs first."""
    result = _get_classifier().classify(state["ecg_id"])

    # Fixed baseline retrieval for the leading finding.
    seed = _get_retriever().retrieve_for_superclass(result["leading_finding"], k=2)
    seed_text = "\n\n".join(f"[{p.citation()}]\n{p.text}" for p in seed)

    pred_lines = "\n".join(
        f"  {n}: p={result['predictions'][n]['probability']:.3f} "
        f"({'POSITIVE' if result['predictions'][n]['positive'] else 'negative'}, "
        f"threshold {result['predictions'][n]['threshold']:.2f})"
        for n in SUPERCLASSES
    )
    findings = result["positive_findings"] or ["none above threshold"]

    human = HumanMessage(content=(
        f"ECG {result['ecg_id']} — patient age {result['age']:.0f}, "
        f"sex {result['sex']}.\n\n"
        f"Classifier output (probability per diagnostic superclass):\n"
        f"{pred_lines}\n\n"
        f"Positive findings: {', '.join(findings)}\n"
        f"Leading finding: {result['leading_finding']}\n\n"
        f"Baseline reference for the leading finding:\n{seed_text}\n\n"
        f"Write the diagnostic report. Retrieve any additional context you "
        f"need with lookup_guidelines before writing."
    ))

    return {
        "classification": result,
        "messages": [SystemMessage(content=AGENT_SYSTEM_PROMPT), human],
    }


def make_agent_node(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0.2).bind_tools([lookup_guidelines])

    def agent_node(state: AgentState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    return agent_node


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "finish"


def build_graph(model_name: str = DEFAULT_MODEL):
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("agent", make_agent_node(model_name))
    graph.add_node("tools", ToolNode([lookup_guidelines]))

    graph.set_entry_point("classify")
    graph.add_edge("classify", "agent")
    graph.add_conditional_edges("agent", should_continue,
                                {"tools": "tools", "finish": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def interpret(ecg_id: int, model_name: str = DEFAULT_MODEL) -> dict:
    """Run the full agent on one ECG. Returns classification, report, trace."""
    graph = build_graph(model_name)
    final = graph.invoke({"ecg_id": ecg_id, "messages": []})

    return {
        "ecg_id": ecg_id,
        "classification": final["classification"],
        "report": final["messages"][-1].content,
        "trace": final["messages"],
    }


def print_trace(result: dict) -> None:
    """Human-readable audit trace of the agent's tool calls and reasoning."""
    print("=" * 70)
    print(f"DECISION TRACE — ECG {result['ecg_id']}")
    print("=" * 70)
    for msg in result["trace"]:
        role = msg.__class__.__name__.replace("Message", "")
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"  [{role}] TOOL CALL: {tc['name']}({tc['args']})")
        elif role == "Tool":
            preview = msg.content[:80].replace("\n", " ")
            print(f"  [Tool] returned: {preview}...")
        elif role == "Human":
            print(f"  [Human] (classifier output + seed context)")
        elif role == "System":
            print(f"  [System] (instructions)")
        elif role == "AI" and msg.content:
            print(f"  [AI] wrote final report ({len(msg.content)} chars)")
    print("=" * 70)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.agent.graph <ecg_id>")
        sys.exit(1)

    ecg_id = int(sys.argv[1])
    result = interpret(ecg_id)

    print_trace(result)
    print("\nCLASSIFIER POSITIVE FINDINGS:",
          result["classification"]["positive_findings"] or "none")
    print("REFERENCE LABELS (ground truth):",
          result["classification"]["reference_labels"] or "none")
    print("\n" + "=" * 70)
    print("GENERATED REPORT")
    print("=" * 70)
    print(result["report"])


if __name__ == "__main__":
    main()