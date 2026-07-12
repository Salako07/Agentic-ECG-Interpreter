"""
FastAPI service for the Agentic ECG Interpreter.

Exposes the hybrid agent as a REST API:
    GET  /health              liveness check
    GET  /records?n=          sample of available ecg_ids with labels
    POST /interpret           run the full agent on one ecg_id

The interpret response includes the classifier output, the generated report,
and a structured decision trace (the agent's tool calls) so the auditability
of the system is visible through the API, not just the CLI.

Run:  uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent.graph import interpret, DEFAULT_MODEL
from src.agent.tools import ECGClassifier

app = FastAPI(
    title="Agentic ECG Interpreter",
    description="Hybrid LangGraph agent: deterministic ECG classification + "
                "retrieval-grounded diagnostic reporting.",
    version="1.0.0",
)

# Allow the Streamlit UI (local) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton for lightweight record listing (no LLM needed).
_classifier = None


def _get_classifier() -> ECGClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ECGClassifier()
    return _classifier


class InterpretRequest(BaseModel):
    ecg_id: int
    model: str = DEFAULT_MODEL


class ToolCall(BaseModel):
    tool: str
    query: str


class InterpretResponse(BaseModel):
    ecg_id: int
    age: float
    sex: str
    positive_findings: list[str]
    reference_labels: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    tool_calls: list[ToolCall]
    report: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/records")
def records(n: int = 20) -> dict:
    """Return a sample of ecg_ids with their ground-truth labels."""
    clf = _get_classifier()
    meta = clf.loader.metadata
    sample = meta.head(n)
    return {
        "records": [
            {"ecg_id": int(idx), "labels": labels or ["(none)"],
             "age": float(row["age"]), "sex": "F" if row["sex"] else "M"}
            for idx, (labels, row) in
            zip(sample.index, zip(sample["labels"], sample.to_dict("records")))
        ]
    }


@app.post("/interpret", response_model=InterpretResponse)
def interpret_ecg(req: InterpretRequest) -> InterpretResponse:
    try:
        result = interpret(req.ecg_id, model_name=req.model)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"ecg_id {req.ecg_id} not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    clf = result["classification"]

    # Extract tool calls from the message trace for auditability.
    tool_calls = []
    for msg in result["trace"]:
        for tc in getattr(msg, "tool_calls", None) or []:
            tool_calls.append(ToolCall(
                tool=tc["name"],
                query=str(tc["args"].get("query", "")),
            ))

    return InterpretResponse(
        ecg_id=clf["ecg_id"],
        age=clf["age"],
        sex=clf["sex"],
        positive_findings=clf["positive_findings"],
        reference_labels=clf["reference_labels"],
        probabilities={k: v["probability"] for k, v in clf["predictions"].items()},
        thresholds={k: v["threshold"] for k, v in clf["predictions"].items()},
        tool_calls=tool_calls,
        report=result["report"],
    )