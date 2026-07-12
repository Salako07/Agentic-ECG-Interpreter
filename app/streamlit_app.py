"""
Streamlit UI for the Agentic ECG Interpreter.

A single-screen demo that makes the whole pipeline legible:
  - pick an ECG record
  - see the 12-lead waveform (Lead II highlighted)
  - see the classifier probabilities vs their tuned thresholds
  - watch the agent's decision trace (its retrieval tool calls)
  - read the grounded, cited diagnostic report

Runs the agent in-process (no separate API needed). To use the FastAPI
service instead, point requests at http://localhost:8000/interpret.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

# Streamlit runs this file from inside app/, so the project root (which
# contains the `src` package) is not on the import path by default. Add it.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.agent.graph import interpret
from src.agent.tools import ECGClassifier

st.set_page_config(page_title="Agentic ECG Interpreter", layout="wide")


@st.cache_resource
def get_classifier() -> ECGClassifier:
    return ECGClassifier()


SUPERCLASS_FULL = {
    "NORM": "Normal", "MI": "Myocardial Infarction", "STTC": "ST/T Change",
    "CD": "Conduction Disturbance", "HYP": "Hypertrophy",
}


def plot_waveform(record) -> plt.Figure:
    """Plot all 12 leads stacked, Lead II highlighted."""
    fig, ax = plt.subplots(figsize=(10, 6))
    signal = record.signal
    t = np.arange(signal.shape[0]) / record.sampling_rate

    offset_step = 2.5
    for i, name in enumerate(record.lead_names):
        offset = -i * offset_step
        is_lead_ii = (name == "II")
        ax.plot(t, signal[:, i] + offset,
                linewidth=1.1 if is_lead_ii else 0.6,
                color="#c0392b" if is_lead_ii else "#2c3e50",
                alpha=1.0 if is_lead_ii else 0.7)
        ax.text(-0.35, offset, name, va="center", ha="right",
                fontsize=9, fontweight="bold" if is_lead_ii else "normal")

    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("12-lead ECG (Lead II highlighted)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_probabilities(probs: dict, thresholds: dict) -> plt.Figure:
    """Horizontal bars of per-class probability with threshold markers."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = list(probs.keys())
    values = [probs[n] for n in names]
    colors = ["#27ae60" if probs[n] >= thresholds[n] else "#bdc3c7"
              for n in names]

    y = np.arange(len(names))
    ax.barh(y, values, color=colors)
    for i, n in enumerate(names):
        ax.plot([thresholds[n], thresholds[n]], [i - 0.4, i + 0.4],
                color="#e74c3c", linewidth=2)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability (red line = decision threshold)")
    ax.set_title("Classifier output")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------- UI ----------------

st.title("Agentic ECG Interpreter")
st.caption("Deterministic classification + retrieval-grounded LLM reporting. "
           "Research prototype — not for clinical use.")

clf = get_classifier()
max_id = int(clf.loader.metadata.index.max())

with st.sidebar:
    st.header("Select record")
    ecg_id = st.number_input("ECG record id", min_value=1, max_value=max_id,
                             value=7, step=1)
    run = st.button("Interpret", type="primary", use_container_width=True)
    st.divider()
    st.caption("Try 7 (HYP false positive) or 3 (NORM) to compare behaviour.")

if run:
    record = clf.loader.load_record(int(ecg_id))

    col_wave, col_prob = st.columns([3, 2])
    with col_wave:
        st.pyplot(plot_waveform(record))
    with col_prob:
        st.markdown(f"**Patient:** age {record.age:.0f}, "
                    f"sex {'F' if record.sex else 'M'}")
        st.markdown(f"**Ground-truth labels:** "
                    f"{', '.join(record.labels) or '(none)'}")

    with st.spinner("Running agent (classify -> retrieve -> report)..."):
        result = interpret(int(ecg_id))

    clf_out = result["classification"]

    with col_prob:
        st.pyplot(plot_probabilities(
            {k: v["probability"] for k, v in clf_out["predictions"].items()},
            {k: v["threshold"] for k, v in clf_out["predictions"].items()},
        ))

    # Decision trace
    st.subheader("Agent decision trace")
    trace_lines = []
    for msg in result["trace"]:
        for tc in getattr(msg, "tool_calls", None) or []:
            trace_lines.append(f"TOOL CALL  {tc['name']}  ->  "
                               f"\"{tc['args'].get('query', '')}\"")
    if trace_lines:
        st.code("\n".join(trace_lines), language="text")
        st.caption(f"The agent issued {len(trace_lines)} retrieval call(s) "
                   f"before writing its report.")
    else:
        st.caption("The agent used only the baseline seed retrieval "
                   "(high-confidence case, no follow-up needed).")

    # Findings vs truth
    c1, c2 = st.columns(2)
    c1.metric("Classifier positive findings",
              ", ".join(clf_out["positive_findings"]) or "none")
    c2.metric("Ground-truth labels",
              ", ".join(clf_out["reference_labels"]) or "none")

    # Report
    st.subheader("Diagnostic report")
    st.markdown(result["report"])
else:
    st.info("Pick a record and press Interpret.")