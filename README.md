# Agentic-ECG-Interpreter

An LLM agent that classifies ECG signals, looks up clinical guidelines, and produces structured diagnostic summaries with citations. Directly maps to your Agentic AI Architect identity.


You should see the superclass distribution, the fold split sizes, and one record loaded with shape (1000, 12) — 10 seconds at 100 Hz across 12 leads.
Three design decisions worth understanding, because they'll come up in interviews
Why filter scp_statements.csv on diagnostic == 1. PTB-XL's SCP codes mix three kinds of statement: diagnostic (what's wrong), form (waveform morphology), and rhythm (rate and regularity). Only diagnostic codes map to the five superclasses. Keeping the others in would silently corrupt your labels. The loader drops them at line one of _load_scp_map.
Why use strat_fold instead of train_test_split. PTB-XL ships ten stratified folds, and folds 9 and 10 were specifically human-validated by cardiologists — the rest include machine-generated annotations. Every published benchmark trains on 1–8, validates on 9, tests on 10. If you invent your own random split, your numbers are not comparable to any paper, and a reviewer who knows the dataset will notice immediately. This is a small thing that signals you read the source paper.
Why the label matrix is multi-hot. A single ECG routinely carries both MI and STTC, or CD and HYP. This is a multi-label problem, not multi-class. It changes your loss function, your metrics (macro AUROC, not accuracy), and your XGBoost setup — we'll handle that in Stage 3.
One thing you'll notice in the output
Roughly nine percent of records come back with an empty label list. Those are ECGs whose SCP codes are all form or rhythm statements with no diagnostic component. Don't silently drop them yet — decide deliberately in Stage 3 whether they're excluded or treated as a sixth "no diagnostic finding" state, and write that decision into your README. Reviewers care more about the reasoning than the choice.
Run it and paste me the output. If the distribution looks right, we go to feature.py