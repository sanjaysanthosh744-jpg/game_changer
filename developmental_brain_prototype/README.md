# Developmental Brain Prototype

A small, inspectable research prototype for:
- persistent state
- self-supervised next-state prediction
- uncertainty-driven exploration
- episodic memory saved to disk
- action → consequence → model update
- transfer evaluation in a changed environment

No LLM, API key, GPU, or third-party Python package is required. Python 3.10+ is recommended.

## Run

```bash
python main.py --episodes 80
```

A report is written to `runs/latest_report.json`; learned transition statistics and episodic memories are saved to `data/brain_state.json`.

Start clean:
```bash
python main.py --episodes 80 --reset
```

Evaluate transfer to a changed world in three separate stages:
```bash
python main.py --episodes 80 --reset --transfer
```

The transfer study reports (1) zero-shot transfer with learning frozen, (2) adaptation with learning enabled, and (3) retention when the adapted agent returns to the original world. The original-world model is saved to `data/brain_state.json`; the adapted copy is saved separately to `data/transfer_adapted_brain_state.json`. The report is saved to `runs/latest_report.json`.

Inspect the saved model:
```bash
python inspect_brain.py
```

Run tests:
```bash
python -m unittest discover -s tests -v
```

## What it learns

The toy world is a small grid with walls, a goal, and a hidden tile mechanic. The agent isn't told the mechanic's rule. It observes state, acts, observes the result, and updates a transition model.

For each state/action pair it stores observed next-state counts. These support next-state prediction, uncertainty estimates, and a diminishing-return exploration bonus. Experience episodes persist to disk.

## Architecture

```text
Observation
    ↓
Compact state representation
    ↓
Persistent transition model + episodic memory
    ↓
Action selection (uncertainty / information-seeking)
    ↓
Action in environment
    ↓
Observed consequence
    ↓
Prediction error + learning-progress statistics
    ↓
Update model and memory
    ↺
```

## Important limitations

This is a toy research baseline, not general intelligence, consciousness, or a 120B-LLM replacement. It uses a finite transition table, receives symbolic state from the environment, has a hand-designed exploration policy, and does not learn vision, language, or its own learning algorithm. The hidden world is intentionally simple.

## Experiments

1. Run with `--reset` several times and compare prediction accuracy.
2. Change `hidden_mechanic` behavior in `brain/environment.py`.
3. Use `--transfer` to compare zero-shot transfer, adaptation, and retention separately.
4. Change exploration weights in `brain/agent.py`.
5. Add memory limits/consolidation.
6. Replace the transition table with a learned neural predictor only after this baseline is measurable.
