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

Run the transfer study:

```bash
python main.py --episodes 80 --reset --transfer
```

The transfer study reports:
1. **Zero-shot goal-reaching:** a frozen model in the changed world.
2. **Online adaptation:** a separate copy learns while acting in the changed world.
3. **Retention:** the adapted episode agent returns to the original world with learning frozen.
4. **Targeted mechanic probe:** the agent is placed immediately above the special tile and forced to move DOWN. This directly tests the transition that differs between the original conveyor mechanic and changed reversal mechanic. A separate copy is used for repeated probe adaptation, so this diagnostic does not leak training into the episode-level adaptation score.
5. **Change-point adaptation and recovery:** after repeated contradictory outcomes, the active transition model can switch to the new outcome. The same probe learner is then tested after the world returns to the original mechanic.

The targeted probe matters because an agent can reach the goal while avoiding the changed mechanic. Goal success alone does not prove that the changed rule was encountered or learned.

The change-point experiment keeps two kinds of evidence:
- **Historical transition counts:** retained as long-term memory and never erased by a detected change.
- **Active transition counts:** used for current predictions and reset for a state/action pair after three consecutive observations contradict the same prediction.

This is a deliberately simple change detector, not a statistically calibrated detector. The threshold is configurable when constructing the agent. It can react to real changes, but repeated noise can also trigger a false change; use repeated seeds and noisy environments before drawing broad conclusions.

Options:
- `--transfer-episodes 20`: episode count for zero-shot and online adaptation.
- `--retention-episodes 20`: episode count for retention.
- `--probe-episodes 30`: number of forced transition observations in the separate mechanic-adaptation diagnostic.
- `--seed 7`: random seed for reproducibility.
- `--state-file data/brain_state.json`: original model state path.
- `--adapted-state-file data/transfer_adapted_brain_state.json`: adapted episode model path.
- `--report-file runs/latest_report.json`: report path.

Inspect the saved model:

```bash
python inspect_brain.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run the multi-seed change-point threshold sweep:

```bash
python run_change_point_sweep.py
```

The sweep trains on the original world using five fixed seeds, then compares thresholds 2, 3, 5, and a very high threshold (a no-change-detection comparison). It reports clean-observation results and a second run with a 10% chance of feedback being flipped to the other mechanic's outcome. In the noisy run, scores compare predictions with the true environment outcome while the learner receives the possibly corrupted observation. Each run probes the changed mechanic and then probes recovery after the original mechanic returns. A third diagnostic keeps the original mechanic unchanged while injecting the same noise, measuring false change detections. This controlled benchmark isolates one state/action transition; it does not measure generalization across states.

Run the repeated noisy change-point stress test:

```bash
python run_repeated_change_sweep.py
```

Compare the default consecutive-contradiction detector with an optional sequential CUSUM detector:

```bash
python run_detector_comparison.py
```

Initial CI result snapshot (20 seeds, 80 training episodes per seed, 30 probes per mechanic phase, 500 stable-world probes per seed):

| Noise | Detector | Mean per-phase accuracy | Mean first-correct probe | Stable-world false changes per seed |
|---|---|---:|---:|---:|
| 0% | Consecutive threshold 3 | 90.0% | 4.0 | 0.00 |
| 0% | CUSUM threshold 5.0 | 83.3% | 6.0 | 0.00 |
| 10% | Consecutive threshold 3 | 88.1% | 4.6 | 0.80 |
| 10% | CUSUM threshold 5.0 | 80.3% | 6.9 | 0.00 |
| 20% | Consecutive threshold 3 | 83.8% | 5.2 | 6.10 |
| 20% | CUSUM threshold 5.0 | 76.7% | 7.9 | 0.45 |

Interpretation: the current CUSUM settings reduce false alarms substantially at high noise, but adapt more slowly and score lower on average when switching mechanics. This is a tradeoff, not a universal win; keep consecutive threshold 3 as the default and treat CUSUM as an optional experiment. The next useful stress test is temporally correlated/bursty noise, because the current benchmark flips each observation independently.

The CUSUM detector accumulates log-likelihood evidence from contradictions and reduces that evidence when an observation matches the active model. The comparison evaluates 20 fixed seeds, four observation-noise rates, repeated mechanic switches, and long stable periods. It compares the default consecutive threshold of 3 with CUSUM thresholds 3.5, 5.0, and 6.5. CUSUM's stable/changed contradiction probabilities (0.25 and 0.80) are explicit assumptions, not calibrated probabilities. The default detector remains consecutive; CUSUM is experimental until results justify changing that default.

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

The targeted probe is a controlled diagnostic of one changed transition, not evidence of general intelligence or broad generalization. The current transition table memorizes state/action outcomes; it does not infer a portable symbolic rule.

## Experiments

1. Run with `--reset` several times and compare prediction accuracy.
2. Change the hidden mechanic behavior in `brain/environment.py`.
3. Use `--transfer` to compare goal-reaching, a controlled mechanic probe, adaptation, recovery after the mechanic reverts, and retention separately.
4. Compare the baseline branch with `experiment/change-point-detection` across multiple fixed seeds.
5. Test change-detection thresholds under noisy observations.
6. Change exploration weights in `brain/agent.py`.
7. Add memory limits/consolidation.
8. Replace the transition table with a learned neural predictor only after this baseline is measurable.
