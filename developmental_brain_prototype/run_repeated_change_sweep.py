"""Repeated change-point stress test.

Run from this directory:
    python run_repeated_change_sweep.py

Tests repeated original/changed mechanic switches and long stable periods under
several observation-noise rates. This is a narrow transition-level experiment,
not a general-intelligence benchmark.
"""
import statistics

from brain.agent import DevelopmentalAgent
from main import run_episode
from brain.environment import TinyWorld
from run_change_point_sweep import (
    PROBE_COUNT,
    SEEDS,
    THRESHOLDS,
    TRAIN_EPISODES,
    mean_or_none,
    pct,
    run_noisy_probe_phase,
    train_original_world,
)

# Avoid the extreme no-change baseline here; this experiment compares usable
# detection thresholds. Keep these settings modest enough for a local run.
TEST_THRESHOLDS = (2, 3, 5)
NOISE_RATES = (0.0, 0.05, 0.10, 0.20)
PHASES = (
    ("changed-1", True),
    ("original-1", False),
    ("changed-2", True),
    ("original-2", False),
)
STABLE_PROBES = 500


def run_repeated_switches(trained_agent, seed, threshold, noise_rate):
    agent = __import__("copy").deepcopy(trained_agent)
    agent.change_point_threshold = threshold
    rows = []
    for phase_index, (phase_name, changed_dynamics) in enumerate(PHASES):
        before_events = len(agent.change_point_events)
        result = run_noisy_probe_phase(
            agent,
            count=PROBE_COUNT,
            changed_dynamics=changed_dynamics,
            seed=seed + 20_000 + phase_index * 1_000,
            noise_rate=noise_rate,
        )
        rows.append({
            "phase": phase_name,
            "accuracy": result["prediction_accuracy_when_known"],
            "first_correct": result["first_correct_probe"],
            "known": result["known_predictions"],
            "change_points": len(agent.change_point_events) - before_events,
            "corruption_rate": result["corruption_rate_observed"],
        })
    return rows


def run_stable_noise(trained_agent, seed, threshold, noise_rate):
    import copy

    agent = copy.deepcopy(trained_agent)
    agent.change_point_threshold = threshold
    before_events = len(agent.change_point_events)
    result = run_noisy_probe_phase(
        agent,
        count=STABLE_PROBES,
        changed_dynamics=False,
        seed=seed + 50_000,
        noise_rate=noise_rate,
    )
    return {
        "accuracy": result["prediction_accuracy_when_known"],
        "false_changes": len(agent.change_point_events) - before_events,
        "corruption_rate": result["corruption_rate_observed"],
        "known": result["known_predictions"],
    }


def print_switch_table(rows, noise_rate):
    print(f"REPEATED SWITCHES — noise {noise_rate:.0%}")
    print(f"{'threshold':>9} {'phase':>12} {'accuracy':>10} {'first correct':>14} {'known':>8} {'change points':>14}")
    for threshold in TEST_THRESHOLDS:
        for phase_name, _ in PHASES:
            subset = [
                row for row in rows
                if row["threshold"] == threshold and row["phase"] == phase_name
            ]
            accuracy = mean_or_none(row["accuracy"] for row in subset)
            first = mean_or_none(row["first_correct"] for row in subset)
            known = sum(row["known"] for row in subset)
            changes = mean_or_none(row["change_points"] for row in subset)
            first_text = "n/a" if first is None else f"{first:.1f}"
            print(
                f"{threshold:>9} {phase_name:>12} {pct(accuracy):>10} "
                f"{first_text:>14} {known:>4}/{len(subset) * PROBE_COUNT:<3} "
                f"{changes:>14.2f}"
            )
    print()


def print_stable_table(rows, noise_rate):
    print(f"STABLE WORLD — noise {noise_rate:.0%}, {STABLE_PROBES} probes per seed")
    print(f"{'threshold':>9} {'accuracy':>10} {'false changes':>14} {'actual noise':>13}")
    for threshold in TEST_THRESHOLDS:
        subset = [row for row in rows if row["threshold"] == threshold]
        print(
            f"{threshold:>9} "
            f"{pct(mean_or_none(row['accuracy'] for row in subset)):>10} "
            f"{mean_or_none(row['false_changes'] for row in subset):>14.2f} "
            f"{pct(mean_or_none(row['corruption_rate'] for row in subset)):>13}"
        )
    print()


def main():
    trained = {seed: train_original_world(seed) for seed in SEEDS}
    print("Repeated change-point stress test")
    print(f"Seeds: {', '.join(map(str, SEEDS))}")
    print(f"Thresholds: {', '.join(map(str, TEST_THRESHOLDS))}")
    print(f"Noise rates: {', '.join(pct(rate) for rate in NOISE_RATES)}")
    print(f"Repeated-switch phases: {len(PHASES)} x {PROBE_COUNT} probes")
    print()

    for noise_rate in NOISE_RATES:
        switch_rows = []
        for threshold in TEST_THRESHOLDS:
            for seed in SEEDS:
                for phase in run_repeated_switches(
                    trained[seed], seed, threshold, noise_rate
                ):
                    switch_rows.append({
                        "seed": seed,
                        "threshold": threshold,
                        **phase,
                    })
        print_switch_table(switch_rows, noise_rate)

        stable_rows = []
        for threshold in TEST_THRESHOLDS:
            for seed in SEEDS:
                stable_rows.append({
                    "seed": seed,
                    "threshold": threshold,
                    **run_stable_noise(trained[seed], seed, threshold, noise_rate),
                })
        print_stable_table(stable_rows, noise_rate)

    print("Interpretation:")
    print("- During switch phases, accuracy is scored against the true mechanic, not the noisy observation.")
    print("- A change-point count is the number of detections within that phase; it is not automatically a false alarm.")
    print("- Stable-world false changes are the key false-alarm measure.")
    print("- Unknown predictions are excluded from accuracy, so always read accuracy alongside known counts.")
    print("- Results concern one forced state/action transition and this specific flip-noise model.")


if __name__ == "__main__":
    main()
