"""Multi-seed sweep for active-model change-point detection.

Run from this directory:
    python run_change_point_sweep.py

This is a controlled probe benchmark, not a general-intelligence benchmark.
"""
import copy
import random
import statistics

from brain.agent import DevelopmentalAgent
from brain.environment import TinyWorld
from main import probe_summary, run_episode, run_mechanic_probe


SEEDS = (7, 17, 27, 37, 47)
TRAIN_EPISODES = 80
PROBE_COUNT = 30
# A very high threshold approximates a no-change-detection baseline.
THRESHOLDS = (2, 3, 5, 10_000)


def train_original_world(seed):
    agent = DevelopmentalAgent()
    env = TinyWorld(seed=seed, changed_dynamics=False)
    rng = random.Random(seed)
    for episode in range(TRAIN_EPISODES):
        env.episode_index = episode
        run_episode(agent, env, max_steps=100, rng=rng, learn=True)
    return agent


def evaluate_threshold(trained_agent, seed, threshold):
    agent = copy.deepcopy(trained_agent)
    agent.change_point_threshold = threshold

    frozen_probe = run_mechanic_probe(
        copy.deepcopy(agent), count=1, learn=False,
        seed=seed + 4000, changed_dynamics=True
    )
    changed = run_mechanic_probe(
        agent, count=PROBE_COUNT, learn=True,
        seed=seed + 4000, changed_dynamics=True
    )
    recovery = run_mechanic_probe(
        agent, count=PROBE_COUNT, learn=True,
        seed=seed + 8000, changed_dynamics=False
    )
    return {
        "seed": seed,
        "threshold": threshold,
        "initial_prediction_known": frozen_probe[0]["prediction"] is not None,
        "initial_prediction_correct": frozen_probe[0]["prediction"] == frozen_probe[0]["actual_next_state"],
        "changed": probe_summary(changed),
        "recovery": probe_summary(recovery),
        "change_points": len(agent.change_point_events),
        "history_preserved": bool(agent.transition_counts),
    }


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def main():
    all_results = []
    trained = {seed: train_original_world(seed) for seed in SEEDS}

    print("Change-point sweep")
    print(f"Training: {TRAIN_EPISODES} original-world episodes per seed")
    print(f"Probe phases: {PROBE_COUNT} changed-mechanic probes, then {PROBE_COUNT} original-mechanic recovery probes")
    print(f"Seeds: {', '.join(map(str, SEEDS))}")
    print()
    print(f"{'threshold':>9} {'known':>7} {'changed acc':>12} {'first correct':>14} "
          f"{'recovery acc':>13} {'recovery first':>15} {'change points':>14}")

    for threshold in THRESHOLDS:
        rows = [evaluate_threshold(trained[seed], seed, threshold) for seed in SEEDS]
        all_results.extend(rows)
        known = sum(row["initial_prediction_known"] for row in rows)
        changed_acc = mean_or_none(row["changed"]["prediction_accuracy_when_known"] for row in rows)
        changed_first = mean_or_none(row["changed"]["first_correct_probe"] for row in rows)
        recovery_acc = mean_or_none(row["recovery"]["prediction_accuracy_when_known"] for row in rows)
        recovery_first = mean_or_none(row["recovery"]["first_correct_probe"] for row in rows)
        change_points = mean_or_none(row["change_points"] for row in rows)
        print(f"{threshold:>9} {known:>4}/{len(rows):<2} {pct(changed_acc):>12} "
              f"{('n/a' if changed_first is None else f'{changed_first:.1f}'):>14} "
              f"{pct(recovery_acc):>13} "
              f"{('n/a' if recovery_first is None else f'{recovery_first:.1f}'):>15} "
              f"{change_points:>14.1f}")

    print()
    print("Interpretation:")
    print("- Lower thresholds adapt faster but may react to noisy observations.")
    print("- The very high threshold is a no-change-detection comparison.")
    print("- Accuracy is averaged per seed over predictions that were known; unknown predictions are not counted as correct.")
    print("- These results measure one forced state/action transition, not broad generalization or goal-reaching.")
    print("- Historical counts are kept by design; this script does not treat retained counts alone as proof of useful memory.")


if __name__ == "__main__":
    main()
