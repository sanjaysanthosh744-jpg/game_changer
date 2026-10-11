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
NOISE_RATE = 0.10
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


def run_noisy_probe_phase(agent, count, changed_dynamics, seed, noise_rate):
    """Probe one fixed transition while occasionally corrupting the observed outcome.

    Evaluation compares predictions with the true environment outcome. Learning receives
    the possibly corrupted outcome, separating model quality from observation quality.
    """
    rng = random.Random(seed)
    results = []
    for i in range(count):
        env = TinyWorld(seed=seed + i, changed_dynamics=changed_dynamics)
        env.episode_index = seed + i
        obs = env.reset(start_position=(3, 3))
        state = env.encode_observation(obs)
        action = "DOWN"
        prediction = agent.predict_next_state(state, action)

        true_next_obs, reward, _, info = env.step(action)
        true_next_state = env.encode_observation(true_next_obs)

        alternate_env = TinyWorld(
            seed=seed + i, changed_dynamics=not changed_dynamics
        )
        alternate_env.reset(start_position=(3, 3))
        alternate_next_obs, _, _, _ = alternate_env.step(action)
        alternate_state = alternate_env.encode_observation(alternate_next_obs)

        corrupted = rng.random() < noise_rate
        observed_next_state = alternate_state if corrupted else true_next_state
        known = prediction is not None
        correct = bool(known and prediction == true_next_state)
        results.append({
            "probe": i + 1,
            "prediction": prediction,
            "actual_next_state": true_next_state,
            "observed_next_state": observed_next_state,
            "prediction_known": known,
            "prediction_correct": correct,
            "observation_corrupted": corrupted,
        })
        agent.observe(
            state, action, observed_next_state, prediction,
            env.episode_index, 0, reward, info
        )
        agent.end_episode()

    known_results = [row for row in results if row["prediction_known"]]
    first_correct = next(
        (row["probe"] for row in results if row["prediction_correct"]), None
    )
    return {
        "probes": len(results),
        "known_predictions": len(known_results),
        "prediction_accuracy_when_known": (
            sum(row["prediction_correct"] for row in known_results) / len(known_results)
            if known_results else None
        ),
        "first_correct_probe": first_correct,
        "corruption_rate_observed": (
            sum(row["observation_corrupted"] for row in results) / len(results)
            if results else None
        ),
        "results": results,
    }



def evaluate_stable_world_noise(trained_agent, seed, threshold, count, noise_rate):
    """Measure false change detections when the true mechanic never changes."""
    agent = copy.deepcopy(trained_agent)
    agent.change_point_threshold = threshold
    initial_events = len(agent.change_point_events)
    result = run_noisy_probe_phase(
        agent, count=count, changed_dynamics=False,
        seed=seed + 12000, noise_rate=noise_rate
    )
    result["false_change_points"] = len(agent.change_point_events) - initial_events
    result["final_prediction"] = agent.predict_next_state(
        "x3y3:EMPTY:uOdOlBrB", "DOWN"
    )
    return result


def evaluate_threshold(trained_agent, seed, threshold, noise_rate=0.0):
    agent = copy.deepcopy(trained_agent)
    agent.change_point_threshold = threshold

    frozen_probe = run_mechanic_probe(
        copy.deepcopy(agent), count=1, learn=False,
        seed=seed + 4000, changed_dynamics=True
    )
    if noise_rate:
        changed = run_noisy_probe_phase(
            agent, PROBE_COUNT, True, seed + 4000, noise_rate
        )
        recovery = run_noisy_probe_phase(
            agent, PROBE_COUNT, False, seed + 8000, noise_rate
        )
    else:
        changed = probe_summary(run_mechanic_probe(
            agent, count=PROBE_COUNT, learn=True,
            seed=seed + 4000, changed_dynamics=True
        ))
        recovery = probe_summary(run_mechanic_probe(
            agent, count=PROBE_COUNT, learn=True,
            seed=seed + 8000, changed_dynamics=False
        ))
    return {
        "seed": seed,
        "threshold": threshold,
        "noise_rate": noise_rate,
        "initial_prediction_known": frozen_probe[0]["prediction"] is not None,
        "initial_prediction_correct": frozen_probe[0]["prediction"] == frozen_probe[0]["actual_next_state"],
        "changed": changed,
        "recovery": recovery,
        "change_points": len(agent.change_point_events),
        "history_preserved": bool(agent.transition_counts),
    }


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def print_table(rows, heading):
    print(heading)
    print(f"{'threshold':>9} {'known':>7} {'changed acc':>12} {'first correct':>14} "
          f"{'recovery acc':>13} {'recovery first':>15} {'change points':>14}")
    for threshold in THRESHOLDS:
        subset = [row for row in rows if row["threshold"] == threshold]
        known = sum(row["initial_prediction_known"] for row in subset)
        changed_acc = mean_or_none(row["changed"]["prediction_accuracy_when_known"] for row in subset)
        changed_first = mean_or_none(row["changed"]["first_correct_probe"] for row in subset)
        recovery_acc = mean_or_none(row["recovery"]["prediction_accuracy_when_known"] for row in subset)
        recovery_first = mean_or_none(row["recovery"]["first_correct_probe"] for row in subset)
        change_points = mean_or_none(row["change_points"] for row in subset)
        print(f"{threshold:>9} {known:>4}/{len(subset):<2} {pct(changed_acc):>12} "
              f"{('n/a' if changed_first is None else f'{changed_first:.1f}'):>14} "
              f"{pct(recovery_acc):>13} "
              f"{('n/a' if recovery_first is None else f'{recovery_first:.1f}'):>15} "
              f"{change_points:>14.1f}")


def main():
    trained = {seed: train_original_world(seed) for seed in SEEDS}
    print("Change-point sweep")
    print(f"Training: {TRAIN_EPISODES} original-world episodes per seed")
    print(f"Probe phases: {PROBE_COUNT} changed-mechanic probes, then {PROBE_COUNT} original-mechanic recovery probes")
    print(f"Seeds: {', '.join(map(str, SEEDS))}")
    print()

    clean_rows = [
        evaluate_threshold(trained[seed], seed, threshold, noise_rate=0.0)
        for threshold in THRESHOLDS for seed in SEEDS
    ]
    print_table(clean_rows, "CLEAN OBSERVATIONS")
    print()

    noisy_rows = [
        evaluate_threshold(trained[seed], seed, threshold, noise_rate=NOISE_RATE)
        for threshold in THRESHOLDS for seed in SEEDS
    ]
    print_table(noisy_rows, f"NOISY OBSERVATIONS ({NOISE_RATE:.0%} chance of a flipped outcome observation)")
    observed_noise = mean_or_none(
        row["changed"]["corruption_rate_observed"] for row in noisy_rows
    )
    print(f"Actual sampled corruption rate in changed phase: {pct(observed_noise)}")
    print()

    stable_rows = []
    stable_count = PROBE_COUNT * 2
    for threshold in THRESHOLDS:
        for seed in SEEDS:
            result = evaluate_stable_world_noise(
                trained[seed], seed, threshold, stable_count, NOISE_RATE
            )
            stable_rows.append({
                "threshold": threshold,
                "seed": seed,
                "accuracy": result["prediction_accuracy_when_known"],
                "first_correct": result["first_correct_probe"],
                "false_change_points": result["false_change_points"],
                "corruption_rate": result["corruption_rate_observed"],
            })
    print(f"STABLE WORLD WITH {NOISE_RATE:.0%} NOISY OBSERVATIONS (no real mechanic change)")
    print(f"{'threshold':>9} {'accuracy':>10} {'false changes':>14} {'actual noise':>13}")
    for threshold in THRESHOLDS:
        subset = [row for row in stable_rows if row["threshold"] == threshold]
        accuracy = mean_or_none(row["accuracy"] for row in subset)
        false_changes = mean_or_none(row["false_change_points"] for row in subset)
        sampled_noise = mean_or_none(row["corruption_rate"] for row in subset)
        print(f"{threshold:>9} {pct(accuracy):>10} {false_changes:>14.1f} {pct(sampled_noise):>13}")
    print()
    print("Interpretation:")
    print("- The noisy phase corrupts feedback to the learner, but scores predictions against the true environment outcome.")
    print("- Lower thresholds adapt faster in clean conditions, but may mistake repeated noise for a real change.")
    print("- The very high threshold is a no-change-detection comparison.")
    print("- Accuracy is averaged per seed over predictions that were known; unknown predictions are not counted as correct.")
    print("- These results measure one forced state/action transition, not broad generalization or goal-reaching.")
    print("- Historical counts are retained, but that alone does not establish useful memory.")


if __name__ == "__main__":
    main()
