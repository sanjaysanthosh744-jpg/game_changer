"""Compare the consecutive baseline with sequential CUSUM detection.

Run from this directory:
    python run_detector_comparison.py

This controlled experiment probes one state/action transition; it is not a
general-intelligence benchmark. Results are averaged across fixed seeds.
"""
import copy

from run_change_point_sweep import (
    PROBE_COUNT, mean_or_none, pct, run_noisy_probe_phase, train_original_world,
)

SEEDS = tuple(7 + 10 * i for i in range(20))
NOISE_RATES = (0.0, 0.05, 0.10, 0.20)
PHASES = (
    ("changed-1", True),
    ("original-1", False),
    ("changed-2", True),
    ("original-2", False),
)
STABLE_PROBES = 500
CONFIGS = (
    ("consecutive-3", "consecutive", 3),
    ("cusum-3.5", "cusum", 3.5),
    ("cusum-5.0", "cusum", 5.0),
    ("cusum-6.5", "cusum", 6.5),
)


def configure(agent, detector, threshold):
    agent.change_detector = detector
    if detector == "consecutive":
        agent.change_point_threshold = int(threshold)
    else:
        agent.cusum_threshold = float(threshold)


def evaluate_switches(trained, seed, detector, threshold, noise_rate):
    agent = copy.deepcopy(trained)
    configure(agent, detector, threshold)
    rows = []
    for index, (name, changed) in enumerate(PHASES):
        before = len(agent.change_point_events)
        result = run_noisy_probe_phase(
            agent, PROBE_COUNT, changed,
            seed=seed + 20_000 + index * 1_000,
            noise_rate=noise_rate,
        )
        rows.append({
            "phase": name,
            "accuracy": result["prediction_accuracy_when_known"],
            "first_correct": result["first_correct_probe"],
            "known": result["known_predictions"],
            "change_points": len(agent.change_point_events) - before,
        })
    return rows


def evaluate_stable(trained, seed, detector, threshold, noise_rate):
    agent = copy.deepcopy(trained)
    configure(agent, detector, threshold)
    before = len(agent.change_point_events)
    result = run_noisy_probe_phase(
        agent, STABLE_PROBES, False, seed=seed + 50_000, noise_rate=noise_rate
    )
    return {
        "accuracy": result["prediction_accuracy_when_known"],
        "false_changes": len(agent.change_point_events) - before,
        "known": result["known_predictions"],
    }


def main():
    trained = {seed: train_original_world(seed) for seed in SEEDS}
    print("CHANGE DETECTOR COMPARISON")
    print(f"Seeds: {len(SEEDS)}; training: 80 original-world episodes per seed")
    print(f"Switch phases: {len(PHASES)} x {PROBE_COUNT} probes; stable phase: {STABLE_PROBES} probes")
    print("CUSUM assumptions: stable contradiction probability=0.25; changed=0.80")
    print("Accuracy is scored against the true mechanic, not potentially corrupted feedback.")
    print()

    for noise_rate in NOISE_RATES:
        print(f"NOISE {noise_rate:.0%}")
        print(f"{'detector':<15} {'switch accuracy':>15} {'first correct':>14} {'changes/phase':>14} {'stable accuracy':>16} {'false changes/seed':>19}")
        for name, detector, threshold in CONFIGS:
            switch_rows = []
            stable_rows = []
            for seed in SEEDS:
                switch_rows.extend(
                    {**row, "seed": seed}
                    for row in evaluate_switches(trained[seed], seed, detector, threshold, noise_rate)
                )
                stable_rows.append(
                    evaluate_stable(trained[seed], seed, detector, threshold, noise_rate)
                )
            avg_accuracy = mean_or_none(row["accuracy"] for row in switch_rows)
            first_correct = mean_or_none(row["first_correct"] for row in switch_rows)
            changes = mean_or_none(row["change_points"] for row in switch_rows)
            stable_accuracy = mean_or_none(row["accuracy"] for row in stable_rows)
            false_changes = mean_or_none(row["false_changes"] for row in stable_rows)
            first_text = "n/a" if first_correct is None else f"{first_correct:.1f}"
            print(
                f"{name:<15} {pct(avg_accuracy):>15} {first_text:>14} "
                f"{changes:>14.2f} {pct(stable_accuracy):>16} {false_changes:>19.2f}"
            )
        print()

    print("Readout: faster detection is useful only if stable-world false alarms stay controlled.")
    print("These measurements cover one forced transition and this specific flip-noise model.")


if __name__ == "__main__":
    main()
