from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
from brain.agent import DevelopmentalAgent
from brain.environment import TinyWorld
from brain.metrics import summarize


def run_episode(agent, env, max_steps, rng, learn=True):
    """Run one episode. With learn=False, evaluate without recording new transitions."""
    obs = env.reset()
    total_reward, correct, predictions = 0.0, 0, 0
    done = False
    steps = 0
    info = {"event": "unknown"}
    for step in range(max_steps):
        state = env.encode_observation(obs)
        action = agent.choose_action(state, env.actions, rng)
        predicted = agent.predict_next_state(state, action)
        next_obs, reward, done, info = env.step(action)
        next_state = env.encode_observation(next_obs)
        if predicted is not None:
            predictions += 1
            correct += int(predicted == next_state)
        if learn:
            agent.observe(state, action, next_state, predicted, env.episode_index,
                          step, reward, info)
        total_reward += reward
        steps += 1
        obs = next_obs
        if done:
            break
    if learn:
        agent.end_episode()
    return {"steps": steps, "reward": total_reward,
            "success": bool(done and info.get("event") == "goal_reached"),
            "prediction_count": predictions, "prediction_correct": correct}


def run_batch(agent, changed_dynamics, start_episode, count, max_steps, seed,
              learn, label=""):
    env = TinyWorld(seed=seed, changed_dynamics=changed_dynamics)
    rng = random.Random(seed + 17)
    results = []
    for i in range(count):
        env.episode_index = start_episode + i
        results.append(run_episode(agent, env, max_steps, rng, learn=learn))
    if label:
        success = sum(r["success"] for r in results) / max(1, len(results))
        mean_steps = sum(r["steps"] for r in results) / max(1, len(results))
        print(f"{label}: success={success:.1%} | mean steps={mean_steps:.1f}")
    return results


def success_rate(results):
    return sum(r["success"] for r in results) / len(results) if results else 0.0


def chunk_success(results, chunk_size=5):
    return [
        {"episodes": f"{start + 1}-{min(start + chunk_size, len(results))}",
         "success_rate": success_rate(results[start:start + chunk_size]),
         "mean_steps": (sum(x["steps"] for x in results[start:start + chunk_size]) /
                        len(results[start:start + chunk_size]))}
        for start in range(0, len(results), chunk_size)
    ]


def run_mechanic_probe(agent, count, learn, seed=9001, changed_dynamics=True):
    """Force the same state/action through the special tile to test its dynamics."""
    results = []
    for i in range(count):
        env = TinyWorld(seed=seed + i, changed_dynamics=changed_dynamics)
        env.episode_index = seed + i
        obs = env.reset(start_position=(3, 3))
        state = env.encode_observation(obs)
        action = "DOWN"
        prediction = agent.predict_next_state(state, action)
        next_obs, reward, done, info = env.step(action)
        next_state = env.encode_observation(next_obs)
        known_prediction = prediction is not None
        correct = known_prediction and prediction == next_state
        results.append({
            "probe": i + 1,
            "prediction": prediction,
            "actual_next_state": next_state,
            "prediction_known": known_prediction,
            "prediction_correct": bool(correct),
            "event": info.get("event"),
        })
        if learn:
            agent.observe(state, action, next_state, prediction, env.episode_index,
                          0, reward, info)
            agent.end_episode()
    return results


def probe_summary(results):
    known = [r for r in results if r["prediction_known"]]
    first_correct = next((r["probe"] for r in results if r["prediction_correct"]), None)
    return {
        "probes": len(results),
        "known_predictions": len(known),
        "prediction_accuracy_when_known": (
            sum(r["prediction_correct"] for r in known) / len(known) if known else None
        ),
        "first_correct_probe": first_correct,
        "initial_prediction": results[0]["prediction"] if results else None,
        "actual_next_state": results[0]["actual_next_state"] if results else None,
        "final_prediction": results[-1]["prediction"] if results else None,
        "event": results[0]["event"] if results else None,
        "results": results,
    }


def run_transfer_study(base_agent, args):
    """Measure ordinary transfer and a targeted probe of the changed mechanic."""
    print("\n=== Transfer study ===")

    # Ordinary goal success can hide transfer failure when the changed tile is avoided.
    # Probe a separate copy directly so the diagnostic cannot leak into episode scores.
    probe_agent = copy.deepcopy(base_agent)
    zero_shot_probe = run_mechanic_probe(probe_agent, count=1, learn=False,
                                         seed=args.seed + 4000)
    probe_adaptation_agent = copy.deepcopy(base_agent)
    probe_adaptation = run_mechanic_probe(
        probe_adaptation_agent, count=args.probe_episodes, learn=True,
        seed=args.seed + 4000
    )
    zero_probe_summary = probe_summary(zero_shot_probe)
    adapt_probe_summary = probe_summary(probe_adaptation)

    # Reverse the mechanic again on the same probe learner to measure recovery.
    recovery_probe = run_mechanic_probe(
        probe_adaptation_agent, count=args.probe_episodes, learn=True,
        seed=args.seed + 8000, changed_dynamics=False
    )
    recovery_probe_summary = probe_summary(recovery_probe)
    print("Targeted mechanic probe (frozen): "
          f"predicted={zero_probe_summary['initial_prediction']} | "
          f"actual={zero_probe_summary['actual_next_state']} | "
          f"correct={zero_probe_summary['results'][0]['prediction_correct']}")
    print("Targeted mechanic probe (learning): "
          f"accuracy when prediction known={adapt_probe_summary['prediction_accuracy_when_known']} | "
          f"first correct probe={adapt_probe_summary['first_correct_probe']}")
    print("Mechanic recovery probe (back to original): "
          f"accuracy when prediction known={recovery_probe_summary['prediction_accuracy_when_known']} | "
          f"first correct probe={recovery_probe_summary['first_correct_probe']} | "
          f"change points={len(probe_adaptation_agent.change_point_events)}")

    # Same initial RNG seed makes the two episode-level runs more comparable.
    # The trajectories may still diverge because the adapted model changes.
    zero_shot_agent = copy.deepcopy(base_agent)
    zero_shot = run_batch(zero_shot_agent, True, args.episodes, args.transfer_episodes,
                          args.max_steps, args.seed + 1000, learn=False,
                          label="Zero-shot transfer (frozen model)")
    adapted_agent = copy.deepcopy(base_agent)
    adaptation = run_batch(adapted_agent, True, args.episodes, args.transfer_episodes,
                           args.max_steps, args.seed + 1000, learn=True,
                           label="Adaptation (learning enabled)")
    retention = run_batch(adapted_agent, False, args.episodes + args.transfer_episodes,
                          args.retention_episodes, args.max_steps, args.seed + 3000,
                          learn=False, label="Retention (original world, frozen model)")

    adapted_path = Path(args.adapted_state_file)
    adapted_agent.save(adapted_path)
    print(f"Adapted brain saved: {adapted_path}")
    return {
        "protocol": "frozen zero-shot evaluation -> online adaptation -> frozen retention",
        "important_caveat": (
            "The ordinary goal-reaching score may not exercise the changed mechanic. "
            "The targeted probe forces the same state/action through the changed tile. "
            "Probe adaptation is diagnostic and is kept separate from the episode adaptation agent."
        ),
        "targeted_mechanic_probe": {
            "zero_shot": zero_probe_summary,
            "adaptation": adapt_probe_summary,
            "recovery_to_original": recovery_probe_summary,
            "change_point_events": list(probe_adaptation_agent.change_point_events),
            "historical_transition_observations": probe_adaptation_agent.transition_observations,
        },
        "zero_shot": {"success_rate": success_rate(zero_shot),
                      "mean_steps": sum(x["steps"] for x in zero_shot) / len(zero_shot),
                      "prediction_accuracy": _prediction_accuracy(zero_shot),
                      "episodes": len(zero_shot)},
        "adaptation": {"success_rate": success_rate(adaptation),
                       "mean_steps": sum(x["steps"] for x in adaptation) / len(adaptation),
                       "prediction_accuracy": _prediction_accuracy(adaptation),
                       "episodes": len(adaptation), "chunks": chunk_success(adaptation)},
        "retention": {"success_rate": success_rate(retention),
                      "mean_steps": sum(x["steps"] for x in retention) / len(retention),
                      "prediction_accuracy": _prediction_accuracy(retention),
                      "episodes": len(retention)},
        "adapted_state_file": str(adapted_path),
    }


def _prediction_accuracy(results):
    count = sum(x["prediction_count"] for x in results)
    correct = sum(x["prediction_correct"] for x in results)
    return correct / count if count else 0.0


def main():
    parser = argparse.ArgumentParser(description="Tiny developmental self-learning prototype")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--transfer", action="store_true",
                        help="run transfer, targeted mechanic probe, adaptation, and retention study")
    parser.add_argument("--transfer-episodes", type=int, default=20,
                        help="number of changed-world episodes for zero-shot and adaptation")
    parser.add_argument("--retention-episodes", type=int, default=20,
                        help="number of original-world episodes for retention test")
    parser.add_argument("--probe-episodes", type=int, default=30,
                        help="number of forced observations used only in the targeted mechanic diagnostic")
    parser.add_argument("--state-file", default="data/brain_state.json")
    parser.add_argument("--adapted-state-file", default="data/transfer_adapted_brain_state.json")
    parser.add_argument("--report-file", default="runs/latest_report.json")
    args = parser.parse_args()
    if min(args.episodes, args.transfer_episodes, args.retention_episodes, args.probe_episodes) < 1:
        parser.error("episode and probe counts must all be at least 1")

    rng = random.Random(args.seed)
    state_path = Path(args.state_file)
    if args.reset and state_path.exists():
        state_path.unlink()
    agent = DevelopmentalAgent.load(state_path)
    env = TinyWorld(seed=args.seed)
    training = []
    for episode in range(args.episodes):
        env.episode_index = episode
        result = run_episode(agent, env, args.max_steps, rng, learn=True)
        training.append(result)
        if (episode + 1) % 10 == 0 or episode == 0:
            recent = training[-10:]
            print(f"episode {episode+1:4d}/{args.episodes} | "
                  f"recent success={sum(x['success'] for x in recent)/len(recent):.0%} | "
                  f"mean steps={sum(x['steps'] for x in recent)/len(recent):.1f} | "
                  f"memories={len(agent.episodes)} | transitions={agent.transition_observations}")

    agent.save(state_path)
    report = summarize(training, agent)
    report["configuration"] = vars(args)
    report["training_episodes_this_run"] = len(training)
    report["note"] = "Toy experiment only; not a general intelligence benchmark."

    if args.transfer:
        report["transfer_study"] = run_transfer_study(agent, args)
        ts = report["transfer_study"]
        print("\n=== Transfer study summary ===")
        print(f"Targeted probe first prediction: {ts['targeted_mechanic_probe']['zero_shot']['initial_prediction']}")
        print(f"Targeted probe actual outcome:   {ts['targeted_mechanic_probe']['zero_shot']['actual_next_state']}")
        print(f"Recovery to original mechanic:  {ts['targeted_mechanic_probe']['recovery_to_original']['prediction_accuracy_when_known']:.1%} accuracy when known")
        print(f"Change points detected:         {len(ts['targeted_mechanic_probe']['change_point_events'])}")
        print(f"Zero-shot transfer: {ts['zero_shot']['success_rate']:.1%}")
        print(f"Adaptation overall: {ts['adaptation']['success_rate']:.1%}")
        print(f"Retention:          {ts['retention']['success_rate']:.1%}")
        print("Adaptation in 5-episode blocks:")
        for chunk in ts["adaptation"]["chunks"]:
            print(f"  episodes {chunk['episodes']:>5}: success={chunk['success_rate']:.0%}, "
                  f"mean steps={chunk['mean_steps']:.1f}")

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== Run summary ===")
    print(f"Success rate (last 20): {report['last_20_success_rate']:.1%}")
    print(f"Prediction accuracy:   {report['prediction_accuracy']:.1%}")
    print(f"Stored memories:       {report['stored_memories']}")
    print(f"Transition samples:    {report['transition_observations']}")
    print(f"Saved brain:           {state_path}")
    print(f"Saved report:          {report_path}")


if __name__ == "__main__":
    main()
