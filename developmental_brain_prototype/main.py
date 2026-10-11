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


def run_transfer_study(base_agent, args):
    """Measure zero-shot transfer, online adaptation, then retention."""
    print("\n=== Transfer study ===")
    # Stage 1: no updates in the changed world. This is the zero-shot baseline.
    zero_shot_agent = copy.deepcopy(base_agent)
    zero_shot = run_batch(zero_shot_agent, True, args.episodes, args.transfer_episodes,
                          args.max_steps, args.seed + 1000, learn=False,
                          label="Zero-shot transfer (frozen model)")

    # Stage 2: a separate copy adapts online in the changed world.
    adapted_agent = copy.deepcopy(base_agent)
    adaptation = run_batch(adapted_agent, True, args.episodes, args.transfer_episodes,
                           args.max_steps, args.seed + 2000, learn=True,
                           label="Adaptation (learning enabled)")

    # Stage 3: freeze the adapted model and return to the original world.
    retention = run_batch(adapted_agent, False, args.episodes + args.transfer_episodes,
                          args.retention_episodes, args.max_steps, args.seed + 3000,
                          learn=False, label="Retention (original world, frozen model)")

    adapted_path = Path(args.adapted_state_file)
    adapted_agent.save(adapted_path)
    print(f"Adapted brain saved: {adapted_path}")
    return {
        "protocol": "zero-shot frozen evaluation -> online adaptation -> frozen retention",
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
                        help="run zero-shot transfer, adaptation, and retention study")
    parser.add_argument("--transfer-episodes", type=int, default=20,
                        help="number of changed-world episodes for zero-shot and adaptation")
    parser.add_argument("--retention-episodes", type=int, default=20,
                        help="number of original-world episodes for retention test")
    parser.add_argument("--state-file", default="data/brain_state.json")
    parser.add_argument("--adapted-state-file", default="data/transfer_adapted_brain_state.json")
    parser.add_argument("--report-file", default="runs/latest_report.json")
    args = parser.parse_args()
    if args.episodes < 1 or args.transfer_episodes < 1 or args.retention_episodes < 1:
        parser.error("episode counts must all be at least 1")

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

    # Save the original-world model before the separate transfer experiments.
    agent.save(state_path)
    report = summarize(training, agent)
    report["configuration"] = vars(args)
    report["training_episodes_this_run"] = len(training)
    report["note"] = "Toy experiment only; not a general intelligence benchmark."

    if args.transfer:
        report["transfer_study"] = run_transfer_study(agent, args)
        ts = report["transfer_study"]
        print("\n=== Transfer study summary ===")
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
