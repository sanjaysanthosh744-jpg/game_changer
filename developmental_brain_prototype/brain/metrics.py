def summarize(results, agent):
    if results:
        last = results[-20:]
        last_success = sum(x["success"] for x in last)/len(last)
        total_pred = sum(x["prediction_count"] for x in results)
        total_correct = sum(x["prediction_correct"] for x in results)
        accuracy = total_correct/total_pred if total_pred else 0.0
        success = sum(x["success"] for x in results)/len(results)
        mean_steps = sum(x["steps"] for x in results)/len(results)
    else:
        last_success = accuracy = success = mean_steps = 0.0
    return {"success_rate":success,"last_20_success_rate":last_success,
        "mean_steps":mean_steps,"prediction_accuracy":accuracy,
        "stored_memories":len(agent.episodes),
        "transition_observations":agent.transition_observations,
        "unique_state_action_pairs":len(agent.transition_counts),
        "unique_states":len(agent.state_visits)}
