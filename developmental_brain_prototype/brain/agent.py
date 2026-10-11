import json, math, random
from collections import defaultdict, deque
from pathlib import Path


class DevelopmentalAgent:
    # Transparent table-based baseline. It is intentionally not an LLM or neural network.
    def __init__(self, memory_limit=5000, change_point_threshold=3):
        # Historical counts are retained as long-term evidence.
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        # Active counts drive predictions and can be reset when a mechanic changes.
        self.active_transition_counts = defaultdict(lambda: defaultdict(int))
        self.contradiction_streaks = {}
        self.change_point_threshold = max(2, int(change_point_threshold))
        self.change_point_events = []
        self.state_visits = defaultdict(int)
        self.action_visits = defaultdict(int)
        self.episodes = deque(maxlen=memory_limit)
        self.recent_errors = defaultdict(lambda: deque(maxlen=30))
        self.total_predictions = 0
        self.correct_predictions = 0
        self.transition_observations = 0
        self.current_episode = []
        self.uncertainty_weight = 0.65
        self.information_weight = 0.30
        self.goal_bias_weight = 0.05
        self.random_exploration = 0.05

    @staticmethod
    def _key(state, action):
        return f"{state}|{action}"

    def _outcomes(self, state, action):
        # Read without creating an empty entry during frozen evaluation.
        key = self._key(state, action)
        active = self.active_transition_counts.get(key)
        if active:
            return active
        # Backward-compatible fallback for old in-memory states.
        return self.transition_counts.get(key, {})

    def predict_next_state(self, state, action):
        outcomes = self._outcomes(state, action)
        return max(outcomes.items(), key=lambda kv: (kv[1], kv[0]))[0] if outcomes else None

    def uncertainty(self, state, action):
        outcomes = self._outcomes(state, action)
        n = sum(outcomes.values())
        if not n:
            return 1.0
        probs = [v / n for v in outcomes.values()]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        return min(1.0, entropy / math.log(max(2, len(probs) + 1)))

    def expected_information_gain(self, state, action):
        n = sum(self._outcomes(state, action).values())
        # Diminishing-return novelty heuristic, not a formal Bayesian estimate.
        return 1.0 / math.sqrt(n + 1)

    def _goal_bias(self, state, action):
        try:
            head = state.split(":")[0]
            xpart, ypart = head.split("y")
            x, y = int(xpart[1:]), int(ypart)
        except Exception:
            return 0.0
        before = abs(5 - x) + abs(5 - y)
        dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0),
                  "RIGHT": (1, 0), "WAIT": (0, 0)}[action]
        after = abs(5 - (x + dx)) + abs(5 - (y + dy))
        return 1.0 if after < before else 0.0

    def choose_action(self, state, actions, rng):
        scores = {}
        for action in actions:
            scores[action] = (
                self.uncertainty_weight * self.uncertainty(state, action)
                + self.information_weight * self.expected_information_gain(state, action)
                + self.goal_bias_weight * self._goal_bias(state, action)
                + rng.random() * self.random_exploration
            )
        best = max(scores.values())
        return rng.choice([a for a, score in scores.items() if abs(score - best) < 1e-12])

    def _update_active_model(self, key, state, action, next_state, prediction):
        active = self.active_transition_counts[key]
        if not active:
            active[next_state] += 1
            self.contradiction_streaks.pop(key, None)
            return

        current_prediction = max(active.items(), key=lambda kv: (kv[1], kv[0]))[0]
        # Count only a consistent run of outcomes that contradict the same active prediction.
        if prediction is not None and next_state != prediction:
            previous = self.contradiction_streaks.get(key)
            if (previous and previous["predicted"] == prediction
                    and previous["observed"] == next_state):
                streak = previous["count"] + 1
            else:
                streak = 1
            self.contradiction_streaks[key] = {
                "predicted": prediction, "observed": next_state, "count": streak
            }
            active[next_state] += 1
            if streak >= self.change_point_threshold:
                old_outcomes = dict(active)
                self.active_transition_counts[key] = defaultdict(int, {next_state: 1})
                self.change_point_events.append({
                    "key": key,
                    "previous_prediction": prediction,
                    "new_prediction": next_state,
                    "contradictions": streak,
                    "previous_active_counts": old_outcomes,
                })
                # Keep the event log bounded; historical transition counts remain complete.
                self.change_point_events = self.change_point_events[-100:]
                self.contradiction_streaks.pop(key, None)
            return

        # A matching observation breaks the consecutive contradiction streak.
        self.contradiction_streaks.pop(key, None)
        active[next_state] += 1

    def observe(self, state, action, next_state, prediction, episode, step, reward, info):
        key = self._key(state, action)
        error = 0.0 if prediction == next_state else 1.0

        # Keep complete long-term evidence, even when the active model changes its belief.
        self.transition_counts[key][next_state] += 1
        self._update_active_model(key, state, action, next_state, prediction)

        self.transition_observations += 1
        self.state_visits[state] += 1
        self.action_visits[action] += 1
        if prediction is not None:
            self.total_predictions += 1
            self.correct_predictions += int(prediction == next_state)
        self.recent_errors[key].append(error)
        self.current_episode.append({
            "state": state, "action": action, "next_state": next_state,
            "prediction": prediction, "prediction_error": error, "reward": reward,
            "event": info.get("event", "unknown"), "episode": episode, "step": step
        })

    def end_episode(self):
        if self.current_episode:
            self.episodes.append(self.current_episode)
        self.current_episode = []

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "transition_counts": {k: dict(v) for k, v in self.transition_counts.items()},
            "active_transition_counts": {k: dict(v) for k, v in self.active_transition_counts.items()},
            "contradiction_streaks": self.contradiction_streaks,
            "change_point_threshold": self.change_point_threshold,
            "change_point_events": self.change_point_events,
            "state_visits": dict(self.state_visits),
            "action_visits": dict(self.action_visits),
            "episodes": list(self.episodes),
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "transition_observations": self.transition_observations,
            "policy": {
                "uncertainty_weight": self.uncertainty_weight,
                "information_weight": self.information_weight,
                "goal_bias_weight": self.goal_bias_weight,
                "random_exploration": self.random_exploration
            }
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        agent = cls(change_point_threshold=payload.get("change_point_threshold", 3))
        agent.transition_counts = defaultdict(lambda: defaultdict(int))
        for key, outcomes in payload.get("transition_counts", {}).items():
            agent.transition_counts[key] = defaultdict(int, outcomes)

        # Old version-1 files had only one table; copy it into the active model.
        agent.active_transition_counts = defaultdict(lambda: defaultdict(int))
        active_payload = payload.get("active_transition_counts")
        if active_payload is None:
            active_payload = payload.get("transition_counts", {})
        for key, outcomes in active_payload.items():
            agent.active_transition_counts[key] = defaultdict(int, outcomes)

        agent.contradiction_streaks = payload.get("contradiction_streaks", {})
        agent.change_point_events = payload.get("change_point_events", [])
        agent.state_visits = defaultdict(int, payload.get("state_visits", {}))
        agent.action_visits = defaultdict(int, payload.get("action_visits", {}))
        agent.episodes = deque(payload.get("episodes", []), maxlen=5000)
        agent.total_predictions = payload.get("total_predictions", 0)
        agent.correct_predictions = payload.get("correct_predictions", 0)
        agent.transition_observations = payload.get("transition_observations", 0)
        for key, value in payload.get("policy", {}).items():
            setattr(agent, key, value)
        return agent
