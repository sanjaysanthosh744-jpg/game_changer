import json, math, random
from collections import defaultdict, deque
from pathlib import Path

class DevelopmentalAgent:
    # Transparent table-based baseline. It is intentionally not an LLM or neural network.
    def __init__(self, memory_limit=5000):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
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
        return self.transition_counts.get(self._key(state, action), {})

    def predict_next_state(self, state, action):
        outcomes = self._outcomes(state, action)
        return max(outcomes.items(), key=lambda kv:(kv[1],kv[0]))[0] if outcomes else None

    def uncertainty(self, state, action):
        outcomes = self._outcomes(state, action)
        n = sum(outcomes.values())
        if not n:
            return 1.0
        probs = [v/n for v in outcomes.values()]
        entropy = -sum(p*math.log(p+1e-12) for p in probs)
        return min(1.0, entropy / math.log(max(2, len(probs)+1)))

    def expected_information_gain(self, state, action):
        n = sum(self._outcomes(state, action).values())
        # Diminishing-return novelty heuristic, not a formal Bayesian estimate.
        return 1.0 / math.sqrt(n+1)

    def _goal_bias(self, state, action):
        try:
            head = state.split(":")[0]
            xpart, ypart = head.split("y")
            x, y = int(xpart[1:]), int(ypart)
        except Exception:
            return 0.0
        before = abs(5-x)+abs(5-y)
        dx,dy = {"UP":(0,-1),"DOWN":(0,1),"LEFT":(-1,0),"RIGHT":(1,0),"WAIT":(0,0)}[action]
        after = abs(5-(x+dx))+abs(5-(y+dy))
        return 1.0 if after < before else 0.0

    def choose_action(self, state, actions, rng):
        scores = {}
        for action in actions:
            scores[action] = (self.uncertainty_weight*self.uncertainty(state,action)
                + self.information_weight*self.expected_information_gain(state,action)
                + self.goal_bias_weight*self._goal_bias(state,action)
                + rng.random()*self.random_exploration)
        best = max(scores.values())
        return rng.choice([a for a,s in scores.items() if abs(s-best)<1e-12])

    def observe(self, state, action, next_state, prediction, episode, step, reward, info):
        key = self._key(state, action)
        error = 0.0 if prediction == next_state else 1.0
        self.transition_counts[key][next_state] += 1
        self.transition_observations += 1
        self.state_visits[state] += 1
        self.action_visits[action] += 1
        if prediction is not None:
            self.total_predictions += 1
            self.correct_predictions += int(prediction == next_state)
        self.recent_errors[key].append(error)
        self.current_episode.append({"state":state,"action":action,"next_state":next_state,
            "prediction":prediction,"prediction_error":error,"reward":reward,
            "event":info.get("event","unknown"),"episode":episode,"step":step})

    def end_episode(self):
        if self.current_episode:
            self.episodes.append(self.current_episode)
        self.current_episode = []

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version":1,
            "transition_counts":{k:dict(v) for k,v in self.transition_counts.items()},
            "state_visits":dict(self.state_visits),"action_visits":dict(self.action_visits),
            "episodes":list(self.episodes),"total_predictions":self.total_predictions,
            "correct_predictions":self.correct_predictions,
            "transition_observations":self.transition_observations,
            "policy":{"uncertainty_weight":self.uncertainty_weight,
                "information_weight":self.information_weight,"goal_bias_weight":self.goal_bias_weight,
                "random_exploration":self.random_exploration}}
        tmp = path.with_suffix(path.suffix+".tmp")
        tmp.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        agent = cls()
        path = Path(path)
        if not path.exists():
            return agent
        payload = json.loads(path.read_text(encoding="utf-8"))
        agent.transition_counts = defaultdict(lambda:defaultdict(int))
        for key, outcomes in payload.get("transition_counts",{}).items():
            agent.transition_counts[key] = defaultdict(int,outcomes)
        agent.state_visits = defaultdict(int,payload.get("state_visits",{}))
        agent.action_visits = defaultdict(int,payload.get("action_visits",{}))
        agent.episodes = deque(payload.get("episodes",[]),maxlen=5000)
        agent.total_predictions = payload.get("total_predictions",0)
        agent.correct_predictions = payload.get("correct_predictions",0)
        agent.transition_observations = payload.get("transition_observations",0)
        for k,v in payload.get("policy",{}).items():
            setattr(agent,k,v)
        return agent
