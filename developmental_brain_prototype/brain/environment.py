import random

class TinyWorld:
    # Tiny symbolic environment with a hidden tile mechanic.
    actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")

    def __init__(self, width=6, height=6, seed=0, changed_dynamics=False):
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.changed_dynamics = changed_dynamics
        self.episode_index = 0
        self.walls = {(2, 1), (2, 2), (2, 3), (4, 3)}
        self.special, self.goal, self.start = (3, 4), (5, 5), (0, 0)
        self.position, self.steps, self._done = self.start, 0, False

    def reset(self, start_position=None):
        """Reset the episode, optionally placing the agent at a valid test position."""
        position = self.start if start_position is None else tuple(start_position)
        if self._blocked(position):
            raise ValueError(f"Cannot start inside a wall or outside the world: {position}")
        self.position, self.steps, self._done = position, 0, False
        return self.observe()

    def observe(self):
        x, y = self.position
        tile = "GOAL" if self.position == self.goal else "SPECIAL" if self.position == self.special else "EMPTY"
        adjacent = {}
        for action, (dx, dy) in {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}.items():
            adjacent[action] = "BLOCKED" if self._blocked((x + dx, y + dy)) else "OPEN"
        return {"x": x, "y": y, "tile": tile, "adjacent": adjacent,
                "goal_dx": self.goal[0] - x, "goal_dy": self.goal[1] - y}

    @staticmethod
    def encode_observation(obs):
        a = obs["adjacent"]
        return f"x{obs['x']}y{obs['y']}:{obs['tile']}:u{a['UP'][0]}d{a['DOWN'][0]}l{a['LEFT'][0]}r{a['RIGHT'][0]}"

    def _blocked(self, p):
        x, y = p
        return x < 0 or y < 0 or x >= self.width or y >= self.height or p in self.walls

    def step(self, action):
        if self._done:
            return self.observe(), 0.0, True, {"event": "already_done"}
        delta = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0), "WAIT": (0, 0)}
        dx, dy = delta[action]
        candidate = (self.position[0] + dx, self.position[1] + dy)
        if not self._blocked(candidate):
            self.position = candidate
        event = "move"
        if self.position == self.special and action != "WAIT":
            if self.changed_dynamics:
                reverse = (self.position[0] - dx, self.position[1] - dy)
                if not self._blocked(reverse):
                    self.position = reverse
                event = "special_reversal"
            else:
                extra = (self.position[0] + dx, self.position[1] + dy)
                if not self._blocked(extra):
                    self.position = extra
                event = "special_conveyor"
        self.steps += 1
        done = self.position == self.goal or self.steps >= 100
        reward = 1.0 if self.position == self.goal else -0.01
        if self.position == self.goal:
            event = "goal_reached"
        elif self.steps >= 100:
            event = "timeout"
        self._done = done
        return self.observe(), reward, done, {"event": event, "action": action}
