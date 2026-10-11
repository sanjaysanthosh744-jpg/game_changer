import random
import unittest
from argparse import Namespace
from pathlib import Path
import tempfile

from brain.agent import DevelopmentalAgent
from brain.environment import TinyWorld
from main import run_episode, run_mechanic_probe, run_transfer_study


class TransferEvaluationTests(unittest.TestCase):
    def test_frozen_episode_does_not_learn(self):
        agent = DevelopmentalAgent()
        env = TinyWorld(seed=2, changed_dynamics=True)
        before = agent.transition_observations
        run_episode(agent, env, 10, random.Random(3), learn=False)
        self.assertEqual(agent.transition_observations, before)
        self.assertEqual(len(agent.episodes), 0)

    def test_transfer_study_separates_stages(self):
        agent = DevelopmentalAgent()
        env = TinyWorld(seed=7)
        rng = random.Random(7)
        for i in range(3):
            env.episode_index = i
            run_episode(agent, env, 20, rng, learn=True)
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(episodes=3, transfer_episodes=4, retention_episodes=4,
                             probe_episodes=6, max_steps=20, seed=7,
                             adapted_state_file=str(Path(tmp) / "adapted.json"))
            result = run_transfer_study(agent, args)
            self.assertEqual(result["zero_shot"]["episodes"], 4)
            self.assertEqual(result["adaptation"]["episodes"], 4)
            self.assertEqual(result["retention"]["episodes"], 4)
            self.assertEqual(result["targeted_mechanic_probe"]["zero_shot"]["probes"], 1)
            self.assertEqual(result["targeted_mechanic_probe"]["adaptation"]["probes"], 6)
            self.assertTrue(Path(result["adapted_state_file"]).exists())

    def test_reset_accepts_valid_probe_position(self):
        env = TinyWorld(changed_dynamics=True)
        obs = env.reset(start_position=(3, 3))
        self.assertEqual((obs["x"], obs["y"]), (3, 3))

    def test_reset_rejects_wall_position(self):
        env = TinyWorld()
        with self.assertRaises(ValueError):
            env.reset(start_position=(2, 2))

    def test_targeted_probe_exposes_changed_mechanic(self):
        env_old = TinyWorld(changed_dynamics=False)
        obs_old = env_old.reset(start_position=(3, 3))
        next_old, _, _, info_old = env_old.step("DOWN")
        env_new = TinyWorld(changed_dynamics=True)
        env_new.reset(start_position=(3, 3))
        next_new, _, _, info_new = env_new.step("DOWN")
        self.assertEqual(info_old["event"], "special_conveyor")
        self.assertEqual(info_new["event"], "special_reversal")
        self.assertNotEqual(
            env_old.encode_observation(next_old),
            env_new.encode_observation(next_new)
        )



    def test_single_contradiction_does_not_trigger_change_point(self):
        agent = DevelopmentalAgent(change_point_threshold=3)
        agent.observe("state", "DOWN", "old", None, 0, 0, 0.0, {"event": "move"})
        agent.observe("state", "DOWN", "new", "old", 1, 0, 0.0, {"event": "noise"})
        self.assertEqual(agent.predict_next_state("state", "DOWN"), "old")
        self.assertEqual(len(agent.change_point_events), 0)

    def test_change_point_switches_active_model_but_preserves_history(self):
        agent = DevelopmentalAgent(change_point_threshold=3)
        agent.observe("state", "DOWN", "old", None, 0, 0, 0.0, {"event": "move"})
        for i in range(3):
            agent.observe("state", "DOWN", "new", "old", 1, i, 0.0, {"event": "changed"})
        self.assertEqual(agent.predict_next_state("state", "DOWN"), "new")
        key = agent._key("state", "DOWN")
        self.assertEqual(agent.transition_counts[key]["old"], 1)
        self.assertEqual(agent.transition_counts[key]["new"], 3)
        self.assertEqual(dict(agent.active_transition_counts[key]), {"new": 1})
        self.assertEqual(len(agent.change_point_events), 1)

    def test_model_recovers_when_original_mechanic_returns(self):
        agent = DevelopmentalAgent(change_point_threshold=3)
        agent.observe("state", "DOWN", "old", None, 0, 0, 0.0, {"event": "move"})
        for i in range(3):
            agent.observe("state", "DOWN", "new", "old", 1, i, 0.0, {"event": "changed"})
        for i in range(3):
            agent.observe("state", "DOWN", "old", "new", 2, i, 0.0, {"event": "reverted"})
        self.assertEqual(agent.predict_next_state("state", "DOWN"), "old")
        key = agent._key("state", "DOWN")
        self.assertEqual(agent.transition_counts[key]["old"], 4)
        self.assertEqual(agent.transition_counts[key]["new"], 3)
        self.assertEqual(len(agent.change_point_events), 2)

    def test_active_model_and_history_survive_persistence(self):
        agent = DevelopmentalAgent()
        agent.observe("state", "DOWN", "old", None, 0, 0, 0.0, {"event": "move"})
        for i in range(3):
            agent.observe("state", "DOWN", "new", "old", 1, i, 0.0, {"event": "changed"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.json"
            agent.save(path)
            loaded = DevelopmentalAgent.load(path)
        self.assertEqual(loaded.predict_next_state("state", "DOWN"), "new")
        key = loaded._key("state", "DOWN")
        self.assertEqual(loaded.transition_counts[key]["old"], 1)
        self.assertEqual(loaded.transition_counts[key]["new"], 3)
        self.assertEqual(len(loaded.change_point_events), 1)

    def test_probe_adaptation_records_transitions(self):
        agent = DevelopmentalAgent()
        before = agent.transition_observations
        results = run_mechanic_probe(agent, count=4, learn=True, seed=10)
        self.assertEqual(len(results), 4)
        self.assertEqual(agent.transition_observations - before, 4)
        self.assertEqual(len(agent.episodes), 4)


if __name__ == "__main__":
    unittest.main()
