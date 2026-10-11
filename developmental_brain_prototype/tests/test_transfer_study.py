import random
import unittest
from brain.agent import DevelopmentalAgent
from brain.environment import TinyWorld
from main import run_episode, run_transfer_study
from argparse import Namespace
from pathlib import Path
import tempfile

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
                             max_steps=20, seed=7,
                             adapted_state_file=str(Path(tmp)/'adapted.json'))
            result = run_transfer_study(agent, args)
            self.assertEqual(result['zero_shot']['episodes'], 4)
            self.assertEqual(result['adaptation']['episodes'], 4)
            self.assertEqual(result['retention']['episodes'], 4)
            self.assertTrue(Path(result['adapted_state_file']).exists())

if __name__ == '__main__':
    unittest.main()
