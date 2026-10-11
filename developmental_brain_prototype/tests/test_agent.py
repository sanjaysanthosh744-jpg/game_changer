import random, tempfile, unittest
from pathlib import Path
from brain.agent import DevelopmentalAgent
from brain.environment import TinyWorld

class AgentTests(unittest.TestCase):
    def test_learns_observed_transition(self):
        a=DevelopmentalAgent()
        a.observe("s0","RIGHT","s1",None,0,0,0.0,{"event":"move"})
        self.assertEqual(a.predict_next_state("s0","RIGHT"),"s1")
    def test_persistence_roundtrip(self):
        a=DevelopmentalAgent()
        a.observe("s0","WAIT","s0",None,0,0,0.0,{"event":"wait"})
        a.end_episode()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"state.json"
            a.save(p)
            b=DevelopmentalAgent.load(p)
            self.assertEqual(b.predict_next_state("s0","WAIT"),"s0")
            self.assertEqual(b.transition_observations,1)
    def test_environment(self):
        env=TinyWorld(seed=1)
        obs=env.reset()
        self.assertEqual((obs["x"],obs["y"]),(0,0))
        nxt,reward,done,info=env.step("RIGHT")
        self.assertEqual(nxt["x"],1)
        self.assertFalse(done)
    def test_action_valid(self):
        a=DevelopmentalAgent(); env=TinyWorld()
        state=env.encode_observation(env.reset())
        self.assertIn(a.choose_action(state,env.actions,random.Random(1)),env.actions)

if __name__ == "__main__":
    unittest.main()
