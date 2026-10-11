import json
from pathlib import Path
path = Path("data/brain_state.json")
if not path.exists():
    print("No saved brain state yet. Run: python main.py --episodes 80")
    raise SystemExit(0)
data = json.loads(path.read_text(encoding="utf-8"))
counts = data.get("transition_counts",{})
episodes = data.get("episodes",[])
print("=== Saved Developmental Brain ===")
print("Format version:         ",data.get("version"))
print("State-action models:    ",len(counts))
print("Transition observations:",data.get("transition_observations",0))
print("Stored episodes:        ",len(episodes))
total, correct = data.get("total_predictions",0), data.get("correct_predictions",0)
print("Online prediction rate: ",f"{correct/total:.1%}" if total else "not available")
print("\nSample learned transitions:")
for key,outcomes in list(counts.items())[:12]:
    print(" ",key,"=>",outcomes)
