import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

model_name = "GPT2-Small-Distilled-100M-dft"
model_dir = Path("../models") / Path(model_name)
checkpoint_dirs = [
    "checkpoint-1998",
    "checkpoint-3996",
    "checkpoint-5994",
    "checkpoint-7992",
    "checkpoint-9990",
    "checkpoint-11988",
    "checkpoint-13986",
    "checkpoint-15984",
    "checkpoint-17982",
    "checkpoint-19980"
]

train_steps = []
train_losses = []

eval_steps = []
eval_losses = []

max_step_seen = 0 

for ckpt in checkpoint_dirs:
    state_path = os.path.join(model_dir / Path(ckpt), "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"Warning: {state_path} not found, skipping.")
        continue

    with open(state_path, "r") as f:
        state = json.load(f)

    entries = state.get("log_history", [])
    entries_sorted = sorted(entries, key=lambda e: e.get("step", 0))

    for entry in entries_sorted:
        step = entry.get("step", None)
        
        if step is None:
            continue


        if step <= max_step_seen:
            continue

        # training loss
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])

        # eval loss
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

        if step > max_step_seen:
            max_step_seen = step

# plot learning curve
plt.figure(figsize=(11, 6))
plt.plot(train_steps, train_losses, label="Training Loss", linewidth=1)
plt.plot(eval_steps, eval_losses, label="Validation Loss", marker='o', linestyle='-', linewidth=2)

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# save figure
plt.savefig(f"{model_name}.jpg")
