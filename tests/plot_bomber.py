"""Extract native terminal snapshots and plot preliminary training diagnostics."""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / "ocean/bomber"
rows = []
for seed in (42, 43, 44):
    text = (root / f"build/bomber_experiment/train_{seed}.txt").read_text()
    for block in text.split("╭"):
        steps = re.search(r"Steps\s+([\d.]+)([KM]?)", block)
        perf = re.search(r"perf\s+([\d.]+)", block)
        death = re.search(r"death_rate\s+([\d.]+)", block)
        timeout = re.search(r"timeout_rate\s+([\d.]+)", block)
        if not all((steps, perf, death, timeout)):
            continue
        n = float(steps[1]) * {"": 1, "K": 1000, "M": 1000000}[steps[2]]
        row = (seed, n, float(perf[1]), float(death[1]), float(timeout[1]))
        if not rows or rows[-1] != row:
            rows.append(row)
with (out / "training_curves.csv").open("w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(("seed", "steps", "success", "death", "timeout"))
    writer.writerows(rows)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for seed in (42, 43, 44):
    points = [r for r in rows if r[0] == seed]
    for index, ax in enumerate(axes):
        ax.plot([r[1] / 1e6 for r in points], [r[index + 2] for r in points], label=str(seed))
for title, ax in zip(("Clearance", "Death", "Timeout"), axes):
    ax.set_title(title)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Episode fraction")
    ax.grid(alpha=0.2)
axes[-1].legend(title="Seed")
fig.suptitle("Bomber: preliminary training snapshots (rounded console metrics)")
fig.tight_layout()
fig.savefig(out / "training_curves.png", dpi=160)
