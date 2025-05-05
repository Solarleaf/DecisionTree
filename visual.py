import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and filter data
df = pd.read_csv("depth_summary.csv")
df = df[df["Accuracy"] > 0]

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
colors = ['blue', 'green', 'orange', 'red']
marker_shapes = {1: 'o', 2: 's', 3: 'D'}  # circle, square, diamond

plt.figure(figsize=(14, 8))

for i, metric in enumerate(metrics):
    for rnd in [1, 2, 3]:
        sub = df[df["Round"] == rnd]
        depths = sub["Depth"].values
        values = sub[metric].values

        if rnd < 3:
            plt.plot(depths, values, color=colors[i], alpha=0.4)
            plt.scatter(depths, values, color=colors[i], marker=marker_shapes[rnd],
                        label=f"{metric} (Round {rnd})", s=60)
        else:
            new_depths = []
            new_values = []
            for d, v in zip(depths, values):
                new_depths.extend([d + 0.0, d + 0.33, d + 0.66])
                new_values.extend([v, v, v])

            plt.plot(new_depths, new_values, color=colors[i], alpha=0.4)
            plt.scatter(new_depths, new_values, edgecolors=colors[i], facecolors='none',
                        marker=marker_shapes[rnd], label=f"{metric} (Round 3)", s=60)

plt.title("Decision Tree Metrics by Depth and Round")
plt.xlabel("Tree Depth")
plt.ylabel("Metric (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(sorted(df["Depth"].unique()))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
