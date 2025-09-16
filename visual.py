import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and filter data
df = pd.read_csv("depth_summary.csv")
df = df[df["Accuracy"] > 0]

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
round_colors = {1: 'blue', 2: 'green'}
round_labels = {1: 'Training Data', 2: 'Sample'}
metric_markers = {'Accuracy': 'o', 'Precision': 's', 'Recall': '^', 'F1': 'D'}

plt.figure(figsize=(14, 8))

for i, metric in enumerate(metrics):
    for rnd in [1, 2]:
        sub = df[df["Round"] == rnd]
        depths = sub["Depth"].values
        values = sub[metric].values

        plt.plot(depths, values, color=round_colors[rnd], alpha=0.4)
        plt.scatter(depths, values, color=round_colors[rnd], marker=metric_markers[metric],
                    label=f"{metric} ({round_labels[rnd]})", s=60)

plt.title("Decision Tree Metrics by Depth and Metric")
plt.xlabel("Tree Depth")
plt.ylabel("Metric (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(sorted(df["Depth"].unique()))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
plt.savefig("graph.png")