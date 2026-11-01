import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(r"E:\weld_project\results\final_evaluation")
CSV = ROOT / "image_predictions.csv"

classes = ["Burn_Through", "Normal_Weld", "Over_Weld", "Spatter"]
matrix = [[0 for _ in classes] for _ in classes]

with open(CSV, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        actual = row["Actual"]
        predicted = row["Predicted"]
        matrix[classes.index(actual)][classes.index(predicted)] += 1

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(matrix, cmap="Blues")

ax.set_xticks(range(4), classes, rotation=25, ha="right")
ax.set_yticks(range(4), classes)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("YOLOv8 Weld Defect Classification - Confusion Matrix")

for i in range(4):
    for j in range(4):
        ax.text(j, i, str(matrix[i][j]), ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(ROOT / "confusion_matrix.png", dpi=300)
plt.close(fig)

print("Confusion matrix saved successfully.")
print(ROOT / "confusion_matrix.png")
