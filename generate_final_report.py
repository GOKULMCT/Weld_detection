from pathlib import Path
import csv
from collections import defaultdict
from ultralytics import YOLO

ROOT = Path(r"E:\weld_project")
MODEL = ROOT / "models" / "trained_4class_v1" / "best.pt"
TEST = ROOT / "dataset" / "weld_defect_dataset_4class_working" / "test"
OUT = ROOT / "results" / "final_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL))
classes = ["Burn_Through", "Normal_Weld", "Over_Weld", "Spatter"]

files = sorted(
    p for p in TEST.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

results = model.predict(
    source=[str(p) for p in files],
    imgsz=224,
    device=0,
    verbose=False
)

matrix = {a: {p: 0 for p in classes} for a in classes}
correct = 0
rows = []

for path, result in zip(files, results):
    actual = path.parent.name
    predicted = model.names[int(result.probs.top1)]
    confidence = float(result.probs.top1conf)
    is_correct = actual == predicted

    matrix[actual][predicted] += 1
    correct += int(is_correct)

    rows.append({
        "Image": path.name,
        "Actual": actual,
        "Predicted": predicted,
        "Confidence": round(confidence, 4),
        "Correct": is_correct
    })

with open(OUT / "image_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with open(OUT / "confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Actual / Predicted"] + classes)
    for actual in classes:
        writer.writerow([actual] + [matrix[actual][p] for p in classes])

lines = []
lines.append("YOLOv8 FOUR-CLASS WELD CLASSIFICATION EVALUATION")
lines.append("=" * 55)
lines.append(f"Total test images: {len(files)}")
lines.append(f"Correct predictions: {correct}")
lines.append(f"Incorrect predictions: {len(files)-correct}")
lines.append(f"Observed test accuracy: {correct/len(files)*100:.2f}%")
lines.append("")
lines.append("CLASS-WISE METRICS")
lines.append("-" * 55)

f1_values = []

for c in classes:
    tp = matrix[c][c]
    fp = sum(matrix[a][c] for a in classes if a != c)
    fn = sum(matrix[c][p] for p in classes if p != c)
    support = sum(matrix[c].values())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    lines.append(
        f"{c}: Precision={precision:.4f}, Recall={recall:.4f}, "
        f"F1={f1:.4f}, Support={support}"
    )

    if support:
        f1_values.append(f1)

lines.append("")
lines.append("Macro F1 (classes with test support): " +
             f"{sum(f1_values)/len(f1_values):.4f}")
lines.append("")
lines.append("LIMITATION: Test data contains only the classes with nonzero support.")
lines.append("This evaluation does not establish generalization for all four classes.")

(OUT / "evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("\nFiles saved in:", OUT)
