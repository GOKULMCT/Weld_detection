import sys
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(r"E:\weld_project\models\trained_4class_v1\best.pt")

CLASSES = [
    "Burn_Through",
    "Normal_Weld",
    "Over_Weld",
    "Spatter"
]

if len(sys.argv) < 2:
    print("Usage: python predict_weld.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not MODEL_PATH.exists():
    print("ERROR: Model not found:", MODEL_PATH)
    sys.exit(1)

if not image_path.is_file():
    print("ERROR: Image not found:", image_path)
    sys.exit(1)

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(image_path),
    imgsz=224,
    device=0,
    verbose=False
)

result = results[0]
probs = result.probs

print("\n" + "=" * 45)
print("       6G PIPE WELD DEFECT DETECTION")
print("=" * 45)

predicted = model.names[int(probs.top1)]
confidence = float(probs.top1conf) * 100

print("Image          :", image_path.name)
print("Predicted Class:", predicted)
print("Confidence     : {:.2f}%".format(confidence))

print("\nCLASS PROBABILITIES")
print("-" * 45)

for index in probs.top5:
    class_name = model.names[int(index)]
    score = float(probs.data[int(index)]) * 100
    print("{:<20} {:.2f}%".format(class_name, score))

print("=" * 45)
