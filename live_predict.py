import cv2
from pathlib import Path
from ultralytics import YOLO

# MODEL
MODEL_PATH = r"E:\weld_project\models\trained_4class_v1\best.pt"

# USB CAMERA
CAMERA_INDEX = 1

# SETTINGS
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50

print("Loading YOLOv8 model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print("Starting USB camera...")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera 1 failed. Trying camera 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera not detected!")
    print("Check USB connection or camera index.")
    exit()

print("Camera connected!")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame not received.")
        break

    # YOLO PREDICTION
    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        device=0,
        verbose=False
    )

    result = results[0]

    predicted_class = model.names[int(result.probs.top1)]
    confidence = float(result.probs.top1conf)

    # DISPLAY
    if confidence >= CONFIDENCE_THRESHOLD:
        label = f"{predicted_class} | {confidence*100:.2f}%"
    else:
        label = f"LOW CONFIDENCE | {predicted_class} | {confidence*100:.2f}%"

    cv2.rectangle(frame, (10, 10), (650, 90), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "6G PIPE WELD - YOLOv8 LIVE PREDICTION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        label,
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255),
        2
    )

    cv2.imshow("6G Pipe Weld Defect Detection", frame)

    # PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Live prediction stopped.")
