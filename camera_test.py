
import cv2

CAMERA_INDEX = 1  # First test camera 1

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Camera not detected")
    exit()

print("Press Q to close preview")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame capture failed")
        break

    cv2.imshow("Weld Camera Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
