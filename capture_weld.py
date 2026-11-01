
import cv2
import serial
import time
import argparse
from pathlib import Path

# ---------------- SETTINGS ----------------
SERIAL_PORT = "COM14"
BAUD_RATE = 115200
CAMERA_INDEX = 1

TOTAL_IMAGES = 36
ANGLE_STEP = 10

BASE_DIR = Path(r"E:\weld_project\new_weld_collection")

# ---------------- SERIAL ----------------
def read_response(ser, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("ESP32:", line)
            if line.startswith("ERROR:"):
                raise RuntimeError(line)
            if line == "ANGLE_REACHED":
                return line
    raise TimeoutError("ESP32 response timeout")

def send_command(ser, command, expected, timeout=10):
    ser.reset_input_buffer()
    ser.write((command + "\n").encode())
    ser.flush()

    start = time.time()
    while time.time() - start < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("ESP32:", line)
            if line.startswith("ERROR:"):
                raise RuntimeError(line)
            if line == expected:
                return line

    raise TimeoutError(f"Expected {expected}, command {command}")

# ---------------- MAIN ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--class-name", required=True,
                    choices=["Spatter", "Over_Weld",
                             "Burn_Through", "Normal_Weld"])
parser.add_argument("--weld", required=True,
                    choices=["Weld1", "Weld2", "Weld3",
                             "Weld4", "Weld5"])
args = parser.parse_args()

save_dir = BASE_DIR / args.class_name / args.weld
save_dir.mkdir(parents=True, exist_ok=True)

if any(save_dir.iterdir()):
    raise RuntimeError(
        f"Folder is not empty: {save_dir}\n"
        "Choose an empty Weld folder."
    )

cap = None
ser = None

try:
    print("Connecting ESP32...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.2)
    time.sleep(2)

    ser.reset_input_buffer()
    send_command(ser, "PING", "PONG")

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Camera not detected")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("\nCheck pipe is at its 0-degree mark.")
    input("Press ENTER when ready to set ZERO...")

    send_command(ser, "ZERO", "ZEROED")

    for i in range(TOTAL_IMAGES):
        angle = i * ANGLE_STEP

        if i > 0:
            send_command(ser, "1", "ANGLE_REACHED", timeout=30)

        # Allow camera exposure to stabilize
        time.sleep(0.5)

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Camera capture failed at {angle} degrees")

        filename = save_dir / f"{angle:03d}.jpg"

        if not cv2.imwrite(str(filename), frame):
            raise RuntimeError(f"Could not save {filename}")

        print(f"Saved {filename.name} ({angle} degrees)")

    print("\nCAPTURE COMPLETED: 36 images saved.")
    print("Folder:", save_dir)

except Exception as e:
    print("\nCAPTURE STOPPED:", e)
    print("Check saved images before restarting.")

finally:
    if cap is not None:
        cap.release()
    if ser is not None:
        ser.close()
    cv2.destroyAllWindows()
