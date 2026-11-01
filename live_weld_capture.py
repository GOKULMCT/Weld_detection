
import cv2
import csv
import time
import serial
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

PORT = "COM10"
BAUD = 115200
CAMERA_INDEX = 1

MODEL_PATH = r"E:\weld_project\models\trained_4class_v1\best.pt"
ROOT = Path(r"E:\weld_project\live_weld_capture")

CONFIDENCE_THRESHOLD = 70.0

SESSION = ROOT / ("session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
RAW = SESSION / "raw_images"
PRED = SESSION / "predictions"

RAW.mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "6G PIPE INSPECTION - YOLOv8"


# ==========================================
# LCD COMMUNICATION
# ==========================================

def lcd_send(ser, line1, line2=""):

    message = f"LCD:{line1}|{line2}\n"

    ser.write(message.encode())
    ser.flush()

    deadline = time.time() + 2

    while time.time() < deadline:

        response = ser.readline().decode(
            errors="ignore"
        ).strip()

        if response == "LCD_UPDATED":
            return

    print("Warning: LCD acknowledgement not received.")


def lcd_status(ser, line1, line2=""):
    lcd_send(ser, line1, line2)


# ==========================================
# ESP32 RESPONSE
# ==========================================

def wait_reply(ser, expected, timeout=30):

    start = time.time()

    while time.time() - start < timeout:

        line = ser.readline().decode(
            errors="ignore"
        ).strip()

        if line:
            print("ESP32:", line)

        if expected in line:
            return

    raise TimeoutError(
        "ESP32 response missing: " + expected
    )


# ==========================================
# PREDICTION DISPLAY NAME
# ==========================================

def short_class_name(name):

    names = {
        "Burn_Through": "BURN THRU",
        "Normal_Weld": "NORMAL",
        "Over_Weld": "OVER WELD",
        "Spatter": "SPATTER"
    }

    return names.get(name, name.upper())


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    camera = None
    ser = None

    counts = {
        "Normal_Weld": 0,
        "Burn_Through": 0,
        "Spatter": 0,
        "Over_Weld": 0
    }

    review_count = 0
    total_images = 0

    try:

        print("\n====================================")
        print("          6G PIPE INSPECTION")
        print("====================================")

        # ----------------------------------
        # STEP 1: ESP32 CONNECTION
        # ----------------------------------

        print("\nSTEP 1: Connecting ESP32...")

        ser = serial.Serial(
            PORT,
            BAUD,
            timeout=1,
            write_timeout=3
        )

        time.sleep(3)
        ser.reset_input_buffer()

        ser.write(b"PING\n")
        ser.flush()

        wait_reply(ser, "PONG", 10)

        print("ESP32 connected successfully.")

        lcd_status(
            ser,
            "6G PIPE INSPECT",
            "PROGRAM STARTED"
        )

        time.sleep(1)

        # ----------------------------------
        # STEP 2: LOAD YOLO MODEL
        # ----------------------------------

        print("\nSTEP 2: Loading YOLOv8 model...")

        lcd_status(
            ser,
            "LOADING YOLOV8",
            "PLEASE WAIT"
        )

        model = YOLO(MODEL_PATH)

        print("YOLOv8 model loaded successfully.")

        lcd_status(
            ser,
            "YOLO MODEL",
            "LOADED SUCCESS"
        )

        time.sleep(1)

        # ----------------------------------
        # STEP 3: CAMERA CONNECTION
        # ----------------------------------

        print("\nSTEP 3: Connecting USB camera...")

        lcd_status(
            ser,
            "CONNECTING",
            "USB CAMERA"
        )

        camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():

            camera.release()

            camera = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW
            )

        if not camera.isOpened():
            raise RuntimeError(
                "USB camera not detected"
            )

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        print("Camera connected successfully.")

        lcd_status(
            ser,
            "CAMERA",
            "CONNECTED"
        )

        time.sleep(1)

        # ----------------------------------
        # STEP 4: SYSTEM READY
        # ----------------------------------

        print("\nSTEP 4: System ready.")

        lcd_status(
            ser,
            "SYSTEM READY",
            "PRESS ENTER"
        )

        input(
            "\nPRESS ENTER TO START INSPECTION: "
        )

        # ----------------------------------
        # STEP 5: SET ZERO
        # ----------------------------------

        print("\nSetting pipe position to ZERO...")

        lcd_status(
            ser,
            "SETTING ZERO",
            "PLEASE WAIT"
        )

        ser.reset_input_buffer()

        ser.write(b"ZERO\n")
        ser.flush()

        wait_reply(
            ser,
            "ZEROED",
            5
        )

        print("Pipe zero position set successfully.")

        csv_path = SESSION / "predictions.csv"

        # ==================================
        # IMAGE CAPTURE AND YOLO INSPECTION
        # ==================================

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            fields = [
                "Angle",
                "Image",
                "Predicted_Class",
                "Confidence_Percent",
                "Inspection_Status"
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()

            # 36 IMAGES: 0 TO 350 DEGREES

            for index in range(36):

                angle = index * 10

                # --------------------------
                # MOTOR ROTATION
                # --------------------------

                if index > 0:

                    print(
                        f"\nRotating to {angle} degrees..."
                    )

                    lcd_status(
                        ser,
                        "ROTATING MOTOR",
                        f"ANGLE: {angle} DEG"
                    )

                    ser.reset_input_buffer()

                    ser.write(b"1\n")
                    ser.flush()

                    wait_reply(
                        ser,
                        "ANGLE_REACHED",
                        30
                    )

                # --------------------------
                # IMAGE CAPTURE
                # --------------------------

                time.sleep(0.3)

                for _ in range(3):
                    camera.grab()

                ret, frame = camera.read()

                if not ret:
                    raise RuntimeError(
                        f"Camera capture failed at {angle} degrees"
                    )

                filename = f"{angle:03d}.jpg"

                raw_path = RAW / filename
                pred_path = PRED / filename

                if not cv2.imwrite(
                    str(raw_path),
                    frame
                ):
                    raise RuntimeError(
                        "Image save failed: " + filename
                    )

                total_images += 1

                # --------------------------
                # YOLO PREDICTION
                # --------------------------

                result = model.predict(
                    source=frame,
                    imgsz=224,
                    device=0,
                    verbose=False
                )[0]

                predicted = model.names[
                    int(result.probs.top1)
                ]

                confidence = (
                    float(result.probs.top1conf) * 100
                )

                # --------------------------
                # CONFIDENCE CHECK
                # --------------------------

                if confidence < CONFIDENCE_THRESHOLD:

                    status = "REVIEW_REQUIRED"
                    review_count += 1

                else:

                    status = "ACCEPTED"

                    if predicted in counts:
                        counts[predicted] += 1

                # --------------------------
                # TERMINAL OUTPUT
                # --------------------------

                print("\n------------------------------------")
                print(f"ANGLE       : {angle} DEG")
                print(f"IMAGE       : {index + 1}/36")
                print(f"CLASS       : {predicted}")
                print(f"CONFIDENCE  : {confidence:.2f}%")
                print(f"STATUS      : {status}")
                print("------------------------------------")

                # --------------------------
                # LCD SCREEN 1: ANGLE
                # --------------------------

                lcd_status(
                    ser,
                    f"ANGLE:{angle:03d} DEG",
                    f"IMAGE:{index + 1:02d}/36"
                )

                # --------------------------
                # CAMERA PREVIEW
                # --------------------------

                display = frame.copy()

                cv2.rectangle(
                    display,
                    (10, 10),
                    (750, 100),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    display,
                    f"ANGLE: {angle} DEG",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"{predicted} | {confidence:.2f}%",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display
                )

                cv2.waitKey(1)

                # --------------------------
                # LCD SCREEN 2: PREDICTION
                # --------------------------

                class_text = short_class_name(
                    predicted
                )

                time.sleep(0.8)

                if status == "REVIEW_REQUIRED":

                    lcd_status(
                        ser,
                        "REVIEW REQUIRED",
                        f"CONF:{confidence:.1f}%"
                    )

                else:

                    lcd_status(
                        ser,
                        "CLASS:" + class_text,
                        f"CONF:{confidence:.1f}%"
                    )

                time.sleep(1.2)

                # --------------------------
                # LCD SCREEN 3: COUNTS
                # --------------------------

                lcd_status(
                    ser,
                    (
                        f"BT:{counts['Burn_Through']:02d} "
                        f"SP:{counts['Spatter']:02d} "
                        f"OW:{counts['Over_Weld']:02d}"
                    ),
                    (
                        f"NW:{counts['Normal_Weld']:02d} "
                        f"REV:{review_count:02d}"
                    )
                )

                time.sleep(0.8)

                # --------------------------
                # SAVE PREDICTION IMAGE
                # --------------------------

                cv2.imwrite(
                    str(pred_path),
                    display
                )

                # --------------------------
                # SAVE CSV
                # --------------------------

                writer.writerow({
                    "Angle": angle,
                    "Image": filename,
                    "Predicted_Class": predicted,
                    "Confidence_Percent": round(
                        confidence, 2
                    ),
                    "Inspection_Status": status
                })

                f.flush()

                print(
                    f"Saved: {filename} | Status: {status}"
                )

        # ==================================
        # FINAL 360 DEGREE ROTATION
        # ==================================

        print(
            "\nCompleting final 10-degree rotation..."
        )

        lcd_status(
            ser,
            "FINAL ROTATION",
            "PLEASE WAIT"
        )

        ser.reset_input_buffer()

        ser.write(b"1\n")
        ser.flush()

        wait_reply(
            ser,
            "ANGLE_REACHED",
            30
        )

        # ==================================
        # FINAL INSPECTION REPORT
        # ==================================

        total_defects = (
            counts["Burn_Through"]
            + counts["Spatter"]
            + counts["Over_Weld"]
        )

        if review_count > 0:

            final_result = "REVIEW REQUIRED"

        elif total_defects > 0:

            final_result = "DEFECT DETECTED"

        else:

            final_result = "NO DEFECT DETECTED"

        print("\n")
        print("====================================")
        print("       FINAL INSPECTION REPORT")
        print("====================================")

        print("Total Images       :", total_images)
        print("Normal Weld        :", counts["Normal_Weld"])
        print("Burn Through       :", counts["Burn_Through"])
        print("Spatter            :", counts["Spatter"])
        print("Over Weld          :", counts["Over_Weld"])
        print("Total Defects      :", total_defects)
        print("Review Required    :", review_count)

        print("------------------------------------")
        print("FINAL RESULT       :", final_result)
        print("------------------------------------")

        print("Session Folder:", SESSION)
        print("CSV Report:", csv_path)

        print("====================================")

        # ==================================
        # SEND FINAL REPORT TO LCD
        # ==================================

        final_command = (
            f"FINAL:{total_images},"
            f"{counts['Normal_Weld']},"
            f"{counts['Burn_Through']},"
            f"{counts['Spatter']},"
            f"{counts['Over_Weld']},"
            f"{review_count},"
            f"{final_result}\n"
        )

        ser.write(final_command.encode())
        ser.flush()

        print("\nFinal report sent to LCD.")

        wait_reply(
            ser,
            "FINAL_REPORT_STARTED",
            5
        )

        print("LCD final report display started.")

        print("\nINSPECTION COMPLETED SUCCESSFULLY.")

        # ==================================
        # KEEP CAMERA OPEN FOR 30 SECONDS
        # ==================================

        print("\nFinal report displayed.")
        print("Camera window will close in 30 seconds...")

        end_time = time.time() + 30

        while time.time() < end_time:

            cv2.waitKey(50)

            # Allow user to close camera window manually
            try:
                visible = cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE
                )

                if visible < 1:
                    break

            except cv2.error:
                break

        print("\n30 seconds completed.")
        print("Closing inspection program...")

    finally:

        # ==================================
        # CLEANUP
        # ==================================

        if camera is not None:
            camera.release()

        if ser is not None and ser.is_open:
            ser.close()

        cv2.destroyAllWindows()

        print("Camera released.")
        print("Serial connection closed.")
        print("Program closed.")


# ==========================================
# PROGRAM ENTRY
# ==========================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print("\nProgram interrupted by user.")

    except Exception as e:

        print("\nERROR:", e)

        print(
            "Check ESP32, LCD, camera and COM port."
        )
