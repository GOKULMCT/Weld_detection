#!/usr/bin/env python3
"""
predict_camera.py - live USB-camera classification of the weld: Burn_Through / Normal_Weld / Spatter.

IMPORTANT: this is an image CLASSIFICATION model. The window shows ONE class for the whole picture, with the
confidence of that class. It does not find or outline defects, so no bounding boxes are drawn.

Keys in the window
    Q or ESC   quit
    S          save the current picture (annotated + original) into --save-dir and add a line to the CSV

Usage
    python predict_camera.py                       # camera index 1 (default), models\\best.pt
    python predict_camera.py --camera 0
    python predict_camera.py --find-camera         # which camera index gives a picture?
    python predict_camera.py --threshold 0.8 --smooth 5
    python predict_camera.py --save-interval 10    # also save a picture every 10 seconds
    python predict_camera.py --source test.mp4 --no-display --max-frames 100   # test with a video file

The camera is asked for 1920x1080, the picture size the model was trained on (16:9). If your camera gives
another shape a warning is printed - the results are then less reliable.
Confidence = softmax output, NOT a calibrated defect probability. Low confidence is shown as "CHECK MANUALLY".
"""
import os

# Media Foundation (a Windows camera back-end) is very slow to open without this.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse
import collections
import csv
import datetime
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from weld_common import MODELS_DIR, ROOT, WeldClassifier, describe_device

try:   # hide OpenCV's own camera warnings; this script prints clearer messages
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

WINDOW = "weld classification (Q = quit, S = save)"
COLORS = {"Burn_Through": (40, 40, 255), "Spatter": (0, 150, 255), "Normal_Weld": (60, 200, 60)}   # BGR
BLANK_MEAN_MIN, BLANK_STD_MIN = 8.0, 3.0
PROBE_SECONDS = 4.0
WARMUP_FRAMES = 20


def frame_is_blank(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    return bool(mean[0][0] < BLANK_MEAN_MIN or std[0][0] < BLANK_STD_MIN)


def camera_modes():
    """(name, backend, force MJPG). On Windows DirectShow is tried first, then Media Foundation."""
    if sys.platform.startswith("win"):
        backends = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("default", None)]
    else:
        backends = [("default", None)]
    for name, backend in backends:
        for mjpg in (True, False):
            yield name, backend, mjpg


def open_camera(index, width, height, verbose=True):
    """Open camera `index` and return a capture that really delivers non-blank pictures."""
    problems = []
    for name, backend, mjpg in camera_modes():
        label = "%s, %s" % (name, "MJPG" if mjpg else "camera's own format")
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            problems.append((label, "could not be opened"))
            continue
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        deadline, why = time.monotonic() + PROBE_SECONDS, "no frames arrived"
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if not frame_is_blank(frame):
                for _ in range(WARMUP_FRAMES):      # let the exposure settle
                    cap.read()
                if verbose:
                    print("Camera %d opened (%s) at %dx%d" % (index, label, frame.shape[1], frame.shape[0]))
                return cap
            why = "pictures are blank"
        cap.release()
        problems.append((label, why))
    detail = "\n".join("    %s: %s" % p for p in problems)
    raise RuntimeError(
        "Camera %d gives no usable picture:\n%s\n"
        "  Run  python predict_camera.py --find-camera  to test other indexes. Close Zoom/Teams/Windows Camera, "
        "check the lens cover and the Windows camera privacy setting (Settings > Privacy & security > Camera > "
        "'Let desktop apps access your camera')." % (index, detail))


def find_cameras(width, height, max_index=5):
    print("Testing camera indexes 0..%d (takes a little while) ..." % max_index)
    found = False
    for i in range(max_index + 1):
        try:
            cap = open_camera(i, width, height, verbose=False)
        except RuntimeError as e:
            print("  camera %d: not usable (%s)" % (i, str(e).splitlines()[1].strip() if "\n" in str(e) else e))
            continue
        ok, frame = cap.read()
        cap.release()
        if ok:
            found = True
            out = ROOT / ("camera_test_%d.jpg" % i)
            cv2.imwrite(str(out), frame)
            print("  camera %d: OK %dx%d, brightness %.0f -> picture saved as %s" % (
                i, frame.shape[1], frame.shape[0], frame.mean(), out))
    if not found:
        print("No camera gave a usable picture.")
        return 1
    print("Open the saved pictures, pick the one that shows your pipe, then run  python predict_camera.py --camera N")
    return 0


def open_source(source, camera, width, height):
    if source:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError("cannot open video file %s" % source)
        return cap, True
    return open_camera(camera, width, height), False


def annotate(frame, probs, names, fps, threshold, max_width=1280):
    """Return a copy of the frame (shrunk to max_width) with the predicted class, confidence, per-class bars and FPS."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / float(w))
    img = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()
    H, W = img.shape[:2]
    top = int(np.argmax(probs))
    name, conf = names[top], float(probs[top])
    low = conf < threshold
    color = COLORS.get(name, (255, 255, 255))
    fs = max(0.6, W / 1280.0)
    th = max(1, int(round(2 * fs)))

    bar_h = int(78 * fs)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, "%s  %.0f%%" % (name, 100 * conf), (14, int(bar_h * 0.55)), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * fs, color, th + 1, cv2.LINE_AA)
    sub = "CHECK MANUALLY (confidence below %.0f%%)" % (100 * threshold) if low else "whole-image class (no defect location)"
    cv2.putText(img, sub, (16, int(bar_h * 0.88)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * fs, (0, 255, 255) if low else (210, 210, 210), max(1, th - 1), cv2.LINE_AA)
    cv2.putText(img, "FPS %.1f" % fps, (W - int(150 * fs), int(bar_h * 0.55)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * fs, (255, 255, 255), th, cv2.LINE_AA)

    # per-class probability bars (bottom left)
    bw, bh, gap = int(260 * fs), int(18 * fs), int(8 * fs)
    y0 = H - 14 - len(names) * (bh + gap)
    for i, n in enumerate(names):
        y = y0 + i * (bh + gap)
        cv2.rectangle(img, (14, y), (14 + bw, y + bh), (40, 40, 40), -1)
        cv2.rectangle(img, (14, y), (14 + int(bw * float(probs[i])), y + bh), COLORS.get(n, (200, 200, 200)), -1)
        cv2.putText(img, "%s %.0f%%" % (n, 100 * float(probs[i])), (14 + bw + 8, y + bh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs, (255, 255, 255), 1, cv2.LINE_AA)
    if low:
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 255, 255), max(2, th * 2))
    return img


def save_frame(raw, shown, name, conf, probs, names, save_dir, csv_path):
    """Save the original frame and the annotated picture; append a line to the CSV. Returns the annotated file name."""
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = "%s_%s_%02d" % (stamp, name, int(round(100 * conf)))
    p_annot, p_raw = save_dir / (base + ".jpg"), save_dir / (base + "_raw.jpg")
    cv2.imwrite(str(p_annot), shown, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(p_raw), raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["time", "annotated_file", "raw_file", "predicted_class", "confidence"] + ["prob_" + n for n in names])
        wr.writerow([datetime.datetime.now().isoformat(timespec="seconds"), p_annot.name, p_raw.name, name,
                     "%.4f" % conf] + ["%.4f" % float(p) for p in probs])
    return p_annot


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live weld classification from a USB camera.")
    ap.add_argument("--camera", type=int, default=1, help="camera index (default %(default)s)")
    ap.add_argument("--find-camera", action="store_true", help="test camera indexes 0-5 and save one picture of each")
    ap.add_argument("--weights", default=str(MODELS_DIR / "best.pt"), help="model file (default %(default)s)")
    ap.add_argument("--threshold", type=float, default=0.70, help="confidence below this -> CHECK MANUALLY (default %(default)s)")
    ap.add_argument("--width", type=int, default=1920, help="requested camera width (default %(default)s)")
    ap.add_argument("--height", type=int, default=1080, help="requested camera height (default %(default)s)")
    ap.add_argument("--imgsz", type=int, default=None, help="model input size (default: the size it was trained at)")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0")
    ap.add_argument("--smooth", type=int, default=1, help="average the probabilities of the last N frames (default 1 = off)")
    ap.add_argument("--save-dir", default=str(ROOT / "predicted_frames"), help="where S saves pictures (default %(default)s)")
    ap.add_argument("--save-interval", type=float, default=0.0, help="also save a picture every N seconds (0 = off)")
    ap.add_argument("--source", default=None, help="use a video file instead of the camera (for testing)")
    ap.add_argument("--no-display", action="store_true", help="no window; print the predictions (for testing)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until Q)")
    args = ap.parse_args(argv)

    if args.find_camera:
        return find_cameras(args.width, args.height)

    clf = WeldClassifier(args.weights, args.imgsz, args.device)
    print("Model: %s | input %d | %s" % (args.weights, clf.imgsz, describe_device(clf.device)))
    try:
        cap, is_file = open_source(args.source, args.camera, args.width, args.height)
    except RuntimeError as e:
        print("\nERROR:", e, file=sys.stderr)
        return 1

    save_dir = Path(args.save_dir)
    csv_path = save_dir / "predictions_camera.csv"
    history = collections.deque(maxlen=max(1, args.smooth))
    fps, stamps, last_autosave, n, warned = 0.0, collections.deque(maxlen=30), time.monotonic(), 0, False
    print("Running. Press Q (or ESC) in the video window to quit, S to save a picture." if not args.no_display else "Running without a window.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file:
                    print("End of video.")
                    break
                print("The camera stopped delivering pictures.", file=sys.stderr)
                return 1
            if not warned and abs(frame.shape[1] / frame.shape[0] - 16 / 9) > 0.05:
                print("WARNING: the picture is %dx%d, not 16:9. The model was trained on 1920x1080 photos; the "
                      "result may be less reliable." % (frame.shape[1], frame.shape[0]))
                warned = True

            history.append(clf.predict_bgr([frame])[0])
            probs = np.mean(history, axis=0)
            stamps.append(time.perf_counter())          # FPS = frames per second over the last ~30 frames
            if len(stamps) >= 2 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            n += 1
            top = int(np.argmax(probs))
            shown = annotate(frame, probs, clf.names, fps, args.threshold)

            if args.save_interval > 0 and time.monotonic() - last_autosave >= args.save_interval:
                last_autosave = time.monotonic()
                print("saved", save_frame(frame, shown, clf.names[top], probs[top], probs, clf.names, save_dir, csv_path))

            if args.no_display:
                print("frame %4d  %-13s %5.1f %%  (%.1f FPS)%s" % (n, clf.names[top], 100 * probs[top], fps,
                                                                  "  CHECK MANUALLY" if probs[top] < args.threshold else ""))
            else:
                cv2.imshow(WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    print("saved", save_frame(frame, shown, clf.names[top], probs[top], probs, clf.names, save_dir, csv_path))
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print("Done: %d frames, last FPS %.1f" % (n, fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
