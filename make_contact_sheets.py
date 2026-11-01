
import cv2
import math
from pathlib import Path

SOURCE = Path(
    r"E:\weld_project\dataset\weld_defect_dataset_4class_working\review_required"
)

OUTPUT = Path(
    r"E:\weld_project\results\review_contact_sheets"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("*.jpg"))

THUMB_W = 320
THUMB_H = 200
COLS = 3
ROWS = 4
PER_SHEET = COLS * ROWS

print("Total images:", len(files))

for sheet_no in range(math.ceil(len(files) / PER_SHEET)):
    batch = files[sheet_no * PER_SHEET:(sheet_no + 1) * PER_SHEET]

    sheet = 255 * (cv2.UMat(THUMB_H * ROWS, THUMB_W * COLS, cv2.CV_8UC3).get() * 0 + 1)
    sheet = sheet.astype("uint8")

    for i, path in enumerate(batch):
        img = cv2.imread(str(path))
        if img is None:
            continue

        thumb = cv2.resize(img, (THUMB_W, THUMB_H))

        x = (i % COLS) * THUMB_W
        y = (i // COLS) * THUMB_H

        sheet[y:y+THUMB_H, x:x+THUMB_W] = thumb

        cv2.putText(
            sheet, path.name, (x+8, y+20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 255), 2
        )

    output_file = OUTPUT / f"review_sheet_{sheet_no+1:02d}.jpg"
    cv2.imwrite(str(output_file), sheet)
    print("Created:", output_file)

print("Contact sheets completed.")
