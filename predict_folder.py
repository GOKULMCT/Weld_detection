#!/usr/bin/env python3
"""
predict_folder.py - classify every image in a folder (batch inference) and save the results as CSV.

Usage
    python predict_folder.py D:\\photos\\new_pipe
    python predict_folder.py D:\\photos --recursive              # include sub-folders
    python predict_folder.py D:\\photos\\new_pipe --threshold 0.8 --batch 32
    python predict_folder.py dataset\\weld_defect_dataset\\review_required --out results\\review_predictions.csv

CSV columns: image, predicted_class, confidence, prob_<class> for every class, needs_manual_review, note.
Images that cannot be read are listed with an ERROR note and do not stop the run.

The confidence is the softmax output of the network - it is NOT a calibrated probability of a defect.
Predictions only classify the whole image; they do not locate a defect.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from weld_common import (CONFIDENCE_NOTE, MODELS_DIR, RESULTS_DIR, WeldClassifier, describe_device, find_images,
                         prediction_fields, result_to_row, write_csv)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Classify all images in a folder.")
    ap.add_argument("folder", help="folder with weld photos")
    ap.add_argument("--recursive", action="store_true", help="also look in sub-folders")
    ap.add_argument("--weights", default=str(MODELS_DIR / "best.pt"), help="model file (default %(default)s)")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="confidence below this -> needs_manual_review (default %(default)s)")
    ap.add_argument("--batch", type=int, default=16, help="images per inference batch (default %(default)s)")
    ap.add_argument("--imgsz", type=int, default=None, help="input size (default: the size the model was trained at)")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0")
    ap.add_argument("--out", default=str(RESULTS_DIR / "predictions_folder.csv"), help="CSV file to write")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit("ERROR: %s is not a folder. For single images use predict_image.py." % folder)
    paths = find_images(folder, args.recursive)
    if not paths:
        sys.exit("ERROR: no images (.jpg .jpeg .png .bmp .tif .webp) found in %s%s" % (
            folder, "" if args.recursive else "  (use --recursive to include sub-folders)"))

    clf = WeldClassifier(args.weights, args.imgsz, args.device)
    print("Model: %s | input %d | %s" % (args.weights, clf.imgsz, describe_device(clf.device)))
    print("Classifying %d images from %s ...\n" % (len(paths), folder))
    results = clf.predict_paths([str(p) for p in paths], batch_size=args.batch)
    by_path = {r["path"]: r for r in results}

    rows, counts, flagged, errors = [], Counter(), 0, 0
    for p in paths:
        r = by_path[str(p)]
        row = result_to_row(r, args.threshold)
        row["image"] = str(p.relative_to(folder)) if args.recursive else p.name
        rows.append(row)
        if "error" in r:
            errors += 1
            print("  ERROR", p.name, "-", r["error"])
            continue
        counts[r["predicted_class"]] += 1
        flagged += r["confidence"] < args.threshold
    write_csv(args.out, rows, prediction_fields(clf.names))

    print("Predicted classes:")
    for name in clf.names:
        print("  %-14s %5d" % (name, counts[name]))
    print("  flagged for manual review (confidence < %.2f): %d | unreadable: %d" % (args.threshold, flagged, errors))
    print("\nSaved:", args.out)
    print(CONFIDENCE_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
