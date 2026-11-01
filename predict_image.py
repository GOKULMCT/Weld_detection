#!/usr/bin/env python3
"""
predict_image.py - classify one or more weld photos: Burn_Through, Normal_Weld or Spatter.

The whole photo is classified (image-level). This is a classification model, not an object detector:
it does not say WHERE a defect is.

Usage
    python predict_image.py photo.jpg
    python predict_image.py a.jpg b.jpg c.jpg
    python predict_image.py "D:\\photos\\*.jpg" --threshold 0.8
    python predict_image.py photo.jpg --weights models\\best.pt --out results\\my_predictions.csv

For every image it prints the predicted class, the confidence and all three class probabilities, and
writes a CSV (default results\\predictions_image.csv). Images whose confidence is below --threshold
are marked  needs_manual_review = YES.

The confidence is the softmax output of the network - it is NOT a calibrated probability of a defect.
"""
import argparse
import sys
from pathlib import Path

from weld_common import (CONFIDENCE_NOTE, MODELS_DIR, RESULTS_DIR, WeldClassifier, describe_device,
                         expand_image_args, prediction_fields, result_to_row, write_csv)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Classify weld photo(s).")
    ap.add_argument("images", nargs="+", help="image file(s); wildcards like *.jpg are allowed")
    ap.add_argument("--weights", default=str(MODELS_DIR / "best.pt"), help="model file (default %(default)s)")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="confidence below this -> needs_manual_review (default %(default)s)")
    ap.add_argument("--imgsz", type=int, default=None, help="input size (default: the size the model was trained at)")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0")
    ap.add_argument("--out", default=str(RESULTS_DIR / "predictions_image.csv"), help="CSV file to write")
    args = ap.parse_args(argv)

    paths = expand_image_args(args.images)
    if not paths:
        sys.exit("ERROR: no image matches %s" % args.images)
    folders = [p for p in paths if p.is_dir()]
    if folders:
        sys.exit("ERROR: %s is a folder. Use  python predict_folder.py <folder>  for whole folders." % folders[0])

    clf = WeldClassifier(args.weights, args.imgsz, args.device)
    print("Model: %s | input %d | %s\n" % (args.weights, clf.imgsz, describe_device(clf.device)))
    results = clf.predict_paths([str(p) for p in paths], batch_size=16)
    by_path = {r["path"]: r for r in results}

    rows, n_bad = [], 0
    for p in paths:
        r = by_path[str(p)]
        rows.append(result_to_row(r, args.threshold))
        if "error" in r:
            n_bad += 1
            print("%-40s ERROR: %s" % (p.name, r["error"]))
            continue
        flag = "   <-- LOW CONFIDENCE: check manually" if r["confidence"] < args.threshold else ""
        print("%-40s %-13s %5.1f %%%s" % (p.name, r["predicted_class"], 100 * r["confidence"], flag))
        print("%-40s   %s" % ("", "   ".join("%s %.1f%%" % (n, 100 * v) for n, v in r["probs"].items())))
    write_csv(args.out, rows, prediction_fields(clf.names))
    print("\nSaved:", args.out)
    print(CONFIDENCE_NOTE)
    return 1 if n_bad == len(paths) else 0


if __name__ == "__main__":
    sys.exit(main())
