#!/usr/bin/env python3
"""
evaluate.py - measure a trained model (best.pt) on the validation weld and on the untouched TEST weld.

For every split it saves
    results\\confusion_matrix.png            (test split)     results\\confusion_matrix_val.png (validation split)
    results\\predictions_<split>.csv         one row per image: true class, predicted class, confidence, all class
                                              probabilities, correct?, flagged for manual review?
    results\\classification_report.csv       precision / recall / F1 / support per class + accuracy,
                                              balanced accuracy, macro-F1 (both splits)
    results\\evaluation_report.md            the same in readable form, with the limitations written out

A class that has NO images in a split (validation has no Burn_Through, test has no Spatter) is reported as
"n/a - not measurable", never as 0.0, and it is left out of the averages.

Usage
    python evaluate.py
    python evaluate.py --weights models\\best.pt --threshold 0.70
    python evaluate.py --splits train,val,test      # (training-set scores only show how well it memorised)
    python evaluate.py --predict-review             # also predict the review_required images (no scores)
"""
import argparse
import datetime
import sys
from pathlib import Path

from weld_common import (CLASSES, CONFIDENCE_NOTE, DEFAULT_DATASET, EVAL_ROW_FIELDS, METRIC_FIELDS, MODELS_DIR,
                         NA, RESULTS_DIR, WeldClassifier, check_dataset_exists, class_counts, confidence_summary,
                         describe_device, evaluate_items, fmt, labelled_items, metrics_rows, plot_confusion_matrix,
                         prediction_fields, read_csv_rows, result_to_row, write_csv)

LIMITATIONS = """\
## How far can these numbers be trusted?

* The dataset contains only **4 independent physical welds** (W1-W4). Photos of one weld taken every 10 degrees
  look very alike, so a split has far fewer *independent* samples than images. Treat every score here as a rough
  indication, not as a guarantee for new pipes.
* **Missing classes:** the validation weld (W3) has no Burn_Through and the test weld (W4) has no Spatter. Those
  cells are *not measurable* and are shown as n/a - they are not zero and not good. The balanced accuracy and
  macro-F1 average only over classes that exist in the split.
* The choice of best.pt was made on the validation weld, which cannot judge Burn_Through. A model that never
  predicts Burn_Through would not have been penalised during that choice.
* "confidence" is the softmax output of the network. It is **not** a calibrated probability of a defect: wrong
  predictions can have high confidence. The review threshold only flags low-confidence images for a human look.
* Group-aware cross-validation (cross_validate.py) uses every weld as test weld once and is the better guide.
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a trained weld classifier.")
    ap.add_argument("--weights", default=str(MODELS_DIR / "best.pt"), help="model file (default %(default)s)")
    ap.add_argument("--data", default=str(DEFAULT_DATASET), help="dataset folder (default %(default)s)")
    ap.add_argument("--splits", default="val,test", help="comma separated: train,val,test (default %(default)s)")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="flag predictions with confidence below this for manual review (default %(default)s)")
    ap.add_argument("--imgsz", type=int, default=None, help="input size (default: the size the model was trained at)")
    ap.add_argument("--batch", type=int, default=16, help="images per inference batch (default %(default)s)")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="where the outputs are saved")
    ap.add_argument("--predict-review", action="store_true",
                    help="also predict the review_required images (saved to predictions_review_required.csv)")
    return ap.parse_args(argv)


def split_welds(items):
    return sorted({i["weld_group"] for i in items})


def md_split_section(name, items, ev, thr):
    m = ev["metrics"]
    cs = confidence_summary(ev, thr)
    lines = ["## %s split  (weld %s, %d images)" % (name, "+".join(split_welds(items)), len(items)), ""]
    lines.append("| class | images | precision | recall | F1 | predicted as this class |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for i, c in enumerate(CLASSES):
        if m["support"][i] > 0:
            lines.append("| %s | %d | %s | %s | %s | %d |" % (c, m["support"][i], fmt(m["precision"][i]),
                                                             fmt(m["recall"][i]), fmt(m["f1"][i]), m["n_predicted"][i]))
        else:
            lines.append("| %s | 0 | n/a | n/a | n/a | %d |" % (c, m["n_predicted"][i]))
    lines += ["",
              "* accuracy: **%s**   balanced accuracy: **%s**   macro-F1: **%s**   (over %s)" % (
                  fmt(m["accuracy"]), fmt(m["balanced_accuracy"]), fmt(m["macro_f1"]), ", ".join(m["classes_present"]))]
    if m["classes_absent"]:
        lines.append("* **NOT MEASURABLE here: %s** (no such images in this split). %d image(s) were predicted as a class "
                     "that does not exist in this split - those are errors." % (", ".join(m["classes_absent"]),
                                                                              m["n_predicted_as_absent_class"]))
    lines.append("* confidence < %.2f -> flagged for manual review: %d of %d images; accuracy of flagged images %s, "
                 "of not-flagged images %s; wrong but NOT flagged: %d" % (
                     thr, cs["n_flagged"], cs["n"], fmt(cs["acc_flagged"]), fmt(cs["acc_not_flagged"]),
                     cs["wrong_but_not_flagged"]))
    lines.append("* mean confidence of correct predictions %s, of wrong predictions %s" % (
        fmt(cs["mean_conf_correct"]), fmt(cs["mean_conf_wrong"])))
    return lines


def main(argv=None):
    args = parse_args(argv)
    check_dataset_exists(args.data)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    clf = WeldClassifier(args.weights, args.imgsz, args.device)
    if sorted(clf.names) != sorted(CLASSES):
        sys.exit("ERROR: the model has classes %s but the dataset has %s." % (clf.names, CLASSES))
    items = labelled_items(args.data)
    print("Model  :", args.weights, "| input size", clf.imgsz, "| device", describe_device(clf.device))
    print(CONFIDENCE_NOTE)

    report_rows, md = [], ["# Evaluation report", "",
                           "* model: `%s`  (input %d x %d)" % (args.weights, clf.imgsz, clf.imgsz),
                           "* evaluated: %s on %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                      describe_device(clf.device)),
                           "* review threshold: confidence < %.2f" % args.threshold, ""]
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        its = [i for i in items if i["split"] == sp]
        if not its:
            print("\n[%s] no images - skipped" % sp)
            continue
        ev = evaluate_items(clf, its, args.threshold, args.batch)
        m = ev["metrics"]
        write_csv(out / ("predictions_%s.csv" % sp), ev["rows"], EVAL_ROW_FIELDS)
        title = "Confusion matrix - %s split (weld %s, %d images)" % (sp, "+".join(split_welds(its)), len(its))
        plot_confusion_matrix(m["confusion_matrix"], out / ("confusion_matrix_%s.png" % sp), title)
        if sp == "test":
            plot_confusion_matrix(m["confusion_matrix"], out / "confusion_matrix.png", title)
        report_rows += metrics_rows(m, sp)
        md += md_split_section(sp, its, ev, args.threshold) + [""]

        print("\n[%s] weld %s, %d images, true counts %s" % (sp, "+".join(split_welds(its)), len(its),
                                                         dict(zip(CLASSES, class_counts(its)))))
        print("   accuracy %s | balanced accuracy %s | macro-F1 %s  (classes present: %s)" % (
            fmt(m["accuracy"]), fmt(m["balanced_accuracy"]), fmt(m["macro_f1"]), ", ".join(m["classes_present"])))
        print("   %-14s %9s %9s %9s %8s" % ("class", "precision", "recall", "F1", "images"))
        for i, c in enumerate(CLASSES):
            if m["support"][i] > 0:
                print("   %-14s %9s %9s %9s %8d" % (c, fmt(m["precision"][i]), fmt(m["recall"][i]), fmt(m["f1"][i]), m["support"][i]))
            else:
                print("   %-14s %9s %9s %9s %8d   <- NOT MEASURABLE (no %s images in this split)" % (c, NA, NA, NA, 0, c))
        print("   confusion matrix (rows = true, cols = predicted; order %s):" % CLASSES)
        for i, c in enumerate(CLASSES):
            print("     %-14s %s" % (c, list(map(int, m["confusion_matrix"][i]))))

    if not report_rows:
        sys.exit("ERROR: nothing was evaluated (check --splits).")
    write_csv(out / "classification_report.csv", report_rows, METRIC_FIELDS)
    md += [LIMITATIONS]
    (out / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")

    if args.predict_review:
        rows = read_csv_rows(Path(args.data) / "review_required_list.csv")
        paths = [str(Path(args.data) / "review_required" / r["image_filename"]) for r in rows]
        res = clf.predict_paths(paths, args.batch)
        by = {r["path"]: r for r in res}
        out_rows = []
        for r, p in zip(rows, paths):
            row = result_to_row(by[p], args.threshold)
            row["image"] = r["image_filename"]
            row["note"] = "check: " + r["what_to_check"]
            out_rows.append(row)
        write_csv(out / "predictions_review_required.csv", out_rows, prediction_fields(clf.names))
        print("\nreview_required predictions saved (these images have NO trusted label, so no score is computed).")

    print("\nSaved in %s: classification_report.csv, evaluation_report.md, predictions_*.csv, confusion_matrix*.png" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
