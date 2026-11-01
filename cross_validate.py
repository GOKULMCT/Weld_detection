#!/usr/bin/env python3
"""
cross_validate.py - leave-one-weld-out cross-validation (group-aware, no data leakage).

The dataset has 4 independent physical welds (W1-W4). Each fold
    * holds out ONE weld as the test weld  (used only after training has finished),
    * picks ONE of the remaining welds as the validation weld (early stopping + choice of best.pt),
    * trains on the other remaining weld(s).
Images of one weld are never split between train / validation / test.

The validation weld is chosen automatically among the training-side welds so that
  1. the training welds still contain all three classes (if possible), then
  2. the validation weld contains as many classes as possible, then
  3. the training set is as large as possible.
Every class that is missing from a fold's training / validation / test weld is reported; scores for a class
without images are "n/a", never 0.0, and are left out of the averages.

Outputs
    results\\cross_validation_results.csv       per fold and class, plus pooled and mean rows
    results\\cross_validation_summary.md        readable summary with the missing-class notes
    results\\cross_validation_confusion_matrix.png   pooled out-of-fold predictions
    results\\cross_validation_scores.png        balanced accuracy per fold
    results\\cv\\fold_<weld>\\                    per-fold predictions, confusion matrix, curves, config
    models\\cv\\fold_<weld>_best.pt              per-fold models

Usage
    python cross_validate.py
    python cross_validate.py --epochs 60 --folds W1,W2
    python cross_validate.py --resume        # skip folds that are already finished
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import train as T
from weld_common import (CLASSES, DEFAULT_DATASET, DEFAULT_IMGSZ, EVAL_ROW_FIELDS, METRIC_FIELDS, MODELS_DIR, NA,
                         RESULTS_DIR, RUNS_DIR, WeldClassifier, _mpl, check_dataset_exists, class_counts,
                         classes_absent, classes_present, compute_metrics, confidence_summary, describe_device,
                         evaluate_items, fmt, labelled_items, metrics_rows, pick_device, plot_confusion_matrix,
                         plot_training_curves, read_csv_rows, write_csv, write_json)


def plan_folds(items):
    """List of dicts: test_weld, val_weld, train_welds (+ the image lists)."""
    welds = sorted({i["weld_group"] for i in items})
    by = {g: [i for i in items if i["weld_group"] == g] for g in welds}
    plans = []
    for test_w in welds:
        rest = [g for g in welds if g != test_w]
        best = None
        for val_w in rest:
            train_ws = [g for g in rest if g != val_w]
            tr = [i for g in train_ws for i in by[g]]
            va = by[val_w]
            key = (len(classes_present(tr)), len(classes_present(va)), len(tr))
            if best is None or key > best[0]:
                best = (key, val_w, train_ws, tr, va)
        _, val_w, train_ws, tr, va = best
        plans.append(dict(test_weld=test_w, val_weld=val_w, train_welds=train_ws,
                          train=tr, val=va, test=by[test_w]))
    for p in plans:   # hard guarantee: no weld in two roles
        roles = [set(p["train_welds"]), {p["val_weld"]}, {p["test_weld"]}]
        assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]), "weld used in two roles"
    return plans


def fold_status(p):
    return dict(classes_absent_in_train=classes_absent(p["train"]), classes_absent_in_val=classes_absent(p["val"]),
                classes_absent_in_test=classes_absent(p["test"]))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Leave-one-weld-out cross-validation.")
    ap.add_argument("--data", default=str(DEFAULT_DATASET))
    ap.add_argument("--model", default="yolov8n-cls.pt")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imbalance", choices=["sampler", "loss", "none"], default="sampler")
    ap.add_argument("--sampler-power", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.70, help="manual-review confidence threshold")
    ap.add_argument("--folds", default="all", help="all, or comma separated test welds e.g. W1,W3")
    ap.add_argument("--resume", action="store_true", help="skip folds whose results are already saved")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    return ap.parse_args(argv)


def load_saved_fold(fold_dir):
    """Re-read a finished fold from its predictions.csv (used by --resume)."""
    rows = read_csv_rows(fold_dir / "predictions.csv")
    y_true = np.array([CLASSES.index(r["true_class"]) for r in rows])
    y_pred = np.array([CLASSES.index(r["predicted_class"]) for r in rows])
    conf = np.array([float(r["confidence"]) for r in rows])
    return dict(rows=rows, metrics=compute_metrics(y_true, y_pred), y_true=y_true, y_pred=y_pred, conf=conf)


def main(argv=None):
    args = parse_args(argv)
    check_dataset_exists(args.data)
    items = labelled_items(args.data)
    plans = plan_folds(items)
    if args.folds != "all":
        wanted = [w.strip() for w in args.folds.split(",")]
        plans = [p for p in plans if p["test_weld"] in wanted]
        if not plans:
            sys.exit("ERROR: --folds %s matches no weld." % args.folds)
    results_dir, models_dir = Path(args.results_dir), Path(args.models_dir)
    cv_dir = results_dir / "cv"
    (models_dir / "cv").mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print("Device:", describe_device(device))

    print("\nFold plan (leave-one-weld-out; a weld is never in two roles)")
    print("%-6s %-8s %-8s %-12s  %-34s %s" % ("fold", "test", "val", "train", "classes missing (train | val | test)", "images (train/val/test)"))
    for p in plans:
        st = fold_status(p)
        print("%-6s %-8s %-8s %-12s  %-34s %d/%d/%d" % (
            "test=" + p["test_weld"], p["test_weld"], p["val_weld"], "+".join(p["train_welds"]),
            "%s | %s | %s" % (",".join(st["classes_absent_in_train"]) or "-", ",".join(st["classes_absent_in_val"]) or "-",
                              ",".join(st["classes_absent_in_test"]) or "-"), len(p["train"]), len(p["val"]), len(p["test"])))

    cfg = vars(args).copy()
    cfg["run_dir"] = str(RUNS_DIR / "cv")
    fold_out = []
    for p in plans:
        w = p["test_weld"]
        fdir = cv_dir / ("fold_" + w)
        fdir.mkdir(parents=True, exist_ok=True)
        st = fold_status(p)
        meta = dict(test_weld=w, val_weld=p["val_weld"], train_welds="+".join(p["train_welds"]),
                    n_train=len(p["train"]), n_val=len(p["val"]), n_test=len(p["test"]),
                    absent_in_train=",".join(st["classes_absent_in_train"]) or "none",
                    absent_in_val=",".join(st["classes_absent_in_val"]) or "none",
                    absent_in_test=",".join(st["classes_absent_in_test"]) or "none")
        if args.resume and (fdir / "predictions.csv").is_file() and (fdir / "training_config.json").is_file():
            print("\n=== fold test=%s: found saved results, skipping training (--resume) ===" % w)
            ev = load_saved_fold(fdir)
            fold_out.append(dict(meta=meta, ev=ev, trained=False))
            continue

        print("\n" + "=" * 78)
        print("FOLD test=%s   train=%s (%d img)   validation=%s (%d img)   test=%s (%d img)" % (
            w, "+".join(p["train_welds"]), len(p["train"]), p["val_weld"], len(p["val"]), w, len(p["test"])))
        for c in st["classes_absent_in_train"]:
            print("  WARNING: no %s in the TRAINING welds of this fold" % c)
        for c in st["classes_absent_in_val"]:
            print("  note: no %s in the validation weld - early stopping / best.pt cannot judge it" % c)
        for c in st["classes_absent_in_test"]:
            print("  note: no %s in the test weld - it cannot be measured in this fold" % c)
        print("=" * 78)

        res = T.train_model(p["train"], p["val"], "cv_test_" + w, cfg)
        best_copy = models_dir / "cv" / ("fold_%s_best.pt" % w)
        shutil.copy2(res["best"], best_copy)
        shutil.copy2(res["results_csv"], fdir / "training_log.csv")
        plot_training_curves(res["results_csv"], fdir / "loss_graph.png", fdir / "accuracy_graph.png",
                             "(fold test=%s, validation=%s)" % (w, p["val_weld"]))

        # The held-out weld is used for the first time HERE, after training has finished.
        clf = WeldClassifier(best_copy, args.imgsz, args.device)
        ev = evaluate_items(clf, p["test"], args.threshold, 16)
        write_csv(fdir / "predictions.csv", ev["rows"], EVAL_ROW_FIELDS)
        plot_confusion_matrix(ev["metrics"]["confusion_matrix"], fdir / "confusion_matrix.png",
                              "Fold test=%s: held-out weld %s (%d images)" % (w, w, len(p["test"])))
        write_json(fdir / "training_config.json", dict(
            fold=meta, class_counts_train=dict(zip(CLASSES, class_counts(p["train"]))),
            class_counts_val=dict(zip(CLASSES, class_counts(p["val"]))),
            class_counts_test=dict(zip(CLASSES, class_counts(p["test"]))),
            settings={k: v for k, v in cfg.items() if k not in ("data",)},
            batch_used=res["batch_used"], epochs_run=res["epochs_run"], best_epoch=res["best_epoch"],
            seconds=round(res["seconds"], 1), device=res["device"], sampler_weights=res["info"]["sampler_weights"],
            class_loss_weights=res["info"]["class_weights"], environment=T.versions(),
            model_selection="best epoch by validation balanced accuracy on the validation weld only"))
        m = ev["metrics"]
        print("\n  -> held-out weld %s: accuracy %s | balanced accuracy %s | macro-F1 %s (classes present: %s)" % (
            w, fmt(m["accuracy"]), fmt(m["balanced_accuracy"]), fmt(m["macro_f1"]), ", ".join(m["classes_present"])))
        meta.update(epochs_run=res["epochs_run"], best_epoch=res["best_epoch"], minutes=round(res["seconds"] / 60, 1))
        fold_out.append(dict(meta=meta, ev=ev, trained=True))

    write_consolidated(fold_out, results_dir, args.threshold)
    return 0


# --------------------------------------------------------------------------- consolidated reports
def mean_over_folds(values):
    v = [x for x in values if x is not None and not np.isnan(x)]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v)), n=len(v))


def write_consolidated(fold_out, results_dir, threshold):
    rows = []
    for f in fold_out:
        extra = {k: f["meta"][k] for k in ("test_weld", "val_weld", "train_welds", "absent_in_train", "absent_in_val", "absent_in_test")}
        for r in metrics_rows(f["ev"]["metrics"], "fold test=" + f["meta"]["test_weld"], extra=extra):
            r["fold"] = "test=" + f["meta"]["test_weld"]
            rows.append(r)

    # pooled out-of-fold predictions: every image predicted once by a model that never saw its weld
    y_true = np.concatenate([f["ev"]["y_true"] for f in fold_out])
    y_pred = np.concatenate([f["ev"]["y_pred"] for f in fold_out])
    pooled = compute_metrics(y_true, y_pred)
    pool_extra = dict(test_weld="all", val_weld="", train_welds="(different in every fold)", absent_in_train="", absent_in_val="", absent_in_test="")
    for r in metrics_rows(pooled, "POOLED out-of-fold", extra=pool_extra):
        r["fold"] = "pooled"
        rows.append(r)

    # mean over folds, using only folds where the number is measurable
    per_class = {}
    for i, c in enumerate(CLASSES):
        for key in ("precision", "recall", "f1"):
            vals = [f["ev"]["metrics"][key][i] if f["ev"]["metrics"]["support"][i] > 0 else None for f in fold_out]
            per_class[(c, key)] = mean_over_folds(vals)
    overall = {k: mean_over_folds([f["ev"]["metrics"][k] for f in fold_out]) for k in ("accuracy", "balanced_accuracy", "macro_f1")}
    for i, c in enumerate(CLASSES):
        meas = sum(1 for f in fold_out if f["ev"]["metrics"]["support"][i] > 0)
        cells = {}
        for key in ("precision", "recall", "f1"):
            s = per_class[(c, key)]
            cells[key] = ("%.3f (min %.3f, max %.3f; %d folds)" % (s["mean"], s["lo"], s["hi"], s["n"])) if s else NA
        rows.append(dict(fold="mean over folds", split="MEAN over folds (measurable folds only)", class_name=c,
                         row_type="class", support=sum(int(f["ev"]["metrics"]["support"][i]) for f in fold_out),
                         n_predicted="", status="class measurable in %d of %d folds" % (meas, len(fold_out)), **cells))
    for k, s in overall.items():
        rows.append(dict(fold="mean over folds", split="MEAN over folds (measurable folds only)", class_name=k,
                         row_type="overall", precision="", recall="",
                         f1=("%.3f (min %.3f, max %.3f; %d folds)" % (s["mean"], s["lo"], s["hi"], s["n"])) if s else NA,
                         support="", n_predicted="", status="average of the per-fold values"))
    fields = ["fold", "test_weld", "val_weld", "train_welds", "absent_in_train", "absent_in_val", "absent_in_test"] + METRIC_FIELDS
    write_csv(results_dir / "cross_validation_results.csv", rows, fields)

    plot_confusion_matrix(pooled["confusion_matrix"], results_dir / "cross_validation_confusion_matrix.png",
                          "Pooled out-of-fold predictions (%d images, each predicted by a model that never saw its weld)" % len(y_true))
    plot_scores(fold_out, results_dir / "cross_validation_scores.png")

    md = ["# Leave-one-weld-out cross-validation", "",
          "Every fold: one weld is the held-out TEST weld, one other weld is the VALIDATION weld (early stopping, choice of "
          "best.pt), the remaining weld(s) are for TRAINING. Nothing from the test weld was used before the final scoring.", "",
          "| fold | train welds | val weld | test weld | images train/val/test | missing in train | missing in val | missing in test | accuracy | balanced acc. | macro-F1 |",
          "|---|---|---|---|---|---|---|---|---:|---:|---:|"]
    for f in fold_out:
        mt, m = f["meta"], f["ev"]["metrics"]
        md.append("| test=%s | %s | %s | %s | %d/%d/%d | %s | %s | %s | %s | %s | %s |" % (
            mt["test_weld"], mt["train_welds"], mt["val_weld"], mt["test_weld"], mt["n_train"], mt["n_val"], mt["n_test"],
            mt["absent_in_train"], mt["absent_in_val"], mt["absent_in_test"],
            fmt(m["accuracy"]), fmt(m["balanced_accuracy"]), fmt(m["macro_f1"])))
    md += ["", "Balanced accuracy and macro-F1 average only over the classes that exist in the test weld of that fold.", "",
           "## Per-class recall in each fold (n/a = class not present in the test weld, not measurable)", "",
           "| fold | " + " | ".join(CLASSES) + " |", "|---|" + "---:|" * len(CLASSES)]
    for f in fold_out:
        m = f["ev"]["metrics"]
        md.append("| test=%s | " % f["meta"]["test_weld"] + " | ".join(
            ("%s (n=%d)" % (fmt(m["recall"][i]), m["support"][i])) if m["support"][i] > 0 else "n/a (none)" for i in range(len(CLASSES))) + " |")
    md += ["", "## Mean over folds (only folds where the number is measurable)", "",
           "| score | mean | min | max | folds |", "|---|---:|---:|---:|---:|"]
    for k, s in overall.items():
        md.append("| %s | %s | %s | %s | %d |" % (k, fmt(s["mean"]) if s else NA, fmt(s["lo"]) if s else NA, fmt(s["hi"]) if s else NA, s["n"] if s else 0))
    for c in CLASSES:
        s = per_class[(c, "recall")]
        md.append("| recall %s | %s | %s | %s | %d |" % (c, fmt(s["mean"]) if s else NA, fmt(s["lo"]) if s else NA, fmt(s["hi"]) if s else NA, s["n"] if s else 0))
    md += ["", "## Pooled out-of-fold predictions", "",
           "All %d labelled images, each predicted once by the model of the fold in which its weld was held out. "
           "This is the only view in which all three classes can be measured. The 4 models differ, so treat it as a "
           "summary, not as the score of one model." % len(y_true), "",
           "accuracy %s, balanced accuracy %s, macro-F1 %s" % (fmt(pooled["accuracy"]), fmt(pooled["balanced_accuracy"]), fmt(pooled["macro_f1"])), "",
           "| class | images | precision | recall | F1 |", "|---|---:|---:|---:|---:|"]
    for i, c in enumerate(CLASSES):
        md.append("| %s | %d | %s | %s | %s |" % (c, pooled["support"][i], fmt(pooled["precision"][i]), fmt(pooled["recall"][i]), fmt(pooled["f1"][i])))
    md += ["", "## Reliability", "",
           "* Only **4 independent welds** exist. Each fold's test score rests on a single weld (28-79 highly similar images); "
           "the folds differ strongly, so the spread between folds is a better indicator than any single number.",
           "* Burn_Through exists only in welds W2 and W4, Spatter only in W1 and W3. When W2 or W4 is held out, only one "
           "Burn_Through weld remains for training, and likewise for Spatter - so the test scores are for a model that has "
           "seen a single example weld of that defect.",
           "* Softmax confidence is not a calibrated probability.", ""]
    (results_dir / "cross_validation_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\nSaved: %s, %s, cross_validation_confusion_matrix.png, cross_validation_scores.png" % (
        results_dir / "cross_validation_results.csv", results_dir / "cross_validation_summary.md"))
    print("\n".join(md[:5 + len(fold_out)]))


def plot_scores(fold_out, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(fold_out))
    vals = [f["ev"]["metrics"]["balanced_accuracy"] for f in fold_out]
    bars = ax.bar(x, vals, color="#4c78a8")
    for xi, v, f in zip(x, vals, fold_out):
        ab = f["ev"]["metrics"]["classes_absent"]
        ax.text(xi, v + 0.02, "%.2f" % v + ("\nno %s" % ", ".join(ab) if ab else ""), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["test=%s" % f["meta"]["test_weld"] for f in fold_out])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("balanced accuracy (present classes)")
    ax.set_title("Held-out weld, balanced accuracy per fold (4 independent welds only)", fontsize=9)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
