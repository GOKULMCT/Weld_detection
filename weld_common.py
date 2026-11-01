#!/usr/bin/env python3
"""
weld_common.py - helpers shared by every script of the weld defect project.

  * folder locations and the class list
  * reading the dataset index (dataset_report.csv) - labelled images only
  * image loading / pre-processing (the SAME function is used for training and for prediction)
  * metrics that never turn a class without samples into a fake "0.0 score"
  * confusion-matrix and training-curve pictures
  * WeldClassifier: loads best.pt and returns class probabilities

Nothing in here changes or moves your original images.
"""
import csv
import json
import math
import os
import random
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset" / "weld_defect_dataset"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

# Alphabetical = the order Ultralytics and torchvision ImageFolder use. Index 0,1,2 below.
CLASSES = ["Burn_Through", "Normal_Weld", "Over_Weld", "Spatter"]
SHORT = {"Burn_Through": "BT", 
         "Normal_Weld": "NW", 
         "Over_Weld": "OW",
         "Spatter": "SR"}
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMGSZ = 224
NA = "n/a"


# --------------------------------------------------------------------------- small utilities
def set_seed(seed):
    """Fix every random number generator we use (python, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(requested="auto"):
    """'auto' -> CUDA device 0 if PyTorch can see a GPU, else CPU. Explicit choices are honoured
    but fall back to CPU (with a warning) when the GPU is not usable."""
    import torch
    req = str(requested).strip().lower() if requested is not None else "auto"
    cuda_ok = torch.cuda.is_available()
    if req in ("auto", ""):
        return 0 if cuda_ok else "cpu"
    if req == "cpu":
        return "cpu"
    if not cuda_ok:
        print("WARNING: CUDA device '%s' requested but PyTorch cannot see a GPU - using the CPU. "
              "See README > Troubleshooting > CUDA." % requested)
        return "cpu"
    return int(req) if req.isdigit() else requested


def describe_device(device):
    import torch
    if device == "cpu":
        return "CPU"
    try:
        idx = int(device)
        return "cuda:%d  %s  (%.1f GB)" % (idx, torch.cuda.get_device_name(idx),
                                          torch.cuda.get_device_properties(idx).total_memory / 1e9)
    except Exception:
        return str(device)


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def fmt(x, digits=3):
    """Number -> text; NaN/None -> 'n/a'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return NA
    return ("%." + str(digits) + "f") % x


# --------------------------------------------------------------------------- images
def read_image_bgr(path):
    """Read an image (works with non-ASCII Windows paths). Raises ValueError if it cannot be decoded."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError("cannot decode image: %s" % path)
    return img


def squash_to_square(img_bgr, imgsz=DEFAULT_IMGSZ):
    """Resize the WHOLE camera frame to imgsz x imgsz.

    Why not Ultralytics' default (resize + centre-crop)? The photos are 16:9 and the defect is often
    near the left/right/top edge; a centre crop would cut off ~44 % of the picture width. Squashing keeps
    everything in view. The same function is used for training, evaluation and every predict_*.py script,
    so the network always sees identical geometry."""
    return cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- dataset index
def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def labelled_items(dataset_dir):
    """All labelled raw images from dataset_report.csv (review_required and crops are NOT included).
    Returns a list of dicts: path, cls_name, cls_idx, split, pipe_id, weld_group, angle_deg."""
    dataset_dir = Path(dataset_dir)
    rows = read_csv_rows(dataset_dir / "dataset_report.csv")
    items = []
    for r in rows:
        if r["image_type"] != "raw" or r["review_status"] != "labelled":
            continue
        cls = r["assigned_class"]
        if cls not in CLASSES:
            raise ValueError("unknown class '%s' in dataset_report.csv (row %s)" % (cls, r["image_filename"]))
        items.append(dict(path=str(dataset_dir / r["split"] / cls / r["image_filename"]),
                          cls_name=cls, cls_idx=CLASSES.index(cls), split=r["split"],
                          pipe_id=r["pipe_id"], weld_group=r["weld_group"],
                          angle_deg=int(r["angle_deg"]), filename=r["image_filename"]))
    return items


def class_counts(items, key="cls_idx"):
    counts = [0] * len(CLASSES)
    for it in items:
        counts[it[key]] += 1
    return counts


def classes_present(items):
    return [CLASSES[i] for i, c in enumerate(class_counts(items)) if c > 0]


def classes_absent(items):
    return [CLASSES[i] for i, c in enumerate(class_counts(items)) if c == 0]


def check_dataset_exists(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / "dataset_report.csv").is_file():
        sys.exit("ERROR: %s does not look like the weld dataset (dataset_report.csv missing).\n"
                 "Run  python prepare_dataset.py  first, or pass --data <folder>." % dataset_dir)


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, classes=CLASSES):
    """Confusion matrix and scores. A class WITHOUT ground-truth samples gets NaN (not 0.0) for precision,
    recall and F1 - nothing can be measured for it - and is left out of the averages.
    A class that has samples but is never predicted gets recall 0, F1 0 and precision NaN (undefined)."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    total = int(cm.sum())
    present = support > 0
    recall = np.full(n, np.nan)
    precision = np.full(n, np.nan)
    f1 = np.full(n, np.nan)
    for i in range(n):
        if not present[i]:
            continue
        recall[i] = tp[i] / support[i]
        if predicted[i] > 0:
            precision[i] = tp[i] / predicted[i]
        p_, r_ = precision[i], recall[i]
        f1[i] = 0.0 if (math.isnan(p_) or (p_ + r_) == 0) else 2 * p_ * r_ / (p_ + r_)
    out = dict(
        confusion_matrix=cm, support=support, n_predicted=predicted, precision=precision, recall=recall, f1=f1,
        total=total,
        accuracy=float(tp.sum() / total) if total else float("nan"),
        balanced_accuracy=float(np.nanmean(recall)) if present.any() else float("nan"),
        macro_f1=float(np.nanmean(f1)) if present.any() else float("nan"),
        classes_present=[classes[i] for i in range(n) if present[i]],
        classes_absent=[classes[i] for i in range(n) if not present[i]],
        n_predicted_as_absent_class=int(predicted[~present].sum()),
    )
    return out


def metrics_rows(m, split_name, classes=CLASSES, extra=None):
    """Rows for a CSV report: one per class, plus overall rows. Absent classes are marked, not zeroed."""
    rows = []
    for i, c in enumerate(classes):
        measured = m["support"][i] > 0
        row = dict(split=split_name, class_name=c, row_type="class",
                   precision=fmt(m["precision"][i]) if measured else NA,
                   recall=fmt(m["recall"][i]) if measured else NA,
                   f1=fmt(m["f1"][i]) if measured else NA,
                   support=int(m["support"][i]), n_predicted=int(m["n_predicted"][i]),
                   status="measured" if measured else "NOT MEASURABLE - no %s images in this set" % c)
        if extra:
            row.update(extra)
        rows.append(row)
    for name, val in (("accuracy", m["accuracy"]), ("balanced_accuracy (mean recall of present classes)", m["balanced_accuracy"]),
                      ("macro_f1 (present classes only)", m["macro_f1"])):
        row = dict(split=split_name, class_name=name, row_type="overall", precision="", recall="", f1=fmt(val),
                   support=m["total"], n_predicted=m["total"],
                   status="over classes present: " + ", ".join(m["classes_present"]))
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


METRIC_FIELDS = ["split", "class_name", "row_type", "precision", "recall", "f1", "support", "n_predicted", "status"]


# --------------------------------------------------------------------------- pictures
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion_matrix(cm, path, title, classes=CLASSES):
    """Rows = true class, columns = predicted class. Cell text: count and (share of the true class).
    A true class with no images is greyed out and labelled 'no samples' - its row is not a measurement."""
    plt = _mpl()
    cm = np.asarray(cm)
    n = len(classes)
    support = cm.sum(axis=1)
    share = np.divide(cm, np.maximum(support[:, None], 1), dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(np.where(support[:, None] > 0, share, 0.0), cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if support[i] == 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#dddddd", zorder=2))
                txt = "-" if cm[i, j] == 0 else str(cm[i, j])
                color = "#555555"
            else:
                txt = "%d\n(%.0f%%)" % (cm[i, j], 100 * share[i, j])
                color = "white" if share[i, j] > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10, zorder=3)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=20)
    ax.set_yticks(range(n))
    ax.set_yticklabels([c if support[k] > 0 else c + "\n(no samples)" for k, c in enumerate(classes)])
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(textwrap.fill(title, 58), fontsize=10)      # long titles are wrapped, not cut off
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def read_results_csv(path):
    """Ultralytics results.csv -> dict of column name -> list of floats (column names stripped)."""
    rows = read_csv_rows(path)
    cols = {}
    for r in rows:
        for k, v in r.items():
            k = (k or "").strip()
            try:
                cols.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                cols.setdefault(k, []).append(float("nan"))
    return cols


def plot_training_curves(results_csv, loss_png, acc_png, title=""):
    """loss_graph.png (training + validation loss) and accuracy_graph.png (validation accuracy)
    made from the results.csv that Ultralytics wrote during the real training run."""
    plt = _mpl()
    cols = read_results_csv(results_csv)
    ep = cols.get("epoch", list(range(1, len(next(iter(cols.values()))) + 1)))
    ep = [int(e) for e in ep]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if "train/loss" in cols:
        ax.plot(ep, cols["train/loss"], label="training loss", marker="o", ms=2.5)
    if "val/loss" in cols:
        ax.plot(ep, cols["val/loss"], label="validation loss", marker="o", ms=2.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(("Loss  " + title).strip())
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    Path(loss_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(loss_png, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if "metrics/accuracy_top1" in cols:
        ax.plot(ep, cols["metrics/accuracy_top1"], label="validation accuracy (top-1)", marker="o", ms=2.5)
    if "metrics/balanced_accuracy" in cols:
        ba = cols["metrics/balanced_accuracy"]
        ax.plot(ep, ba, label="validation balanced accuracy (used to pick best.pt)", marker="o", ms=2.5)
        if not all(math.isnan(v) for v in ba):
            best = int(np.nanargmax(ba))
            ax.axvline(ep[best], color="grey", ls="--", lw=1)
            ax.text(ep[best], 0.02, " best epoch %d" % ep[best], color="grey", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(("Validation accuracy  " + title).strip())
    ax.grid(alpha=.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(acc_png, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- inference
class WeldClassifier:
    """Loads a trained classification checkpoint (best.pt) and returns class probabilities.
    Uses exactly the same pre-processing as training (squash_to_square)."""

    def __init__(self, weights, imgsz=None, device="auto"):
        weights = Path(weights)
        if not weights.is_file():
            sys.exit("ERROR: model file not found: %s\nTrain first (python train.py) or pass --weights <file>." % weights)
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        task = getattr(self.model, "task", "classify")
        if task != "classify":
            sys.exit("ERROR: %s is a '%s' model, this project needs a classification model." % (weights, task))
        names = self.model.names
        self.names = [names[i] for i in range(len(names))]
        self.imgsz = int(imgsz or (getattr(self.model, "overrides", {}) or {}).get("imgsz") or DEFAULT_IMGSZ)
        self.device = pick_device(device)

    def predict_bgr(self, images_bgr):
        """list of BGR numpy images (any size) -> array (N, n_classes) of softmax probabilities."""
        if not images_bgr:
            return np.zeros((0, len(self.names)))
        batch = [squash_to_square(im, self.imgsz) for im in images_bgr]
        results = self.model.predict(source=batch, imgsz=self.imgsz, device=self.device, verbose=False)
        return np.stack([r.probs.data.detach().cpu().numpy() for r in results]).astype(float)

    def predict_paths(self, paths, batch_size=16):
        """list of file paths -> list of result dicts (unreadable files get an 'error' entry)."""
        out = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            imgs, ok_idx = [], []
            for k, p in enumerate(chunk):
                try:
                    imgs.append(read_image_bgr(p))
                    ok_idx.append(k)
                except (ValueError, OSError) as e:
                    out.append(dict(path=str(p), error=str(e)))
            if imgs:
                probs = self.predict_bgr(imgs)
                for k, pr in zip(ok_idx, probs):
                    out.append(self.result_from_probs(chunk[k], pr))
        return out

    def result_from_probs(self, path, probs):
        top = int(np.argmax(probs))
        return dict(path=str(path), predicted_class=self.names[top], confidence=float(probs[top]),
                    probs={n: float(p) for n, p in zip(self.names, probs)})


def evaluate_items(clf, items, threshold, batch_size=16):
    """Run the classifier on labelled items (dicts from labelled_items) and compute the metrics.
    Returns dict(rows=per-image rows, metrics=compute_metrics(...), y_true, y_pred, conf)."""
    results = clf.predict_paths([it["path"] for it in items], batch_size)
    by_path = {r["path"]: r for r in results}
    y_true, y_pred, conf, rows = [], [], [], []
    for it in items:
        r = by_path[str(it["path"])]
        if "error" in r:
            raise RuntimeError("could not evaluate %s: %s" % (it["path"], r["error"]))
        if r["predicted_class"] not in CLASSES:
            raise RuntimeError("the model predicts class '%s' which is not one of %s" % (r["predicted_class"], CLASSES))
        p_idx = CLASSES.index(r["predicted_class"])
        y_true.append(it["cls_idx"])
        y_pred.append(p_idx)
        conf.append(r["confidence"])
        row = dict(image=it["filename"], pipe_id=it["pipe_id"], weld_group=it["weld_group"], angle_deg=it["angle_deg"],
                   true_class=it["cls_name"], predicted_class=r["predicted_class"], confidence="%.4f" % r["confidence"])
        for n in CLASSES:
            row["prob_" + n] = "%.4f" % r["probs"].get(n, float("nan"))
        row["correct"] = "yes" if p_idx == it["cls_idx"] else "NO"
        row["needs_manual_review"] = "YES" if r["confidence"] < threshold else "no"
        rows.append(row)
    return dict(rows=rows, metrics=compute_metrics(y_true, y_pred), y_true=np.array(y_true), y_pred=np.array(y_pred),
                conf=np.array(conf))


EVAL_ROW_FIELDS = (["image", "pipe_id", "weld_group", "angle_deg", "true_class", "predicted_class", "confidence"]
                   + ["prob_" + c for c in CLASSES] + ["correct", "needs_manual_review"])


def confidence_summary(ev, threshold):
    """How the softmax confidence behaves: mean for right/wrong answers and accuracy of flagged / not flagged."""
    ok = ev["y_true"] == ev["y_pred"]
    low = ev["conf"] < threshold
    out = dict(n=int(len(ok)), n_flagged=int(low.sum()))
    out["mean_conf_correct"] = float(ev["conf"][ok].mean()) if ok.any() else float("nan")
    out["mean_conf_wrong"] = float(ev["conf"][~ok].mean()) if (~ok).any() else float("nan")
    out["acc_flagged"] = float(ok[low].mean()) if low.any() else float("nan")
    out["acc_not_flagged"] = float(ok[~low].mean()) if (~low).any() else float("nan")
    out["wrong_but_not_flagged"] = int((~ok & ~low).sum())
    return out


def find_images(folder, recursive=False):
    """Sorted list of image files in a folder (optionally including sub-folders)."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def expand_image_args(args):
    """Command-line image arguments -> list of paths. Wildcards (*.jpg) are expanded here because the Windows
    command prompt does not do it; folders are not accepted (use predict_folder.py)."""
    import glob
    paths = []
    for a in args:
        hits = sorted(glob.glob(a)) if any(ch in a for ch in "*?[") else [a]
        paths += hits
    return [Path(p) for p in paths]


def result_to_row(res, threshold):
    """One CSV row for a predict_* result dict."""
    if "error" in res:
        return dict(image=res["path"], predicted_class="", confidence="", needs_manual_review="", note="ERROR: " + res["error"])
    row = dict(image=res["path"], predicted_class=res["predicted_class"], confidence="%.4f" % res["confidence"])
    for n, p in res["probs"].items():
        row["prob_" + n] = "%.4f" % p
    low = res["confidence"] < threshold
    row["needs_manual_review"] = "YES" if low else "no"
    row["note"] = "confidence below %.2f - check by eye" % threshold if low else ""
    return row


def prediction_fields(class_names):
    return ["image", "predicted_class", "confidence"] + ["prob_" + n for n in class_names] + ["needs_manual_review", "note"]


CONFIDENCE_NOTE = ("NOTE: 'confidence' is the network's softmax output. It is NOT a calibrated probability that "
                   "a defect is present - a wrong prediction can have a high value. Use it only to rank/flag.")
