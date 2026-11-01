#!/usr/bin/env python3
"""
train.py - train a YOLOv8 image CLASSIFICATION model (default yolov8n-cls.pt) on the weld dataset.

Classes: Burn_Through, Normal_Weld, Spatter.  Split used: the dataset's own weld-wise split
(train = welds W1+W2, val = weld W3, test = weld W4). The test set is NOT touched here.

What is special compared with a plain `yolo classify train`
  * whole camera frame is used (squashed to imgsz x imgsz) - nothing is centre-cropped away
  * augmentation (brightness, contrast, small rotation, mild blur, noise) is applied to TRAINING images only
  * class imbalance: --imbalance sampler (default) = weighted random sampling,
                     --imbalance loss               = class-weighted cross-entropy,
                     --imbalance none
    (Ultralytics' classification trainer has no class-weight option of its own - verified in the installed
     version - so both are implemented here in plain PyTorch.)
  * best.pt is the epoch with the best validation BALANCED accuracy (mean recall of the classes present in the
    validation set), not plain accuracy; ties -> lower validation loss
  * AdamW, lr 0.001, early stopping, fixed seed, GPU if available else CPU
  * if the GPU runs out of memory the batch size is halved and training restarts

Usage
    python train.py
    python train.py --epochs 100 --batch 16 --imgsz 224
    python train.py --imbalance loss          # class-weighted loss instead of weighted sampling
    python train.py --device cpu
Outputs: models\\best.pt, models\\last.pt, results\\loss_graph.png, results\\accuracy_graph.png,
         results\\training_config.json, and the full Ultralytics run folder runs\\original_split\\
"""
import argparse
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from weld_common import (CLASSES, DEFAULT_DATASET, DEFAULT_IMGSZ, MODELS_DIR, RESULTS_DIR, RUNS_DIR,
                         check_dataset_exists, class_counts, classes_absent, compute_metrics, describe_device,
                         labelled_items, pick_device, plot_training_curves, read_image_bgr, read_results_csv,
                         set_seed, squash_to_square, write_json)

# --------------------------------------------------------------------------- augmentation
# "moderate" values: the camera sessions in the dataset really differ in brightness/contrast,
# the pipe is not always positioned identically, and the photos are slightly out of focus.
AUGMENT = dict(
    brightness=0.25,        # multiply by U(0.75, 1.25)
    contrast=0.25,          # stretch around the image mean by U(0.75, 1.25)
    rotate_deg=10.0,        # rotate by U(-10, +10) degrees
    shift=0.05,             # translate by up to +-5 % of the width/height   (small, goes with the rotation)
    zoom=0.10,              # scale by U(0.9, 1.1)                          (small, goes with the rotation)
    blur_p=0.30,            # probability of a mild Gaussian blur
    blur_sigma=(0.4, 1.2),
    noise_p=0.50,           # probability of additive Gaussian noise
    noise_sigma=(2.0, 8.0),  # on the 0-255 scale
)


def augment_rgb(img, cfg=AUGMENT):
    """img: uint8 HxWx3 RGB array (already squashed to imgsz). Returns an augmented uint8 copy.
    Uses numpy's global random state (seeded by set_seed; DataLoader workers get their own seeds)."""
    h, w = img.shape[:2]
    ang = np.random.uniform(-cfg["rotate_deg"], cfg["rotate_deg"])
    scale = 1.0 + np.random.uniform(-cfg["zoom"], cfg["zoom"])
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, scale)
    m[0, 2] += np.random.uniform(-cfg["shift"], cfg["shift"]) * w
    m[1, 2] += np.random.uniform(-cfg["shift"], cfg["shift"]) * h
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    f = out.astype(np.float32)
    contrast = 1.0 + np.random.uniform(-cfg["contrast"], cfg["contrast"])
    bright = 1.0 + np.random.uniform(-cfg["brightness"], cfg["brightness"])
    mean = f.mean()
    f = ((f - mean) * contrast + mean) * bright

    if np.random.rand() < cfg["blur_p"]:
        sigma = np.random.uniform(*cfg["blur_sigma"])
        f = cv2.GaussianBlur(f, (0, 0), sigma)
    if np.random.rand() < cfg["noise_p"]:
        sigma = np.random.uniform(*cfg["noise_sigma"])
        f = f + np.random.normal(0.0, sigma, f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- dataset
_CACHE = {}   # (path, imgsz) -> RGB array; the 1920x1080 originals are decoded only once per process


def load_square_rgb(path, imgsz):
    key = (str(path), int(imgsz))
    if key not in _CACHE:
        _CACHE[key] = cv2.cvtColor(squash_to_square(read_image_bgr(path), imgsz), cv2.COLOR_BGR2RGB)
    return _CACHE[key]


class WeldDataset(Dataset):
    """Returns the dict Ultralytics' classification trainer expects: img (3,H,W float 0-1, RGB), cls, im_file."""

    def __init__(self, items, imgsz, augment):
        self.items = list(items)
        self.imgsz = int(imgsz)
        self.augment = bool(augment)
        self.labels = [it["cls_idx"] for it in self.items]
        self.samples = [(it["path"], it["cls_idx"]) for it in self.items]   # Ultralytics looks for this attribute
        for it in self.items:                                                 # fail early on unreadable images
            load_square_rgb(it["path"], self.imgsz)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = load_square_rgb(self.items[i]["path"], self.imgsz)
        if self.augment:
            img = augment_rgb(img)
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        return {"img": tensor, "cls": self.labels[i], "im_file": self.items[i]["path"]}


class _Loader(DataLoader):
    def reset(self):      # Ultralytics may call loader.reset(); a plain DataLoader has nothing to reset
        pass


def balance_weights(items, power=1.0):
    """per-sample sampling weights = (1 / class count) ** power. power=1 -> every class is drawn equally often."""
    counts = class_counts(items)
    return [(1.0 / counts[it["cls_idx"]]) ** power for it in items], counts


def class_loss_weights(items):
    """Class weights for the cross-entropy loss: N / (K * n_c) over the K classes present in the training data
    (classes with no training image get weight 1 - they never occur as a target anyway)."""
    counts = class_counts(items)
    n, k = sum(counts), sum(1 for c in counts if c > 0)
    return [n / (k * c) if c > 0 else 1.0 for c in counts]


class WeightedClassificationLoss:
    """Same as Ultralytics' v8ClassificationLoss but with per-class weights. Training-only object:
    Ultralytics removes the criterion from the model before best.pt / last.pt are written."""

    calls = 0   # how many batches used this loss (written to training_config.json as proof it was active)

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        WeightedClassificationLoss.calls += 1
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], weight=self.weights.to(preds.device),
                                                 reduction="mean")
        return loss, {"loss": loss.detach()}


# --------------------------------------------------------------------------- trainer
def make_trainer_class(train_items, val_items, imgsz, imbalance, sampler_power, seed, augment):
    """Build a ClassificationTrainer subclass that reads the given image lists instead of scanning folders."""
    from ultralytics.models.yolo.classify import ClassificationTrainer

    info = dict(class_weights=None, sampler_weights=None, best_epoch=None, train_counts=class_counts(train_items),
                val_counts=class_counts(val_items))

    class WeldTrainer(ClassificationTrainer):
        weld_info = info

        def get_dataset(self):
            return {"train": "weld-train-list", "val": "weld-val-list", "test": "weld-val-list",
                    "nc": len(CLASSES), "names": dict(enumerate(CLASSES)), "channels": 3}

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            workers = int(self.args.workers)
            pin = torch.cuda.is_available() and self.device.type == "cuda"
            if mode == "train":
                ds = WeldDataset(train_items, imgsz, augment=augment)
                bs = max(2, min(int(batch_size), len(ds)))
                gen = torch.Generator()
                gen.manual_seed(seed)
                sampler = None
                if imbalance == "sampler":
                    w, counts = balance_weights(train_items, sampler_power)
                    info["sampler_weights"] = {c: round((1.0 / counts[i]) ** sampler_power, 6)
                                               for i, c in enumerate(CLASSES) if counts[i] > 0}
                    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                                    replacement=True, generator=gen)
                return _Loader(ds, batch_size=bs, shuffle=sampler is None, sampler=sampler, num_workers=workers,
                               drop_last=True,       # a last batch of 1 image would break BatchNorm
                               pin_memory=pin, persistent_workers=workers > 0, generator=gen)
            ds = WeldDataset(val_items, imgsz, augment=False)
            return _Loader(ds, batch_size=max(1, min(int(batch_size), len(ds))), shuffle=False,
                           num_workers=workers, drop_last=False, pin_memory=pin, persistent_workers=workers > 0)

        def get_model(self, cfg=None, weights=None, verbose=True):
            """Build the model as usual, then (for --imbalance loss) give it the class-weighted loss.
            Done here, not in a newer-version-only hook, so it also works with older Ultralytics releases."""
            model = super().get_model(cfg, weights, verbose)
            if imbalance == "loss":
                w = class_loss_weights(train_items)
                info["class_weights"] = {c: round(w[i], 4) for i, c in enumerate(CLASSES)}
                WeightedClassificationLoss.calls = 0
                model.criterion = WeightedClassificationLoss(torch.tensor(w, dtype=torch.float32))
            return model

        def final_eval(self):
            """The stock version re-validates best.pt by re-reading the dataset FOLDER, which does not match the
            image lists used here (and prints misleading 'requires 3 classes' errors because the validation and
            test weld lack a class). The real evaluation is done by evaluate.py / cross_validate.py, so only the
            optimizer state is stripped from last.pt / best.pt here (the stock first step)."""
            from ultralytics.utils.torch_utils import strip_optimizer
            ckpt = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": (ckpt or {}).get("train_results")})

        def validate(self):
            """Same as Ultralytics' validate(), but the fitness that picks best.pt / drives early stopping is the
            validation BALANCED accuracy = mean recall over the classes present in the validation set
            (+ a tiny loss-based tie-breaker). Plain accuracy would reward ignoring the small classes."""
            metrics = self.validator(self)
            if metrics is None:
                return None, None
            metrics.pop("fitness", None)
            pred = torch.cat(self.validator.pred)[:, 0].numpy()
            true = torch.cat(self.validator.targets).numpy()
            bal = compute_metrics(true, pred)["balanced_accuracy"]
            metrics["metrics/balanced_accuracy"] = round(float(bal), 5)
            fitness = float(bal) + 1e-3 / (1.0 + float(metrics.get("val/loss", 0.0)))
            if self.best_fitness is None or fitness > self.best_fitness:
                self.best_fitness = fitness
                info["best_epoch"] = self.epoch + 1
            return metrics, fitness

    return WeldTrainer


def resolve_pretrained(model_arg):
    """'yolov8n-cls.pt' -> models/pretrained/yolov8n-cls.pt (downloaded from the official Ultralytics release if missing)."""
    p = Path(model_arg)
    if p.is_file():
        return str(p)
    if p.name == str(model_arg):
        target = MODELS_DIR / "pretrained" / p.name
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading pretrained weights %s ..." % p.name)
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(str(target))
        if target.is_file():
            return str(target)
    sys.exit("ERROR: pretrained model '%s' not found and could not be downloaded. Check the internet connection "
             "or download it from https://github.com/ultralytics/assets/releases and pass --model <file>." % model_arg)


def _is_oom(exc):
    s = str(exc).lower()
    return "out of memory" in s or "cublas_status_alloc_failed" in s


def train_model(train_items, val_items, name, cfg):
    """Train once. Returns a dict with the paths of the real outputs and what actually happened.
    Retries with half the batch size when the GPU runs out of memory."""
    from ultralytics import YOLO

    device = pick_device(cfg["device"])
    pretrained = resolve_pretrained(cfg["model"])
    batch = int(cfg["batch"])
    run_dir = Path(cfg.get("run_dir") or RUNS_DIR)
    attempt = 0
    while True:
        attempt += 1
        WeightedClassificationLoss.calls = 0
        set_seed(cfg["seed"])
        trainer_cls = make_trainer_class(train_items, val_items, cfg["imgsz"], cfg["imbalance"],
                                         cfg["sampler_power"], cfg["seed"], not cfg["no_augment"])
        model = YOLO(pretrained)
        t0 = time.time()
        try:
            model.train(
                trainer=trainer_cls, data=str(cfg["data"]), epochs=int(cfg["epochs"]), imgsz=int(cfg["imgsz"]),
                batch=batch, nbs=batch,                 # nbs=batch: one optimizer step per batch (small dataset)
                optimizer="AdamW", lr0=float(cfg["lr0"]), momentum=0.9, warmup_bias_lr=0.0,
                patience=int(cfg["patience"]), pretrained=True, device=device, seed=int(cfg["seed"]),
                deterministic=True, workers=int(cfg["workers"]), project=str(run_dir), name=name, exist_ok=True,
                plots=False, close_mosaic=0, dropout=float(cfg["dropout"]), amp=not cfg["no_amp"], verbose=True)
            break
        except RuntimeError as e:
            if not _is_oom(e) or batch <= 2:
                raise
            new_batch = max(2, batch // 2)
            print("\n*** GPU out of memory with batch=%d -> restarting with batch=%d (attempt %d) ***\n"
                  % (batch, new_batch, attempt))
            batch = new_batch
            del model, trainer_cls
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    tr = model.trainer
    save_dir = Path(tr.save_dir)
    best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
    if not best.is_file():
        raise RuntimeError("Training did not produce %s - it did not finish." % best)
    csv_path = save_dir / "results.csv"
    cols = read_results_csv(csv_path)
    return dict(save_dir=save_dir, best=best, last=last, results_csv=csv_path, batch_used=batch,
                epochs_run=len(cols.get("epoch", [])), best_epoch=tr.weld_info["best_epoch"],
                seconds=time.time() - t0, device=describe_device(device), info=tr.weld_info,
                weighted_loss_batches=WeightedClassificationLoss.calls, final_args=dict(vars(tr.args)))


def versions():
    import ultralytics
    return dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                torchvision=__import__("torchvision").__version__, ultralytics=ultralytics.__version__,
                opencv=cv2.__version__, numpy=np.__version__, cuda_available=torch.cuda.is_available(),
                cuda_build=torch.version.cuda)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train YOLOv8 classification on the weld dataset.")
    ap.add_argument("--data", default=str(DEFAULT_DATASET), help="dataset folder (default %(default)s)")
    ap.add_argument("--model", default="yolov8n-cls.pt", help="pretrained model (default %(default)s)")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="input size (default %(default)s)")
    ap.add_argument("--epochs", type=int, default=100, help="maximum epochs (default %(default)s)")
    ap.add_argument("--batch", type=int, default=16, help="batch size, halved automatically on GPU out-of-memory")
    ap.add_argument("--lr0", type=float, default=0.001, help="initial learning rate (default %(default)s)")
    ap.add_argument("--patience", type=int, default=20, help="early stopping: epochs without improvement")
    ap.add_argument("--imbalance", choices=["sampler", "loss", "none"], default="sampler",
                    help="class-imbalance handling (default %(default)s)")
    ap.add_argument("--sampler-power", type=float, default=1.0,
                    help="sampler weight = (1/class count)**power; 1 = classes drawn equally often, 0.5 = softer")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout in the classifier head (default 0)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default %(default)s)")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 (default auto = GPU 0 if available)")
    ap.add_argument("--workers", type=int, default=2, help="data-loader workers on GPU (CPU uses 0). If Windows "
                    "complains about multiprocessing use --workers 0")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (try if you see NaN or AMP errors)")
    ap.add_argument("--no-augment", action="store_true", help="switch off augmentation (for experiments)")
    ap.add_argument("--name", default="original_split", help="run name inside runs\\ (default %(default)s)")
    ap.add_argument("--models-dir", default=str(MODELS_DIR), help="where best.pt / last.pt are copied")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="where graphs and the config are saved")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_dataset_exists(args.data)
    items = labelled_items(args.data)
    train_items = [i for i in items if i["split"] == "train"]
    val_items = [i for i in items if i["split"] == "val"]
    if not train_items or not val_items:
        sys.exit("ERROR: the dataset has no %s images." % ("train" if not train_items else "validation"))
    if len({i["weld_group"] for i in train_items} & {i["weld_group"] for i in val_items}):
        sys.exit("ERROR: a weld group is in both train and val - refusing to train (data leakage).")

    device = pick_device(args.device)
    print("Device :", describe_device(device))
    print("Train  : %d images, welds %s, class counts %s" % (
        len(train_items), sorted({i["weld_group"] for i in train_items}), dict(zip(CLASSES, class_counts(train_items)))))
    print("Val    : %d images, welds %s, class counts %s" % (
        len(val_items), sorted({i["weld_group"] for i in val_items}), dict(zip(CLASSES, class_counts(val_items)))))
    for c in classes_absent(train_items):
        print("WARNING: no %s images in the TRAINING set - the model cannot learn this class." % c)
    for c in classes_absent(val_items):
        print("WARNING: no %s images in the VALIDATION set - validation scores and the choice of best.pt say "
              "nothing about %s." % (c, c))

    cfg = vars(args)
    res = train_model(train_items, val_items, args.name, cfg)

    models_dir, results_dir = Path(args.models_dir), Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(res["best"], models_dir / "best.pt")
    shutil.copy2(res["last"], models_dir / "last.pt")
    plot_training_curves(res["results_csv"], results_dir / "loss_graph.png", results_dir / "accuracy_graph.png",
                         "(original split: train W1+W2, validation W3)")
    shutil.copy2(res["results_csv"], results_dir / "training_log.csv")
    shutil.copy2(res["save_dir"] / "args.yaml", results_dir / "training_args_ultralytics.yaml")

    write_json(results_dir / "training_config.json", dict(
        what="YOLOv8 classification training on the original weld-wise split",
        settings=dict(model=args.model, imgsz=args.imgsz, epochs_max=args.epochs, batch_requested=args.batch,
                      batch_used=res["batch_used"], optimizer="AdamW", lr0=args.lr0, patience=args.patience,
                      pretrained=True, seed=args.seed, deterministic=True, imbalance=args.imbalance,
                      sampler_power=args.sampler_power, dropout=args.dropout, augmentation=None if args.no_augment else AUGMENT,
                      amp=not args.no_amp, nbs=res["batch_used"]),
        model_selection="best.pt = epoch with the highest validation balanced accuracy (mean recall over the classes "
                        "present in the validation set); ties -> lower validation loss",
        data=dict(train_images=len(train_items), val_images=len(val_items),
                  train_class_counts=dict(zip(CLASSES, class_counts(train_items))),
                  val_class_counts=dict(zip(CLASSES, class_counts(val_items))),
                  train_welds=sorted({i["weld_group"] for i in train_items}),
                  val_welds=sorted({i["weld_group"] for i in val_items}),
                  classes_absent_in_train=classes_absent(train_items), classes_absent_in_val=classes_absent(val_items),
                  sampler_weights=res["info"]["sampler_weights"], class_loss_weights=res["info"]["class_weights"],
                  weighted_loss_batches_used=res["weighted_loss_batches"]),
        result=dict(epochs_run=res["epochs_run"], best_epoch=res["best_epoch"], seconds=round(res["seconds"], 1),
                    device=res["device"], run_folder=str(res["save_dir"])),
        environment=versions()))

    print("\n" + "=" * 70)
    print("TRAINING FINISHED (%d epochs run, best epoch %s, %.1f min on %s)" % (
        res["epochs_run"], res["best_epoch"], res["seconds"] / 60, res["device"]))
    print("  %s\n  %s" % (models_dir / "best.pt", models_dir / "last.pt"))
    print("  %s, %s, %s" % (results_dir / "loss_graph.png", results_dir / "accuracy_graph.png",
                            results_dir / "training_config.json"))
    print("Next: python evaluate.py     (scores on the validation AND the untouched test weld)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
