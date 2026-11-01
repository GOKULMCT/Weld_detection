# Weld defect classification with YOLOv8 (6G pipe-weld inspection)

Classifies a whole weld photo as **Burn_Through**, **Spatter** or **Normal_Weld** with a YOLOv8 *classification*
model (`yolov8n-cls.pt`, fine-tuned). It says *what* is in the picture, not *where* - there are no bounding boxes.

Target PC: Windows 10, Python 3.10, NVIDIA RTX 3050 Laptop GPU (6 GB), VS Code, project folder `D:\weld_project`.

---

## 1. Status and results - please read this first

**What was actually done.** All scripts were written and then really executed end to end in the environment where
this project was built: dataset check, training, evaluation, leave-one-weld-out cross-validation, image / folder /
(video-file) prediction. The files in `models\` and `results\` are the real outputs of those runs; nothing was
written by hand and no number in this README was typed from memory (every number comes from the CSV files in
`results\`). That environment was a **Linux machine with a CPU only**, Python 3.13, PyTorch 2.14.1, Ultralytics
8.4.188. It was **not** Windows 10, Python 3.10 or your RTX 3050 - see section 9 for the full list of what is untested.

**What the results say.** The model does **not** generalise to a weld it has not seen. It fits its training welds
almost perfectly (training accuracy 98 %) but, on unseen welds, it mostly answers "Normal_Weld", finds few
Burn_Through images and almost no Spatter. **Do not use this model for real accept / reject decisions.** The
code and the evaluation are sound; the limit is the data: the dataset has only **4 independent physical welds**, and
each defect type is present on only two of them.

### 1a. Original weld-wise split (`python train.py`, then `python evaluate.py`)

Training welds W1 + W2 (279 images), validation weld W3 (66 images, used to pick `best.pt`), untouched test weld W4
(28 images). Training stopped by early stopping after 26 epochs, `best.pt` = epoch 6 (CPU, about 4 minutes).

| split | images | accuracy | balanced accuracy | macro-F1 | recall Burn_Through | recall Normal_Weld | recall Spatter |
|---|---:|---:|---:|---:|---:|---:|---:|
| train (W1+W2) - memorisation, not a quality score | 279 | 0.982 | 0.992 | 0.971 | 1.000 | 0.976 | 1.000 |
| validation (W3) | 66 | 0.742 | 0.528 | 0.477 | n/a (no such images) | 1.000 | 0.056 (1 of 18) |
| **test (W4)** | 28 | **0.429** | **0.500** | 0.333 | **0.000 (0 of 16)** | 1.000 | n/a (no such images) |

* **n/a is not zero.** W3 has no Burn_Through and W4 has no Spatter, so those cells cannot be measured; they are
  shown as n/a and are left out of the balanced accuracy and macro-F1. (On W4, 4 images were predicted as Spatter, a
  class that does not exist there - these are counted as errors, separately.)
* On W4, 12 of the 16 Burn_Through images were called Normal_Weld and 4 were called Spatter.
* Confidence does not warn you: wrong answers had a mean confidence of 0.91 (W3) and 0.88 (W4). With the review
  threshold at 0.70, 16 of the 17 wrong answers on W3 and 14 of the 16 on W4 were *not* flagged.
* Files: `results\confusion_matrix.png` (test), `confusion_matrix_val.png`, `confusion_matrix_train.png`,
  `classification_report.csv`, `evaluation_report.md`, `predictions_*.csv`, `loss_graph.png`, `accuracy_graph.png`,
  `training_log.csv`, `training_config.json`.

### 1b. Leave-one-weld-out cross-validation (`python cross_validate.py`) - the better guide

Each weld is the test weld once. In every fold one *other* weld is the validation weld (early stopping and choice of
that fold's best model) and the rest train. The test weld is only used for the final score. Images of one weld are never split.

| test weld | train welds | validation weld | accuracy | balanced accuracy | macro-F1 | not measurable in this test weld |
|---|---|---|---:|---:|---:|---|
| W1 (200 images) | W2+W3 | W4 | 0.545 | 0.415 | 0.446 | Burn_Through |
| W2 (79) | W1+W4 | W3 | 0.671 | 0.675 | 0.749 | Spatter |
| W3 (66) | W1+W2 | W4 | 0.727 | 0.500 | 0.421 | Burn_Through |
| W4 (28) | W1+W2 | W3 | 0.536 | 0.594 | 0.482 | Spatter |
| **mean of the 4 folds** | | | **0.620** (0.536-0.727) | **0.546** (0.415-0.675) | **0.524** (0.421-0.749) | |

Recall per class on the held-out weld (n/a = class not present there):

| test weld | Burn_Through | Normal_Weld | Spatter |
|---|---:|---:|---:|
| W1 | n/a | 0.604 (n=169) | 0.226 (n=31) |
| W2 | 0.561 (n=41) | 0.789 (n=38) | n/a |
| W3 | n/a | 1.000 (n=48) | 0.000 (n=18) |
| W4 | 0.188 (n=16) | 1.000 (n=12) | n/a |

Pooled over all 373 labelled images (each predicted once by the model that had not seen its weld; four different
models, so this is a summary and not the score of one model): accuracy 0.603, balanced accuracy 0.439, macro-F1 0.436;
recall Burn_Through 0.456 (26 of 57), Normal_Weld 0.719 (192 of 267), Spatter 0.143 (7 of 49).
For comparison, a "model" that always answers Normal_Weld would reach an accuracy of 0.716 (267 of 373) on these
images - higher than 0.603 - with a balanced accuracy of 0.333. So the accuracy numbers alone flatter the model.

Files: `results\cross_validation_results.csv`, `cross_validation_summary.md`, `cross_validation_confusion_matrix.png`,
`cross_validation_scores.png`, `results\cv\fold_W*\` (per-fold predictions, curves, config) and
`models\cv\fold_W*_best.pt` (one model per fold).

### 1c. How far to trust these numbers

* **Few welds.** Every score rests on 1-4 welds. Photos of one weld taken a few degrees apart look alike, so 373 images are
  far fewer than 373 independent samples. The spread between folds (balanced accuracy 0.415 to 0.675) is as important as
  the mean.
* **Not repeatable to the last digit.** The W4 test in section 1a and the "test = W4" fold in 1b use the same training
  images, the same validation weld and the same settings - they differ only in the order of the training list, which
  changes the random draws of the sampler and augmentation. The first gave test balanced accuracy 0.500 (Burn_Through
  recall 0 of 16), the second 0.594 (3 of 16). With this little data, one training run is a noisy measurement. A GPU
  will not reproduce these exact numbers either.
* **Probable reason for the failure** (consistent with the numbers, but not proven by a separate experiment): when a
  defect type is available on only one training weld, the network can learn what *that weld* looks like (lighting,
  surface, pipe position) instead of the defect. Held-out Burn_Through is found better when W4 is in training and W2 is held out
  (recall 0.561, precision 1.000) than the other way round (0.188), and Spatter is hardly found on an unseen weld (recall 0.226 and 0.000).
* **Model selection.** `best.pt` was chosen on the validation weld, which has no Burn_Through, so a model that ignores
  Burn_Through is not penalised during selection. This is a consequence of the data; the cross-validation
  output lists for every fold which class its validation weld lacks.
* The softmax confidence is **not** a calibrated probability of a defect.

---

## 2. Quick start (details in sections 4 and 5)

Use **Command Prompt** (in VS Code: Terminal > the drop-down next to `+` > *Command Prompt*). Lines that start with `::`
are comments; PowerShell does not understand them, so either use Command Prompt or leave those lines out.

```bat
cd D:\weld_project
py -3.10 -m venv .venv
.venv\Scripts\activate
:: confirm the command on pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python check_environment.py
:: copy weld_defect_dataset.zip into D:\weld_project first, then extract + verify it
python prepare_dataset.py
:: train       -> models\best.pt, models\last.pt, results\*
python train.py
:: evaluate    -> results\confusion_matrix.png, classification_report.csv ...
python evaluate.py
:: leave-one-weld-out -> results\cross_validation_results.csv ...
python cross_validate.py
python predict_image.py photo.jpg
python predict_folder.py D:\photos\new_pipe
:: camera index 1, press Q to quit
python predict_camera.py
```

The ZIP already contains a trained `models\best.pt`, so the prediction commands work right after the setup, without
training. Running `python train.py` again replaces `models\best.pt`, `models\last.pt` and the training graphs in
`results\` with your own run (on your GPU the numbers will differ somewhat from section 1). Run `python evaluate.py`
afterwards, because the evaluation files in `results\` belong to the model that was in `models\` when they were made.

---

## 3. Folder layout

```
weld_project\
  weld_defect_dataset.zip        <- put your dataset ZIP here (or in dataset\)
  prepare_dataset.py             1. extract + verify the dataset
  train.py                       2. train (original weld-wise split)
  evaluate.py                    3. evaluate best.pt on validation + test weld
  cross_validate.py              4. leave-one-weld-out cross-validation
  predict_image.py               classify one or more photos
  predict_folder.py              classify a whole folder (batch) -> CSV
  predict_camera.py              live USB-camera classification
  check_environment.py           "why does it not work?" checker
  weld_common.py                 shared code (metrics, plots, model wrapper) - do not run
  requirements.txt
  README.md
  dataset\weld_defect_dataset\   <- created by prepare_dataset.py (the dataset itself is not in this ZIP)
  models\best.pt, last.pt        <- trained weights (see section 1)
  models\pretrained\yolov8n-cls.pt  <- the pretrained starting model (train.py downloads it if missing)
  models\cv\fold_W*_best.pt      <- one model per cross-validation fold
  results\                       <- graphs, CSV files, reports
  runs\                          <- created by train.py / cross_validate.py (raw Ultralytics run folders; not in this ZIP)
```

`models\best.pt` and `models\last.pt` in this ZIP are real outputs of the training run described in section 1.
`results\` contains only files that were produced by actually running the scripts - nothing was written by hand.

---

## 4. Setup on Windows 10 (Python 3.10, NVIDIA RTX 3050 Laptop GPU)

**Step 1 - NVIDIA driver.** Install or update the driver from <https://www.nvidia.com/drivers>. Then open
*Command Prompt* and run `nvidia-smi`. It must print a table with your GPU. (You do **not** need to install the
CUDA Toolkit separately - the PyTorch wheels contain the CUDA runtime.)

**Step 2 - Python 3.10 (64-bit) and a virtual environment.** Install Python 3.10 from
<https://www.python.org/downloads/windows/> and tick *"Add python.exe to PATH"*. Check with `py -3.10 --version`.
Then (VS Code > Terminal > *Command Prompt* or PowerShell):

```bat
cd D:\weld_project
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

In PowerShell the activation command is `.venv\Scripts\Activate.ps1`. If PowerShell says *"running scripts is
disabled"*, run once `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or simply use Command Prompt.
In VS Code press `Ctrl+Shift+P` > *Python: Select Interpreter* > `.venv`.

**Step 3 - PyTorch WITH CUDA (do this before step 4).**
Go to <https://pytorch.org/get-started/locally/>, choose *Stable - Windows - Pip - Python - CUDA*, and copy the
command it shows. It looks like this (the CUDA number changes with each PyTorch release):

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For PyTorch 2.14 the Windows wheels are announced for CUDA 12.6, 13.0 and 13.2; `cu126` works with the widest
range of drivers. *I could not test this command on Windows - always prefer the command the selector shows you.*
If you run only `pip install ultralytics` first, pip installs a **CPU-only** PyTorch and the GPU is never used.

**Step 4 - the other packages:**

```bat
pip install -r requirements.txt
```

**Step 5 - check everything:**

```bat
python check_environment.py
```

It prints `[OK]` / `[!!]` for Python, every package, the GPU (it runs a small calculation on it), the dataset
and the model, and a *TO DO* list for whatever is missing.

**Step 6 (optional) - one-epoch test of the training on your PC**, without touching the delivered model:

```bat
python train.py --epochs 1 --name smoke --models-dir runs\smoke_models --results-dir runs\smoke_results
```

It should finish within a minute or two (on the CPU used here: under a minute in total) and write its files into `runs\smoke_models` and `runs\smoke_results`.
A normal `python train.py` **overwrites** `models\best.pt`, `models\last.pt` and the training graphs in `results\`; copy
`models\best.pt` somewhere first if you want to keep the model that came with this project.

---

## 5. Commands (run in the activated environment, inside `D:\weld_project`)

| Task | Command |
|---|---|
| Extract + validate the dataset | `python prepare_dataset.py` |
| (already extracted?) only check | `python prepare_dataset.py --check-only` |
| Train | `python train.py` |
| Evaluate best.pt (validation + test weld) | `python evaluate.py` |
| Leave-one-weld-out cross-validation | `python cross_validate.py` |
| Classify one image | `python predict_image.py photo.jpg` |
| Classify a folder | `python predict_folder.py D:\photos\new_pipe` |
| Live camera (index 1) | `python predict_camera.py` |

Put `weld_defect_dataset.zip` next to the scripts (or into `dataset\`) first. Useful options:

```bat
python prepare_dataset.py --zip D:\downloads\weld_defect_dataset.zip
:: the defaults
python train.py --epochs 100 --batch 16 --imgsz 224
:: smaller batch if the GPU memory is tight
python train.py --batch 8
:: class-weighted loss instead of weighted sampling
python train.py --imbalance loss
:: force the CPU
python train.py --device cpu
:: if Windows complains about multiprocessing
python train.py --workers 0
:: also training-set score + the review_required images
python evaluate.py --splits train,val,test --predict-review
:: only some folds / skip finished folds
python cross_validate.py --folds W1,W3 --resume
:: several photos, stricter review threshold
python predict_image.py "D:\photos\*.jpg" --threshold 0.8
python predict_folder.py D:\photos --recursive --batch 32 --out results\my_folder.csv
:: which camera index gives a picture?
python predict_camera.py --find-camera
:: other camera, average the last 5 frames
python predict_camera.py --camera 0 --smooth 5
:: also save a picture every 10 s
python predict_camera.py --save-interval 10
```

In the camera window: **Q** (or Esc) quits, **S** saves the current picture (annotated + original) into
`predicted_frames\` and appends a line to `predicted_frames\predictions_camera.csv`. The window shows the class, the
confidence and the FPS. It shows **one class for the whole picture and no bounding boxes** - this is a
classification model, not a detector.

### What the prediction CSV contains
`image, predicted_class, confidence, prob_Burn_Through, prob_Normal_Weld, prob_Spatter, needs_manual_review, note`.
`needs_manual_review` is `YES` when the top confidence is below `--threshold` (default 0.70). The confidence is the
softmax output of the network, **not a calibrated probability** of a defect - the evaluation shows wrong
predictions with confidence above 0.9, so a high confidence does not prove that the answer is right.

---

## 6. How it works (design decisions)

* **Classification, not detection.** `yolov8n-cls.pt` (pretrained on ImageNet) is fine-tuned to put each *whole photo*
  into one of three classes. Your 6G pipe-weld labels are image-level, so a detector was not possible.
* **Whole frame, not centre crop.** Ultralytics' default classification pipeline resizes and centre-crops to a square;
  for a 16:9 frame that discards about 44 % of the picture, where a defect may be. Here the full frame is
  squashed to 224 x 224, identically in training, evaluation and prediction (`weld_common.py`).
* **No leakage.** The dataset's splits are weld-wise: all images of a physical weld (W1-W4) stay in one split.
  Photos of one weld taken a few degrees apart look almost identical, so a random split would leak. This project
  never re-shuffles images between the splits. `prepare_dataset.py` re-checks this on every run.
* **`review_required` images are excluded** from training, validation and testing. They are only predicted when
  you ask for it (`evaluate.py --predict-review`), and then without any score.
* **Augmentation - training images only.** brightness +/-25 %, contrast +/-25 %, rotation +/-10 degrees, small
  shift (5 %) and zoom (10 %), Gaussian blur (30 % of the images, sigma 0.4-1.2), Gaussian noise (50 % of the
  images, sigma 2-8 grey levels). Validation, test and prediction images are never augmented. There is no flipping, mosaic or
  mixup: the training images come from this project's own data pipeline (`train.py`, `WeldDataset`), not from
  Ultralytics' augmentation code, so Ultralytics' augmentation settings that are printed in its log are not used.
* **Class imbalance.** Normal_Weld is far more common than the two defects. Default: a PyTorch
  `WeightedRandomSampler` (weight = 1 / class count), so each class is drawn about equally often.
  `--imbalance loss` uses a class-weighted cross-entropy instead. Ultralytics' classification trainer has no
  class-weight option of its own (checked in the installed version 8.4.188), so both were implemented in PyTorch
  and tested (the weights and the number of weighted loss calls are written to `training_config.json`).
* **Choosing best.pt.** The epoch with the highest validation **balanced accuracy** (mean recall over the classes
  that exist in the validation weld) - plain accuracy would reward predicting Normal_Weld all the time. The
  test weld is never used for this choice.
* **Training settings.** AdamW, lr 0.001, up to 100 epochs, early stopping (patience 20), pretrained weights,
  batch 16 (halved automatically when the GPU runs out of memory), seed 0, mixed precision on GPU. `nbs` is set equal
  to the batch size, otherwise Ultralytics accumulates gradients over 4 batches with such a small dataset, and
  `warmup_bias_lr` is kept at 0 for AdamW.
* **Missing classes are not zeros.** If a split has no image of a class, that class is `n/a` ("not measurable"),
  it is left out of the macro averages and the balanced accuracy, and predictions into such a class are
  counted separately as errors.

---

## 7. Troubleshooting

**`torch.cuda.is_available()` is False / training says "CPU"**
1. `nvidia-smi` must work. If not: install/update the NVIDIA driver, restart Windows.
2. `python check_environment.py` shows *"PyTorch ... is a CPU-ONLY build"*? Then PyTorch was installed without CUDA:
   ```bat
   pip uninstall -y torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
   (use the command from the pytorch.org selector).
3. A laptop with two GPUs may use the Intel GPU for apps: in *Windows Settings > System > Display > Graphics*
   set VS Code / python.exe to *High performance*.

**`CUDA error: no kernel image is available` / `CUDA driver version is insufficient`**
The PyTorch CUDA build is newer than the driver supports. Update the NVIDIA driver, or install a PyTorch build
with an older CUDA number (e.g. `cu126`). `nvidia-smi` shows the highest CUDA version the driver supports.

**`CUDA out of memory`** (6 GB is normally enough for batch 16 at 224 px)
`train.py` halves the batch size automatically and restarts. You can also start smaller: `python train.py --batch 8`.
Close other GPU programs (browsers with video, games) first. Check with `nvidia-smi`.
*The automatic halving was tested with a simulated error on the CPU; a real GPU out-of-memory error has not been seen.*

**`ERROR: No matching distribution found for torch` / `Could not find a version that satisfies the requirement torch`**
The newest PyTorch may not offer a wheel for your Python version or the CUDA index you named. Look at the command on
<https://pytorch.org/get-started/locally/> and, if needed, at <https://pytorch.org/get-started/previous-versions/>
for an older release that supports Python 3.10; then run `python check_environment.py` and the one-epoch test (step 6).
*This project was run with Python 3.13 and PyTorch 2.14.1; the combination Python 3.10 + your PyTorch has not been tested.*

**`ModuleNotFoundError: No module named 'ultralytics'` (or cv2, matplotlib, torch)**
The virtual environment is not active (`(.venv)` is missing in the prompt) or the package is not installed:
`.venv\Scripts\activate`, then `pip install -r requirements.txt` (for torch see step 3). In VS Code select the
`.venv` interpreter. `python check_environment.py` lists what is missing.

**`ModuleNotFoundError: No module named 'weld_common'` (or `'train'`)**
All `.py` files must stay together in one folder (`weld_common.py` is shared by all scripts and is not run itself).
The scripts find `models\`, `results\` and `dataset\` relative to their own folder, wherever you start them from.

**Dataset not found / wrong dataset path**
Scripts expect `dataset\weld_defect_dataset\` (with `train\`, `val\`, `test\`, `review_required\`,
`dataset_report.csv`, `weld_groups.csv`). Put `weld_defect_dataset.zip` next to the scripts and run
`python prepare_dataset.py`, or give a path: `python prepare_dataset.py --zip D:\downloads\weld_defect_dataset.zip`.
For an already extracted dataset elsewhere use `--data D:\path\to\weld_defect_dataset` with `train.py`,
`evaluate.py` and `cross_validate.py`.

**`prepare_dataset.py` reports problems**
Read the messages: they name the missing class folder, a corrupted image, a CSV row without a file, or a weld
that appears in two splits. Re-extract with `python prepare_dataset.py --force`.

**Model file not found (`models\best.pt`)**
Run `python train.py` first, or point to another file with `--weights`.

**Windows: `BrokenPipeError`, `RuntimeError: ... multiprocessing` or a hang at the first epoch**
Use `python train.py --workers 0`. (On CPU the scripts already use no worker processes.)

**NaN loss or AMP errors** - `python train.py --no-amp`.

**Camera: black or blank picture, wrong camera, "cannot open camera"**
`python predict_camera.py --find-camera` tests indexes 0-5, saves one picture of each as `camera_test_N.jpg` in the project folder and
tells you which ones work (blank pictures are detected). Then use `--camera N`. Close other programs that use
the camera (Zoom, Teams, the capture program). The script tries the DirectShow and Media Foundation back-ends
automatically. It asks the camera for 1920 x 1080 (the size the model was trained on).

**Everything is classified as Normal_Weld**
That is what the evaluation shows for unseen welds (section 1) - it is a data limitation, not a bug. See section 8.

---

## 8. How to get a model that works on new welds

The scores in section 1 say clearly: with two training welds the model learns *those welds*, not the defects.
What helps, in this order:

1. **More independent welds**, especially with every defect type on several different welds. Right now Burn_Through
   exists on only two welds (W2, W4) and Spatter on only two (W1, W3). A new weld is worth far more than 100 more
   photos of an old one. When you capture them, keep each weld in one split, as `weld_groups.csv` does.
2. The `review_required` images: once you have labelled them, they are extra data.
3. Keep the camera, lighting and distance identical to the later use, or train on photos taken in the same setup.
4. Re-run `python cross_validate.py` after every new weld - it is the number to trust.
5. Only then try bigger models (`--model yolov8s-cls.pt`), higher resolution (`--imgsz 384`), or
   `--imbalance loss`. Do not tune these on the test weld.

---

## 9. What was tested and what was not

**Tested (actually executed):** dataset extraction and verification; training; evaluation; leave-one-weld-out
cross-validation; `predict_image.py`, `predict_folder.py`; `predict_camera.py` on a video file instead of a camera;
a set of 36 automated checks (a development aid, not included in this ZIP) of the metric code, the missing-class handling, the augmentation (training only), the
weighted sampler/loss, the leak check and the prediction path (the prediction path reproduces the training
pipeline's probabilities to 3e-7).

**Not tested (could not be tested where this was built):**
* Windows 10 itself (the code was run on Linux; paths use `pathlib`, camera back-ends are chosen per OS).
* **Python 3.10** - the run used Python 3.13; the files were syntax-checked for 3.10 only.
* **The RTX 3050 / any CUDA GPU**, mixed precision (AMP) and the real GPU out-of-memory path. No GPU was available,
  so the training time on your GPU is unknown (on the CPU used here, training 26 epochs took about 4 minutes and the 4 cross-validation folds together roughly 20 minutes).
* A **real USB camera** (the live loop was run on a video file; the blank-frame and back-end fallback logic is
  untested on hardware).
* The Windows PyTorch/CUDA install commands.

If something fails on your PC, `python check_environment.py` and the troubleshooting section are the first steps.
