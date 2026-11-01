#!/usr/bin/env python3
"""
prepare_dataset.py - extract weld_defect_dataset.zip, verify it, and draw the class distribution.

What it does
  1. finds weld_defect_dataset.zip and extracts it into  dataset\\  (skipped if already extracted)
  2. checks the folder structure and the three class names
  3. reads dataset_report.csv and weld_groups.csv and compares them with the files on disk
  4. opens EVERY image to find corrupted files; finds exact duplicates (MD5)
  5. checks that no physical weld (weld group) and no pipe folder is in more than one split
  6. excludes review_required images (they are never used for training or testing)
  7. prints the image counts per class and split and saves results\\class_distribution.png

Your original images are only read, never modified, moved or re-saved.

Usage
    python prepare_dataset.py
    python prepare_dataset.py --zip D:\\downloads\\weld_defect_dataset.zip
    python prepare_dataset.py --check-only          # only verify an already extracted dataset
Exit code 0 = dataset is fine, 1 = a problem was found (the messages say which).
"""
import argparse
import hashlib
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from weld_common import (CLASSES, DEFAULT_DATASET, IMAGE_EXTS, RESULTS_DIR, ROOT, SPLITS, SHORT,
                         labelled_items, read_csv_rows, read_image_bgr, write_csv, write_json, _mpl)


def find_zip(arg):
    candidates = [Path(arg)] if arg else []
    candidates += [ROOT / "weld_defect_dataset.zip", ROOT / "dataset" / "weld_defect_dataset.zip"]
    for c in candidates:
        if c.is_file():
            return c
    return None


def safe_extract(zip_path, dest, overwrite):
    """Extract zip_path into dest. Refuses entries that would be written outside dest (zip-slip)."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n_new = n_skip = 0
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            sys.exit("ERROR: the ZIP file is damaged (first bad entry: %s). Download it again." % bad)
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                sys.exit("ERROR: unsafe path inside the ZIP: %s" % info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and not overwrite:
                n_skip += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
            n_new += 1
    print("Extracted %d files into %s (%d already existed and were left untouched)." % (n_new, dest, n_skip))


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def images_in(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main():
    ap = argparse.ArgumentParser(description="Extract and verify weld_defect_dataset.zip")
    ap.add_argument("--zip", help="path of weld_defect_dataset.zip (default: next to this script or in dataset\\)")
    ap.add_argument("--data", default=str(DEFAULT_DATASET), help="dataset folder (default %(default)s)")
    ap.add_argument("--check-only", action="store_true", help="do not extract, only verify")
    ap.add_argument("--force", action="store_true", help="re-extract and overwrite existing files from the ZIP")
    ap.add_argument("--results", default=str(RESULTS_DIR), help="where to save the check report and chart")
    args = ap.parse_args()

    data = Path(args.data)
    results = Path(args.results)
    errors, warnings = [], []

    # 1. extraction --------------------------------------------------------------------------
    if not args.check_only:
        if data.is_dir() and (data / "dataset_report.csv").is_file() and not args.force:
            print("Dataset already extracted at %s (use --force to extract again)." % data)
        else:
            zp = find_zip(args.zip)
            if zp is None:
                sys.exit("ERROR: weld_defect_dataset.zip not found. Put it in %s or in %s, or use --zip <path>."
                         % (ROOT, ROOT / "dataset"))
            print("Using ZIP:", zp)
            safe_extract(zp, data.parent, overwrite=args.force)
    if not (data / "dataset_report.csv").is_file():
        sys.exit("ERROR: %s is not the weld dataset (dataset_report.csv missing). Check --data / the ZIP location." % data)

    # 2. structure ----------------------------------------------------------------------------
    for sp in SPLITS:
        if not (data / sp).is_dir():
            errors.append("missing folder: %s" % (data / sp))
            continue
        found = sorted(p.name for p in (data / sp).iterdir() if p.is_dir())
        if found != sorted(CLASSES):
            errors.append("folder %s has class folders %s, expected %s" % (sp, found, sorted(CLASSES)))
    if not (data / "review_required").is_dir():
        warnings.append("no review_required folder (nothing to exclude)")
    for name in ("dataset_report.csv", "weld_groups.csv", "class_summary.csv"):
        if not (data / name).is_file():
            errors.append("missing file: " + name)
    if errors:
        return finish(errors, warnings)

    # 3. CSV reports vs files on disk ----------------------------------------------------------
    report = read_csv_rows(data / "dataset_report.csv")
    raw = [r for r in report if r["image_type"] == "raw"]
    labelled = [r for r in raw if r["review_status"] == "labelled"]
    review = [r for r in raw if r["review_status"] == "review_required"]
    classes_in_csv = sorted({r["assigned_class"] for r in labelled})
    if classes_in_csv != sorted(CLASSES):
        errors.append("class names in dataset_report.csv are %s, expected %s" % (classes_in_csv, sorted(CLASSES)))

    on_disk = {}
    for sp in SPLITS:
        for c in CLASSES:
            for p in images_in(data / sp / c):
                on_disk[(sp, c, p.name)] = p
    for r in labelled:
        key = (r["split"], r["assigned_class"], r["image_filename"])
        if key not in on_disk:
            errors.append("listed in dataset_report.csv but not found on disk: %s/%s/%s" % key)
    listed = {(r["split"], r["assigned_class"], r["image_filename"]) for r in labelled}
    for key in on_disk:
        if key not in listed:
            errors.append("image on disk but not listed as labelled in dataset_report.csv: %s/%s/%s" % key)
    review_on_disk = {p.name for p in images_in(data / "review_required")}
    review_listed = {r["image_filename"] for r in review}
    if review_on_disk != review_listed:
        errors.append("review_required folder (%d files) and dataset_report.csv (%d rows) disagree"
                      % (len(review_on_disk), len(review_listed)))
    if len({r["image_filename"] for r in raw}) != len(raw):
        errors.append("duplicate image file names in dataset_report.csv")

    groups = read_csv_rows(data / "weld_groups.csv")
    folder_to_group = {}
    for g in groups:
        for folder in g["folders"].split():
            folder_to_group[folder] = (g["weld_group"], g["split"])
    for r in labelled:
        gsp = folder_to_group.get(r["pipe_id"])
        if gsp is None:
            errors.append("pipe %s is not listed in weld_groups.csv" % r["pipe_id"])
        elif gsp != (r["weld_group"], r["split"]):
            errors.append("%s: dataset_report.csv says group/split %s/%s but weld_groups.csv says %s/%s"
                          % (r["image_filename"], r["weld_group"], r["split"], gsp[0], gsp[1]))

    # 4. open every image; MD5 for duplicates -------------------------------------------------------
    manifest, corrupted = [], []
    all_files = [(sp, c, p) for (sp, c, _), p in on_disk.items()] + [("review_required", "", data / "review_required" / n)
                                                                    for n in sorted(review_on_disk)]
    hashes = defaultdict(list)
    for sp, c, p in all_files:
        row = dict(split=sp, class_name=c, file=str(p.relative_to(data)).replace("\\", "/"), md5="", width="", height="", status="ok")
        try:
            img = read_image_bgr(p)
            row["height"], row["width"] = img.shape[:2]
            row["md5"] = md5_of(p)
            hashes[row["md5"]].append(row)
        except (ValueError, OSError) as e:
            row["status"] = "CORRUPTED: %s" % e
            corrupted.append(row)
        manifest.append(row)
    dup_groups = [rows for rows in hashes.values() if len(rows) > 1]
    for rows in dup_groups:
        splits = {r["split"] for r in rows}
        msg = "identical files: " + ", ".join(r["file"] for r in rows)
        if len(splits) > 1:
            errors.append("LEAKAGE - " + msg + " (in different splits)")
        else:
            warnings.append(msg)
    for r in corrupted:
        (errors if r["split"] != "review_required" else warnings).append("corrupted image: %s (%s)" % (r["file"], r["status"]))
    sizes = Counter("%sx%s" % (r["width"], r["height"]) for r in manifest if r["status"] == "ok")

    # 5. leakage between physical welds / folders --------------------------------------------------
    group_splits, pipe_splits = defaultdict(set), defaultdict(set)
    for r in labelled:
        group_splits[r["weld_group"]].add(r["split"])
        pipe_splits[r["pipe_id"]].add(r["split"])
    for g, sps in group_splits.items():
        if len(sps) > 1:
            errors.append("LEAKAGE: weld group %s appears in splits %s" % (g, sorted(sps)))
    for p, sps in pipe_splits.items():
        if len(sps) > 1:
            errors.append("LEAKAGE: pipe folder %s appears in splits %s" % (p, sorted(sps)))

    # 6. counts, chart, reports ----------------------------------------------------------------------
    counts = {sp: {c: 0 for c in CLASSES} for sp in SPLITS}
    for r in labelled:
        counts[r["split"]][r["assigned_class"]] += 1
    print("\nImage counts (labelled images only; review_required are excluded)")
    print("%-8s" % "split" + "".join("%14s" % c for c in CLASSES) + "%8s" % "total" + "   weld groups")
    for sp in SPLITS:
        wg = sorted({r["weld_group"] for r in labelled if r["split"] == sp})
        print("%-8s" % sp + "".join("%14d" % counts[sp][c] for c in CLASSES) + "%8d" % sum(counts[sp].values()) + "   " + "+".join(wg))
    tot = {c: sum(counts[sp][c] for sp in SPLITS) for c in CLASSES}
    print("%-8s" % "all" + "".join("%14d" % tot[c] for c in CLASSES) + "%8d" % sum(tot.values()))
    print("\nraw photos: %d | labelled: %d | review_required (excluded): %d | corrupted: %d | exact-duplicate groups: %d"
          % (len(raw), len(labelled), len(review), len(corrupted), len(dup_groups)))
    print("image sizes found:", dict(sizes))
    print("weld groups:", {g: sorted(s) for g, s in group_splits.items()})

    absent = [(sp, c) for sp in SPLITS for c in CLASSES if counts[sp][c] == 0]
    for sp, c in absent:
        warnings.append("class %s has NO images in the %s split - nothing can be measured for it there" % (c, sp))

    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "dataset_manifest.csv", manifest, ["split", "class_name", "file", "md5", "width", "height", "status"])
    write_csv(results / "corrupted_images.csv", corrupted, ["split", "class_name", "file", "md5", "width", "height", "status"])
    write_json(results / "dataset_check.json", dict(
        dataset=str(data), raw_photos=len(raw), labelled=len(labelled), review_required_excluded=len(review),
        corrupted=len(corrupted), duplicate_groups=len(dup_groups), counts=counts, image_sizes=dict(sizes),
        weld_groups={g: sorted(s) for g, s in group_splits.items()}, errors=errors, warnings=warnings))
    plot_distribution(counts, len(review), results / "class_distribution.png")
    print("\nSaved: %s, %s, %s, %s" % (results / "class_distribution.png", results / "dataset_manifest.csv",
                                      results / "corrupted_images.csv", results / "dataset_check.json"))
    return finish(errors, warnings)


def plot_distribution(counts, n_review, path):
    import numpy as np
    plt = _mpl()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(CLASSES))
    w = 0.26
    colors = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}
    for k, sp in enumerate(SPLITS):
        vals = [counts[sp][c] for c in CLASSES]
        bars = a1.bar(x + (k - 1) * w, vals, w, label=sp, color=colors[sp])
        for b, v, c in zip(bars, vals, CLASSES):
            a1.text(b.get_x() + b.get_width() / 2, v + 2, str(v) if v else "none", ha="center", fontsize=8,
                    color="black" if v else "#b00020")
    a1.set_xticks(x)
    a1.set_xticklabels(CLASSES)
    a1.set_ylabel("images")
    a1.set_title("Images per class in each split (original weld-wise split)")
    a1.legend()
    a1.grid(axis="y", alpha=.3)
    tot = [sum(counts[sp][c] for sp in SPLITS) for c in CLASSES]
    labels = CLASSES + ["review_required\n(excluded)"]
    vals = tot + [n_review]
    bars = a2.bar(range(len(vals)), vals, color=["#4c78a8", "#72b7b2", "#e45756", "#bbbbbb"])
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center", fontsize=9)
    a2.set_xticks(range(len(vals)))
    a2.set_xticklabels(labels, fontsize=8, rotation=15)
    a2.set_title("Total labelled images (%d)" % sum(tot))
    a2.grid(axis="y", alpha=.3)
    missing = [("%s has no %s" % (sp, c)) for sp in SPLITS for c in CLASSES if counts[sp][c] == 0]
    if missing:
        fig.text(0.5, 0.005, "Missing in the original split: " + "; ".join(missing) +
                 "  (only 4 independent welds - see README)", ha="center", fontsize=8, color="#b00020")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def finish(errors, warnings):
    print()
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:  ", e)
    if errors:
        print("\nDATASET CHECK FAILED (%d error(s))." % len(errors))
        return 1
    print("\nDATASET CHECK PASSED: all images readable, files match the CSV reports, "
          "no physical weld is shared between splits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
