#!/usr/bin/env python3
"""
check_environment.py - find out in 10 seconds why something does not work.

Prints the Python version, the installed packages, whether PyTorch can use the NVIDIA GPU (and really runs a
small calculation on it), whether the dataset and the trained model are where the scripts expect them, and
what to do about each problem.

Usage
    python check_environment.py
"""
import importlib
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
problems = []


def line(ok, text):
    print("  [%s] %s" % ("OK" if ok else "!!", text))


def main():
    print("Python / system")
    v = sys.version_info
    line(v[:2] >= (3, 8), "Python %s  (%s)" % (platform.python_version(), platform.platform()))
    if v[:2] != (3, 10):
        print("       note: this project was written for Python 3.10; %d.%d usually works too." % v[:2])
    line(sys.maxsize > 2 ** 32, "64-bit Python" if sys.maxsize > 2 ** 32 else "32-bit Python: PyTorch needs 64-bit Python")

    print("\nPackages")
    mods = [("numpy", "numpy"), ("cv2", "opencv-python"), ("matplotlib", "matplotlib"), ("torch", "torch"),
            ("torchvision", "torchvision"), ("ultralytics", "ultralytics")]
    loaded = {}
    for mod, pipname in mods:
        try:
            m = importlib.import_module(mod)
            loaded[mod] = m
            line(True, "%-12s %s" % (pipname, getattr(m, "__version__", "?")))
        except Exception as e:   # not only ImportError: a broken DLL raises OSError
            line(False, "%-12s MISSING or broken (%s: %s)" % (pipname, type(e).__name__, str(e).splitlines()[0][:100]))
            problems.append("pip install %s   (for torch see README step 3)" % pipname)
    if "ultralytics" in loaded and loaded["ultralytics"].__version__ != "8.4.188":
        print("       note: tested with ultralytics 8.4.188; you have %s. If train.py fails with an AttributeError or "
              "TypeError inside ultralytics, run  pip install ultralytics==8.4.188" % loaded["ultralytics"].__version__)

    print("\nGPU")
    line(shutil.which("nvidia-smi") is not None, "nvidia-smi found (NVIDIA driver installed)"
         if shutil.which("nvidia-smi") else "nvidia-smi not found: install/update the NVIDIA driver (geforce.com/drivers)")
    torch = loaded.get("torch")
    if torch is not None:
        build = torch.__version__
        cuda_build = torch.version.cuda
        line(cuda_build is not None, "PyTorch %s, built for CUDA %s" % (build, cuda_build) if cuda_build
             else "PyTorch %s is a CPU-ONLY build -> it can never use the GPU" % build)
        if cuda_build is None:
            problems.append("Reinstall PyTorch with CUDA support (README step 3), e.g. "
                            "pip uninstall -y torch torchvision  then the command from pytorch.org")
        avail = torch.cuda.is_available()
        line(avail, "torch.cuda.is_available() = %s" % avail)
        if avail:
            try:
                idx = 0
                props = torch.cuda.get_device_properties(idx)
                free, total = torch.cuda.mem_get_info(idx)
                line(True, "GPU 0: %s, %.1f GB total, %.1f GB free, compute capability %d.%d" % (
                    props.name, total / 1e9, free / 1e9, props.major, props.minor))
                x = torch.randn(512, 512, device="cuda")
                y = (x @ x).sum().item()
                torch.cuda.synchronize()
                line(True, "a test calculation on the GPU worked")
                if free / 1e9 < 2.0:
                    print("       note: less than 2 GB of GPU memory is free. Close other programs, or use --batch 8.")
            except Exception as e:
                line(False, "GPU test calculation FAILED: %s" % str(e).splitlines()[0][:120])
                problems.append("GPU is visible but unusable: update the NVIDIA driver, or install the PyTorch build "
                                "that matches your driver (README > Troubleshooting > CUDA)")
        elif cuda_build is not None:
            problems.append("PyTorch has CUDA but cannot find a GPU: update the NVIDIA driver and restart Windows; "
                            "check that your driver supports CUDA %s (nvidia-smi shows the maximum)" % cuda_build)
        if not avail:
            print("       Without a GPU the scripts still work, but training is much slower (CPU fallback is automatic).")

    print("\nProject files")
    ds = ROOT / "dataset" / "weld_defect_dataset"
    line((ds / "dataset_report.csv").is_file(), "dataset: %s" % ds if (ds / "dataset_report.csv").is_file()
         else "dataset not found at %s  -> put weld_defect_dataset.zip next to the scripts and run  python prepare_dataset.py" % ds)
    if not (ds / "dataset_report.csv").is_file():
        problems.append("python prepare_dataset.py   (extracts the dataset to %s)" % ds)
    line((ROOT / "models" / "best.pt").is_file(), "trained model models\\best.pt present" if (ROOT / "models" / "best.pt").is_file()
         else "models\\best.pt not found -> run  python train.py  (needed for evaluate / predict scripts)")
    pre = ROOT / "models" / "pretrained" / "yolov8n-cls.pt"
    line(pre.is_file(), "pretrained yolov8n-cls.pt present" if pre.is_file()
         else "yolov8n-cls.pt not downloaded yet (train.py downloads it automatically; needs internet once)")

    print()
    if problems:
        print("TO DO:")
        for p in problems:
            print("  -", p)
        return 1
    print("Everything needed is in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
