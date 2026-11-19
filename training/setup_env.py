"""Install what Lindy's policy training needs, on either machine.

    python training/setup_env.py sim   # this laptop (no NVIDIA GPU): run policies in MuJoCo
    python training/setup_env.py gpu   # the borrowed NVIDIA PC: Isaac Sim 5.1 + Isaac Lab 2.3.2 + booster_train

Everything lands in external/ next to this repo, which git ignores. Safe to re-run:
finished steps are skipped, so after a failure fix the cause and run it again.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO / "external"
SIM_VENV = EXTERNAL / "sim-venv"
GPU_VENV = EXTERNAL / "gpu-venv"

# github.com/BoosterRobotics, pinned to the commits this repo was written against (2026-11-19).
BOOSTER = {
    "booster_assets": "12a97516f57469078707afb587a5f84c1cb110d1",
    "booster_train": "83037fde138c49cd8d6459a4b592538e6e979ffb",   # tested by Booster with Isaac Lab 2.3.2 / Isaac Sim 5.1.0
    "booster_deploy": "7bb1462e2ac742eba5c01659b815e2e99fbe617f",
}
ISAACLAB_TAG = "v2.3.2"
ISAACSIM = "isaacsim[all,extscache]==5.1.0"
NVIDIA_INDEX = "https://pypi.nvidia.com"
TORCH_GPU = ["torch==2.7.0", "torchvision==0.22.0", "--index-url", "https://download.pytorch.org/whl/cu128"]
TORCH_CPU = ["torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cpu"]
SIM_PACKAGES = ["numpy", "mujoco==3.3.7", "scipy", "pyyaml", "onnxruntime"]  # booster_deploy's list, minus the robot-only evdev
NO_RT_CORES = re.compile(r"\b(A100|A800|A30|H100|H800|H200|GH200|B100|B200|B300|V100|P100)\b")


class SetupError(Exception):
    pass


def say(msg):
    print("\n==> " + msg, flush=True)


def run(cmd, **kw):
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        raise SetupError("command failed (exit %d): %s" % (r.returncode, " ".join(str(c) for c in cmd)))


def venv_python(venv):
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed_version(py, dist):
    """Version of `dist` inside the interpreter `py`, or None."""
    r = subprocess.run([str(py), "-c", "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))", dist],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def pip(py, *args):
    run([py, "-m", "pip", "install", *args])


def git_checkout(url, dest, commit=None, tag=None):
    """Clone `url` into `dest` at `commit` (or `tag`); skip when it is already there."""
    if (dest / ".git").exists():
        want = commit or tag
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if commit and head == commit:
            print("   %s already at %s" % (dest.name, commit[:10]))
            return
        if tag:
            at = subprocess.run(["git", "-C", str(dest), "describe", "--tags", "--exact-match"],
                                capture_output=True, text=True).stdout.strip()
            if at == tag:
                print("   %s already at %s" % (dest.name, tag))
                return
        run(["git", "-C", dest, "fetch", "--tags", "origin"])
        run(["git", "-C", dest, "checkout", "--detach", want])
        return
    if tag:
        run(["git", "clone", "--depth", "1", "--branch", tag, url, dest])
    else:
        run(["git", "clone", url, dest])
        run(["git", "-C", dest, "checkout", "--detach", commit])


def booster(name):
    git_checkout("https://github.com/BoosterRobotics/%s.git" % name, EXTERNAL / name, commit=BOOSTER[name])
    return EXTERNAL / name


def make_venv(venv, need=None):
    """Create a venv. `need` = (major, minor) Python it must have; uses uv to fetch it when this Python differs."""
    py = venv_python(venv)
    if py.exists():
        print("   %s exists" % venv.name)
        return py
    if need is None or sys.version_info[:2] == need:
        run([sys.executable, "-m", "venv", venv])
    elif shutil.which("uv"):
        run(["uv", "venv", "--python", "%d.%d" % need, "--seed", venv])
    else:
        raise SetupError(
            "Isaac Sim 5.1 needs Python %d.%d and this is Python %d.%d. Either run this script with Python %d.%d, "
            "or install uv (it downloads the right Python by itself) and run this again:\n"
            "  Linux:   curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  Windows: powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\""
            % (need + sys.version_info[:2] + need))
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    return py


# ---------------------------------------------------------------- GPU PC checks


def parse_nvidia_smi(text):
    """'name, memory MiB, driver' lines -> [(name, gib, driver)]."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            mem = re.match(r"(\d+)", parts[1])
            gpus.append((parts[0], int(mem.group(1)) / 1024 if mem else 0.0, parts[2]))
    return gpus


def gpu_problems(gpus):
    """(errors, warnings) for running Isaac Sim on these GPUs."""
    if not gpus:
        return ["no NVIDIA GPU found (nvidia-smi printed nothing)"], []
    name, gib, driver = gpus[0]
    errors, warnings = [], []
    if NO_RT_CORES.search(name):
        errors.append("%s has no RT cores, and Isaac Sim does not run on it. Use an RTX card "
                      "(GeForce RTX 30/40/50, RTX A-series/Ada/PRO)." % name)
    elif "RTX" not in name.upper():
        warnings.append("%s is not an RTX card; Isaac Sim needs RT cores, so this may not work." % name)
    if gib < 15.5:
        warnings.append("%.0f GB of GPU memory (16 GB recommended): train with fewer robots, e.g. --envs 2048." % gib)
    major = re.match(r"(\d+)", driver)
    if major and int(major.group(1)) < 570:
        warnings.append("driver %s is old; Isaac Sim 5.1 was tested with 580. Update the NVIDIA driver if it fails." % driver)
    return errors, warnings


def check_gpu_pc():
    say("Checking this PC")
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise SetupError("nvidia-smi not found: this needs an NVIDIA GPU with its driver installed.")
    out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    gpus = parse_nvidia_smi(out)
    for g in gpus:
        print("   GPU: %s, %.0f GB, driver %s" % g)
    errors, warnings = gpu_problems(gpus)
    if platform.system() == "Linux":
        libc, ver = platform.libc_ver()
        if libc == "glibc" and tuple(int(x) for x in ver.split(".")[:2]) < (2, 35):
            errors.append("glibc %s is too old for the Isaac Sim pip package (needs 2.35+, Ubuntu 22.04 or newer)." % ver)
        if not shutil.which("cmake") or not shutil.which("g++"):
            warnings.append("cmake/g++ missing; if a pip step fails, run: sudo apt install cmake build-essential")
    elif platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
                if winreg.QueryValueEx(k, "LongPathsEnabled")[0] != 1:
                    warnings.append(
                        "Windows long paths are off, and the Isaac Sim install can fail on long file names. "
                        "To turn them on, run this in an admin PowerShell, then reboot:\n"
                        "     New-ItemProperty -Path HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
                        "-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force")
        except OSError:
            pass
    free = shutil.disk_usage(EXTERNAL.parent).free / 2**30
    if free < 50:
        warnings.append("only %.0f GB free disk; Isaac Sim needs about 50 GB." % free)
    for w in warnings:
        print("   WARNING: " + w)
    if errors:
        raise SetupError("\n".join(errors))


# ---------------------------------------------------------------- the two setups


def setup_sim():
    EXTERNAL.mkdir(exist_ok=True)
    say("Booster's model files and deploy code")
    assets = booster("booster_assets")
    booster("booster_deploy")
    say("Python environment for MuJoCo (CPU only): " + str(SIM_VENV))
    py = make_venv(SIM_VENV)
    if not installed_version(py, "torch"):
        pip(py, *TORCH_CPU)
    pip(py, *SIM_PACKAGES)
    if not installed_version(py, "booster_assets"):
        pip(py, "-e", assets)
    say("Done. Try Booster's own trained T1 dance:\n"
        "     python training/lindy_sim2sim.py check\n"
        "     python training/lindy_sim2sim.py watch")


def setup_gpu():
    EXTERNAL.mkdir(exist_ok=True)
    check_gpu_pc()
    say("Python 3.11 environment: " + str(GPU_VENV))
    py = make_venv(GPU_VENV, need=(3, 11))

    say("Isaac Sim 5.1.0 (large download, ~15 GB)")
    if installed_version(py, "isaacsim") != "5.1.0":
        pip(py, ISAACSIM, "--extra-index-url", NVIDIA_INDEX)
    say("PyTorch 2.7.0 with CUDA 12.8")
    if not (installed_version(py, "torch") or "").startswith("2.7.0"):
        pip(py, *TORCH_GPU)

    say("Isaac Lab " + ISAACLAB_TAG)
    lab = EXTERNAL / "IsaacLab"
    git_checkout("https://github.com/isaac-sim/IsaacLab.git", lab, tag=ISAACLAB_TAG)
    if not installed_version(py, "isaaclab_rl"):
        src = lab / "source"
        pip(py, "-e", src / "isaaclab", "-e", src / "isaaclab_assets", "-e", src / "isaaclab_tasks",
            "-e", str(src / "isaaclab_rl") + "[rsl_rl]")

    say("Booster's training code and robot models")
    assets = booster("booster_assets")
    train = booster("booster_train")
    if not installed_version(py, "booster_assets"):
        pip(py, "-e", assets)
    if not installed_version(py, "booster_train"):
        pip(py, "-e", train / "source" / "booster_train")

    say("Done. The first Isaac Sim start asks you to accept NVIDIA's licence (type Yes) and then takes\n"
        "    5-10 minutes to warm up. Next:\n"
        "     python training/lindy_train.py check\n"
        "     python training/lindy_train.py motion          # Booster's T1 dance -> motions/t1_dance.npz\n"
        "     python training/lindy_train.py train t1_dance")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machine", choices=("sim", "gpu"), help="sim = this laptop, gpu = the NVIDIA PC")
    args = ap.parse_args(argv)
    if not shutil.which("git"):
        print("git is needed: install it from https://git-scm.com and run this again.")
        return 1
    try:
        (setup_sim if args.machine == "sim" else setup_gpu)()
    except SetupError as e:
        print("\nSTOPPED: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
