"""Train Lindy (Booster T1) to copy a motion, on the NVIDIA PC.

Wraps Booster's booster_train (BeyondMimic motion tracking on Isaac Lab):

    python training/lindy_train.py check                  # what is installed, which GPU
    python training/lindy_train.py motion                 # Booster's T1 dance CSV -> training/motions/t1_dance.npz
    python training/lindy_train.py motion my_move.csv --fps 30
    python training/lindy_train.py train t1_dance         # hours; saves a checkpoint every 1000 iterations
    python training/lindy_train.py export t1_dance        # -> training/policies/t1_dance/

Bring training/policies/<name>/ back to the laptop and test it there first:
    python training/lindy_sim2sim.py check training/policies/<name>

Add --dry-run to print the commands without running them. Set up first with:
    python training/setup_env.py gpu
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINING = REPO / "training"
EXTERNAL = REPO / "external"
GPU_VENV = EXTERNAL / "gpu-venv"
BOOSTER_TRAIN = EXTERNAL / "booster_train"
BOOSTER_ASSETS = EXTERNAL / "booster_assets"
MOTIONS = TRAINING / "motions"
POLICIES = TRAINING / "policies"
RUNS = TRAINING / "logs" / "rsl_rl" / "t1_motion_tracking"  # booster_train writes logs/rsl_rl/<experiment> under cwd

TASK = "Booster-T1-MotionTracking-v0"        # T1 23 DoF, rough floor, no base-velocity input (Lindy has no estimator)
PLAY_TASK = TASK + "-Play"
DEMO_CSV = BOOSTER_ASSETS / "motions" / "T1" / "t1_dance.csv"  # 50 Hz, the motion Booster's own t1_dance.pt was trained on
CSV_COLUMNS = 7 + 23  # root xyz, root quaternion xyzw, 23 joints in Booster's T1 order


def venv_python(venv):
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def motion_override(npz):
    path = Path(npz).resolve().as_posix()
    if "'" in path:
        raise SystemExit("the motion path can't contain a ' character: %s" % path)
    return "env.commands.motion.motion_file='%s'" % path


def check_csv(csv):
    """Catch a wrong CSV before Isaac Sim spends minutes starting up."""
    with open(csv) as f:
        first = f.readline().strip().split(",")
    try:
        [float(v) for v in first]
    except ValueError:  # a header row; booster_train checks the joint names itself
        return
    if len(first) != CSV_COLUMNS:
        raise SystemExit("%s has %d columns; a T1 motion needs %d (root xyz, quaternion xyzw, 23 joints)"
                         % (csv, len(first), CSV_COLUMNS))


def motion_cmd(py, csv, fps, name):
    out = MOTIONS / (name + ".npz")
    return [str(py), str(BOOSTER_TRAIN / "scripts" / "csv_to_npz.py"), "--headless", "--robot", "t1",
            "--input_file", str(Path(csv).resolve()), "--input_fps", str(fps),
            "--output_file", str(out), "--output_fps", "50"], out


def find_motion(name):
    npz = Path(name)
    if npz.suffix != ".npz":
        npz = MOTIONS / (name + ".npz")
    if not npz.is_file():
        raise SystemExit("no motion %s; make it first: python training/lindy_train.py motion ..." % npz)
    return npz


def train_cmd(py, name, envs, iterations, resume, extra):
    cmd = [str(py), str(BOOSTER_TRAIN / "scripts" / "rsl_rl" / "train.py"), "--task", TASK, "--headless",
           "--num_envs", str(envs), "--run_name", name]
    if iterations:
        cmd += ["--max_iterations", str(iterations)]
    if resume:
        cmd += ["--resume", "--load_run", latest_run(name).name]
    return cmd + list(extra) + [motion_override(find_motion(name))]


def latest_run(name, runs=RUNS):
    """Newest training run folder for this motion (folders are <date>_<time>_<name>)."""
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_" + re.escape(name))
    found = sorted(p for p in runs.iterdir() if p.is_dir() and pattern.fullmatch(p.name)) if runs.is_dir() else []
    if not found:
        raise SystemExit("no training run for %s in %s" % (name, runs))
    return found[-1]


def latest_checkpoint(run):
    ckpts = [(int(m.group(1)), p) for p in run.glob("model_*.pt") if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))]
    if not ckpts:
        raise SystemExit("no model_*.pt in %s yet (the first one is written at iteration 0)" % run)
    return max(ckpts)


def export_cmd(py, name, checkpoint):
    return [str(py), str(BOOSTER_TRAIN / "scripts" / "rsl_rl" / "play.py"), "--task", PLAY_TASK, "--headless",
            "--num_envs", "1", "--checkpoint", str(checkpoint), motion_override(find_motion(name))]


def bundle(name, run, iteration, checkpoint, motion, dest_root=POLICIES):
    """Copy play.py's exported files plus the motion into one folder the laptop can use."""
    exported = run / "exported"
    stem = "t1_motion_tracking_" + run.name
    dest = dest_root / name
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ("pt", "onnx"):
        src = exported / ("%s.%s" % (stem, ext))
        if not src.is_file():
            raise SystemExit("export did not write %s" % src)
        shutil.copyfile(src, dest / ("policy." + ext))
    shutil.copyfile(motion, dest / "motion.npz")
    commit = subprocess.run(["git", "-C", str(BOOSTER_TRAIN), "rev-parse", "HEAD"], capture_output=True, text=True)
    info = {
        "motion": name,
        "task": TASK,
        "iteration": iteration,
        "checkpoint": str(checkpoint),
        "booster_train_commit": commit.stdout.strip() or None,
        "exported": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (dest / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return dest


def status():
    """Print what is installed. Does not start Isaac Sim (that takes minutes)."""
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        print("GPU           : " + (out or "none found"))
    else:
        print("GPU           : nvidia-smi not found")
    import importlib.metadata as md
    ok = True
    for dist in ("isaacsim", "isaaclab", "isaaclab_rl", "rsl-rl-lib", "torch", "booster_train", "booster_assets"):
        try:
            v = md.version(dist)
        except md.PackageNotFoundError:
            v, ok = "MISSING", False
        print("%-14s: %s" % (dist, v))
    try:
        import torch
        print("CUDA in torch : %s" % torch.cuda.is_available())
        ok = ok and torch.cuda.is_available()
    except ImportError:
        pass
    print("demo motion   : %s" % ("ok" if DEMO_CSV.is_file() else "MISSING " + str(DEMO_CSV)))
    print("\n" + ("Ready." if ok else "Something is missing: run  python training/setup_env.py gpu"))
    return 0 if ok else 1


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    argv = raw
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="show what is installed")
    m = sub.add_parser("motion", help="turn a T1 motion CSV into the .npz training needs")
    m.add_argument("csv", nargs="?", default=str(DEMO_CSV), help="default: Booster's T1 dance")
    m.add_argument("--fps", type=int, default=50, help="frame rate of the CSV (Booster's are 50)")
    m.add_argument("--name", help="motion name (default: the CSV file name)")
    t = sub.add_parser("train", help="train a policy that copies the motion",
                       epilog="Anything after -- goes to booster_train's train.py, e.g. -- --seed 1")
    t.add_argument("name", help="motion name, e.g. t1_dance")
    t.add_argument("--envs", type=int, default=4096, help="robots simulated at once (use 2048 on a <16 GB GPU)")
    t.add_argument("--iterations", type=int, help="stop after this many (Booster's default: 50000)")
    t.add_argument("--resume", action="store_true", help="continue the newest run of this motion")
    e = sub.add_parser("export", help="export the newest checkpoint for the laptop and the robot")
    e.add_argument("name")
    extra = []
    if "--" in argv:
        extra = argv[argv.index("--") + 1:]
        argv = argv[:argv.index("--")]
    dry_run = "--dry-run" in argv  # allowed anywhere before --
    args = ap.parse_args([a for a in argv if a != "--dry-run"])
    args.dry_run = dry_run

    py = venv_python(GPU_VENV)
    if not args.dry_run and Path(sys.prefix).resolve() != GPU_VENV.resolve():
        if not py.exists():
            print("Set up first:  python training/setup_env.py gpu")
            return 1
        return subprocess.call([str(py), str(Path(__file__).resolve()), *raw])

    if args.cmd == "check":
        return status()

    if args.cmd == "motion":
        check_csv(args.csv)
        name = args.name or Path(args.csv).stem
        cmd, out = motion_cmd(py, args.csv, args.fps, name)
        steps = [(cmd, None)]
        done = "Motion saved: %s\nNext:  python training/lindy_train.py train %s" % (out, name)
    elif args.cmd == "train":
        steps = [(train_cmd(py, args.name, args.envs, args.iterations, args.resume, extra), TRAINING)]
        done = ("Training stopped. Export the newest checkpoint with:  python training/lindy_train.py export %s"
                % args.name)
    else:
        run = latest_run(args.name)
        iteration, ckpt = latest_checkpoint(run)
        print("Exporting %s (iteration %d)" % (ckpt, iteration))
        steps = [(export_cmd(py, args.name, ckpt), TRAINING)]
        done = None

    for cmd, cwd in steps:
        print("$ " + subprocess.list2cmdline(cmd))
        if args.dry_run:
            continue
        MOTIONS.mkdir(parents=True, exist_ok=True)
        code = subprocess.call(cmd, cwd=cwd)
        if code != 0 and not (args.cmd == "train" and code in (130, -2, 3221225786)):  # Ctrl+C ends training
            print("\nFailed with exit code %d" % code)
            return code
    if args.dry_run:
        return 0
    if args.cmd == "export":
        dest = bundle(args.name, run, iteration, ckpt, find_motion(args.name))
        done = ("Policy ready: %s\nOn the laptop:  python training/lindy_sim2sim.py check %s"
                % (dest, dest.relative_to(REPO).as_posix()))
    print("\n" + done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
