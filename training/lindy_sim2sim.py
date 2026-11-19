"""Run a T1 motion-tracking policy in MuJoCo on this laptop (CPU only), before it goes anywhere near Lindy.

    python training/lindy_sim2sim.py check                               # Booster's own trained T1 dance, no window
    python training/lindy_sim2sim.py watch                               # the same in a MuJoCo window (space = pause)
    python training/lindy_sim2sim.py check training/policies/t1_dance    # a policy we trained (policy.pt + motion.npz)

Isaac Lab trains in PhysX; MuJoCo is a different physics engine. A policy that only works in the
simulator it was trained in will fall over here, and would fall over on the real robot too.
`check` runs the whole motion as fast as the CPU allows and passes only if the robot stays up.
It drives Booster's booster_deploy code, installed by:  python training/setup_env.py sim
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO / "external"
DEPLOY = EXTERNAL / "booster_deploy"
SIM_VENV = EXTERNAL / "sim-venv"
TASK = "t1_motion_tracking"
BEYOND_MIMIC = DEPLOY / "tasks" / "beyond_mimic"  # booster_deploy resolves motion paths relative to this
MIN_TRUNK_HEIGHT = 0.40  # m; Lindy's trunk is ~0.65 m up when standing
EXTRA_SECONDS = 2.0      # keep balancing on the last frame this long after the motion ends
MAX_JOINT_ERROR = 0.20   # rad, average; Booster's trained dance gets ~0.07, a robot that ignores the motion ~0.3


def venv_python(venv):
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def in_venv(venv):
    return Path(sys.prefix).resolve() == venv.resolve()


def find_policy(folder):
    """A folder made by `lindy_train.py export` -> (policy file, motion.npz)."""
    d = Path(folder)
    policy = next((d / n for n in ("policy.pt", "policy.onnx") if (d / n).is_file()), None)
    motion = d / "motion.npz"
    if policy is None or not motion.is_file():
        raise SystemExit("%s needs policy.pt (or policy.onnx) and motion.npz in it" % d)
    return policy, motion


def motion_path_for_deploy(motion):
    """booster_deploy wants the motion path relative to its tasks/beyond_mimic folder."""
    motion = Path(motion).resolve()
    try:
        return os.path.relpath(motion, BEYOND_MIMIC).replace(os.sep, "/")
    except ValueError:  # Windows: another drive; copy it next to Booster's motions instead
        dest = BEYOND_MIMIC / "robots" / "t1" / "motions" / ("lindy_" + motion.name)
        shutil.copyfile(motion, dest)
        return os.path.relpath(dest, BEYOND_MIMIC).replace(os.sep, "/")


def load_task(policy=None, motion=None, ghost=True):
    """Booster's T1 motion-tracking deploy config, pointed at our policy when given."""
    sys.path.insert(0, str(DEPLOY))
    import importlib
    importlib.import_module("tasks.beyond_mimic.robots.t1")  # registers t1_motion_tracking
    from booster_deploy.utils.registry import get_task
    cfg = get_task(TASK)
    cfg.policy.device = "cpu"
    cfg.mujoco.visualize_reference_ghost = ghost
    if policy is not None:
        cfg.policy.checkpoint_path = str(Path(policy).resolve())
        cfg.policy.motion_path = motion_path_for_deploy(motion)
    return cfg


def trunk_up(quat_wxyz):
    """z of the trunk's up axis: 1 upright, 0 lying down."""
    _, x, y, _ = quat_wxyz
    return 1.0 - 2.0 * (x * x + y * y)


def check(cfg, extra_seconds=EXTRA_SECONDS):
    """Run the motion headless. Returns a dict with the verdict and numbers."""
    from booster_deploy.controllers.mujoco_controller import MujocoController
    c = MujocoController(cfg)
    c.update_state()
    c.start()
    frames = int(c.policy.motion.time_step_total)
    steps = frames + int(round(extra_seconds / cfg.policy_dt))
    real2sim = c.robot.data.real2sim_joint_indexes
    low, worst_tilt, errors, step, fell_at = math.inf, 0.0, [], 0, None
    while c.is_running and step < steps:
        c.update_state()
        targets = c.policy_step()
        if step < frames:
            q = c.robot.data.joint_pos[real2sim]
            errors.append(float((q - c.policy.cmd_dof_pos).abs().mean()))
        c.ctrl_step(targets)
        z = float(c.mj_data.qpos[2])
        low = min(low, z)
        worst_tilt = max(worst_tilt, math.degrees(math.acos(max(-1.0, min(1.0, trunk_up(c.mj_data.qpos[3:7]))))))
        step += 1
        if z < MIN_TRUNK_HEIGHT and fell_at is None:
            fell_at = step * cfg.policy_dt
            break
    stopped = step < steps and fell_at is None  # booster_deploy's own safety stop: trunk >60 deg off the motion
    result = {
        "frames": frames,
        "seconds": frames * cfg.policy_dt,
        "ran_seconds": step * cfg.policy_dt,
        "lowest_trunk": low,
        "worst_tilt_deg": worst_tilt,
        "mean_joint_error": sum(errors) / len(errors) if errors else float("nan"),
        "fell_at": fell_at,
        "safety_stop_at": step * cfg.policy_dt if stopped else None,
    }
    result["lost_motion"] = not result["mean_joint_error"] <= MAX_JOINT_ERROR
    result["passed"] = fell_at is None and not stopped and not result["lost_motion"]
    return result


def report(name, r):
    print("Policy : %s" % name)
    print("Motion : %d frames, %.1f s (then %.0f s holding the last pose)" % (r["frames"], r["seconds"], EXTRA_SECONDS))
    print("Trunk  : lowest %.2f m, most tilted %.0f deg" % (r["lowest_trunk"], r["worst_tilt_deg"]))
    print("Joints : %.3f rad average distance from the motion" % r["mean_joint_error"])
    if r["passed"]:
        print("Result : PASS, stayed up through the whole motion in MuJoCo")
    elif r["fell_at"] is not None:
        print("Result : FAIL, fell at %.1f s" % r["fell_at"])
    elif r["safety_stop_at"] is not None:
        print("Result : FAIL, booster_deploy's safety stop fired at %.1f s (trunk >60 deg off the motion)"
              % r["safety_stop_at"])
    else:
        print("Result : FAIL, stayed up but did not follow the motion (joints off by more than %.2f rad)"
              % MAX_JOINT_ERROR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "watch"))
    ap.add_argument("policy", nargs="?", help="folder from `lindy_train.py export` (default: Booster's T1 dance)")
    args = ap.parse_args(argv)

    if not in_venv(SIM_VENV):
        py = venv_python(SIM_VENV)
        if not (py.exists() and DEPLOY.exists()):
            print("Set up first:  python training/setup_env.py sim")
            return 1
        return subprocess.call([str(py), str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])

    policy = motion = None
    name = "Booster's trained T1 dance (booster_deploy t1_dance.pt)"
    if args.policy:
        policy, motion = find_policy(args.policy)
        name = str(policy)
    cfg = load_task(policy, motion, ghost=args.mode == "watch")
    if args.mode == "watch":
        from booster_deploy.controllers.mujoco_controller import MujocoController
        print("Solid robot = physics with the policy; see-through robot = the motion it should follow. Close the window to stop.")
        MujocoController(cfg).run()
        return 0
    r = check(cfg)
    report(name, r)
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
