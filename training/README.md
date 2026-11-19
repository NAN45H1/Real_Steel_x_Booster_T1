# Training whole-body policies for Lindy

Lindy is our Booster T1 (23 joints). This folder trains a policy that makes Lindy copy a recorded motion with its whole body and keep its balance while doing it. Training runs on an NVIDIA RTX PC. Every policy is tested on the laptop in MuJoCo before it gets anywhere near the robot.

It wraps Booster's own code rather than replacing it:
- [booster_train](https://github.com/BoosterRobotics/booster_train): BeyondMimic motion tracking on Isaac Lab, with a T1 task.
- [booster_deploy](https://github.com/BoosterRobotics/booster_deploy): runs a policy in MuJoCo or on the robot.
- [booster_assets](https://github.com/BoosterRobotics/booster_assets): robot models and a T1 dance.

The setup script downloads all three at pinned commits into `external/`, which git ignores.

```
motion CSV ──► motion .npz ──► train (hours) ──► export ──► check in MuJoCo ──► Lindy (later, on the gantry)
          └──────────────── NVIDIA PC ────────────────┘    └──── laptop ────┘
```

| Script | Runs on | What it does |
|---|---|---|
| `setup_env.py sim` / `gpu` | laptop / NVIDIA PC | Installs everything into `external/`. Safe to re-run. |
| `lindy_train.py` | NVIDIA PC | Converts a motion, trains, and exports the policy to `training/policies/<name>/`. |
| `lindy_sim2sim.py` | laptop | `check` gives a pass/fail without opening a window. `watch` opens a MuJoCo window. |

**What this is and isn't.** BeyondMimic trains **one policy per motion**: the T1 dance, a punch combo, and so on. It is the first real step: the same Isaac Lab setup, robot model, motor limits, MuJoCo check and deploy path are what the live "Real Steel" shadowing needs. Live shadowing itself needs a general tracker trained on thousands of motions. That is the next stage, and it builds on this.

## 1. On the laptop (works now, no GPU)

```powershell
python training/setup_env.py sim
python training/lindy_sim2sim.py check
python training/lindy_sim2sim.py watch
```

The setup downloads about 300 MB: CPU PyTorch, MuJoCo, and Booster's repos.

`check` runs Booster's own trained T1 dance (`t1_dance.pt`) through all 33 seconds of the motion in MuJoCo. It takes about 7 s:

```
Trunk  : lowest 0.63 m, most tilted 12 deg
Joints : 0.066 rad average distance from the motion
Result : PASS, stayed up through the whole motion in MuJoCo
```

A policy fails if it falls, if Booster's safety stop fires (trunk more than 60° off the motion), or if its joints are on average more than 0.2 rad off the motion. A policy that just stands still fails within 2 s.

In `watch`, the solid robot is the physics simulation and the see-through robot is the motion it should follow. Space pauses.

## 2. On the borrowed NVIDIA PC

**What it needs** (Isaac Sim 5.1's requirements):
- An NVIDIA **RTX** card with **16 GB+** GPU memory: RTX 4080/4090/5080/5090, RTX A5000/A6000/Ada. Cards with 8–12 GB usually work with `--envs 2048`. A100/H100/V100 do **not** work, because they have no RT cores.
- Ubuntu 22.04/24.04 or Windows 10/11.
- 32 GB RAM, about 60 GB free disk, and NVIDIA driver 580 or newer.
- `git`, plus Python 3.11. If Python isn't 3.11, install [uv](https://docs.astral.sh/uv/) and the script gets 3.11 by itself:
  - Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

```bash
git clone https://github.com/NAN45H1/Real_Steel_x_Booster_T1.git
cd Real_Steel_x_Booster_T1
python training/setup_env.py gpu        # 30-60 min, ~20 GB download
python training/lindy_train.py check    # everything installed? CUDA visible?
```

On Ubuntu, type `python3` where these commands say `python`. `setup_env.py gpu` checks the GPU, driver, disk, and (Windows) long-path support or (Linux) glibc before downloading anything. It stops with the reason if something is wrong. You don't need to activate the environment: `lindy_train.py` switches to `external/gpu-venv` by itself.

### Smoke test first (15–30 minutes, mostly the first Isaac Sim start)

Run the whole pipeline on a tiny training run before starting a long one:

```bash
python training/lindy_train.py motion                                  # Booster's T1 dance -> training/motions/t1_dance.npz
python training/lindy_train.py train t1_dance --iterations 50 --envs 1024
python training/lindy_train.py export t1_dance
```

The first Isaac Sim start asks you to **accept NVIDIA's licence** (type `Yes`), then spends 5–10 minutes loading. After that it starts much faster. A 50-iteration policy will fail the laptop check, and that's expected. This run only proves the tools work end to end.

### The real run

```bash
python training/lindy_train.py train t1_dance             # use --envs 2048 on a card with less than 16 GB
```

- The log prints the time per iteration and an ETA. Booster's setting is 50,000 iterations.
- A checkpoint is saved every 1,000 iterations, so you can stop with **Ctrl+C** at any point and export the newest one.
- To continue later: `train t1_dance --resume`.
- Reward curves: `external/gpu-venv/bin/python -m tensorboard.main --logdir training/logs` (on Windows the Python is `external\gpu-venv\Scripts\python.exe`). If it says no module, `pip install tensorboard` into that Python first.

```bash
python training/lindy_train.py export t1_dance            # -> training/policies/t1_dance/
```

The export folder holds:
- `policy.pt` (TorchScript)
- `policy.onnx`
- `motion.npz`: the reference the policy follows.
- `info.json`: which checkpoint and iteration it came from.

It's a few MB, so commit and push it, then `git pull` on the laptop and run:

```powershell
python training/lindy_sim2sim.py check training/policies/t1_dance
python training/lindy_sim2sim.py watch training/policies/t1_dance
```

Booster ships a trained `t1_dance.pt` for the same dance, so it's the benchmark: ours should pass `check` at least as well.

Add `--dry-run` to any `lindy_train.py` command to see exactly what it would run. Anything after `--` goes to booster_train's `train.py`, for example `train t1_dance -- --seed 1`.

## 3. Our own motions

`lindy_train.py motion my_move.csv --fps 30` takes any motion in Booster's T1 CSV format:
- one row per frame, no header;
- root position `x y z` (m);
- root orientation quaternion `x y z w`;
- the 23 joint angles (rad) in this order: `head_yaw, head_pitch, left_shoulder_pitch, left_shoulder_roll, left_elbow_pitch, left_elbow_yaw, right_shoulder_pitch, right_shoulder_roll, right_elbow_pitch, right_elbow_yaw, waist_yaw, left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee_pitch, left_ankle_pitch, left_ankle_roll`, then the same six for the right leg.

Turning a Rokoko suit take into this needs a whole-body retargeter (human skeleton to T1 joints), which isn't written yet. [GMR](https://github.com/YanjieZe/GMR) can retarget SMPL-X/AMASS data to `booster_t1` today. Motions the robot can't physically do won't train, whatever the GPU.

## 4. Before a policy goes on Lindy

The robot side isn't written yet. When it is:
- It goes through booster_deploy, which needs Booster firmware 1.7.2 or newer and Booster's ROS 2 interface on the robot.
- The order is: pass `check` here, watch it, then run on the gantry with the remote in hand.
- booster_deploy's README warns about one thing in particular: the ankles are parallel joints, and the damping used in simulation must not be copied to the motors.

## Versions (pinned in `setup_env.py`)

| | Version |
|---|---|
| Isaac Sim | 5.1.0 (pip, Python 3.11) |
| Isaac Lab | v2.3.2, with rsl_rl |
| PyTorch (GPU PC) | 2.7.0 + CUDA 12.8 |
| booster_train | `83037fd` (2026-10-27), Booster tested it on Isaac Lab 2.3.2 / Isaac Sim 5.1.0 |
| booster_deploy | `7bb1462` (2026-09-13) |
| booster_assets | `12a9751` (2026-09-26), the same model as `lindy_viewer` |
| Laptop | PyTorch 2.9.1 CPU, MuJoCo 3.3.7 |

**Not verified yet:** nothing in section 2 has run, because there's no NVIDIA PC here. The commands follow Booster's and Isaac Lab's own instructions, and the tests check the commands and file handling, but expect to fix small things on the first real run. Section 1 runs and is tested on the laptop (`python -m unittest discover -s training`).
