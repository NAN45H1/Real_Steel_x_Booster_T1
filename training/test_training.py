"""Tests for the training scripts. Run:  python -m unittest discover -s training

The GPU parts can't run on the laptop, so those tests check the commands and file handling.
The MuJoCo tests run Booster's real trained T1 dance when `setup_env.py sim` has been done.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import lindy_sim2sim as sim
import lindy_train as train
import setup_env

HERE = Path(__file__).resolve().parent


class GpuCheckTests(unittest.TestCase):
    def test_parses_nvidia_smi(self):
        gpus = setup_env.parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564 MiB, 580.88\n")
        self.assertEqual(gpus, [("NVIDIA GeForce RTX 4090", 24564 / 1024, "580.88")])

    def test_good_card_has_no_problems(self):
        self.assertEqual(setup_env.gpu_problems([("NVIDIA GeForce RTX 4090", 24.0, "580.88")]), ([], []))

    def test_cards_without_rt_cores_are_refused(self):
        for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3", "Tesla V100-SXM2-32GB"):
            errors, _ = setup_env.gpu_problems([(name, 80.0, "580.65")])
            self.assertEqual(len(errors), 1, name)

    def test_small_card_and_old_driver_warn(self):
        errors, warnings = setup_env.gpu_problems([("NVIDIA GeForce RTX 3070", 8.0, "535.10")])
        self.assertEqual(errors, [])
        self.assertTrue(any("--envs 2048" in w for w in warnings))
        self.assertTrue(any("driver" in w for w in warnings))

    def test_no_gpu(self):
        self.assertEqual(len(setup_env.gpu_problems([])[0]), 1)

    def test_every_booster_repo_is_pinned_to_a_full_commit(self):
        for name, commit in setup_env.BOOSTER.items():
            self.assertRegex(commit, r"^[0-9a-f]{40}$", name)


class TrainCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_motion_override_is_quoted_for_hydra(self):
        o = train.motion_override(self.dir / "my motions" / "t1_dance.npz")
        self.assertTrue(o.startswith("env.commands.motion.motion_file='"))
        self.assertTrue(o.endswith("/my motions/t1_dance.npz'"))
        self.assertNotIn("\\", o)

    def test_motion_command(self):
        cmd, out = train.motion_cmd("py", self.dir / "move.csv", 30, "move")
        self.assertEqual(out, train.MOTIONS / "move.npz")
        self.assertIn("--headless", cmd)
        self.assertEqual(cmd[cmd.index("--robot") + 1], "t1")
        self.assertEqual(cmd[cmd.index("--input_fps") + 1], "30")
        self.assertEqual(cmd[cmd.index("--output_fps") + 1], "50")  # the policy runs at 50 Hz

    def test_csv_check(self):
        good = self.dir / "good.csv"
        good.write_text(",".join(["0"] * 30) + "\n")
        train.check_csv(good)
        header = self.dir / "header.csv"
        header.write_text("x,y,z\n1,2,3\n")
        train.check_csv(header)  # booster_train checks named columns itself
        bad = self.dir / "bad.csv"
        bad.write_text(",".join(["0"] * 29) + "\n")
        with self.assertRaises(SystemExit):
            train.check_csv(bad)

    def test_booster_demo_csv_has_the_right_columns(self):
        csv = setup_env.EXTERNAL / "booster_assets" / "motions" / "T1" / "t1_dance.csv"
        if not csv.is_file():
            self.skipTest("booster_assets not downloaded")
        train.check_csv(csv)

    def test_train_command(self):
        npz = self.dir / "t1_dance.npz"
        npz.write_bytes(b"x")
        cmd = train.train_cmd("py", str(npz), 2048, 20000, False, ["--seed", "1"])
        self.assertEqual(cmd[cmd.index("--task") + 1], "Booster-T1-MotionTracking-v0")
        self.assertIn("--headless", cmd)
        self.assertEqual(cmd[cmd.index("--num_envs") + 1], "2048")
        self.assertEqual(cmd[cmd.index("--max_iterations") + 1], "20000")
        self.assertIn("--seed", cmd)
        self.assertTrue(cmd[-1].startswith("env.commands.motion.motion_file="))  # Hydra overrides go last

    def test_missing_motion_says_how_to_make_it(self):
        with self.assertRaises(SystemExit) as e:
            train.find_motion("nope_not_here")
        self.assertIn("lindy_train.py motion", str(e.exception))

    def test_finds_newest_run_and_checkpoint(self):
        for run in ("2026-11-20_10-00-00_dance", "2026-11-21_09-00-00_dance", "2026-11-22_09-00-00_my_dance"):
            (self.dir / run).mkdir()
        run = train.latest_run("dance", self.dir)
        self.assertEqual(run.name, "2026-11-21_09-00-00_dance")  # not the newer run of another motion
        for i in (0, 1000, 999, 12000):
            (run / ("model_%d.pt" % i)).write_bytes(b"")
        self.assertEqual(train.latest_checkpoint(run), (12000, run / "model_12000.pt"))
        with self.assertRaises(SystemExit):
            train.latest_run("waltz", self.dir)

    def test_bundle_collects_what_the_laptop_needs(self):
        run = self.dir / "2026-11-21_09-00-00_dance"
        (run / "exported").mkdir(parents=True)
        for ext in ("pt", "onnx"):
            (run / "exported" / ("t1_motion_tracking_%s.%s" % (run.name, ext))).write_bytes(ext.encode())
        motion = self.dir / "dance.npz"
        motion.write_bytes(b"npz")
        dest = train.bundle("dance", run, 12000, run / "model_12000.pt", motion, self.dir / "policies")
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["info.json", "motion.npz", "policy.onnx", "policy.pt"])
        self.assertEqual((dest / "policy.pt").read_bytes(), b"pt")
        info = json.loads((dest / "info.json").read_text())
        self.assertEqual(info["iteration"], 12000)
        self.assertEqual(info["task"], train.TASK)
        sim.find_policy(dest)  # the laptop side accepts it

    def test_dry_run_prints_and_runs_nothing(self):
        r = subprocess.run([sys.executable, str(HERE / "lindy_train.py"), "motion", "--dry-run"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("csv_to_npz.py", r.stdout)


class Sim2SimHelperTests(unittest.TestCase):
    def test_trunk_up(self):
        self.assertAlmostEqual(sim.trunk_up((1, 0, 0, 0)), 1.0)
        self.assertAlmostEqual(sim.trunk_up((0.7071068, 0.7071068, 0, 0)), 0.0, places=6)  # lying on its side

    def test_motion_path_is_relative_to_booster_deploy(self):
        p = sim.motion_path_for_deploy(sim.BEYOND_MIMIC / "robots" / "t1" / "motions" / "t1_dance.npz")
        self.assertEqual(p, "robots/t1/motions/t1_dance.npz")
        p = sim.motion_path_for_deploy(HERE / "policies" / "x" / "motion.npz")
        self.assertTrue((sim.BEYOND_MIMIC / p).resolve() == (HERE / "policies" / "x" / "motion.npz").resolve())

    def test_policy_folder_needs_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "policy.pt").write_bytes(b"")
            with self.assertRaises(SystemExit):
                sim.find_policy(d)


@unittest.skipUnless(sim.venv_python(sim.SIM_VENV).exists() and sim.DEPLOY.exists(),
                     "run `python training/setup_env.py sim` first")
class Sim2SimRunTests(unittest.TestCase):
    """Booster's trained dance must pass, a policy that ignores the motion must fail."""

    def run_check(self, *policy):
        return subprocess.run([sys.executable, str(HERE / "lindy_sim2sim.py"), "check", *policy],
                              capture_output=True, text=True, timeout=300)

    def test_booster_dance_passes(self):
        r = self.run_check()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)

    def test_standing_still_fails(self):
        with tempfile.TemporaryDirectory() as d:
            script = ("import sys, torch\n"
                      "class Still(torch.nn.Module):\n"
                      "    def forward(self, obs):\n"
                      "        return torch.zeros(obs.shape[0], 23)\n"
                      "torch.jit.script(Still()).save(sys.argv[1])\n")
            maker = Path(d, "make_still.py")  # TorchScript needs the source in a real file
            maker.write_text(script)
            subprocess.run([str(sim.venv_python(sim.SIM_VENV)), str(maker), os.path.join(d, "policy.pt")], check=True)
            src = sim.BEYOND_MIMIC / "robots" / "t1" / "motions" / "t1_dance.npz"
            Path(d, "motion.npz").write_bytes(src.read_bytes())
            r = self.run_check(d)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("FAIL", r.stdout)


if __name__ == "__main__":
    unittest.main()
