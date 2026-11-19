"""Virtual Lindy: model download, demo source, and the arm math checked against Booster's MuJoCo model."""
import os
import random
import tempfile
import unittest

from inspire_hand_ros2 import lindy_viewer as lv, t1_arm_follow, t1_arms

try:
    import mujoco  # noqa: F401
    HAVE_MUJOCO = True
except ImportError:
    HAVE_MUJOCO = False


class DownloadTests(unittest.TestCase):
    def test_downloads_model_and_every_mesh_once(self):
        xml = b'<mujoco><asset><mesh file="A.STL"/><mesh file="B.STL"/><mesh file="A.STL"/></asset></mujoco>'
        fetched = []

        def fetch(url):
            fetched.append(url)
            return xml if url.endswith(lv.MODEL_FILE) else b"solid"

        with tempfile.TemporaryDirectory() as d:
            path = lv.ensure_model(d, fetch)
            self.assertTrue(path.endswith(lv.MODEL_FILE) and os.path.exists(path))
            meshes = sorted(os.listdir(os.path.join(os.path.dirname(path), "meshes")))
            self.assertEqual(meshes, ["A.STL", "B.STL"])
            self.assertEqual(len(fetched), 3)
            self.assertTrue(all(lv.ASSET_COMMIT in u for u in fetched))  # pinned, not a moving branch
            lv.ensure_model(d, fetch)
            self.assertEqual(len(fetched), 3)  # cached


class DemoTests(unittest.TestCase):
    def test_demo_runs_through_the_real_mapping_within_limits(self):
        src = lv.DemoSource(loop=False)
        src.reset(t1_arms.LEFT_REST + t1_arms.RIGHT_REST)
        labels = set()
        t = 0.0
        while not src.finished:
            q = src.target(t)
            labels.add(src.label)
            for a, (lo, hi) in zip(q, t1_arms.ALL_LIMITS):
                self.assertTrue(lo - 1e-9 <= a <= hi + 1e-9)
            t += 0.05
            self.assertLess(t, 60)
        self.assertEqual(labels, {d[1] for d in lv.DEMO})

    def test_demo_poses_land_where_they_should(self):
        m = t1_arms.ArmMapper(handedness=1)
        poses = {d[1]: d for d in lv.DEMO}
        q = m.angles(lv.synthetic_body(poses["left punch"][2], poses["left punch"][3]))
        u, f = t1_arms.arm_directions("left", q[:4])
        self.assertGreater(u[0], 0.99)
        self.assertGreater(f[0], 0.99)
        q = m.angles(lv.synthetic_body(poses["both up"][2], poses["both up"][3]))
        for side, qs in (("left", q[:4]), ("right", q[4:])):
            self.assertGreater(t1_arms.arm_directions(side, qs)[0][2], 0.99)


def _model_or_skip(test):
    if not HAVE_MUJOCO:
        test.skipTest("mujoco not installed")
    try:
        return lv.ensure_model()
    except OSError as e:
        test.skipTest("T1 model not cached and can't be downloaded: %s" % e)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lindy = None

    def setUp(self):
        if ModelTests.lindy is None:
            ModelTests.lindy = lv.VirtualLindy(_model_or_skip(self))
        self.lindy = ModelTests.lindy

    def test_joint_names_and_limits_match_the_model(self):
        import mujoco
        m = self.lindy.model
        for name, lim in zip(lv.ARM_JOINT_NAMES, t1_arms.ALL_LIMITS):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.assertGreaterEqual(jid, 0, name)
            self.assertAlmostEqual(m.jnt_range[jid][0], lim[0], places=6)
            self.assertAlmostEqual(m.jnt_range[jid][1], lim[1], places=6)

    def test_analytic_arm_kinematics_match_mujoco(self):
        rng = random.Random(3)
        worst = 0.0
        for _ in range(200):
            q = [rng.uniform(lo, hi) for lo, hi in t1_arms.ALL_LIMITS]
            self.lindy.set_arms(q)
            for side, qs in (("left", q[:4]), ("right", q[4:])):
                got = self.lindy.arm_directions(side)
                want = t1_arms.arm_directions(side, qs)
                worst = max(worst, max(abs(a - b) for g, w in zip(got, want) for a, b in zip(g, w)))
        self.assertLess(worst, 0.005)  # left: the model's tiny 0.00025 m / 0.05 deg offsets

    def test_stands_on_the_floor(self):
        self.lindy.set_arms(t1_arms.LEFT_REST + t1_arms.RIGHT_REST)
        self.assertAlmostEqual(self.lindy._lowest_foot_point(), 0.0, places=4)
        self.assertGreater(self.lindy.data.qpos[2], 0.5)

    def test_viewer_follows_like_the_robot(self):
        self.lindy.set_arms(t1_arms.LEFT_REST + t1_arms.RIGHT_REST)
        io = t1_arm_follow.FakeArms(self.lindy.arms())
        follow = t1_arm_follow.ArmFollow(io, lv.DemoSource(loop=False), max_speed=1.0, ramp=0.5)
        follow.start(0.0)
        t = 0.0
        while not follow.done and t < 60:
            t += 0.01
            follow.tick(t)
        self.assertTrue(follow.done)
        for (a, _), (b, _) in zip(io.sent, io.sent[1:]):
            self.assertTrue(all(abs(x - y) <= 0.01 + 1e-9 for x, y in zip(a, b)))


if __name__ == "__main__":
    unittest.main()
