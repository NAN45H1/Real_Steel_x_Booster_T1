"""Suit -> T1 arm mapping and the arm-follow controller, on synthetic Rokoko frames and a fake robot."""
import json
import math
import os
import random
import socket
import tempfile
import time
import unittest

from inspire_hand_ros2 import t1_arms, t1_arm_follow
from inspire_hand_ros2.t1_arms import arm_directions, solve_arm

HALF_PI = math.pi / 2


def close(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# -- a synthetic person, built in robot axes (x forward, y left, z up) then placed in a world ------

def unity_world(p):
    """Left-handed, Y up, Z forward (how Rokoko describes its JSON v3 space)."""
    return (-p[1], p[2], p[0])


def right_handed_world(p):
    """Right-handed, Y up: x right, y up, z backward."""
    return (-p[1], p[2], -p[0])


def make_frame(left=((0, 0, -1), (0, 0, -1)), right=((0, 0, -1), (0, 0, -1)), world=unity_world,
               yaw_deg=30.0, lean_deg=0.0, offset=(1.5, 0.0, -2.0), actor="Actor1", drop=()):
    """left/right: (upper arm direction, forearm direction) in the person's torso frame."""
    yaw, lean = math.radians(yaw_deg), math.radians(lean_deg)

    def place(p):
        p = t1_arms.rot_y(lean, p)  # lean the whole torso forward about the hips
        p = t1_arms.rot_z(yaw, p)   # turn to face somewhere else
        w = world(p)
        return tuple(a + b for a, b in zip(w, offset))

    pts = {"hip": (0, 0, 1.0), "neck": (0, 0, 1.5),
           "leftUpperArm": (0, 0.18, 1.45), "rightUpperArm": (0, -0.18, 1.45)}
    for side, (u, f) in (("left", left), ("right", right)):
        shoulder = pts[side + "UpperArm"]
        elbow = tuple(s + 0.28 * c for s, c in zip(shoulder, t1_arms.unit(u)))
        pts[side + "LowerArm"] = elbow
        pts[side + "Hand"] = tuple(e + 0.25 * c for e, c in zip(elbow, t1_arms.unit(f)))
    body = {}
    for name, p in pts.items():
        if name in drop:
            continue
        x, y, z = place(p)
        body[name] = {"position": {"x": x, "y": y, "z": z}, "rotation": {"w": 1, "x": 0, "y": 0, "z": 0}}
    return {"version": 3, "fps": 60, "scene": {"timestamp": 0.0, "props": [], "actors": [
        {"name": actor, "meta": {"hasGloves": False, "hasBody": True}, "body": body}]}}


ARMS_FORWARD = ((1, 0, 0), (1, 0, 0))
DOWN_FOREARM_FORWARD = ((0, 0, -1), (1, 0, 0))


def body_of(frame):
    return t1_arms.find_body(frame)[1]


class KinematicsTests(unittest.TestCase):
    def test_rest_pose_is_arms_down_elbows_slightly_forward(self):
        for side, q in (("left", t1_arms.LEFT_REST), ("right", t1_arms.RIGHT_REST)):
            u, f = arm_directions(side, q)
            self.assertLess(u[2], -0.9)  # upper arm hangs down
            self.assertGreater(f[0], 0.2)  # forearm tilts forward
            self.assertLess(f[2], -0.8)

    def test_mirror_symmetry(self):
        u, f = arm_directions("left", (0.3, -1.0, 0.4, -0.9))
        ur, fr = arm_directions("right", (0.3, 1.0, 0.4, 0.9))
        self.assertTrue(close(ur, t1_arms.mirror(u)) and close(fr, t1_arms.mirror(f)))

    def test_round_trip_random_poses(self):
        rng = random.Random(1)
        n = 0
        while n < 400:
            side = rng.choice(("left", "right"))
            limits = t1_arms.LEFT_LIMITS if side == "left" else t1_arms.RIGHT_LIMITS
            q = tuple(rng.uniform(lo, hi) for lo, hi in limits)
            if abs(math.sin(q[1])) < 0.35 or abs(math.sin(q[3])) < 0.35:
                continue  # near a singularity the answer is deliberately smoothed, not exact
            n += 1
            u, f = arm_directions(side, q)
            prev = tuple(a + rng.uniform(-0.05, 0.05) for a in q)
            got = solve_arm(side, u, f, prev)
            gu, gf = arm_directions(side, got)
            self.assertTrue(close(gu, u, 1e-6) and close(gf, f, 1e-6), (side, q, got))
            self.assertTrue(close(got, q, 1e-6), (side, q, got))  # continuity picks the same branch
            for a, (lo, hi) in zip(got, limits):
                self.assertTrue(lo - 1e-6 <= a <= hi + 1e-6)

    def test_known_pose_arms_down_forearms_forward(self):
        self.assertTrue(close(solve_arm("left", (0, 0, -1), (1, 0, 0)), (0, -HALF_PI, 0, -HALF_PI), 1e-9))
        self.assertTrue(close(solve_arm("right", (0, 0, -1), (1, 0, 0)), (0, HALF_PI, 0, HALF_PI), 1e-9))

    def test_arm_straight_forward(self):
        q = solve_arm("left", (1, 0, 0), (1, 0, 0))
        u, f = arm_directions("left", q)
        self.assertTrue(close(u, (1, 0, 0), 1e-6) and close(f, (1, 0, 0), 1e-6))
        self.assertAlmostEqual(q[0], -HALF_PI, places=6)  # negative pitch = forward (Booster's docs)

    def test_overhead_stays_in_limits(self):
        for side in ("left", "right"):
            q = solve_arm(side, (0, 0, 1), (0, 0, 1))
            u, _ = arm_directions(side, q)
            self.assertGreater(u[2], 0.99)

    def test_unreachable_is_clamped_not_crashing(self):
        # behind the back and across: outside the T1's range, must come back inside the limits
        q = solve_arm("left", (-0.7, -0.7, 0.1), (0.0, -1.0, 0.0))
        for a, (lo, hi) in zip(q, t1_arms.LEFT_LIMITS):
            self.assertTrue(lo - 1e-9 <= a <= hi + 1e-9)


class HumanMappingTests(unittest.TestCase):
    def mapped(self, world, **pose):
        m = t1_arms.ArmMapper()
        m.calibrate(body_of(make_frame(left=ARMS_FORWARD, right=ARMS_FORWARD, world=world)))
        return m, m.angles(body_of(make_frame(world=world, **pose)))

    def test_handedness_detected_for_both_world_conventions(self):
        a = t1_arms.detect_handedness(body_of(make_frame(left=ARMS_FORWARD, right=ARMS_FORWARD,
                                                         world=unity_world)))
        b = t1_arms.detect_handedness(body_of(make_frame(left=ARMS_FORWARD, right=ARMS_FORWARD,
                                                         world=right_handed_world)))
        self.assertEqual(a, -b)  # mirrored worlds need opposite signs

    def test_forearms_forward_in_both_worlds(self):
        for world in (unity_world, right_handed_world):
            _, q = self.mapped(world, left=DOWN_FOREARM_FORWARD, right=DOWN_FOREARM_FORWARD)
            self.assertTrue(close(q[:4], (0, -HALF_PI, 0, -HALF_PI), 1e-6), q)
            self.assertTrue(close(q[4:], (0, HALF_PI, 0, HALF_PI), 1e-6), q)

    def test_left_and_right_are_not_swapped(self):
        # only the LEFT arm raised forward
        _, q = self.mapped(unity_world, left=ARMS_FORWARD)
        lu, _ = arm_directions("left", q[:4])
        ru, _ = arm_directions("right", q[4:])
        self.assertGreater(lu[0], 0.99)
        self.assertLess(ru[2], -0.99)

    def test_leaning_and_turning_dont_move_the_arms(self):
        _, straight = self.mapped(unity_world, left=DOWN_FOREARM_FORWARD, right=DOWN_FOREARM_FORWARD)
        m, _ = self.mapped(unity_world)
        leaned = m.angles(body_of(make_frame(left=DOWN_FOREARM_FORWARD, right=DOWN_FOREARM_FORWARD,
                                             lean_deg=25, yaw_deg=-70)))
        self.assertTrue(close(straight, leaned, 1e-6))

    def test_calibration_rejects_wrong_pose(self):
        with self.assertRaises(ValueError):
            t1_arms.detect_handedness(body_of(make_frame()))  # arms down, not forward

    def test_uncalibrated_refuses(self):
        with self.assertRaises(RuntimeError):
            t1_arms.ArmMapper().angles(body_of(make_frame()))

    def test_missing_bones_reported(self):
        frame = make_frame(drop=("neck",))
        self.assertEqual(t1_arms.find_body(frame), (None, None))
        self.assertEqual(t1_arms.missing_bones(frame), ["neck"])

    def test_calibration_file_round_trip(self):
        m = t1_arms.ArmMapper(handedness=-1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal.json")
            t1_arms.save_calibration(m, path)
            n = t1_arms.ArmMapper()
            self.assertTrue(t1_arms.load_calibration(n, path))
            self.assertEqual(n.handedness, -1)


class ShaperTests(unittest.TestCase):
    def test_speed_limit_and_joint_limits(self):
        start = list(t1_arms.LEFT_REST + t1_arms.RIGHT_REST)
        s = t1_arms.Shaper(start, max_speed=1.0, smoothing=0.0)
        far = [10.0] * 8
        q = s.step(far, 0.01)
        self.assertTrue(all(abs(a - b) <= 0.01 + 1e-12 for a, b in zip(q, start)))
        for _ in range(2000):
            q = s.step(far, 0.01)
        for a, (lo, hi) in zip(q, t1_arms.ALL_LIMITS):
            self.assertAlmostEqual(a, hi - 0.05)


class ScriptedSource:
    def __init__(self, targets):
        self.targets, self.finished = targets, False

    def reset(self, current):
        self.reset_with = list(current)

    def target(self, now):
        return self.targets(now)

    def status(self, now):
        return "scripted"


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.io = t1_arm_follow.FakeArms()
        self.start = self.io.measured_arms()

    def drive(self, follow, seconds, t0=0.0, dt=0.01):
        t = t0
        while t < t0 + seconds and not follow.done:
            t += dt
            follow.tick(t)
        return t

    def test_full_cycle(self):
        goal = list(self.start)
        goal[0] -= 0.5  # left shoulder forward
        f = t1_arm_follow.ArmFollow(self.io, ScriptedSource(lambda now: goal), max_speed=1.0, ramp=2.0)
        f.start(0.0)
        self.assertTrue(self.io.enabled)
        self.assertEqual(f.source.reset_with, self.start)
        t = self.drive(f, 2.0)
        weights = [w for _, w in self.io.sent]
        self.assertTrue(all(b >= a for a, b in zip(weights, weights[1:])))
        self.assertLess(weights[0], 0.02)
        self.assertTrue(all(close(q, self.start) for q, _ in self.io.sent))  # holds during ramp-in
        self.assertEqual(f.phase, "follow")
        t = self.drive(f, 1.0, t)
        self.assertAlmostEqual(self.io.q[0], goal[0], places=6)  # 0.5 rad at 1 rad/s: there by now
        for (a, _), (b, _) in zip(self.io.sent, self.io.sent[1:]):
            self.assertTrue(all(abs(x - y) <= 0.01 + 1e-9 for x, y in zip(a, b)))  # never faster
        f.stop("test")
        t = self.drive(f, 10.0, t)
        self.assertTrue(f.done)
        self.assertFalse(self.io.enabled)
        self.assertTrue(close(self.io.q, self.start, 0.03))
        self.assertLess(self.io.sent[-1][1], 0.02)  # weight ramped back down

    def test_stream_loss_holds_still(self):
        state = {"goal": list(self.start)}
        f = t1_arm_follow.ArmFollow(self.io, ScriptedSource(lambda now: state["goal"]), ramp=0.1)
        f.start(0.0)
        t = self.drive(f, 0.2)
        state["goal"] = [a + 0.3 for a in self.start]
        t = self.drive(f, 0.1, t)
        held = list(self.io.q)
        state["goal"] = None
        t = self.drive(f, 1.0, t)
        self.assertTrue(close(self.io.q, held, 1e-9))
        self.assertEqual(f.phase, "follow")

    def test_robot_problem_hands_back_at_once(self):
        f = t1_arm_follow.ArmFollow(self.io, ScriptedSource(lambda now: self.start), ramp=0.1)
        f.start(0.0)
        self.drive(f, 0.5)
        self.io.problem = "robot is tilting"
        f.tick(1.0)
        self.assertTrue(f.done)
        self.assertFalse(self.io.enabled)
        self.assertEqual(f.reason, "robot is tilting")

    def test_selftest_moves_left_shoulder_forward_and_back(self):
        src = t1_arm_follow.SelfTestSource(period=2.0, amplitude=0.3)
        f = t1_arm_follow.ArmFollow(self.io, src, max_speed=2.0, ramp=0.1)
        f.start(0.0)
        self.drive(f, 30.0)
        self.assertTrue(f.done)
        pitches = [q[0] for q, _ in self.io.sent]
        self.assertAlmostEqual(min(pitches), self.start[0] - 0.3, delta=0.02)
        self.assertTrue(all(close(q[1:], self.start[1:]) for q, _ in self.io.sent))  # nothing else moved
        self.assertTrue(close(self.io.q, self.start, 0.03))


class FakeSdk:
    """Just enough of booster_robotics_sdk_python (1.6.3 names) to exercise BoosterArms."""

    class _Obj:
        pass

    class MotorCmd(_Obj):
        pass

    class LowCmd(_Obj):
        pass

    class LowCmdType:
        PARALLEL, SERIAL = "PARALLEL", "SERIAL"

    def __init__(self, enable_results):
        sdk = self
        self.enable_results = list(enable_results)  # per call: int, or an Exception to raise
        self.enable_calls, self.written = [], []
        self.state_cb = None

        class ChannelFactory:
            @staticmethod
            def Instance():
                f = FakeSdk._Obj()
                f.Init = lambda domain, net: setattr(sdk, "net", net)
                return f

        class B1LocoClient:
            def Init(self):
                pass

            def UpperBodyCustomControl(self, on):
                sdk.enable_calls.append(on)
                r = sdk.enable_results.pop(0) if sdk.enable_results else 0
                if isinstance(r, Exception):
                    raise r
                return r

        class B1LowCmdPublisher:
            def InitChannel(self):
                pass

            def Write(self, cmd):
                sdk.written.append(cmd)

            def CloseChannel(self):
                pass

        class B1LowStateSubscriber:
            def __init__(self, cb):
                sdk.state_cb = cb

            def InitChannel(self):
                pass

            def CloseChannel(self):
                pass

        self.ChannelFactory, self.B1LocoClient = ChannelFactory, B1LocoClient
        self.B1LowCmdPublisher, self.B1LowStateSubscriber = B1LowCmdPublisher, B1LowStateSubscriber

    def push_state(self, q, rpy=(0.0, 0.0, 0.0)):
        msg = self._Obj()
        msg.motor_state_parallel = []
        for a in q:
            m = self._Obj()
            m.q = a
            msg.motor_state_parallel.append(m)
        msg.imu_state = self._Obj()
        msg.imu_state.rpy = list(rpy)
        self.state_cb(msg)


class BoosterArmsTests(unittest.TestCase):
    def make(self, enable_results=()):
        import sys
        sdk = FakeSdk(enable_results)
        sys.modules["booster_robotics_sdk_python"] = sdk
        self.addCleanup(sys.modules.pop, "booster_robotics_sdk_python", None)
        return sdk, t1_arm_follow.BoosterArms(net="127.0.0.1", kp=60.0, kd=3.0)

    def test_state_send_and_enable(self):
        sdk, arms = self.make()
        q = [0.01 * i for i in range(23)]
        sdk.push_state(q)
        arms.wait_for_state(timeout=0.1)
        self.assertEqual(arms.measured_arms(), q[2:10])
        self.assertIsNone(arms.check())
        arms.enable(True)
        self.assertTrue(arms.enabled)
        arms.send([1.0 + i for i in range(8)], 0.5)
        cmd = sdk.written[-1]
        self.assertEqual(cmd.cmd_type, "PARALLEL")
        self.assertEqual(len(cmd.motor_cmd), 23)
        for i, m in enumerate(cmd.motor_cmd):
            if 2 <= i <= 9:
                self.assertEqual((m.q, m.kp, m.kd, m.weight), (1.0 + i - 2, 60.0, 3.0, 0.5))
            else:
                self.assertEqual((m.q, m.kp, m.kd, m.weight), (0.0, 0.0, 0.0, 0.0))  # Booster keeps these

    def test_tilt_and_wrong_robot(self):
        sdk, arms = self.make()
        sdk.push_state([0.0] * 23, rpy=(0.0, 0.8, 0.0))
        self.assertIn("tilting", arms.check())
        sdk.push_state([0.0] * 29)
        with self.assertRaises(RuntimeError):
            arms.wait_for_state(timeout=0.1)  # 7-DoF-arm layout is refused

    def test_enable_retries_timeouts_and_reports_refusal(self):
        import unittest.mock
        with unittest.mock.patch("time.sleep"):
            sdk, arms = self.make([RuntimeError("rpc failed, code = 100"), 0])
            arms.enable(True)
            self.assertEqual(sdk.enable_calls, [True, True])
            sdk, arms = self.make([501])
            with self.assertRaises(RuntimeError) as e:
                arms.enable(True)
            self.assertIn("501", str(e.exception))


class SuitSourceTests(unittest.TestCase):
    def test_udp_stream_calibrate_follow_and_relay(self):
        relay_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        relay_sock.bind(("127.0.0.1", 0))
        relay_sock.settimeout(1.0)
        with tempfile.TemporaryDirectory() as d:
            src = t1_arm_follow.SuitSource(port=0, relay=relay_sock.getsockname(),
                                           calibration_file=os.path.join(d, "cal.json"))
            out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            dest = ("127.0.0.1", src.receiver.port)
            try:
                self.assertFalse(src.mapper.calibrated)
                out.sendto(json.dumps(make_frame(left=ARMS_FORWARD, right=ARMS_FORWARD)).encode(), dest)
                src.calibrate(wait=2.0)
                self.assertTrue(src.mapper.calibrated)
                self.assertTrue(os.path.exists(os.path.join(d, "cal.json")))
                self.assertIn(b'"version"', relay_sock.recv(1 << 17))  # forwarded for the glove teleop

                src.reset(t1_arms.LEFT_REST + t1_arms.RIGHT_REST)
                out.sendto(json.dumps(make_frame(left=DOWN_FOREARM_FORWARD,
                                                 right=DOWN_FOREARM_FORWARD)).encode(), dest)
                q = None
                end = time.monotonic() + 2.0
                while q is None and time.monotonic() < end:
                    q = src.target(time.monotonic())
                    time.sleep(0.01)
                self.assertTrue(close(q[:4], (0, -HALF_PI, 0, -HALF_PI), 1e-6), q)
                time.sleep(src.timeout + 0.1)
                self.assertIsNone(src.target(time.monotonic()))  # stale stream -> hold
                self.assertIn("stopped", src.status(time.monotonic()))
            finally:
                out.close()
                src.close()
                relay_sock.close()

    def test_bad_stream_reports_missing_bones(self):
        src = t1_arm_follow.SuitSource(port=0, calibration_file=os.path.join(tempfile.gettempdir(), "x.json"))
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            out.sendto(json.dumps(make_frame(drop=("hip",))).encode(), ("127.0.0.1", src.receiver.port))
            end = time.monotonic() + 2.0
            while src.error is None and time.monotonic() < end:
                src.poll(time.monotonic())
                time.sleep(0.01)
            self.assertIn("hip", src.error)
        finally:
            out.close()
            src.close()


if __name__ == "__main__":
    unittest.main()
