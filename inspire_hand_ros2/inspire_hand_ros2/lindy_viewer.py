"""Virtual Lindy: Booster's T1 model in MuJoCo, with its arms driven by your Rokoko suit.

    pip install mujoco
    python -m inspire_hand_ros2.lindy_viewer                 # live suit (same calibration as t1_arm_follow)
    python -m inspire_hand_ros2.lindy_viewer --demo          # no suit: built-in arm moves
    python -m inspire_hand_ros2.lindy_viewer --demo --snapshot lindy.png --seconds 8

It runs the same ArmFollow controller as the real robot (fade-in, speed limit, joint limits, hold on
stream loss), just with a virtual robot, so what you see is what Lindy's arms would do. Only the
arms move; head, waist and legs stay in Booster's standing pose. It's a pose viewer, not a physics
simulation: nothing here can fall over, so it says nothing about balance.

The model (T1_23dof.xml + meshes, BSD-3-Clause) is downloaded once from Booster's booster_assets
repository, pinned to one commit, into ~/.cache/lindy_viewer.
"""
import argparse
import math
import os
import re
import sys
import time
import urllib.request

from . import t1_arm_follow, t1_arms

ASSET_COMMIT = "12a97516f57469078707afb587a5f84c1cb110d1"
ASSET_URL = "https://raw.githubusercontent.com/BoosterRobotics/booster_assets/%s/robots/T1/" % ASSET_COMMIT
MODEL_FILE = "T1_23dof.xml"
DEFAULT_CACHE = "~/.cache/lindy_viewer"

ARM_JOINT_NAMES = ["%s_%s_joint" % (side, j) for side in ("left", "right") for j in t1_arms.ARM_JOINTS]
# Standing pose of Booster's own T1 walking deployment (booster_gym deploy/configs/T1.yaml default_qpos).
STAND_POSE = {"left_hip_pitch_joint": -0.2, "left_knee_pitch_joint": 0.4, "left_ankle_pitch_joint": -0.25,
              "right_hip_pitch_joint": -0.2, "right_knee_pitch_joint": 0.4, "right_ankle_pitch_joint": -0.25}
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lindy-viewer"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def ensure_model(cache=DEFAULT_CACHE, fetch=_download):
    """Path to T1_23dof.xml, downloading it and the meshes it uses on first call."""
    folder = os.path.join(os.path.expanduser(cache), ASSET_COMMIT)
    xml_path = os.path.join(folder, MODEL_FILE)
    if os.path.exists(xml_path):
        return xml_path
    os.makedirs(os.path.join(folder, "meshes"), exist_ok=True)
    xml = fetch(ASSET_URL + MODEL_FILE)
    for name in sorted(set(re.findall(r'file="([^"]+)"', xml.decode("utf-8")))):
        dest = os.path.join(folder, "meshes", name)
        if not os.path.exists(dest):
            data = fetch(ASSET_URL + "meshes/" + name)
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
    with open(xml_path + ".part", "wb") as f:  # xml last: its presence means everything is there
        f.write(xml)
    os.replace(xml_path + ".part", xml_path)
    return xml_path


class VirtualLindy:
    """The T1 model, standing on the floor, with settable arm joints."""

    def __init__(self, xml_path):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.arm_adr = [self._qadr(n) for n in ARM_JOINT_NAMES]
        for name, value in STAND_POSE.items():
            self.data.qpos[self._qadr(name)] = value
        self.data.qpos[0:7] = (0, 0, 1.0, 1, 0, 0, 0)  # free base: position, quaternion (w x y z)
        self.set_arms(t1_arms.LEFT_REST + t1_arms.RIGHT_REST)
        self.data.qpos[2] -= self._lowest_foot_point()  # stand on the floor
        mujoco.mj_forward(self.model, self.data)

    def _qadr(self, joint):
        jid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            raise ValueError("model has no joint %r" % joint)
        return self.model.jnt_qposadr[jid]

    def _body(self, name):
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)

    def _lowest_foot_point(self):
        feet = {self._body(n) for n in FOOT_BODIES}
        low = None
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] not in feet:
                continue
            center, half = self.model.geom_aabb[g][:3], self.model.geom_aabb[g][3:]
            rot, pos = self.data.geom_xmat[g].reshape(3, 3), self.data.geom_xpos[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corner = [center[0] + sx * half[0], center[1] + sy * half[1], center[2] + sz * half[2]]
                        z = pos[2] + sum(rot[2][k] * corner[k] for k in range(3))
                        low = z if low is None else min(low, z)
        return low

    def set_arms(self, q):
        for adr, a in zip(self.arm_adr, q):
            self.data.qpos[adr] = a
        self.mj.mj_forward(self.model, self.data)

    def arms(self):
        return [float(self.data.qpos[adr]) for adr in self.arm_adr]

    def arm_directions(self, side):
        """(upper arm, forearm) unit directions in the trunk frame, measured from the model itself."""
        trunk = self.data.xmat[self._body("trunk")].reshape(3, 3)

        def in_trunk(v):
            return t1_arms.unit(tuple(sum(trunk[k][i] * v[k] for k in range(3)) for i in range(3)))

        # from the roll joint: the 4.7 cm stub between the pitch and roll joints only turns with pitch
        shoulder = self.data.xpos[self._body(side + "_shoulder_roll_link")]
        elbow_body = self._body(side + "_elbow_yaw_link")
        upper = in_trunk([self.data.xpos[elbow_body][k] - shoulder[k] for k in range(3)])
        fm = self.data.xmat[elbow_body].reshape(3, 3)
        sign = 1.0 if side == "left" else -1.0  # the forearm runs along +y (left) / -y (right) of its body
        forearm = in_trunk([sign * fm[k][1] for k in range(3)])
        return upper, forearm

    def snapshot(self, path, width=800, height=600, azimuth=150.0, elevation=-12.0, distance=2.2):
        from PIL import Image
        vis = self.model.vis.global_  # the model's off-screen buffer is 640x480; grow it to fit
        vis.offwidth, vis.offheight = max(vis.offwidth, width), max(vis.offheight, height)
        renderer = self.mj.Renderer(self.model, height, width)
        try:
            cam = self.mj.MjvCamera()
            cam.lookat[:] = self.data.xpos[self._body("trunk")]
            cam.lookat[2] -= 0.15
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
            renderer.update_scene(self.data, camera=cam)
            Image.fromarray(renderer.render()).save(path)
        finally:
            renderer.close()


# -- built-in demo (no suit needed) ---------------------------------------------------------------

DOWN, FWD, UP, OUT_L, OUT_R = (0, 0, -1), (1, 0, 0), (0, 0, 1), (0, 1, 0), (0, -1, 0)
REST = (DOWN, t1_arms.unit((0.3, 0, -1)))
# (seconds, label, left (upper arm, forearm), right (upper arm, forearm)), directions in your torso frame
DEMO = [
    (2.0, "rest", REST, REST),
    (2.0, "left punch", (FWD, FWD), REST),
    (2.0, "right punch", REST, (FWD, FWD)),
    (2.5, "guard", (t1_arms.unit((0.5, 0.3, -0.8)), t1_arms.unit((0.6, -0.2, 0.75))),
     (t1_arms.unit((0.5, -0.3, -0.8)), t1_arms.unit((0.6, 0.2, 0.75)))),
    (2.5, "arms out", (OUT_L, OUT_L), (OUT_R, OUT_R)),
    (2.5, "right wave", REST, (t1_arms.unit((0, -1, 0.1)), UP)),
    (2.5, "both up", (UP, UP), (UP, UP)),
    (2.0, "rest", REST, REST),
]


def _nlerp(a, b, s):
    v = tuple(x + (y - x) * s for x, y in zip(a, b))
    return t1_arms.unit(v) if t1_arms.norm(v) > 1e-6 else b


def synthetic_body(left, right):
    """Rokoko-style body in a right-handed world aligned with the torso (handedness +1)."""
    pts = {"hip": (0, 0, 1.0), "neck": (0, 0, 1.5),
           "leftUpperArm": (0, 0.18, 1.45), "rightUpperArm": (0, -0.18, 1.45)}
    for side, (u, f) in (("left", left), ("right", right)):
        s = pts[side + "UpperArm"]
        e = tuple(a + 0.28 * b for a, b in zip(s, t1_arms.unit(u)))
        pts[side + "LowerArm"] = e
        pts[side + "Hand"] = tuple(a + 0.25 * b for a, b in zip(e, t1_arms.unit(f)))
    return {k: {"position": {"x": p[0], "y": p[1], "z": p[2]}} for k, p in pts.items()}


class DemoSource:
    """Plays DEMO through the real suit mapping (ArmMapper), as if someone wore the suit."""

    def __init__(self, loop=True):
        self.mapper = t1_arms.ArmMapper(handedness=1)
        self.loop, self.finished = loop, False
        self.t0 = None
        self.label = DEMO[0][1]
        self.length = sum(d for d, *_ in DEMO)

    def reset(self, current):
        self.mapper.reset(current)
        self.t0 = None

    def pose(self, t):
        """(label, left dirs, right dirs) at demo time t; each move blends in over its first second."""
        if self.loop:
            t %= self.length
        prev = DEMO[0]
        for step in DEMO:
            if t < step[0]:
                s = min(1.0, t / 1.0)
                s = s * s * (3 - 2 * s)
                blend = lambda a, b: (_nlerp(a[0], b[0], s), _nlerp(a[1], b[1], s))
                return step[1], blend(prev[2], step[2]), blend(prev[3], step[3])
            t -= step[0]
            prev = step
        self.finished = True
        return DEMO[-1][1], DEMO[-1][2], DEMO[-1][3]

    def target(self, now):
        if self.t0 is None:
            self.t0 = now
        self.label, left, right = self.pose(now - self.t0)
        return self.mapper.angles(synthetic_body(left, right))

    def status(self, now):
        return "demo: " + self.label

    def close(self):
        pass


# -- running --------------------------------------------------------------------------------------

def run_window(lindy, follow, rate=60.0, out=sys.stdout):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(lindy.model, lindy.data) as view:
        view.cam.azimuth, view.cam.elevation, view.cam.distance = 150.0, -12.0, 2.4
        view.cam.lookat[:] = (0.0, 0.0, 0.75)
        follow.start(time.monotonic())
        next_print = 0.0
        while view.is_running() and not follow.done:
            now = time.monotonic()
            follow.tick(now)
            with view.lock():
                lindy.set_arms(follow.io.q)
            view.sync()
            if now >= next_print:
                next_print = now + 0.25
                out.write("\r[%-8s] %s  (%s)   " % (follow.phase, t1_arm_follow.format_angles(follow.io.q),
                                                     follow.source.status(now)))
                out.flush()
            time.sleep(max(0.0, 1.0 / rate - (time.monotonic() - now)))
    out.write("\n")


def run_snapshot(lindy, follow, path, seconds, rate=100.0):
    follow.start(time.monotonic())
    end = time.monotonic() + seconds
    while time.monotonic() < end and not follow.done:
        follow.tick(time.monotonic())
        time.sleep(1.0 / rate)
    lindy.set_arms(follow.io.q)
    lindy.snapshot(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demo", action="store_true", help="no suit: play built-in arm moves")
    ap.add_argument("--snapshot", metavar="PNG", help="no window: run for --seconds, then save a picture")
    ap.add_argument("--seconds", type=float, default=5.0, help="how long to run before --snapshot")
    ap.add_argument("--port", type=int, default=t1_arm_follow.rokoko.DEFAULT_PORT,
                    help="UDP port Rokoko Studio streams to")
    ap.add_argument("--relay", metavar="HOST:PORT", help="forward the Rokoko stream here too")
    ap.add_argument("--actor", help="Rokoko actor name")
    ap.add_argument("--calibrate", action="store_true", help="redo the arms-forward calibration")
    ap.add_argument("--calibration", default=t1_arms.DEFAULT_CALIBRATION, help="calibration file")
    ap.add_argument("--max-speed", type=float, default=1.0,
                    help="top joint speed, rad/s (default 1.0, same as the robot)")
    ap.add_argument("--model", help="path to a T1_23dof.xml (default: download Booster's once)")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="where the downloaded model is kept")
    args = ap.parse_args(argv)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        raise SystemExit("MuJoCo isn't installed: pip install mujoco")
    source = None
    try:
        xml = args.model or ensure_model(args.cache)
        lindy = VirtualLindy(xml)
        if args.demo:
            source = DemoSource(loop=not args.snapshot)
        else:
            source = t1_arm_follow.SuitSource(args.port, t1_arm_follow._relay(args.relay), args.actor,
                                              args.calibration)
            t1_arm_follow._calibrate_if_needed(source, args.calibrate)
        follow = t1_arm_follow.ArmFollow(t1_arm_follow.FakeArms(lindy.arms()), source,
                                         max_speed=args.max_speed, ramp=0.5)
        if args.snapshot:
            run_snapshot(lindy, follow, args.snapshot, args.seconds)
            print("saved %s" % args.snapshot)
        else:
            run_window(lindy, follow)
    except (RuntimeError, ValueError, OSError) as e:
        raise SystemExit("error: %s" % e)
    except KeyboardInterrupt:
        pass
    finally:
        if source is not None:
            source.close()


if __name__ == "__main__":
    main()
