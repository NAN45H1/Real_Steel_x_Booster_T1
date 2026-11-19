"""Rokoko Smartsuit arms -> Booster T1 arm joints (4 per arm). No ROS or Booster SDK dependency.

Human side: Rokoko Studio's live stream (same JSON v3 frames the glove code reads) gives every
body bone a world position. The upper arm runs from the upperArm bone to the lowerArm bone, and
the forearm from lowerArm to hand. Both are expressed in a torso frame built from the hips, neck
and shoulders (x forward, y left, z up), so leaning or turning your whole body doesn't swing the
robot's arms.

Robot side: the T1 arm chain from Booster's T1_23dof.urdf (booster_assets). At zero every joint is
0 and the arms point straight out sideways (T-pose). Per arm, in the SDK's joint order:

    shoulder pitch (axis y) -> shoulder roll (axis x) -> "elbow pitch" (axis y) -> "elbow yaw" (axis z)

With the arm along y, the third joint turns the upper arm about its own length (a twist) and the
fourth one bends the elbow. Upper arm direction  u = Ry(pitch) Rx(roll) y,
forearm direction  f = Ry(pitch) Rx(roll) Ry(twist) Rz(elbow) y.  Each arm is solved in closed
form; out of the two mirror solutions, the one inside the joint limits and closest to the previous
answer wins, so the arms move smoothly instead of flipping. The right arm is the left arm mirrored
across the robot's x-z plane: pitch and twist stay, roll and elbow change sign.

Calibration: the world's handedness (which way is "forward") is taken from one pose: stand facing
your forward direction with both arms straight out in front of you.
"""
import json
import math
import os

DEFAULT_CALIBRATION = "~/.rokoko_t1_arms_calibration.json"

# SDK joint indices (booster/robot/b1/b1_api_const.hpp JointIndex): left arm 2..5, right arm 6..9.
LEFT_ARM_INDEX, RIGHT_ARM_INDEX = 2, 6
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow_pitch", "elbow_yaw")  # URDF names, per arm

# Left arm limits (rad) from T1_23dof.urdf. The right arm's are the same, with roll and elbow yaw mirrored.
LEFT_LIMITS = ((-3.276, 1.196), (-1.623, 1.736), (-2.235, 2.235), (-2.077, 1.658))
RIGHT_LIMITS = (LEFT_LIMITS[0], (-LEFT_LIMITS[1][1], -LEFT_LIMITS[1][0]), LEFT_LIMITS[2],
                (-LEFT_LIMITS[3][1], -LEFT_LIMITS[3][0]))
# Arms-down standing pose used by Booster's own T1 walking deployment (booster_gym deploy/configs/T1.yaml).
LEFT_REST = (0.2, -1.35, 0.0, -0.5)
RIGHT_REST = (0.2, 1.35, 0.0, 0.5)

# Rokoko Studio bone names (JSON v3), as listed in Rokoko Studio 2.6.2's Studio.LivestreamFrames.dll.
# Also available for later (walking, kicks): spine, chest, head, *Shoulder, *UpLeg, *Leg, *Foot, *Toe, *ToeEnd.
TORSO_BONES = ("hip", "neck", "leftUpperArm", "rightUpperArm")
ARM_BONES = {side: (side + "UpperArm", side + "LowerArm", side + "Hand") for side in ("left", "right")}
REQUIRED_BONES = set(TORSO_BONES) | {b for bones in ARM_BONES.values() for b in bones}

SINGULAR = 0.15  # below this sine, the angle the solver would need is ill-defined (arm or elbow nearly straight)


# -- small vector / rotation helpers (pure python; runs anywhere) ---------------------------------

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n < 1e-9:
        raise ValueError("zero-length vector")
    return scale(a, 1.0 / n)


def rot_x(t, v):
    c, s = math.cos(t), math.sin(t)
    return (v[0], c * v[1] - s * v[2], s * v[1] + c * v[2])


def rot_y(t, v):
    c, s = math.cos(t), math.sin(t)
    return (c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2])


def rot_z(t, v):
    c, s = math.cos(t), math.sin(t)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def mirror(v):
    """Reflect across the robot's x-z plane (left <-> right)."""
    return (v[0], -v[1], v[2])


def wrap(t):
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(t), math.cos(t))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# -- robot kinematics -----------------------------------------------------------------------------

def left_arm_directions(q):
    """(upper arm, forearm) unit directions in the T1 trunk frame for left-arm angles q."""
    pitch, roll, twist, elbow = q
    u = rot_y(pitch, rot_x(roll, (0.0, 1.0, 0.0)))
    f = rot_y(pitch, rot_x(roll, rot_y(twist, rot_z(elbow, (0.0, 1.0, 0.0)))))
    return u, f


def arm_directions(side, q):
    if side == "left":
        return left_arm_directions(q)
    u, f = left_arm_directions(mirror_angles(q))
    return mirror(u), mirror(f)


def mirror_angles(q):
    """Left-arm angles <-> right-arm angles for mirrored poses."""
    return (q[0], -q[1], q[2], -q[3])


def _in_limits(x, lim, tol=1e-6):
    return lim[0] - tol <= x <= lim[1] + tol


def _fit(angle, lim, prev):
    """angle +/- 2pi that lies in lim, nearest prev; None if none fits."""
    best = None
    for k in (-1, 0, 1):
        a = angle + 2 * math.pi * k
        if _in_limits(a, lim) and (best is None or abs(a - prev) < abs(best - prev)):
            best = a
    return best


def _blend(prev, new, s):
    """Trust a new angle fully when its sine is big, hold the previous one near the singularity."""
    w = clamp((abs(s) - SINGULAR) / SINGULAR, 0.0, 1.0)
    return prev + w * wrap(new - prev)


def solve_left_arm(u, f, prev=LEFT_REST, limits=LEFT_LIMITS):
    """Left-arm angles whose upper arm points along u and forearm along f (trunk frame).

    Every mirror solution inside the limits is tried; the one closest to prev wins. If none fits,
    the closest one is clamped into the limits.
    """
    u, f = unit(u), unit(f)
    candidates = []
    for roll_sign in (-1.0, 1.0):
        roll = roll_sign * math.acos(clamp(u[1], -1.0, 1.0))
        sr = math.sin(roll)
        pitch = math.atan2(u[0] / sr, u[2] / sr) if abs(sr) > 1e-6 else prev[0]
        pitch = _blend(prev[0], pitch, sr)
        # forearm in the upper arm's frame: (-sin(elbow) cos(twist), cos(elbow), sin(elbow) sin(twist))
        fl = rot_x(-roll, rot_y(-pitch, f))
        for elbow_sign in (-1.0, 1.0):
            elbow = elbow_sign * math.acos(clamp(fl[1], -1.0, 1.0))
            se = math.sin(elbow)
            twist = math.atan2(fl[2] / se, -fl[0] / se) if abs(se) > 1e-6 else prev[2]
            twist = _blend(prev[2], twist, se)
            candidates.append((pitch, roll, twist, elbow))
    best, best_cost = None, None
    for c in candidates:
        fitted = []
        for i, (a, lim) in enumerate(zip(c, limits)):
            a = _fit(a, lim, prev[i])
            if a is None:
                break
            fitted.append(a)
        inside = len(fitted) == 4
        if not inside:
            fitted = [clamp(_fit(a, (-1e9, 1e9), prev[i]), *lim)
                      for i, (a, lim) in enumerate(zip(c, limits))]
        cost = sum((a - p) ** 2 for a, p in zip(fitted, prev)) + (0.0 if inside else 1e6)
        if best_cost is None or cost < best_cost:
            best, best_cost = tuple(fitted), cost
    return best


def solve_arm(side, u, f, prev=None):
    if side == "left":
        return solve_left_arm(u, f, LEFT_REST if prev is None else prev)
    prev = RIGHT_REST if prev is None else prev
    return mirror_angles(solve_left_arm(mirror(u), mirror(f), mirror_angles(prev)))


# -- human side (Rokoko Studio frames) ------------------------------------------------------------

def find_body(frame, actor=None):
    """(actor name, body) of the first actor with every bone this module needs, else (None, None)."""
    scene = frame.get("scene", frame) if isinstance(frame, dict) else {}
    for a in scene.get("actors") or []:
        if actor and a.get("name") != actor:
            continue
        body = a.get("body") or {}
        if REQUIRED_BONES <= body.keys():
            return a.get("name"), body
    return None, None


def missing_bones(frame):
    """Bones the first actor lacks (for error messages)."""
    scene = frame.get("scene", frame) if isinstance(frame, dict) else {}
    actors = scene.get("actors") or []
    if not actors:
        return sorted(REQUIRED_BONES)
    return sorted(REQUIRED_BONES - (actors[0].get("body") or {}).keys())


def _pos(body, bone):
    p = body[bone]["position"]
    return (float(p["x"]), float(p["y"]), float(p["z"]))


def torso_axes(body, handedness):
    """Human torso frame (forward, left, up) as world vectors.

    handedness is +1 or -1: which way cross(left, up) points (forward or backward) depends on the
    world's handedness, so it comes from calibration.
    """
    up = unit(sub(_pos(body, "neck"), _pos(body, "hip")))
    left = sub(_pos(body, "leftUpperArm"), _pos(body, "rightUpperArm"))
    left = unit(sub(left, scale(up, dot(left, up))))
    forward = scale(cross(left, up), handedness)
    return forward, left, up


def to_torso(v, axes):
    return (dot(v, axes[0]), dot(v, axes[1]), dot(v, axes[2]))


def human_arm_directions(body, side, axes):
    shoulder, elbow, wrist = (_pos(body, b) for b in ARM_BONES[side])
    return to_torso(unit(sub(elbow, shoulder)), axes), to_torso(unit(sub(wrist, elbow)), axes)


def detect_handedness(body):
    """+1 or -1 from a pose with both arms straight out in front. Raises if the pose doesn't look like that."""
    up = unit(sub(_pos(body, "neck"), _pos(body, "hip")))
    left = sub(_pos(body, "leftUpperArm"), _pos(body, "rightUpperArm"))
    left = unit(sub(left, scale(up, dot(left, up))))
    raw_forward = cross(left, up)
    reach = (0.0, 0.0, 0.0)
    for side in ("left", "right"):
        shoulder, _, wrist = (_pos(body, b) for b in ARM_BONES[side])
        reach = tuple(r + c for r, c in zip(reach, unit(sub(wrist, shoulder))))
    reach = unit(reach)
    along = dot(reach, raw_forward)
    if abs(along) < 0.6:
        raise ValueError("arms aren't pointing straight forward (or straight back); "
                         "face forward and hold both arms straight out in front of you")
    return 1 if along > 0 else -1


class ArmMapper:
    """Rokoko body -> 8 T1 arm angles (left pitch, roll, twist, elbow, then right), in SDK order."""

    def __init__(self, handedness=None):
        self.handedness = handedness
        self.prev = {"left": LEFT_REST, "right": RIGHT_REST}

    @property
    def calibrated(self):
        return self.handedness in (1, -1)

    def calibrate(self, body):
        self.handedness = detect_handedness(body)

    def reset(self, current=None):
        """Seed the solver (e.g. with the robot's measured arm angles) so the first answer is the nearest one."""
        if current is None:
            self.prev = {"left": LEFT_REST, "right": RIGHT_REST}
        else:
            self.prev = {"left": tuple(current[:4]), "right": tuple(current[4:8])}

    def angles(self, body):
        if not self.calibrated:
            raise RuntimeError("not calibrated: hold both arms straight forward and calibrate")
        axes = torso_axes(body, self.handedness)
        out = []
        for side in ("left", "right"):
            u, f = human_arm_directions(body, side, axes)
            q = solve_arm(side, u, f, self.prev[side])
            self.prev[side] = q
            out.extend(q)
        return out

    def to_json(self):
        return {"handedness": self.handedness}

    def load_json(self, data):
        h = data.get("handedness")
        if h not in (1, -1):
            raise ValueError("calibration file has no handedness")
        self.handedness = h


def load_calibration(mapper, path=DEFAULT_CALIBRATION):
    path = os.path.expanduser(path)
    if os.path.exists(path):
        with open(path) as f:
            mapper.load_json(json.load(f))
        return True
    return False


def save_calibration(mapper, path=DEFAULT_CALIBRATION):
    with open(os.path.expanduser(path), "w") as f:
        json.dump(mapper.to_json(), f, indent=1)


# -- motion shaping -------------------------------------------------------------------------------

ALL_LIMITS = LEFT_LIMITS + RIGHT_LIMITS


class Shaper:
    """Keeps commands safe: joint limits with a margin, smoothing, and a top speed per joint."""

    def __init__(self, start, max_speed=1.0, smoothing=0.3, margin=0.05):
        self.q = list(start)
        self.max_speed, self.smoothing, self.margin = max_speed, smoothing, margin

    def step(self, target, dt):
        step_max = self.max_speed * dt
        for i, (t, (lo, hi)) in enumerate(zip(target, ALL_LIMITS)):
            t = clamp(t, lo + self.margin, hi - self.margin)
            want = self.q[i] + (1.0 - self.smoothing) * (t - self.q[i])
            self.q[i] += clamp(want - self.q[i], -step_max, step_max)
        return list(self.q)
