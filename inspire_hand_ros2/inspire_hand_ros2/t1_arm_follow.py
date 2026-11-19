"""Lindy (Booster T1) arms follow a Rokoko Smartsuit, live. Booster's own controller keeps balance.

    python -m inspire_hand_ros2.t1_arm_follow dry-run            # any PC: print the joint angles, no robot
    python -m inspire_hand_ros2.t1_arm_follow selftest           # on Lindy: nudge the left shoulder, no suit
    python -m inspire_hand_ros2.t1_arm_follow follow             # on Lindy: arms follow the suit

How it drives the robot (same as Booster's own Python ArmController in booster_robotics_sdk_python):
B1LocoClient.UpperBodyCustomControl(True), then LowCmd on rt/joint_ctrl at 100 Hz with position
targets for joints 2..9 (both arms, kp 60, kd 3) and zero gains for head, waist and legs, which
Booster's controller keeps running. UpperBodyCustomControl(False) on exit.

Safety: Lindy on the gantry first. Commands start from where the arms are, move at most
--max-speed rad/s, stay inside the URDF joint limits, hold still when the suit stream drops, and
on exit (Ctrl-C) the arms go back to where they started before control is handed back.
A second Ctrl-C hands control back immediately. A tilting robot (IMU roll/pitch > 0.6 rad) stops it.
"""
import argparse
import math
import sys
import threading
import time

from . import rokoko, t1_arms

ARM_SLICE = slice(t1_arms.LEFT_ARM_INDEX, t1_arms.RIGHT_ARM_INDEX + 4)  # joints 2..9
T1_JOINTS = 23
TILT_LIMIT = 0.6        # rad of IMU roll or pitch before we stop
STATE_TIMEOUT = 0.5     # s without robot state before we stop
RETURN_TIMEOUT = 6.0    # s allowed for the arms to get back to the start pose


# -- inputs ---------------------------------------------------------------------------------------

class SuitSource:
    """Rokoko Studio stream -> 8 arm targets, or None while the stream is stale."""

    def __init__(self, port=rokoko.DEFAULT_PORT, relay=None, actor=None,
                 calibration_file=t1_arms.DEFAULT_CALIBRATION, timeout=0.3):
        self.receiver = rokoko.StudioReceiver(port, relay=relay)
        self.mapper = t1_arms.ArmMapper()
        self.calibration_file = calibration_file
        t1_arms.load_calibration(self.mapper, calibration_file)
        self.actor_filter, self.timeout = actor, timeout
        self.body = self.actor = self.error = None
        self.seen = -1e9
        self.finished = False

    def poll(self, now):
        try:
            frame = self.receiver.poll()
        except ValueError as e:
            frame, self.error = None, str(e)
        if frame is not None:
            name, body = t1_arms.find_body(frame, self.actor_filter)
            if body is None:
                self.error = "stream is missing bones: " + ", ".join(t1_arms.missing_bones(frame))
            else:
                self.actor, self.body, self.seen, self.error = name, body, now, None

    def live(self, now):
        return self.body is not None and now - self.seen < self.timeout

    def reset(self, current):
        self.mapper.reset(current)

    def target(self, now):
        self.poll(now)
        if not self.live(now):
            return None
        try:
            return self.mapper.angles(self.body)
        except ValueError as e:  # degenerate pose (e.g. two bones on top of each other): hold
            self.error = "can't read this pose: %s" % e
            return None

    def calibrate(self, wait=5.0):
        """Wait for a live frame, then take the arms-forward pose. Saves the result."""
        end = time.monotonic() + wait
        while time.monotonic() < end:
            self.poll(time.monotonic())
            if self.live(time.monotonic()):
                self.mapper.calibrate(self.body)
                t1_arms.save_calibration(self.mapper, self.calibration_file)
                return
            time.sleep(0.02)
        raise RuntimeError(self.status(time.monotonic()))

    def status(self, now):
        if self.error:
            return self.error
        if self.body is None:
            return "waiting for Rokoko Studio on UDP port %d" % self.receiver.port
        if not self.live(now):
            return "suit stream stopped - holding"
        return "suit: %s" % self.actor

    def close(self):
        self.receiver.close()


class SelfTestSource:
    """No suit: swing the left shoulder 0.3 rad forward and back once, over `period` seconds."""

    def __init__(self, period=6.0, amplitude=0.3):
        self.period, self.amplitude = period, amplitude
        self.start = self.t0 = None
        self.finished = False

    def reset(self, current):
        self.start = list(current)
        self.t0 = None

    def target(self, now):
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        if t >= self.period:
            self.finished = True
            t = self.period
        q = list(self.start)
        q[0] -= self.amplitude * (1 - math.cos(2 * math.pi * t / self.period)) / 2  # negative pitch = forward
        return q

    def status(self, now):
        return "self-test" if not self.finished else "self-test done"

    def close(self):
        pass


# -- outputs --------------------------------------------------------------------------------------

class BoosterArms:
    """Lindy through booster_robotics_sdk_python (pip install booster_robotics_sdk_python)."""

    def __init__(self, net="127.0.0.1", kp=60.0, kd=3.0):
        import booster_robotics_sdk_python as br  # only on the robot
        self.br, self.kp, self.kd = br, kp, kd
        self._lock = threading.Lock()
        self._q = self._rpy = None
        self._stamp = -1e9
        self.enabled = False
        br.ChannelFactory.Instance().Init(0, net)
        self.client = br.B1LocoClient()
        self.client.Init()
        self.pub = br.B1LowCmdPublisher()
        self.pub.InitChannel()
        self.sub = br.B1LowStateSubscriber(self._on_state)
        self.sub.InitChannel()

    def _on_state(self, msg):
        q = [m.q for m in msg.motor_state_parallel]
        rpy = list(msg.imu_state.rpy)
        with self._lock:
            self._q, self._rpy, self._stamp = q, rpy, time.monotonic()

    def wait_for_state(self, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._q is not None:
                    if len(self._q) != T1_JOINTS:
                        raise RuntimeError("robot reports %d joints, this script is for the 23-joint T1 "
                                           "(the 7-DoF-arm T1 uses another joint layout)" % len(self._q))
                    return
            time.sleep(0.02)
        raise RuntimeError("no robot state on rt/low_state; is the SDK talking to Lindy (--net)?")

    def measured_arms(self):
        with self._lock:
            return list(self._q[ARM_SLICE])

    def check(self):
        """None if fine, else why we must stop."""
        with self._lock:
            age, rpy = time.monotonic() - self._stamp, self._rpy
        if age > STATE_TIMEOUT:
            return "lost robot state for %.1f s" % age
        if abs(rpy[0]) > TILT_LIMIT or abs(rpy[1]) > TILT_LIMIT:
            return "robot is tilting (roll %.2f, pitch %.2f rad)" % (rpy[0], rpy[1])
        return None

    def enable(self, on, retries=5):
        br = self.br
        for attempt in range(retries):
            try:
                try:
                    ret = self.client.UpperBodyCustomControl(on)
                except TypeError:  # older bindings take the parameter object
                    ret = self.client.UpperBodyCustomControl(br.UpperBodyCustomControlParameter(on))
            except RuntimeError as e:
                if "code = 100" in str(e) and attempt + 1 < retries:  # RPC timeout: try again
                    time.sleep(1.0)
                    continue
                raise RuntimeError("UpperBodyCustomControl(%s) failed: %s" % (on, e))
            if isinstance(ret, int) and ret != 0:
                if ret == 100 and attempt + 1 < retries:
                    time.sleep(1.0)
                    continue
                raise RuntimeError("UpperBodyCustomControl(%s) returned %d - is Lindy standing "
                                   "(Prepare or Walking mode)?" % (on, ret))
            self.enabled = on
            return
        raise RuntimeError("UpperBodyCustomControl(%s) timed out" % on)

    def send(self, arm_q, weight):
        br = self.br
        motors = []
        for i in range(T1_JOINTS):
            m = br.MotorCmd()
            m.mode = 0x0A  # position mode, as in Booster's ArmController
            m.dq = m.tau = 0.0
            if ARM_SLICE.start <= i < ARM_SLICE.stop:
                m.q, m.kp, m.kd, m.weight = arm_q[i - ARM_SLICE.start], self.kp, self.kd, weight
            else:  # head, waist, legs: zero gains, Booster's controller keeps them
                m.q = m.kp = m.kd = m.weight = 0.0
            motors.append(m)
        cmd = br.LowCmd()
        cmd.cmd_type = br.LowCmdType.PARALLEL
        cmd.motor_cmd = motors
        self.pub.Write(cmd)

    def close(self):
        for ch in (self.sub, self.pub):
            try:
                ch.CloseChannel()
            except Exception:
                pass


class FakeArms:
    """Stand-in robot for dry runs and tests: starts in the arms-down pose and records commands."""

    def __init__(self, start=None):
        self.q = list(start or (t1_arms.LEFT_REST + t1_arms.RIGHT_REST))
        self.enabled = False
        self.sent = []
        self.problem = None

    def wait_for_state(self, timeout=0):
        pass

    def measured_arms(self):
        return list(self.q)

    def check(self):
        return self.problem

    def enable(self, on):
        self.enabled = on

    def send(self, arm_q, weight):
        self.q = list(arm_q)
        self.sent.append((list(arm_q), weight))

    def close(self):
        pass


# -- the controller -------------------------------------------------------------------------------

class ArmFollow:
    """Phases: ramp_in (hold, weight 0 -> max) -> follow -> back (return to start pose) ->
    ramp_out (weight max -> 0) -> done (control handed back)."""

    def __init__(self, io, source, max_speed=1.0, max_weight=1.0, ramp=2.0):
        self.io, self.source = io, source
        self.max_speed, self.max_weight, self.ramp = max_speed, max_weight, ramp
        self.phase = None
        self.reason = None

    def start(self, now):
        self.start_q = self.io.measured_arms()
        self.shaper = t1_arms.Shaper(self.start_q, max_speed=self.max_speed)
        self.source.reset(self.start_q)
        self.io.enable(True)
        self.phase, self.t_phase, self.last = "ramp_in", now, now

    @property
    def done(self):
        return self.phase == "done"

    def stop(self, reason):
        """Graceful stop: arms back to the start pose, then control handed back."""
        if self.phase in ("ramp_in", "follow"):
            self.reason = reason
            self._enter("back", None)

    def abort(self, reason):
        """Robot problem: hand control back to Booster's controller right away."""
        self.reason = reason
        if self.phase != "done":
            self.io.enable(False)
            self.phase = "done"

    def _enter(self, phase, now):
        self.phase, self.t_phase = phase, now

    def tick(self, now):
        dt = max(0.0, min(0.05, now - self.last))
        self.last = now
        problem = self.io.check()
        if problem:
            self.abort(problem)
            return self.shaper.q
        if self.t_phase is None:
            self.t_phase = now
        elapsed = now - self.t_phase
        weight = self.max_weight
        target = self.start_q
        if self.phase == "ramp_in":
            weight = self.max_weight * min(1.0, elapsed / self.ramp)
            if elapsed >= self.ramp:
                self._enter("follow", now)
        elif self.phase == "follow":
            target = self.source.target(now)
            if target is None:
                target = self.shaper.q  # stream lost: hold still
            if self.source.finished:
                self.stop("finished")
        elif self.phase == "back":
            if max(abs(a - b) for a, b in zip(self.shaper.q, self.start_q)) < 0.02 or elapsed > RETURN_TIMEOUT:
                self._enter("ramp_out", now)
        elif self.phase == "ramp_out":
            weight = self.max_weight * max(0.0, 1.0 - elapsed / self.ramp)
            if elapsed >= self.ramp:
                self.io.enable(False)
                self.phase = "done"
                return self.shaper.q
        else:
            return self.shaper.q
        q = self.shaper.step(target, dt)
        self.io.send(q, weight)
        return q


def format_angles(q):
    deg = ["%6.1f" % math.degrees(a) for a in q]
    return "L pitch %s roll %s twist %s elbow %s | R pitch %s roll %s twist %s elbow %s" % tuple(deg)


def run(follow, rate=100.0, print_every=0.25, out=sys.stdout):
    period = 1.0 / rate
    follow.start(time.monotonic())
    next_tick = next_print = time.monotonic()
    try:
        while not follow.done:
            try:
                now = time.monotonic()
                q = follow.tick(now)
                if now >= next_print:
                    next_print = now + print_every
                    out.write("\r[%-8s] %s  (%s)   " % (follow.phase, format_angles(q),
                                                         follow.source.status(now)))
                    out.flush()
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
            except KeyboardInterrupt:
                if follow.phase in ("ramp_in", "follow"):
                    out.write("\nCtrl-C: bringing the arms back, then handing control back "
                              "(Ctrl-C again to hand back now)\n")
                    follow.stop("Ctrl-C")
                else:
                    raise
    finally:
        if follow.io.enabled:
            follow.io.enable(False)
        out.write("\n%s\n" % ("stopped: %s" % follow.reason if follow.reason else "done"))


# -- command line ---------------------------------------------------------------------------------

def _relay(text):
    if not text:
        return None
    host, _, port = text.rpartition(":")
    return (host or "127.0.0.1", int(port))


def _confirm(args):
    if args.yes:
        return
    print("Lindy's arms will move. Lindy should be on the gantry, standing (Prepare or Walking mode),")
    print("with nobody within arm's reach. Type 'yes' to continue: ", end="", flush=True)
    if input().strip().lower() != "yes":
        raise SystemExit("cancelled")


def _calibrate_if_needed(source, force):
    if source.mapper.calibrated and not force:
        return
    print("Calibration: face the direction you call forward, hold BOTH arms straight out in front")
    print("of you at shoulder height, then press Enter.", end="", flush=True)
    input()
    source.calibrate()
    print("calibrated (saved to %s)" % source.calibration_file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("dry-run", "selftest", "follow"),
                    help="dry-run: no robot, print angles; selftest: nudge the left shoulder, no suit; "
                         "follow: arms follow the suit")
    ap.add_argument("--net", default="127.0.0.1", help="network interface/IP for the Booster SDK "
                    "(127.0.0.1 when running on Lindy)")
    ap.add_argument("--port", type=int, default=rokoko.DEFAULT_PORT, help="UDP port Rokoko Studio streams to")
    ap.add_argument("--relay", metavar="HOST:PORT", help="forward the Rokoko stream here too, e.g. "
                    "127.0.0.1:14044 for the glove teleop running at the same time")
    ap.add_argument("--actor", help="Rokoko actor name (default: first actor with a full body)")
    ap.add_argument("--calibrate", action="store_true", help="redo the arms-forward calibration")
    ap.add_argument("--calibration", default=t1_arms.DEFAULT_CALIBRATION, help="calibration file")
    ap.add_argument("--max-speed", type=float, default=1.0, help="top joint speed, rad/s (default 1.0)")
    ap.add_argument("--max-weight", type=float, default=1.0,
                    help="how much our arm commands override Booster's (0..1, default 1)")
    ap.add_argument("--kp", type=float, default=60.0, help="arm joint stiffness (default 60, as Booster's)")
    ap.add_argument("--kd", type=float, default=3.0, help="arm joint damping (default 3, as Booster's)")
    ap.add_argument("--yes", action="store_true", help="skip the safety prompt")
    args = ap.parse_args(argv)
    if not 0.0 <= args.max_weight <= 1.0 or not 0.0 < args.max_speed <= 5.0:
        ap.error("--max-weight must be 0..1 and --max-speed 0..5 rad/s")

    source = io = None
    try:
        if args.mode == "selftest":
            source = SelfTestSource()
        else:
            source = SuitSource(args.port, _relay(args.relay), args.actor, args.calibration)
            _calibrate_if_needed(source, args.calibrate)
        if args.mode == "dry-run":
            io = FakeArms()
        else:
            _confirm(args)
            io = BoosterArms(args.net, args.kp, args.kd)
            io.wait_for_state()
        run(ArmFollow(io, source, args.max_speed, args.max_weight))
    except (RuntimeError, ValueError) as e:
        raise SystemExit("error: %s" % e)
    except KeyboardInterrupt:
        raise SystemExit("stopped (control handed back)")
    finally:
        if io is not None:
            io.close()
        if source is not None:
            source.close()


if __name__ == "__main__":
    main()
