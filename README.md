# Inspire RH56DFX-2L on the Booster T1 (ROS 2)

`inspire_hand_ros2/` is a ROS 2 package with:

| Executable | What it does |
|---|---|
| `driver` | Talks to the hand over RS485 (USB-RS485 adapter) and exposes it as ROS 2 topics. |
| `booster_bridge` | Same command topics, forwarded to the T1's own hand control (`/booster_rpc_service`). Use it only if the hand is plugged into the robot's hand port. |
| `hand_gui` | Manual control panel: a slider per finger, gestures, speed/force, live feedback. Works through ROS 2, or straight to the hand over serial (no ROS, works on Windows), or against a simulator. Add `--rokoko` to drive it from a Rokoko Smartglove. |
| `rokoko_teleop` | Follows a Rokoko Smartglove (Rokoko Studio's live stream) and publishes `cmd_angle`. See [section 7](#7-rokoko-smartglove-teleoperation). |
| `t1_arm_follow` | Lindy's (the T1's) arms follow a Rokoko Smartsuit live, while Booster's controller keeps balance. No ROS needed. See [section 8](#8-lindys-arms-follow-the-rokoko-suit). |
| `lindy_viewer` | A virtual Lindy (Booster's T1 model in MuJoCo) whose arms follow your suit, so you can check the mapping on any PC before using the robot. |

The protocol code (`rh56.py`) follows the RH56 manual V1.0.9 and is tested against the manual's worked examples byte for byte (`test/test_rh56.py`).

## 1. Pick how the hand connects

**A. USB-RS485 adapter → T1 onboard computer → `driver`** (most likely for a hand bought separately)
You own the RS485 link and get full feedback (angle, force, current, temperature, errors).

**B. The T1's own hand port → `booster_bridge`**
Booster's SDK supports Inspire hands natively (`ControlDexterousHand`, API 2013), but only if the robot's factory config lists the hand (`B1LocoClient::GetHands()`). Then the robot owns the RS485 link. The bridge sends commands only, with no feedback topics. Ask Booster whether your T1's wrist has a hand port and whether they can enable an Inspire hand on it.

When in doubt, use A. It needs nothing from Booster.

## 2. Wiring (option A)

GX12 5-pin plug on the hand (manual, Figure 2):

| Pin | Signal | Connect to |
|---|---|---|
| 1 | GND | supply 0 V **and** adapter GND |
| 2 | VCC | +24 V |
| 3 | RS485 A+ | adapter A / D+ |
| 4 | RS485 B− | adapter B / D− |
| 5 | GND | supply 0 V |

- Power: 24 V. The hand draws 0.12 A idle, 0.6 A moving, **2 A at max grip**, so use a 24 V supply or DC-DC converter rated ≥ 3 A per hand. Check with Booster which T1 power output you can tap, or use a separate converter.
- Any CH340/FTDI/CP210x USB-RS485 adapter works. Default bus is 115200 8N1, hand ID 1.
- The manual doesn't say whether the pin drawing shows the plug or the socket face. Check pin numbers against the cable/connector markings, or with a multimeter, before powering up.

## 3. Bench test on your Windows PC first

1. Use the Inspire PC software (`北京因时机器人灵巧手上位机.exe`) to confirm the hand moves and to read its ID and baud rate. Then **close it**, because only one program can hold the COM port.
2. Run the GUI straight on the serial port:

```bash
pip install pyserial
```
```bash
cd G:\robothand\inspire_hand_ros2
```
```bash
python -m inspire_hand_ros2.hand_gui --serial COM5 --id 1
```

Use your adapter's COM port from Device Manager. Add `--sim` instead of `--serial` to try the GUI with no hardware.

## 4. Install on the T1 (Ubuntu 22.04, ROS 2 Humble)

```bash
mkdir -p ~/ros2_ws/src && cp -r inspire_hand_ros2 ~/ros2_ws/src/
```
```bash
sudo apt install python3-serial python3-tk && sudo usermod -aG dialout $USER
```
(log out and back in after `usermod`)
```bash
cd ~/ros2_ws && colcon build --packages-select inspire_hand_ros2 && source install/setup.bash
```

Find the adapter's stable path (it doesn't change between reboots, unlike `/dev/ttyUSB0`):

```bash
ls -l /dev/serial/by-id/
```

Start the driver:

```bash
ros2 launch inspire_hand_ros2 hand.launch.py port:=/dev/serial/by-id/usb-XXXX hand_id:=1
```

Other launch options: `simulate:=true` (no hardware), `namespace:=right_hand`, `config:=/path/to.yaml`, `backend:=booster` (option B; build Booster's `booster_interface` from [booster_robotics_sdk_ros2](https://github.com/BoosterRobotics/booster_robotics_sdk_ros2) in the same workspace first).

## 5. Manual control

GUI through ROS 2. Run it on the T1 with a display, or on any Linux laptop on the same network with the same `ROS_DOMAIN_ID`:

```bash
ros2 run inspire_hand_ros2 hand_gui --ns /left_hand
```

- Moving a slider sends **only that finger**. The others hold, so nothing jumps.
- **Open / Fist / Point / Scissors / Thumbs up** are presets. For closing gestures the thumb rotates in after the fingers so it doesn't collide.
- **Stop** freezes every finger where it is. **Clear errors** resets stall and over-current faults.
- Speed and force-limit sliders apply to all six DOF. Force limit is how hard a finger squeezes before it stops (grams).

From the command line:

```bash
ros2 topic pub --once /left_hand/cmd_angle std_msgs/msg/Int16MultiArray "{data: [0, 0, 0, 0, 0, 0]}"
```
```bash
ros2 topic pub --once /left_hand/cmd_angle std_msgs/msg/Int16MultiArray "{data: [-1, -1, -1, 1000, -1, -1]}"
```
```bash
ros2 topic echo /left_hand/angle_raw
```

## 6. ROS interface (namespace `/left_hand` by default)

DOF order everywhere: little, ring, middle, index, thumb bend, thumb rotate. Raw angle 1000 = open, 0 = closed, −1 = don't move.

| | Topic / service | Type | Notes |
|---|---|---|---|
| sub | `cmd_angle` | `std_msgs/Int16MultiArray` | 6 raw targets |
| sub | `cmd_joint` | `sensor_msgs/JointState` | radians by joint name (`left_index`, …), partial OK |
| sub | `cmd_speed` | `std_msgs/Int16MultiArray` | 6 × 0..1000 (1000 ≈ full stroke in 0.8 s) |
| sub | `cmd_force` | `std_msgs/Int16MultiArray` | 6 × grams (fingers ≤ 1000, thumb ≤ 1500) |
| pub | `joint_states` | `sensor_msgs/JointState` | position rad, effort N (driver only) |
| pub | `angle_raw`, `force_raw` | `std_msgs/Int16MultiArray` | raw feedback (driver only) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | status, errors, current, temperature (driver only) |
| srv | `clear_error` | `std_srvs/Trigger` | driver only |

Tuning lives in [config/left_hand.yaml](inspire_hand_ros2/config/left_hand.yaml): default speed/force, joint name prefix, and the raw↔radian mapping. The default mapping is a linear flexion angle from the manual's ranges (fingers 19°–176.7°, thumb bend −13°–53.6°, thumb rotation 90°–165°). Change `joint_open_rad` / `joint_closed_rad` to match your URDF.

## 7. Rokoko Smartglove teleoperation

The robot hand follows your glove. The glove goes to Rokoko Studio as usual. Studio streams the finger bones over UDP, and our code turns finger curl into RH56 targets. Only the curl is used: turning or moving your whole hand doesn't move the robot's fingers.

| Robot DOF | Measured from |
|---|---|
| little, ring, middle, index | the finger's three joints (knuckle, middle, tip) |
| thumb bend | the thumb's middle and tip joints |
| thumb rotate | the thumb's base joint (swinging across the palm) |

**1. Turn on streaming in Rokoko Studio.** Rokoko lists live streaming as a Studio Plus (or higher) feature.
1. Open the project and scene with your glove actor.
2. Open the **Streaming** panel on the right, find **Custom**, and click its wrench icon.
3. Set **Forward IP** to the machine running the bridge (`127.0.0.1` if it's this PC, or the T1's IP), port **14043**, format **JSON**. LZ4 also works after `pip install lz4`.
4. Click **Activate**.

**2a. Test on this PC with the hand on COM4 (no ROS):**

```bash
cd G:\robothand\inspire_hand_ros2; python -m inspire_hand_ros2.hand_gui --serial COM4 --rokoko
```

Allow Python through Windows Firewall if it asks. When the glove row says `left glove of <actor>`:
1. Hold your hand flat with the fingers straight and the thumb alongside, then click **Calibrate open**.
2. Make a fist with the thumb folded across the fingers, then click **Calibrate fist**.
3. Tick **Follow glove**.

**Stop** or any gesture button stops following. The calibration is saved to `~/.rokoko_rh56_calibration.json` and reloaded next time. Use `--glove right` to drive the hand with a right glove.

**2b. On the T1.** Point Studio's Forward IP at the T1, then:

```bash
ros2 launch inspire_hand_ros2 hand.launch.py port:=/dev/serial/by-id/usb-XXXX rokoko:=true
```
```bash
ros2 service call /left_hand/calibrate_open std_srvs/srv/Trigger
```
```bash
ros2 service call /left_hand/calibrate_closed std_srvs/srv/Trigger
```
```bash
ros2 service call /left_hand/enable std_srvs/srv/SetBool "{data: false}"
```

The last command pauses following; `{data: true}` resumes it. Or copy the calibration file you made on the PC to the T1's home folder. Settings (`glove_side`, `udp_port`, `smoothing`, `calibration_file`) are in [config/left_hand.yaml](inspire_hand_ros2/config/left_hand.yaml).

If the robot doesn't follow, check the status line or the node log:
- `waiting for Rokoko Studio`: no packets are arriving. Check the Forward IP, the port and the firewall.
- `stream has no left glove`: wrong glove side or actor.
- `not calibrated yet`: it works, but uses rough default ranges until you calibrate.

## 8. Lindy's arms follow the Rokoko suit

Your arms drive Lindy's arms live. Head, waist and legs stay with Booster's own controller, so Lindy keeps its balance (and can walk with the remote) the whole time. This is the first step of whole-body teleoperation: arms copied, legs left to Booster.

**How it works**
- **Your side:** Rokoko Studio streams every bone's position. The script measures your upper arm (shoulder → elbow) and forearm (elbow → wrist) directions relative to your chest. Leaning or turning your whole body therefore doesn't swing Lindy's arms.
- **Lindy's side:** it solves the 4 arm joints per arm (shoulder pitch, shoulder roll, upper-arm twist, elbow) so Lindy's arm points the same way. Your arm length and weight don't matter, because only directions are copied. The geometry and joint limits come from Booster's `T1_23dof.urdf`.
- **Sending:** the same method as Booster's own Python `ArmController`. It calls `UpperBodyCustomControl(True)`, then sends joints 2–9 on `rt/joint_ctrl` at 100 Hz (kp 60, kd 3) with zero gains for everything else.

**Safety built in**
- Commands start from wherever the arms are now, with control faded in over 2 s.
- Each joint moves at most `--max-speed` (default 1 rad/s) and stays 0.05 rad inside its limits.
- If the suit stream stops, the arms hold still.
- **Ctrl-C:** the arms go back to where they started, control fades out, and it's handed back to Booster. A second Ctrl-C hands it back immediately.
- If Lindy tilts more than 0.6 rad, or its state stops arriving, control is handed back at once.

**1. Rokoko Studio.** Use the same Custom stream as for the gloves (JSON, port 14043, Activate), with your suit actor live. Set Forward IP to the machine that runs the script.

**2. Dry run on this PC first (no robot).** This prints Lindy's joint angles while you move:

```bash
cd G:\robothand\inspire_hand_ros2
```
```bash
python -m inspire_hand_ros2.t1_arm_follow dry-run
```

The first time, it asks for the calibration pose: face your forward direction, hold **both arms straight out in front of you** at shoulder height, and press Enter. This is saved to `~/.rokoko_t1_arms_calibration.json`; use `--calibrate` to redo it. Check these:
- **Arms hanging down:** L roll and R roll near ∓80–90°.
- **Left arm raised forward:** L pitch goes negative, toward −90°.
- **Elbow bent:** L elbow negative, R elbow positive.
- **Raising only your left arm moves only the L numbers.** If the right numbers move instead, tell me.

If it says `stream is missing bones: ...`, Rokoko names the bones differently from what the script expects. Send me the list it prints.

**2b. Virtual Lindy (recommended before the real robot).** Same as the dry run, but you *see* a 3D Lindy copying you. It runs on any PC; no NVIDIA GPU is needed.

```bash
pip install mujoco
```
```bash
python -m inspire_hand_ros2.lindy_viewer
```

Use `--demo` to watch built-in moves without the suit (punches, guard, arms out, wave, arms up). `--demo --snapshot lindy.png --seconds 8` saves a picture instead of opening a window. The first run downloads Booster's T1 model (BSD-3, pinned to one commit) into `~/.cache/lindy_viewer`.

The viewer runs the *same* controller as the robot, with the same fade-in, speed limit and joint limits, so what you see is what Lindy's arms will do. Only the arms move. It's a pose viewer, not a physics simulation: nothing can fall over here, so it can't tell you anything about balance. The real robot (step 3) answers that.

**3. On Lindy.** Lindy goes on the gantry, standing (Prepare mode, from the app or the remote), with nobody within arm's reach.

```bash
pip install booster_robotics_sdk_python --user
```
```bash
cd ~/ros2_ws/src/inspire_hand_ros2 && python3 -m inspire_hand_ros2.t1_arm_follow selftest
```

The self-test uses no suit. The left shoulder swings about 17° forward and back once, then control is handed back. If that works, arm control over the SDK works. Then follow the suit, slowly at first:

```bash
python3 -m inspire_hand_ros2.t1_arm_follow follow --max-speed 0.5
```

Raise `--max-speed` once it behaves. Copy `~/.rokoko_t1_arms_calibration.json` from the PC so you don't have to calibrate again on Lindy.

**4. Hands at the same time.** Only one program can listen on port 14043, so the arm script forwards the stream to the glove teleop:

```bash
python3 -m inspire_hand_ros2.t1_arm_follow follow --relay 127.0.0.1:14044
```

Then run the glove teleop on port 14044: set `udp_port: 14044` in [config/left_hand.yaml](inspire_hand_ros2/config/left_hand.yaml), or use `hand_gui --rokoko --rokoko-port 14044`.

**5. Then the real question.** Switch Lindy to Walking mode while the arms follow, still on the gantry. Does Booster keep balance while we move the arms? If yes, walking (from your hips) and the other "shadow" pieces can go on top.

Options: `--net` (SDK network interface, default `127.0.0.1` on Lindy), `--port`, `--actor`, `--max-weight` (0..1, default 1), `--kp`, `--kd`, `--yes` (skip the safety prompt).

## 9. Limits and what's not verified

- Tested here: the protocol against the manual's example frames, the full stack against the built-in simulator, the GUI against the simulator, and the driver/bridge/teleop node logic against a stubbed `rclpy`. The GUI has connected to the real hand on Windows. **The ROS 2 nodes haven't run on real ROS 2 yet.** First real run: bench test (step 3), then `simulate:=true` on the T1, then the real hand.
- Glove teleop is tested with synthetic Rokoko Studio frames (same JSON layout and reference T-pose as Rokoko's own Blender plugin), not yet with a live glove. Thumb rotation is the roughest mapping. Calibrate before relying on it.
- One hand per serial port. For two hands, use two adapters (and give each its own namespace and `joint_prefix`).
- The Booster bridge is written from Booster's SDK headers (`b1_loco_api.hpp`) and their ROS 2 RPC example. It hasn't been run against a T1.
- The arm math is checked against Booster's own MuJoCo model of the T1 (`T1_23dof.xml`): 200 random arm poses agree within 0.005 (about 0.3°), and rendered poses (punches, guard, wave, arms up) look right from the front and the side.
- Arm follow (section 8) is tested against synthetic suit frames (both left- and right-handed world axes, leaning and turning), 400 random poses through the arm math, a fake robot, and a stand-in for Booster's Python SDK with the 1.6.3 names. It **hasn't moved a real T1 yet**, and these points are open:
  - Whether `UpperBodyCustomControl` also works in Walking mode.
  - What `MotorCmd.weight` really does on the T1. Booster's examples disagree: one ramps it to 0.5, another uses 1.0, and their `ArmController` leaves it at 0.
  - The bone names (`hip`, `neck`, `leftUpperArm`, `leftLowerArm`, `leftHand`, and the same on the right) are checked against the names inside Rokoko Studio 2.6.2 (`Studio.LivestreamFrames.dll`), not yet against a live suit stream.
  - The 4-joint arm can't match every human pose. Reaches behind the back are clamped to the joint limits.
- Force readings depend on the hand's force-sensor calibration. Recalibrate from the Inspire PC software with the hand unloaded if forces look off at rest.
