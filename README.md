# Hivemind Intersection Controller

A fully autonomous intersection controller simulation built in Python. No traffic lights, no stop signs — a centralized AI continuously schedules every vehicle's precise arrival time at the intersection box and modulates their speed so they glide through without stopping.

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![numpy](https://img.shields.io/badge/numpy-required-orange) ![matplotlib](https://img.shields.io/badge/matplotlib-required-orange)

---

## What It Does

Vehicles approach from all four directions across six lanes per road (three ingress, three egress). The hive mind knows where every vehicle is, how fast it's moving, and what conflicts exist between crossing paths. Instead of reacting to vehicles at the stop line, it plans several seconds ahead and assigns each vehicle a conflict-free time slot — then the vehicle simply adjusts its cruise speed to arrive exactly on time.

At normal traffic loads this means **no stops, no hard braking, no waiting at a red light**. A vehicle 50 metres out that needs to wait four seconds just drives at ~12 mph instead of ~27 mph and arrives right as the box clears. When unexpected backlogs form, the controller detects them and switches to a high-throughput burst mode that rapidly clears queues as tight platoons.

---

## Architecture

### Normal Mode — Predictive Arrival-Time Scheduling

The scheduler runs every 0.12 seconds and rebuilds a conflict-free slot table from scratch:

1. For each lane's head vehicle, compute the natural arrival time at current speed
2. Sort all movements by effective priority — earliest natural arrival minus a starvation bonus that grows with wait time and queue depth, preventing any movement from being indefinitely blocked by right-turners
3. Walk the list greedily: assign each movement the earliest time window that doesn't overlap with any geometrically conflicting movement already scheduled
4. Compatible movements (e.g. N→S and S→N, which cross paths safely) get concurrent slots — they go simultaneously
5. Broadcast `t_arrive` to each vehicle; followers in the same lane get back-to-back slots for smooth platooning

Each vehicle then computes every physics tick:

```
v_needed = dist_to_front / (t_arrive - now)
v_needed = clamp(v_needed, V_MIN=3.0 m/s, V_MAX=12.96 m/s)
```

A gentle acceleration/deceleration envelope (`ACCEL_MAX = 2.5 m/s²`, `DECEL_MAX = 4.0 m/s²`) is applied so speed changes are gradual and never abrupt.

### Flush Mode — Maximum Concurrent Throughput

Triggered automatically when any single lane reaches **≥ 6 vehicles queued**. At that depth, the predictive scheduler's slot-pushing logic starts to cascade delays. Flush mode bypasses slot scheduling entirely:

1. Find the largest geometrically conflict-free set of movements that have vehicles ready
2. Signal all of them **GO** simultaneously — vehicles cruise to the box at full speed
3. Schedule their followers in tight back-to-back slots (headway × 0.35 instead of full headway)
4. Re-evaluate every scheduler tick so new concurrent movements are added the moment they become safe
5. Anti-starvation: any movement that hasn't entered the box in 3.5 seconds gets forced to the front of the next GO set
6. Exit back to normal mode when all queues drop to ≤ 3 vehicles

The HUD shows `⚡ FLUSH` or `  normal` in real time so you can watch the mode switch during heavy traffic.

### Safety Backstop

Both modes share a final physical safety gate at the box edge. The front bumper position is tracked directly — a vehicle can only enter when its front bumper reaches the box edge AND no conflicting vehicle is in the box or committed to entering this same physics tick (`tick_reserved`). This is the ground truth; the scheduler is an optimization layer on top of it.

---

## Road & Vehicle Geometry

All dimensions are real-world scale:

| Parameter | Value | Imperial |
|---|---|---|
| Lane width | 3.66 m | 12 ft |
| Lanes per road | 6 (3 ingress + 3 egress) | — |
| Intersection box half-size | 11.0 m | ~36 ft |
| Vehicle width | 2.44 m | 8 ft |
| Vehicle length | 4.27 m | 14 ft |
| Approach length | 90.0 m | ~295 ft |
| Stop line clearance | 0.61 m | 2 ft |

Vehicles are rendered as oriented 8×14 ft rectangles that rotate smoothly with their heading. During box traversal the heading is computed from the Bézier tangent so vehicles visually track their curve through the intersection.

Lane markings are dashed grey lines at each **boundary** between lanes (not centrelines), matching real road striping. A gold centreline divides ingress from egress halves. Background is dark asphalt (`#1c1c1e`).

---

## Conflict Model

All 12 possible movements (4 approaches × 3 turn types) were geometrically verified. Bézier paths through the box were sampled at 80 points and minimum separations computed. The result: **50 safe pairs** (paths that never come within 2 m of each other) and **16 conflicting pairs** (separation < 0.11 m — essentially crossing paths).

The conflict check is O(1) via a frozen set lookup against the verified safe pairs — fast enough to run thousands of times per second without impacting frame rate.

---

## Performance

At default arrival rate (0.22 veh/s per approach ≈ 0.88 veh/s total):

- Average induced delay: **~1.6 seconds**
- Maximum observed delay: **~12 seconds** (rare, during random arrival clusters)
- Box utilization: ~85%
- Flush mode engagement: 0%

At rate 0.25 veh/s per approach:

- Average induced delay: **~2.4 seconds**
- Flush mode engages briefly during random spikes: ~5%

At rate 0.35 veh/s per approach (heavy traffic):

- Average induced delay: **~6 seconds**
- Flush mode: ~55% of the time — dual-mode hybrid in action

At rate 0.50 veh/s per approach the intersection is at or above theoretical saturation — straights alone require ~82% of available box time — so delays grow unboundedly regardless of algorithm.

---

## Tunable Parameters

All in the `TUNABLES` block at the top of `hivemind.py`:

```python
ARRIVAL_RATE  = 0.22    # vehicles/s/approach — try 0.25–0.35 for heavier load
SPEED_DESIRED = 12.0    # m/s (~27 mph) free-flow speed
V_MIN         = 3.0     # m/s minimum modulation speed before stopping
HEADWAY       = 0.55    # seconds minimum gap between consecutive box entries
FLUSH_Q       = 6       # queue depth to enter flush mode
FLUSH_EXIT_Q  = 3       # queue depth to exit flush mode (hysteresis)
MAX_HOLDOUT   = 3.5     # seconds max starvation time in flush mode
TURN_P        = {"S": 0.55, "L": 0.25, "R": 0.20}  # turn distribution
```

---

## Usage

```bash
pip install numpy matplotlib
python hivemind.py
```

Press **Q** to quit. On close the simulation writes `hivemind_log.csv` with a full event log including per-vehicle induced delay, entry time, movement, and mode (normal/FLUSH).

---

## Dependencies

- Python 3.8+
- numpy
- matplotlib

---

*Author: semantical-monster*
