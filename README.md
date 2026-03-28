# hivemind

> A centralized autonomous intersection controller — simulation prototype exploring whether a "hive mind" coordinating all vehicles through a shared conflict zone can outperform conventional traffic lights.

---


https://github.com/user-attachments/assets/a4886fc3-448c-4f9c-a8dc-cdc1f4b76a4d


---

## The Problem

Traditional traffic lights are blunt instruments. They stop entire approach directions for fixed durations — even when only one car needs to cross, even when opposing movements could safely overlap, even when a left-turner and a right-turner from opposite directions have paths that never come close to touching.

The question this project asks:

> What if every vehicle approaching an intersection reported its position, speed, and intended movement to a centralized controller — and that controller continuously solved for the maximum number of vehicles that could safely traverse the intersection simultaneously?

That is the hive mind.

---

## How It Works

### Road Layout

Each approach direction (N/S/E/W) has **6 lanes** — 3 ingress, 3 egress — centered on the road axis:

```
-3W  -2.5W -1.5W -0.5W  │  +0.5W +1.5W +2.5W  +3W
 |  [R-in][ST-in][LT-in] │ [LT-out][ST-out][R-out] |
        ingress (3)       ↑        egress (3)
                     gold centreline
```

Vehicles self-sort into the correct lane by intention — right-turners, straights, and left-turners each have their own independent queue. A right-turner never waits behind a stopped left-turner.

### Conflict Model

All 12 possible movements (4 directions × 3 intentions) were audited geometrically — 80 Bezier path samples per movement, minimum separation computed for all 66 pairs:

- **16 genuine conflicts** (paths come within 2.0m of each other)
- **50 safe pairs** (paths stay separated by ≥ 2.0m throughout the box)

This replaces hand-reasoned compatibility rules with verified geometry. Every decision the hive mind makes is grounded in actual path data.

### The Control Algorithm

Every `SCHED_DT` seconds, and immediately whenever a vehicle enters or exits the box:

1. **Identify ready movements** — any lane whose head vehicle is within `READY_DIST` metres and has no conflicting vehicle currently in the box
2. **Find the maximum conflict-free subset** — exhaustive search over combinations of ready movements (N ≤ 12, 2¹² = 4096 max subsets, runs in < 2ms)
3. **Grant GO** to that subset. Everything else holds.
4. **Additive pass** — any ready movement compatible with the entire current GO set gets added, even if it wasn't in the primary subset
5. **Anti-starvation** — movements not seen GO in `STARVATION_LIMIT` seconds force a switch; `MAX_HOLDOUT` is an absolute hard cap

No phases. No scoring. No weights. Just: *who's ready, and what's the largest safe set?*

### Vehicle Physics

| State | Behavior |
|-------|----------|
| **GO** (head of lane) | Cruise at `SPEED_DESIRED`, enter box when `dist_in ≤ 0` |
| **HOLD** (head of lane) | Cruise at full speed until physics require braking, then stop at `STOP_LINE` |
| **Follower** | Cruise freely until within `FOLLOW_COMFORT` metres of leader, then blend toward leader speed |
| **In box** | Accelerate back to `SPEED_DESIRED` throughout traversal |
| **Egress** | Constant `SPEED_DESIRED` in assigned lane — no speed variation, no overtaking |

The car-following model is intentionally **distance-gated**: following vehicles only react to the vehicle ahead when within 18m. Beyond that they cruise freely at the speed limit. This prevents a stopped head vehicle from cascading a slow-speed wave back through the entire approach queue.

### Platooning

When a movement is in GO and the next vehicle in that lane is also approaching, the hive mind keeps that movement active — allowing the follower to enter without re-evaluating the phase. Same-movement vehicles platoon through at `PLATOON_GAP` seconds apart.

---

## Performance

Measured over 60-second simulations, ~0.88 vehicles/second arrival rate:

| Metric | Result |
|--------|--------|
| Avg induced delay | **~0.8s** |
| Max induced delay | **~5.3s** |
| Zero-delay throughput | **~50%** of vehicles |
| Platoon events | **6–11** per run |

*Induced delay = actual box entry time minus what it would have been at free-flow with no other traffic.*

---

## Running It

```bash
pip install -r requirements.txt
python hivemind.py
```

Close the pyplot window to write `hivemind_log.csv` with a per-vehicle event breakdown and summary stats.

### Key Tunables

```python
ARRIVAL_RATE    = 0.22   # vehicles/s/approach — increase to stress-test
SIM_DURATION    = 60.0   # seconds
SPEED_DESIRED   = 12.0   # m/s (~27 mph)
STARVATION_LIMIT = 2.5   # seconds before a waiting movement forces a switch
MAX_HOLDOUT     = 4.0    # hard cap — no movement waits longer than this
READY_DIST      = 40.0   # metres — how close before a lane is "ready"
```

---

## Architecture

```
hivemind.py
├── Geometry          lane entry/exit coordinates, Bezier box paths
├── Conflict model    50 geometrically-verified safe movement pairs
├── HiveMind          max concurrent movement controller
│   ├── update()      ready detection → max_safe_subset → GO set
│   ├── Additive      opportunistically add compatible movements
│   └── Anti-starve   STARVATION_LIMIT + MAX_HOLDOUT enforcement
├── Vehicle           dataclass with approach/box/egress state machine
├── step()            physics tick — kinematics, box traversal, spawning
└── Animation         matplotlib real-time visualization
```

---

## What This Is (And Isn't)

This is a **simulation prototype**, not a production control system. It demonstrates the control concept and validates that maximum-concurrent-movement scheduling outperforms naive phase-based approaches. A real deployment would require:

- V2X communication infrastructure
- Sub-100ms round-trip latency between vehicles and controller
- Sensor fusion for position/speed verification
- Safety-certified conflict model validation
- Graceful degradation for non-connected vehicles

The scheduling algorithm itself — find the max conflict-free set of ready movements and grant them GO simultaneously — is the transferable idea.

---

## Stack

`Python` · `NumPy` · `Matplotlib` · `Combinatorics` · `Control Systems` · `Autonomous Vehicles`
