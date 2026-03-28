"""
hivemind.py  —  v5  (maximum concurrent movement)
───────────────────────────────────────────────────
Centralized hive-mind intersection controller.

Core principle
──────────────
At every moment, the hive mind grants GO to the LARGEST set of
movements that are simultaneously safe — no scoring, no phases,
no weights. Just: who's ready + what's the max conflict-free set?

  1. Every SCHED_DT, identify "ready" movements:
       any movement where the head-of-lane vehicle is within
       READY_DIST metres of the box AND no conflicting vehicle
       is currently in the box.

  2. Find the maximum conflict-free subset of ready movements
       using exhaustive search over combinations (N≤12, fast enough).

  3. Grant GO to that subset. Everything else holds.

  4. A movement stays GO until either:
       • its head vehicle clears the box, or
       • a newly arriving conflicting movement creates an unsafe situation
         (re-evaluate immediately when box occupancy changes)

Vehicle physics
───────────────
GO  → cruise at SPEED_DESIRED, enter box when dist_in ≤ 0
HOLD→ decelerate smoothly to stop at STOP_LINE metres from box
BOX → accelerate to SPEED_DESIRED, traverse, exit at SPEED_DESIRED
EGRESS → constant SPEED_DESIRED, stay in assigned lane

Car following enforces minimum physical gap between vehicles
in the same lane — no stacking.

Road: 6 lanes per road (3 ingress + 3 egress), centered at axis=0.
Conflict model: geometrically verified, 50 safe pairs, 16 conflicts.

Usage:  python hivemind.py   (close window → writes hivemind_log.csv)
Dependencies: numpy, matplotlib
Author: semantical-monster
"""

import csv, random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLES
# ═══════════════════════════════════════════════════════════════════════════════

SIM_DURATION  = 60.0
DT            = 0.05        # physics timestep (s)
SCHED_DT      = 0.15        # hive mind re-evaluates every N seconds

ARRIVAL_RATE  = 0.22        # vehicles/s/approach
SPEED_DESIRED = 12.0        # m/s
ACCEL_MAX     = 3.0         # m/s²
DECEL_MAX     = 5.0         # m/s²
BOX_ACCEL     = 5.0         # m/s² — recover speed quickly inside box

BOX_HALF      = 8.0
LANE_W        = 2.5
APPROACH_LEN  = 90.0
EXIT_LEN      = 90.0

STOP_LINE     = 1.0         # metres from box edge to stop at when holding
READY_DIST    = 40.0        # head-of-lane within this distance → movement is "ready"
MIN_GAP       = 5.0         # minimum physical gap between vehicles in same lane
PLATOON_GAP   = 0.5         # seconds of box time before same-movement follower enters
STARVATION_LIMIT = 2.5      # seconds a movement waits before forcing a phase switch
MAX_HOLDOUT      = 4.0      # absolute max seconds any movement can be held out of go_set

BOX_T = {"straight": 1.5, "right": 1.2, "left": 2.0}
TURN_P = {"S": 0.55, "L": 0.25, "R": 0.20}
COLOR_FOR = {"straight": "tab:green", "right": "tab:blue", "left": "tab:orange"}

random.seed(7); np.random.seed(7)

# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def _lane_entry():
    lw = LANE_W
    r = {}
    for d in "NSEW":
        r[d] = {}
        if d == "N":
            r[d]["right"]    = (-2.5*lw,  BOX_HALF)
            r[d]["straight"] = (-1.5*lw,  BOX_HALF)
            r[d]["left"]     = (-0.5*lw,  BOX_HALF)
        elif d == "S":
            r[d]["right"]    = ( 2.5*lw, -BOX_HALF)
            r[d]["straight"] = ( 1.5*lw, -BOX_HALF)
            r[d]["left"]     = ( 0.5*lw, -BOX_HALF)
        elif d == "E":
            r[d]["right"]    = ( BOX_HALF,  2.5*lw)
            r[d]["straight"] = ( BOX_HALF,  1.5*lw)
            r[d]["left"]     = ( BOX_HALF,  0.5*lw)
        else:
            r[d]["right"]    = (-BOX_HALF, -2.5*lw)
            r[d]["straight"] = (-BOX_HALF, -1.5*lw)
            r[d]["left"]     = (-BOX_HALF, -0.5*lw)
    return r

def _lane_exit():
    lw = LANE_W
    r = {}
    for dest in "NSEW":
        r[dest] = {}
        if dest == "N":
            r[dest]["left"]     = ( 0.5*lw,  BOX_HALF)
            r[dest]["straight"] = ( 1.5*lw,  BOX_HALF)
            r[dest]["right"]    = ( 2.5*lw,  BOX_HALF)
        elif dest == "S":
            r[dest]["left"]     = (-0.5*lw, -BOX_HALF)
            r[dest]["straight"] = (-1.5*lw, -BOX_HALF)
            r[dest]["right"]    = (-2.5*lw, -BOX_HALF)
        elif dest == "E":
            r[dest]["left"]     = ( BOX_HALF, -0.5*lw)
            r[dest]["straight"] = ( BOX_HALF, -1.5*lw)
            r[dest]["right"]    = ( BOX_HALF, -2.5*lw)
        else:
            r[dest]["left"]     = (-BOX_HALF,  0.5*lw)
            r[dest]["straight"] = (-BOX_HALF,  1.5*lw)
            r[dest]["right"]    = (-BOX_HALF,  2.5*lw)
    return r

LANE_ENTRY = _lane_entry()
LANE_EXIT  = _lane_exit()

def opposite(d): return {"N":"S","S":"N","E":"W","W":"E"}[d]

def move_type(frm, to):
    if to == opposite(frm): return "straight"
    return "right" if to == {"N":"W","S":"E","E":"N","W":"S"}[frm] else "left"

def mkey(f, t): return f"{f}->{t}"

def lane_entry_point(frm, turn, dist):
    x0, y0 = LANE_ENTRY[frm][turn]
    if frm=="N": return (x0, y0+dist)
    if frm=="S": return (x0, y0-dist)
    if frm=="E": return (x0+dist, y0)
    return (x0-dist, y0)

def lane_exit_point(dest, turn, dist):
    x0, y0 = LANE_EXIT[dest][turn]
    if dest=="N": return (x0, y0+dist)
    if dest=="S": return (x0, y0-dist)
    if dest=="E": return (x0+dist, y0)
    return (x0-dist, y0)

def bezier_path(frm, to, u):
    mt = move_type(frm, to)
    sx, sy = LANE_ENTRY[frm][mt]
    ex, ey = LANE_EXIT[to][mt]
    if mt == "straight":
        return (sx+(ex-sx)*u, sy+(ey-sy)*u)
    if mt == "right":
        c = (sx,ey) if frm in ("N","S") else (ex,sy)
        c1 = c2 = c
    else:
        if frm=="N":   c1=(sx,sy-BOX_HALF*0.6); c2=(BOX_HALF*0.3, 0.0)
        elif frm=="S": c1=(sx,sy+BOX_HALF*0.6); c2=(-BOX_HALF*0.3,0.0)
        elif frm=="E": c1=(sx-BOX_HALF*0.6,sy); c2=(0.0,-BOX_HALF*0.3)
        else:          c1=(sx+BOX_HALF*0.6,sy); c2=(0.0, BOX_HALF*0.3)
    x = (1-u)**3*sx+3*(1-u)**2*u*c1[0]+3*(1-u)*u**2*c2[0]+u**3*ex
    y = (1-u)**3*sy+3*(1-u)**2*u*c1[1]+3*(1-u)*u**2*c2[1]+u**3*ey
    return x, y

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFLICT MODEL  (geometrically verified)
# ═══════════════════════════════════════════════════════════════════════════════

_SAFE_PAIRS = {frozenset(p) for p in [
    ("N->S","N->E"),("N->S","N->W"),("N->S","E->S"),("N->S","W->S"),
    ("N->E","W->E"),("N->W","E->W"),
    ("S->N","S->W"),("S->N","S->E"),("S->N","E->N"),("S->N","W->N"),
    ("S->W","E->W"),("S->E","W->E"),
    ("E->W","E->N"),("E->W","E->S"),
    ("W->E","W->S"),("W->E","W->N"),
    ("N->E","S->W"),("E->S","W->N"),
    ("N->E","N->W"),("N->E","S->E"),
    ("N->W","S->W"),("S->W","S->E"),
    ("E->N","E->S"),("E->N","W->N"),
    ("E->S","W->S"),("W->S","W->N"),
    ("N->E","E->N"),("N->W","W->N"),
    ("S->W","W->S"),("S->E","E->S"),
    ("N->S","S->N"),("E->W","W->E"),
    ("N->S","S->E"),("N->S","E->N"),
    ("N->W","S->N"),("N->W","W->E"),
    ("S->N","W->S"),("S->E","E->W"),
    ("E->W","W->S"),("E->N","W->E"),
    ("N->E","W->S"),("N->W","E->S"),
    ("S->W","E->N"),("S->E","W->N"),
    ("N->W","E->N"),("N->W","W->S"),
    ("S->E","E->N"),("S->E","W->S"),
    ("N->W","S->E"),("E->N","W->S"),
]}

def movement_conflict(m1, m2):
    if m1 == m2: return False
    return frozenset({m1,m2}) not in _SAFE_PAIRS

def max_safe_subset(mvs: List[str]) -> List[str]:
    """
    Find the largest subset of movements that are all mutually non-conflicting.
    Exhaustive search — N≤12, 2^12=4096 max, very fast in practice.
    Returns the first (lexicographically stable) maximum safe subset found.
    """
    if not mvs:
        return []
    for size in range(len(mvs), 0, -1):
        for combo in combinations(mvs, size):
            if all(not movement_conflict(a,b) for a,b in combinations(combo,2)):
                return list(combo)
    return []

# ═══════════════════════════════════════════════════════════════════════════════
#  VEHICLE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Vehicle:
    frm:     str
    to:      str
    t_spawn: float
    dist_in: float = APPROACH_LEN
    v:       float = SPEED_DESIRED
    phase:   str   = "approach"
    box_elapsed: float = 0.0
    dist_out:    float = 0.0
    done:        bool  = False
    t_freeflow:  float = 0.0
    induced_delay: float = 0.0

    def __post_init__(self):
        self.t_freeflow = self.t_spawn + self.dist_in / SPEED_DESIRED

    @property
    def mv(self)   -> str:  return mkey(self.frm, self.to)
    @property
    def turn(self) -> str:  return move_type(self.frm, self.to)
    def lane_key(self) -> Tuple[str,str]: return (self.frm, self.turn)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

_log:   List[dict] = []
_stats: dict       = defaultdict(int)

def log_event(now, v, event, detail=""):
    _log.append({"t":round(now,3),"mv":v.mv,"lane":str(v.lane_key()),
                 "dist_in":round(v.dist_in,2),"v_ms":round(v.v,2),
                 "ind_dly":round(v.induced_delay,3),"event":event,"detail":detail})
    _stats[event] += 1

def write_log(path="hivemind_log.csv"):
    if not _log: print("No events."); return
    fields = ["t","mv","lane","dist_in","v_ms","ind_dly","event","detail"]
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(_log)
    print(f"\n{'═'*60}")
    print(f"  Log → {path}  ({len(_log)} events)")
    print(f"{'═'*60}")
    for k,c in sorted(_stats.items(),key=lambda x:-x[1]):
        print(f"  {k:<25} {c:>6}")
    print(f"{'═'*60}\n")

# ═══════════════════════════════════════════════════════════════════════════════
#  HIVE MIND — maximum concurrent movement
# ═══════════════════════════════════════════════════════════════════════════════

class HiveMind:
    """
    Maximum concurrent movement with queue exhaustion + fairness.

    The key insight: once a movement is granted GO, it should stay GO
    until its queue is drained — not flip-flop with a competing movement
    that could have waited one more cycle.

    Algorithm each tick:
      1. Find all "ready" movements (head of lane within READY_DIST, box safe).
      2. Among ready movements, find the max conflict-free subset.
      3. If any movement in the NEW best subset is NOT in the current go_set,
         only switch if:
           a) The current go_set has no more queued vehicles ready to go, OR
           b) A competing movement has been waiting > STARVATION_LIMIT seconds
         This allows platooning — same-direction vehicles flow through
         back-to-back before yielding to the opposing stream.
    """
    def __init__(self):
        self.go_set:      Set[str]          = set()
        self.go_since:    Dict[str, float]  = {}
        self.last_go:     Dict[str, float]  = {}  # last time each mv was in go_set

    def update(self, vehicles: List[Vehicle], now: float = 0.0):
        by_lane: Dict[Tuple, List[Vehicle]] = defaultdict(list)
        for v in vehicles:
            if v.phase == "approach" and not v.done:
                by_lane[v.lane_key()].append(v)

        # Head of each lane (closest to box)
        lane_heads: Dict[str, Vehicle] = {}
        for lane_vehs in by_lane.values():
            head = min(lane_vehs, key=lambda v: v.dist_in)
            # Use the movement key — if two lanes share mv, keep closest
            if head.mv not in lane_heads or head.dist_in < lane_heads[head.mv].dist_in:
                lane_heads[head.mv] = head

        in_box: Set[str] = {v.mv for v in vehicles if v.phase == "box" and not v.done}

        # Determine which movements are "ready"
        ready: List[str] = []
        for mv, head in lane_heads.items():
            if head.dist_in > READY_DIST:
                continue
            safe = True
            for box_mv in in_box:
                if box_mv == mv:
                    box_v = next((v for v in vehicles
                                  if v.phase == "box" and v.mv == mv and not v.done), None)
                    if box_v and box_v.box_elapsed < PLATOON_GAP:
                        safe = False; break
                elif movement_conflict(mv, box_mv):
                    safe = False; break
            if safe:
                ready.append(mv)

        # Update last_go for movements currently in go_set
        for mv in self.go_set:
            self.last_go[mv] = now

        if not ready:
            self.go_set = set()
            return

        # Find max conflict-free subset of ready movements
        best = set(max_safe_subset(ready))

        # Queue depth per movement (how many vehicles are lined up)
        mv_queue_depth: Dict[str, int] = defaultdict(int)
        for lk, lvs in by_lane.items():
            mv = lvs[0].mv if lvs else None
            if mv:
                mv_queue_depth[mv] = len(lvs)

        # Check if current go_set still has work to do —
        # includes vehicles currently in box (they count as "still serving")
        # AND close followers about to enter (platoon)
        current_still_ready = self.go_set & set(ready)
        current_in_box      = self.go_set & in_box
        current_has_follower = any(
            mv_queue_depth.get(mv, 0) >= 2 and
            any(lvs[1].dist_in < READY_DIST
                for lk, lvs in by_lane.items()
                if len(lvs) >= 2 and lvs[0].mv == mv)
            for mv in self.go_set
        )
        current_has_queue = (
            bool(current_still_ready) or
            bool(current_in_box) or
            current_has_follower
        )

        # Hard holdout cap: any ready movement that hasn't had GO in MAX_HOLDOUT
        # seconds gets absolute priority
        overdue = [
            mv for mv in ready
            if mv not in self.go_set
            and (now - self.last_go.get(mv, now)) > MAX_HOLDOUT
        ]
        if overdue:
            most_overdue = max(overdue, key=lambda mv: now - self.last_go.get(mv, now))
            forced = set(max_safe_subset([most_overdue] + [m for m in ready if m != most_overdue]))
            self.go_set = forced
            for mv in forced:
                self.go_since[mv] = now

        # Starvation: any ready movement not seen GO in STARVATION_LIMIT seconds
        elif any(
            mv not in self.go_set
            and mv in best
            and (now - self.last_go.get(mv, now)) > STARVATION_LIMIT
            for mv in ready
        ):
            self.go_set = best
            for mv in best:
                self.go_since[mv] = now

        elif not current_has_queue or not (self.go_set & best):
            self.go_set = best
            for mv in best:
                self.go_since[mv] = now

        # Additive: include any ready movement compatible with entire go_set
        current_go = set(self.go_set)
        for mv in ready:
            if mv not in current_go:
                if all(not movement_conflict(mv, existing) for existing in current_go):
                    current_go.add(mv)
        self.go_set = current_go

    def is_go(self, mv: str) -> bool:
        return mv in self.go_set

# ═══════════════════════════════════════════════════════════════════════════════
#  ARRIVALS
# ═══════════════════════════════════════════════════════════════════════════════

def poisson_times(rate, T):
    t, out = 0.0, []
    while True:
        t += np.random.exponential(1.0/rate)
        if t > T: break
        out.append(t)
    return out

def rand_dest(frm):
    r = np.random.rand()
    if r < TURN_P["S"]: return opposite(frm)
    r2 = (r-TURN_P["S"])/(1-TURN_P["S"])
    lo = {"N":"E","S":"W","E":"S","W":"N"}
    ro = {"N":"W","S":"E","E":"N","W":"S"}
    return lo[frm] if r2 < TURN_P["L"]/(TURN_P["L"]+TURN_P["R"]) else ro[frm]

# ═══════════════════════════════════════════════════════════════════════════════
#  SIM STATE
# ═══════════════════════════════════════════════════════════════════════════════

directions   = list("NSEW")
arrivals     = {d: poisson_times(ARRIVAL_RATE, SIM_DURATION) for d in directions}
vehicles: List[Vehicle] = []
hivemind     = HiveMind()
t_sched_next = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP
# ═══════════════════════════════════════════════════════════════════════════════

def spawn(now):
    for d in directions:
        while arrivals[d] and arrivals[d][0] <= now:
            arrivals[d].pop(0)
            vehicles.append(Vehicle(frm=d, to=rand_dest(d), t_spawn=now))

def step(now):
    global t_sched_next
    spawn(now)

    if now >= t_sched_next:
        hivemind.update(vehicles, now)
        t_sched_next = now + SCHED_DT

    # ── Per-lane sorted queues ────────────────────────────────────────────────
    by_lane: Dict[Tuple,List[Vehicle]] = defaultdict(list)
    for v in vehicles:
        if v.phase == "approach" and not v.done:
            by_lane[v.lane_key()].append(v)
    for lk in by_lane:
        by_lane[lk].sort(key=lambda v: v.dist_in)

    # ── Approach kinematics ───────────────────────────────────────────────────
    for lane_vehs in by_lane.values():
        for i, v in enumerate(lane_vehs):
            at_front = (i == 0)
            go       = hivemind.is_go(v.mv)

            # Physical minimum gap enforcement — no stacking
            if i > 0:
                lead      = lane_vehs[i-1]
                max_dist  = lead.dist_in + MIN_GAP   # can't be closer than MIN_GAP
                if v.dist_in < max_dist:
                    v.dist_in = max_dist

            # Speed target
            if at_front and go:
                # ── GO — head of lane ─────────────────────────────────────────
                if v.dist_in <= 0.5:
                    v.induced_delay = max(0.0, now - v.t_freeflow)
                    v.phase         = "box"
                    v.box_elapsed   = 0.0
                    v.dist_in       = 0.0
                    v.v             = SPEED_DESIRED
                    log_event(now, v, "BOX_ENTRY",
                              f"ind_delay={v.induced_delay:.2f}s")
                    hivemind.update(vehicles, now)
                    continue
                v_target = SPEED_DESIRED

            elif at_front and not go:
                # ── HOLD — head of lane ───────────────────────────────────────
                # Cruise at full speed until braking is physically necessary.
                # Only start braking when we MUST to stop by STOP_LINE.
                # Required braking distance: d = v² / (2·DECEL_MAX)
                brake_dist = (v.v ** 2) / (2.0 * DECEL_MAX)
                remaining  = v.dist_in - STOP_LINE

                if remaining <= brake_dist:
                    # Inside braking zone — decelerate to stop
                    v_target = (2.0 * DECEL_MAX * max(0.0, remaining)) ** 0.5
                    v_target = min(v_target, v.v)  # never accelerate in HOLD
                else:
                    # Outside braking zone — cruise at full speed
                    v_target = SPEED_DESIRED

                # Failsafe: if stopped at stop line and box is physically clear,
                # enter regardless of go_set (prevents permanent stall)
                if v.dist_in <= STOP_LINE + 0.5 and v.v < 0.5:
                    box_clear = all(
                        not movement_conflict(v.mv, o.mv)
                        for o in vehicles if o.phase == "box" and not o.done
                    )
                    if box_clear:
                        v.induced_delay = max(0.0, now - v.t_freeflow)
                        v.phase         = "box"
                        v.box_elapsed   = 0.0
                        v.dist_in       = 0.0
                        v.v             = SPEED_DESIRED
                        log_event(now, v, "BOX_ENTRY_FAILSAFE",
                                  f"ind_delay={v.induced_delay:.2f}s")
                        hivemind.update(vehicles, now)
                        continue

            else:
                # ── Not head of lane — follow leader ─────────────────────────
                # Only react to the leader when within FOLLOW_COMFORT metres.
                # Beyond that, cruise at SPEED_DESIRED independently.
                # This prevents a stopped head vehicle from cascading slow
                # speeds back through the entire queue far from the box.
                lead          = lane_vehs[i-1]
                gap           = v.dist_in - lead.dist_in
                FOLLOW_COMFORT = 18.0   # metres — start reacting to leader

                if gap > FOLLOW_COMFORT:
                    # Far enough back — cruise freely at speed limit
                    v_target = SPEED_DESIRED
                else:
                    # Within following zone — smoothly match leader speed
                    # t=0 at FOLLOW_COMFORT (full speed), t=1 at MIN_GAP (full match)
                    t = 1.0 - max(0.0, (gap - MIN_GAP) / (FOLLOW_COMFORT - MIN_GAP))
                    v_target = SPEED_DESIRED * (1.0 - t) + lead.v * t
                    v_target = max(0.0, v_target)

            # Apply acceleration/deceleration limits
            dv   = np.clip(v_target - v.v, -DECEL_MAX*DT, ACCEL_MAX*DT)
            v.v  = max(0.0, v.v + dv)
            v.dist_in = max(
                0.0 if at_front else (lane_vehs[i-1].dist_in + MIN_GAP),
                v.dist_in - v.v*DT
            )

    # ── Box traversal ─────────────────────────────────────────────────────────
    for v in vehicles:
        if v.done or v.phase != "box": continue
        dv  = min(BOX_ACCEL*DT, SPEED_DESIRED - v.v)
        v.v = min(SPEED_DESIRED, v.v + dv)
        v.box_elapsed += DT
        if v.box_elapsed >= BOX_T[v.turn]:
            v.phase    = "egress"
            v.dist_out = 0.0
            v.v        = SPEED_DESIRED
            # Re-evaluate as box just freed up
            hivemind.update(vehicles, now)

    # ── Egress ────────────────────────────────────────────────────────────────
    for v in vehicles:
        if v.done or v.phase != "egress": continue
        v.v        = SPEED_DESIRED
        v.dist_out += v.v * DT
        if v.dist_out >= EXIT_LEN:
            v.done = True

# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_roads(ax):
    ax.plot([-BOX_HALF,BOX_HALF,BOX_HALF,-BOX_HALF,-BOX_HALF],
            [-BOX_HALF,-BOX_HALF,BOX_HALF,BOX_HALF,-BOX_HALF],
            lw=2, color="0.25")
    lw, edge = LANE_W, 3.0*LANE_W
    for s,e in [(BOX_HALF,BOX_HALF+APPROACH_LEN),(-BOX_HALF,-BOX_HALF-APPROACH_LEN)]:
        ax.plot([-edge,-edge],[s,e],lw=1.2,color="0.35")
        ax.plot([edge,edge],[s,e],lw=1.2,color="0.35")
        ax.plot([0,0],[s,e],lw=1.0,color="gold",alpha=0.8)
    for s,e in [(BOX_HALF,BOX_HALF+APPROACH_LEN),(-BOX_HALF,-BOX_HALF-APPROACH_LEN)]:
        ax.plot([s,e],[-edge,-edge],lw=1.2,color="0.35")
        ax.plot([s,e],[edge,edge],lw=1.2,color="0.35")
        ax.plot([s,e],[0,0],lw=1.0,color="gold",alpha=0.8)

def draw_lanes(ax):
    si = dict(linestyle="--",lw=0.5,alpha=0.3,color="gray")
    so = dict(linestyle="--",lw=0.5,alpha=0.4,color="steelblue")
    for d in "NSEW":
        for t in ("right","straight","left"):
            x0,y0 = LANE_ENTRY[d][t]
            if d=="N":   ax.plot([x0,x0],[y0,y0+APPROACH_LEN],**si)
            elif d=="S": ax.plot([x0,x0],[y0,y0-APPROACH_LEN],**si)
            elif d=="E": ax.plot([x0,x0+APPROACH_LEN],[y0,y0],**si)
            else:        ax.plot([x0,x0-APPROACH_LEN],[y0,y0],**si)
    for dest in "NSEW":
        for t in ("right","straight","left"):
            x0,y0 = LANE_EXIT[dest][t]
            if dest=="N":   ax.plot([x0,x0],[y0,y0+EXIT_LEN],**so)
            elif dest=="S": ax.plot([x0,x0],[y0,y0-EXIT_LEN],**so)
            elif dest=="E": ax.plot([x0,x0+EXIT_LEN],[y0,y0],**so)
            else:           ax.plot([x0,x0-EXIT_LEN],[y0,y0],**so)

def veh_xy(v):
    if v.phase=="approach": return lane_entry_point(v.frm,v.turn,v.dist_in)
    if v.phase=="box":
        u = np.clip(v.box_elapsed/max(BOX_T[v.turn],1e-6),0.0,1.0)
        return bezier_path(v.frm,v.to,u)
    return lane_exit_point(v.to,v.turn,v.dist_out)

# ═══════════════════════════════════════════════════════════════════════════════
#  ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

VIEW = APPROACH_LEN + 8
fig,ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-VIEW,VIEW); ax.set_ylim(-VIEW,VIEW)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_facecolor("#f7f7f5")
draw_roads(ax); draw_lanes(ax)
ax.legend(handles=[
    Line2D([0],[0],marker='o',linestyle='None',color=COLOR_FOR["straight"],label='Straight'),
    Line2D([0],[0],marker='o',linestyle='None',color=COLOR_FOR["right"],   label='Right'),
    Line2D([0],[0],marker='o',linestyle='None',color=COLOR_FOR["left"],    label='Left'),
],loc='upper right',frameon=True,fontsize=8)

scat   = ax.scatter([],[],s=30,zorder=5)
t_text = ax.text(-VIEW+3,VIEW-3,"",ha="left",va="top",fontsize=8,family="monospace")
go_txt = ax.text(-VIEW+3,-VIEW+3,"",ha="left",va="bottom",fontsize=7,
                 color="0.4",family="monospace")

def init():
    scat.set_offsets(np.empty((0,2))); scat.set_facecolors([])
    t_text.set_text(""); go_txt.set_text("")
    return scat,t_text,go_txt

def animate(frame):
    now    = frame*DT
    step(now)
    active = [v for v in vehicles if not v.done]

    if active:
        scat.set_offsets(np.array([veh_xy(v) for v in active]))
        scat.set_facecolors([COLOR_FOR[v.turn] for v in active])
    else:
        scat.set_offsets(np.empty((0,2))); scat.set_facecolors([])

    n_done = sum(v.done for v in vehicles)
    n_box  = sum(v.phase=="box"      for v in vehicles if not v.done)
    n_app  = sum(v.phase=="approach" for v in vehicles if not v.done)
    thru   = n_done/max(now,1.0)

    delayed  = [v.induced_delay for v in vehicles
                if v.induced_delay>0 and (v.phase in ("box","egress") or v.done)]
    late_app = [now-v.t_freeflow for v in vehicles
                if v.phase=="approach" and not v.done and now>v.t_freeflow]
    all_d    = delayed+late_app
    avg_d    = sum(all_d)/len(all_d) if all_d else 0.0

    t_text.set_text(
        f"t={now:5.1f}s  done={n_done:3d}  approach={n_app:2d}  box={n_box}\n"
        f"throughput={thru:.2f} veh/s   induced_delay={avg_d:.1f}s"
    )
    go_txt.set_text(f"go: {' | '.join(sorted(hivemind.go_set)[:4])}"
                    + (" ..." if len(hivemind.go_set)>4 else ""))
    return scat,t_text,go_txt

anim = animation.FuncAnimation(
    fig,animate,init_func=init,
    frames=int(SIM_DURATION/DT),
    interval=DT*1000,blit=True,
)
fig.canvas.mpl_connect("close_event",lambda e: write_log("hivemind_log.csv"))
plt.tight_layout()
plt.show()
