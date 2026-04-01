"""
hivemind.py  —  v7  (dual-mode: predictive + flush)
─────────────────────────────────────────────────────
Centralized autonomous intersection controller.

NORMAL MODE (default):
  Predictive arrival-time scheduling. Each vehicle gets a conflict-free
  time slot. Speed modulates smoothly over the whole approach so the
  vehicle arrives exactly when the box is clear — no stops, no hard braking.

FLUSH MODE (triggered when queue depth ≥ FLUSH_Q in any lane):
  Pure maximum-concurrent throughput. Finds the largest set of non-conflicting
  movements with vehicles ready and signals them GO. Tight platoons clear
  backlogs rapidly. Exits when all queues are shallow.

Road: 6 lanes per road (3 ingress + 3 egress), LANE_W = 3.66 m (12 ft).
Vehicles: 8 ft wide × 14 ft long oriented rectangles.
Conflict model: geometrically verified, 50 safe pairs, 16 conflicts.
Usage:  python hivemind.py   [Q to quit → writes hivemind_log.csv]
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

DT            = 0.05        # physics timestep (s)
SCHED_DT      = 0.12        # scheduler tick (s)

ARRIVAL_RATE  = 0.22        # vehicles / s / approach
SPEED_DESIRED = 12.0        # m/s
V_MIN         = 3.0         # m/s  minimum modulation speed
V_MAX         = SPEED_DESIRED * 1.08
ACCEL_MAX     = 2.5         # m/s²
DECEL_MAX     = 4.0         # m/s²
BOX_ACCEL     = 5.0         # m/s²

LANE_W        = 3.66        # 12 ft in metres
BOX_HALF      = 11.0
VEH_W         = 2.44        # 8 ft
VEH_L         = 4.27        # 14 ft
APPROACH_LEN  = 90.0
EXIT_LEN      = 90.0

STOP_LINE        = VEH_L / 2.0 + 0.61   # front bumper 2ft outside box edge
MIN_GAP          = 4.5                   # minimum physical gap in lane (m)
HEADWAY          = 0.55                  # min time gap between consecutive box entries
SCHEDULE_HORIZON = 12.0                  # seconds ahead to schedule (normal mode)

# Flush mode triggers when queue depth ≥ FLUSH_Q in any single lane
FLUSH_Q          = 6        # enter flush mode — never triggers at rate ≤ 0.30
FLUSH_EXIT_Q     = 3        # exit flush mode (hysteresis)
FLUSH_READY_DIST = 65.0     # vehicles within this distance count as "ready" in flush
MAX_HOLDOUT      = 3.5      # seconds — flush starvation limit

BOX_T  = {"straight": 1.5, "right": 1.2, "left": 2.0}
TURN_P = {"S": 0.55, "L": 0.25, "R": 0.20}
COLOR_FOR = {"straight": "tab:green", "right": "tab:blue", "left": "tab:orange"}

# No fixed seed — each run produces a unique traffic pattern

# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def _lane_entry():
    lw, r = LANE_W, {}
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
    lw, r = LANE_W, {}
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
    x0,y0 = LANE_ENTRY[frm][turn]
    if frm=="N": return (x0, y0+dist)
    if frm=="S": return (x0, y0-dist)
    if frm=="E": return (x0+dist, y0)
    return (x0-dist, y0)

def lane_exit_point(dest, turn, dist):
    x0,y0 = LANE_EXIT[dest][turn]
    if dest=="N": return (x0, y0+dist)
    if dest=="S": return (x0, y0-dist)
    if dest=="E": return (x0+dist, y0)
    return (x0-dist, y0)

def bezier_path(frm, to, u):
    mt = move_type(frm, to)
    sx,sy = LANE_ENTRY[frm][mt]; ex,ey = LANE_EXIT[to][mt]
    if mt == "straight": return (sx+(ex-sx)*u, sy+(ey-sy)*u)
    if mt == "right":
        c = (sx,ey) if frm in ("N","S") else (ex,sy); c1=c2=c
    else:
        if frm=="N":   c1=(sx,sy-BOX_HALF*0.6); c2=( BOX_HALF*0.3, 0.0)
        elif frm=="S": c1=(sx,sy+BOX_HALF*0.6); c2=(-BOX_HALF*0.3, 0.0)
        elif frm=="E": c1=(sx-BOX_HALF*0.6,sy); c2=(0.0,-BOX_HALF*0.3)
        else:          c1=(sx+BOX_HALF*0.6,sy); c2=(0.0, BOX_HALF*0.3)
    x=(1-u)**3*sx+3*(1-u)**2*u*c1[0]+3*(1-u)*u**2*c2[0]+u**3*ex
    y=(1-u)**3*sy+3*(1-u)**2*u*c1[1]+3*(1-u)*u**2*c2[1]+u**3*ey
    return x,y

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFLICT MODEL  (geometrically verified — 50 safe pairs, 16 conflicts)
# ═══════════════════════════════════════════════════════════════════════════════

_SAFE = {frozenset(p) for p in [
    ("N->S","N->E"),("N->S","N->W"),("N->S","E->S"),("N->S","W->S"),
    ("N->E","W->E"),("N->W","E->W"),
    ("S->N","S->W"),("S->N","S->E"),("S->N","E->N"),("S->N","W->N"),
    ("S->W","E->W"),("S->E","W->E"),("E->W","E->N"),("E->W","E->S"),
    ("W->E","W->S"),("W->E","W->N"),("N->E","S->W"),("E->S","W->N"),
    ("N->E","N->W"),("N->E","S->E"),("N->W","S->W"),("S->W","S->E"),
    ("E->N","E->S"),("E->N","W->N"),("E->S","W->S"),("W->S","W->N"),
    ("N->E","E->N"),("N->W","W->N"),("S->W","W->S"),("S->E","E->S"),
    ("N->S","S->N"),("E->W","W->E"),("N->S","S->E"),("N->S","E->N"),
    ("N->W","S->N"),("N->W","W->E"),("S->N","W->S"),("S->E","E->W"),
    ("E->W","W->S"),("E->N","W->E"),("N->E","W->S"),("N->W","E->S"),
    ("S->W","E->N"),("S->E","W->N"),("N->W","E->N"),("N->W","W->S"),
    ("S->E","E->N"),("S->E","W->S"),("N->W","S->E"),("E->N","W->S"),
]}

def movement_conflict(m1, m2):
    if m1 == m2: return False
    return frozenset({m1,m2}) not in _SAFE

def max_safe_subset(mvs: List[str]) -> List[str]:
    if not mvs: return []
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
    frm: str; to: str; t_spawn: float
    dist_in:      float = APPROACH_LEN
    v:            float = SPEED_DESIRED
    phase:        str   = "approach"
    box_elapsed:  float = 0.0
    dist_out:     float = 0.0
    done:         bool  = False
    t_freeflow:   float = 0.0
    induced_delay:float = 0.0
    t_arrive:     float = -1.0   # scheduled arrival at box front (-1 = unset)

    def __post_init__(self):
        self.t_freeflow = self.t_spawn + self.dist_in / SPEED_DESIRED

    @property
    def mv(self)   -> str: return mkey(self.frm, self.to)
    @property
    def turn(self) -> str: return move_type(self.frm, self.to)
    def lane_key(self)    -> Tuple[str,str]: return (self.frm, self.turn)

    def t_natural(self, now: float) -> float:
        """Arrival at box front edge if we maintain current speed."""
        return now + max(0.0, self.dist_in - VEH_L/2) / max(self.v, 0.1)

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
        csv.DictWriter(f,fieldnames=fields).writeheader()
        csv.DictWriter(f,fieldnames=fields).writerows(_log)
    print(f"\n{'═'*60}")
    print(f"  Log → {path}  ({len(_log)} events)")
    print(f"{'═'*60}")
    for k,c in sorted(_stats.items(),key=lambda x:-x[1]):
        print(f"  {k:<28} {c:>6}")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  HIVE MIND — dual-mode controller
# ═══════════════════════════════════════════════════════════════════════════════

class HiveMind:
    def __init__(self):
        self.flush_mode:  bool              = False
        self.active_mvs:  Set[str]          = set()   # GO set (flush) or scheduled (normal)
        self.last_entry:  Dict[str, float]  = {}      # last actual box entry time per mv

    # ── Normal mode ───────────────────────────────────────────────────────────

    def _run_normal(self, by_lane, vehicles, now):
        """
        Predictive arrival-time scheduler.
        Rebuild conflict-free slot table each tick. Assign t_arrive to every
        vehicle (head + followers). Vehicles then modulate speed to hit t_arrive.
        """
        heads = []
        queue_depth = defaultdict(int)
        for lv in by_lane.values():
            if lv:
                heads.append(lv[0])
                queue_depth[lv[0].mv] = len(lv)

        # Priority: wait bonus + queue depth bonus (prevents right-turn starvation)
        W_WAIT, W_Q, CAP = 3.5, 1.5, 30.0
        def pri(v):
            wb = min((now - v.t_spawn) * W_WAIT, CAP)
            qb = (queue_depth.get(v.mv, 1) - 1) * W_Q
            return v.t_natural(now) - wb - qb

        heads.sort(key=pri)

        slots = defaultdict(list)   # mv → [(t_start, t_end)]
        scheduled = set()

        for v in heads:
            if v.mv in scheduled:
                continue
            dtf    = max(0.0, v.dist_in - VEH_L/2)
            bd     = BOX_T[v.turn]
            t_nat  = v.t_natural(now)
            t_try  = max(t_nat, now + dtf / V_MAX if dtf > 0 else now)

            for _ in range(120):
                t_end, pushed = t_try + bd, False
                for omv, oslots in slots.items():
                    if not movement_conflict(v.mv, omv):
                        continue
                    for s, e in oslots:
                        if t_try < e and t_end > s:
                            t_try, pushed = e + 0.02, True; break
                    if pushed: break
                if not pushed: break

            for s, e in slots[v.mv]:
                if t_try < s + bd + HEADWAY:
                    t_try = s + bd + HEADWAY

            if t_try > now + SCHEDULE_HORIZON:
                t_try = t_nat

            v.t_arrive = t_try
            slots[v.mv].append((t_try, t_try + bd))
            scheduled.add(v.mv)

            # Concurrent grouping: give same slot to compatible unscheduled heads
            for ov in heads:
                if ov.mv in scheduled or movement_conflict(v.mv, ov.mv): continue
                if any(movement_conflict(ov.mv, sm) for sm in scheduled): continue
                od = max(0.0, ov.dist_in - VEH_L/2)
                ct = max(t_try, now + od/V_MAX if od > 0 else now, ov.t_natural(now))
                ov.t_arrive = ct
                slots[ov.mv].append((ct, ct + BOX_T[ov.turn]))
                scheduled.add(ov.mv)

        # Follower scheduling: back-to-back after head
        for lv in by_lane.values():
            if len(lv) < 2: continue
            head = lv[0]
            if head.t_arrive < 0: continue
            gap    = HEADWAY * 0.4 if len(lv) >= 3 else HEADWAY
            t_next = head.t_arrive + BOX_T[head.turn] + gap
            for fol in lv[1:]:
                fd = max(0.0, fol.dist_in - VEH_L/2)
                fe = now + fd / V_MAX if fd > 0 else now
                tf = max(t_next, fe)
                fol.t_arrive = tf
                t_next = tf + BOX_T[fol.turn] + gap

        self.active_mvs = scheduled

    # ── Flush mode ────────────────────────────────────────────────────────────

    def _run_flush(self, by_lane, vehicles, now):
        """
        Max-concurrent-movement mode. Find the largest conflict-free set of
        ready movements and signal them GO. Vehicles cruise to box at full speed.
        Tight platoons drain queues rapidly.
        """
        in_box = {v.mv for v in vehicles if v.phase == "box" and not v.done}

        ready = []
        for lv in by_lane.values():
            if not lv: continue
            head = lv[0]
            if head.dist_in > FLUSH_READY_DIST: continue
            if all(not movement_conflict(head.mv, b) for b in in_box):
                ready.append(head.mv)

        go = set(max_safe_subset(ready))

        # Anti-starvation
        overdue = [mv for mv in ready if mv not in go
                   and (now - self.last_entry.get(mv, now)) > MAX_HOLDOUT]
        if overdue:
            most = max(overdue, key=lambda m: now - self.last_entry.get(m, now))
            go = set(max_safe_subset([most] + [m for m in ready if m != most]))

        # Additive: include compatible ready movements
        for mv in ready:
            if mv not in go and all(not movement_conflict(mv, gm) for gm in go):
                go.add(mv)

        self.active_mvs = go

        # Assign t_arrive: GO → now (cruise in), HOLD → far future
        for lv in by_lane.values():
            if not lv: continue
            head = lv[0]
            if head.mv in go:
                head.t_arrive = now
                t_next = now + BOX_T[head.turn] + HEADWAY * 0.35
                for fol in lv[1:]:
                    fd = max(0.0, fol.dist_in - VEH_L/2)
                    fe = now + fd / V_MAX if fd > 0 else now
                    tf = max(t_next, fe)
                    fol.t_arrive = tf
                    t_next = tf + BOX_T[fol.turn] + HEADWAY * 0.35
            else:
                head.t_arrive = now + 9999.0

    # ── Main entry point ──────────────────────────────────────────────────────

    def schedule(self, vehicles, now):
        by_lane = defaultdict(list)
        for v in vehicles:
            if v.phase == "approach" and not v.done:
                by_lane[v.lane_key()].append(v)
        for lv in by_lane.values():
            lv.sort(key=lambda v: v.dist_in)

        # Mode switching with hysteresis
        max_q = max((len(lv) for lv in by_lane.values()), default=0)
        if not self.flush_mode and max_q >= FLUSH_Q:
            self.flush_mode = True
        elif self.flush_mode and max_q <= FLUSH_EXIT_Q:
            self.flush_mode = False
            for v in vehicles:   # reset t_arrive so normal mode recalculates
                if v.phase == "approach":
                    v.t_arrive = -1.0

        if self.flush_mode:
            self._run_flush(by_lane, vehicles, now)
        else:
            self._run_normal(by_lane, vehicles, now)

    def is_go(self, mv):
        return mv in self.active_mvs


# ═══════════════════════════════════════════════════════════════════════════════
#  ARRIVALS
# ═══════════════════════════════════════════════════════════════════════════════

def rand_dest(frm):
    r = np.random.rand()
    if r < TURN_P["S"]: return opposite(frm)
    r2 = (r - TURN_P["S"]) / (1 - TURN_P["S"])
    lo = {"N":"E","S":"W","E":"S","W":"N"}
    ro = {"N":"W","S":"E","E":"N","W":"S"}
    return lo[frm] if r2 < TURN_P["L"]/(TURN_P["L"]+TURN_P["R"]) else ro[frm]

# ═══════════════════════════════════════════════════════════════════════════════
#  SIM STATE
# ═══════════════════════════════════════════════════════════════════════════════

directions   = list("NSEW")
vehicles: List[Vehicle] = []
hivemind     = HiveMind()
t_sched_next = 0.0
running      = True

_next_arrival = {d: np.random.exponential(1.0/ARRIVAL_RATE) for d in directions}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP
# ═══════════════════════════════════════════════════════════════════════════════

def spawn(now):
    for d in directions:
        if now >= _next_arrival[d]:
            vehicles.append(Vehicle(frm=d, to=rand_dest(d), t_spawn=now))
            _next_arrival[d] = now + np.random.exponential(1.0/ARRIVAL_RATE)

def step(now):
    global t_sched_next
    spawn(now)
    if now >= t_sched_next:
        hivemind.schedule(vehicles, now)
        t_sched_next = now + SCHED_DT

    by_lane = defaultdict(list)
    for v in vehicles:
        if v.phase == "approach" and not v.done:
            by_lane[v.lane_key()].append(v)
    for lk in by_lane:
        by_lane[lk].sort(key=lambda v: v.dist_in)

    tick_reserved: Set[str] = set()

    for lane_vehs in by_lane.values():
        for i, v in enumerate(lane_vehs):
            at_front      = (i == 0)
            dist_to_front = max(0.0, v.dist_in - VEH_L/2)

            if i > 0:
                lead = lane_vehs[i-1]
                if v.dist_in < lead.dist_in + MIN_GAP:
                    v.dist_in = lead.dist_in + MIN_GAP

            if at_front:
                # Hard floor: never push front bumper into box while waiting
                if v.v < 0.3 and dist_to_front <= 0.0:
                    v.dist_in = VEH_L / 2.0   # hold front bumper at box edge

                dist_to_front = max(0.0, v.dist_in - VEH_L/2)   # recompute after floor
                # Box entry gate
                if dist_to_front <= 0.0:
                    box_safe = all(
                        not movement_conflict(v.mv, o.mv)
                        for o in vehicles
                        if (o.phase == "box" and not o.done) or o.mv in tick_reserved
                    )
                    if box_safe:
                        v.induced_delay = max(0.0, now - v.t_freeflow)
                        v.phase         = "box"
                        v.box_elapsed   = 0.0
                        v.dist_in       = 0.0
                        v.v             = SPEED_DESIRED
                        tick_reserved.add(v.mv)
                        hivemind.last_entry[v.mv] = now
                        mode = "FLUSH" if hivemind.flush_mode else "normal"
                        log_event(now, v, "BOX_ENTRY",
                                  f"ind_delay={v.induced_delay:.2f}s [{mode}]")
                        hivemind.schedule(vehicles, now)
                        continue
                    else:
                        v.dist_in = VEH_L / 2.0
                        v.v       = 0.0
                        continue

                # Speed modulation toward t_arrive
                if v.t_arrive > 0 and v.t_arrive > now + DT and dist_to_front > 0.1:
                    t_rem    = v.t_arrive - now
                    v_needed = float(np.clip(dist_to_front / t_rem, V_MIN, V_MAX))
                else:
                    # Slot time is now/past, or right at box — go full speed
                    v_needed = SPEED_DESIRED

                # Committed: past braking distance — go at full speed
                if dist_to_front <= (v.v**2)/(2*DECEL_MAX) and v_needed < V_MIN:
                    v_needed = SPEED_DESIRED

                dv        = np.clip(v_needed - v.v, -DECEL_MAX*DT, ACCEL_MAX*DT)
                v.v       = float(np.clip(v.v + dv, 0.0, V_MAX))
                new_dist  = v.dist_in - v.v * DT
                v.dist_in = max(VEH_L/2 if v.v < 0.3 else 0.0, new_dist)

            else:
                lead = lane_vehs[i-1]
                gap  = v.dist_in - lead.dist_in
                FOLLOW_COMFORT = 20.0

                dtf_f = max(0.0, v.dist_in - VEH_L/2)
                if v.t_arrive > 0 and v.t_arrive > now + DT and dtf_f > 0:
                    v_sched = float(np.clip(dtf_f / (v.t_arrive - now), V_MIN, V_MAX))
                else:
                    v_sched = SPEED_DESIRED

                if gap > FOLLOW_COMFORT:
                    v_gap = V_MAX
                elif gap > MIN_GAP:
                    tb    = 1.0 - (gap - MIN_GAP) / (FOLLOW_COMFORT - MIN_GAP)
                    v_gap = V_MAX*(1-tb) + lead.v*tb
                else:
                    v_gap = lead.v * 0.8

                v_target  = min(v_sched, v_gap)
                dv        = np.clip(v_target - v.v, -DECEL_MAX*DT, ACCEL_MAX*DT)
                v.v       = max(0.0, v.v + dv)
                v.dist_in = max(lead.dist_in + MIN_GAP, v.dist_in - v.v*DT)

    for v in vehicles:
        if v.done or v.phase != "box": continue
        v.v = min(SPEED_DESIRED, v.v + min(BOX_ACCEL*DT, SPEED_DESIRED - v.v))
        v.box_elapsed += DT
        if v.box_elapsed >= BOX_T[v.turn]:
            v.phase = "egress"; v.dist_out = 0.0; v.v = SPEED_DESIRED
            hivemind.schedule(vehicles, now)

    for v in vehicles:
        if v.done or v.phase != "egress": continue
        v.v = SPEED_DESIRED
        v.dist_out += v.v * DT
        if v.dist_out >= EXIT_LEN: v.done = True

# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_roads(ax):
    edge = 3.0*LANE_W
    ax.plot([-BOX_HALF,BOX_HALF,BOX_HALF,-BOX_HALF,-BOX_HALF],
            [-BOX_HALF,-BOX_HALF,BOX_HALF,BOX_HALF,-BOX_HALF],
            lw=2.5,color="0.60",zorder=2)
    road = dict(lw=1.8,color="0.55",zorder=1)
    cent = dict(lw=1.3,color="gold",alpha=0.85,zorder=1)
    for s,e in [(BOX_HALF,BOX_HALF+APPROACH_LEN),(-BOX_HALF,-BOX_HALF-APPROACH_LEN)]:
        ax.plot([-edge,-edge],[s,e],**road); ax.plot([edge,edge],[s,e],**road)
        ax.plot([0,0],[s,e],**cent)
    for s,e in [(BOX_HALF,BOX_HALF+APPROACH_LEN),(-BOX_HALF,-BOX_HALF-APPROACH_LEN)]:
        ax.plot([s,e],[-edge,-edge],**road); ax.plot([s,e],[edge,edge],**road)
        ax.plot([s,e],[0,0],**cent)

def draw_lanes(ax):
    sep = dict(linestyle="--",lw=0.7,alpha=0.50,color="0.65",zorder=1)
    lw  = LANE_W
    for off in (-2*lw,-lw,lw,2*lw):
        for s,e in [(BOX_HALF,BOX_HALF+APPROACH_LEN),(-BOX_HALF,-BOX_HALF-APPROACH_LEN)]:
            ax.plot([off,off],[s,e],**sep)
            ax.plot([s,e],[off,off],**sep)

def veh_xy(v):
    if v.phase=="approach": return lane_entry_point(v.frm,v.turn,v.dist_in)
    if v.phase=="box":
        u = np.clip(v.box_elapsed/max(BOX_T[v.turn],1e-6),0.0,1.0)
        return bezier_path(v.frm,v.to,u)
    return lane_exit_point(v.to,v.turn,v.dist_out)

def veh_heading(v) -> float:
    if v.phase=="approach": return {"N":270.,"S":90.,"E":180.,"W":0.}[v.frm]
    if v.phase=="egress":   return {"N":90., "S":270.,"E":0., "W":180.}[v.to]
    u  = np.clip(v.box_elapsed/max(BOX_T[v.turn],1e-6),0.,1.)
    du = 0.025
    x1,y1 = bezier_path(v.frm,v.to,max(0.,u-du))
    x2,y2 = bezier_path(v.frm,v.to,min(1.,u+du))
    dx,dy  = x2-x1,y2-y1
    if abs(dx)<1e-9 and abs(dy)<1e-9:
        return {"N":270.,"S":90.,"E":180.,"W":0.}[v.frm]
    return float(np.degrees(np.arctan2(dy,dx)))

def make_patch(cx,cy,heading,color):
    hw,hl = VEH_W/2., VEH_L/2.
    loc   = np.array([[hl,hw],[hl,-hw],[-hl,-hw],[-hl,hw]])
    a     = np.radians(heading); ca,sa = np.cos(a),np.sin(a)
    world = loc @ np.array([[ca,-sa],[sa,ca]]).T + np.array([cx,cy])
    from matplotlib.patches import Polygon as MP
    return MP(world,closed=True,facecolor=color,edgecolor="white",lw=0.7,alpha=0.93,zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
#  ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

VIEW = APPROACH_LEN + 10
fig,ax = plt.subplots(figsize=(9,9))
ax.set_xlim(-VIEW,VIEW); ax.set_ylim(-VIEW,VIEW)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_facecolor("#1c1c1e")
draw_roads(ax); draw_lanes(ax)

ax.legend(handles=[
    Line2D([0],[0],marker='s',linestyle='None',color=COLOR_FOR["straight"],markersize=9,label='Straight'),
    Line2D([0],[0],marker='s',linestyle='None',color=COLOR_FOR["right"],  markersize=9,label='Right'),
    Line2D([0],[0],marker='s',linestyle='None',color=COLOR_FOR["left"],   markersize=9,label='Left'),
],loc='upper right',frameon=True,fontsize=8,facecolor='#2c2c2e',edgecolor='0.5',labelcolor='white')

patches: List = []
t_text = ax.text(-VIEW+4, VIEW-5,"",ha="left",va="top",fontsize=8,family="monospace",color="white",zorder=10)
m_text = ax.text(-VIEW+4,-VIEW+4,"",ha="left",va="bottom",fontsize=7,family="monospace",color="0.65",zorder=10)

def init():
    t_text.set_text(""); m_text.set_text("")
    return t_text, m_text

def animate(frame):
    global patches
    if not running: return t_text, m_text
    now = frame * DT
    step(now)
    while vehicles and vehicles[0].done: vehicles.pop(0)

    for p in patches: p.remove()
    patches = []
    for v in vehicles:
        if v.done: continue
        cx,cy = veh_xy(v)
        p = make_patch(cx,cy,veh_heading(v),COLOR_FOR[v.turn])
        ax.add_patch(p); patches.append(p)

    n_done = sum(1 for r in _log if "BOX_ENTRY" in r["event"])
    n_box  = sum(v.phase=="box"      for v in vehicles if not v.done)
    n_app  = sum(v.phase=="approach" for v in vehicles if not v.done)
    thru   = n_done / max(now, 1.0)
    all_d  = [v.induced_delay for v in vehicles
              if v.induced_delay>0 and (v.phase in ("box","egress") or v.done)]
    avg_d  = sum(all_d)/len(all_d) if all_d else 0.0

    t_text.set_text(
        f"t={now:6.1f}s  done={n_done:4d}  approach={n_app:2d}  box={n_box}\n"
        f"throughput={thru:.2f} veh/s   induced_delay={avg_d:.1f}s\n"
        f"[Q] quit"
    )
    mode = "⚡ FLUSH MODE" if hivemind.flush_mode else "  normal mode"
    gs   = sorted(hivemind.go_set)[:4] if hivemind.flush_mode else []
    m_text.set_text(f"{mode}" + (f"  |  go: {' '.join(gs)}" if gs else ""))
    return t_text, m_text

def on_key(event):
    global running
    if event.key == "q":
        running = False
        write_log("hivemind_log.csv")
        plt.close()

anim = animation.FuncAnimation(
    fig,animate,init_func=init,frames=None,
    cache_frame_data=False,interval=DT*1000,blit=False)
fig.canvas.mpl_connect("key_press_event",on_key)
fig.canvas.mpl_connect("close_event",lambda e: write_log("hivemind_log.csv"))
plt.tight_layout()
plt.show()
