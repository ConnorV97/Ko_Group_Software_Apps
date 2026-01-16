# drift_tracker.py
import os
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

# Define Dx, Dy tuple

DxDy = Tuple[float, float]


# Get Creation time of file for velocity calculation
def _mtime(path: str) -> float:
    return os.path.getmtime(path)

# calculate median of drift values
def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if (len(s) % 2 == 1) else 0.5 * (s[m - 1] + s[m])

@dataclass

# Create DriftKeyFrameTracker Object

class DriftKeyframeTracker:


    calculate_translation: Callable[[str, str], DxDy]

    # keyframe control knobs
    anchor_every_n: int = 3          # hard safety cadence
    vel_window: int = 10             # median window for velocity
    dmax_px: float = 12.0            # max allowed drift (px) between anchors for "suggested K"

    # state containers
    ref_path: Optional[str] = None
    prev_path: Optional[str] = None
    prev_time: Optional[float] = None
    idx: int = 0

    drift_accum_px: DxDy = (0.0, 0.0)       # best estimate drift vs ref
    vx_hist: List[float] = field(default_factory=list)
    vy_hist: List[float] = field(default_factory=list)
    dt_hist: List[float] = field(default_factory=list)

    def update(self, current_path: str) -> dict:
        """Call once per new image path (absolute path)."""

        t = _mtime(current_path)

        # first image initializes reference + previous
        if self.ref_path is None:
            self.ref_path = current_path
            self.prev_path = current_path
            self.prev_time = t
            self.idx = 0
            self.drift_accum_px = (0.0, 0.0)
            return {
                "idx": self.idx,
                "anchored": True,
                "drift_ref_px": (0.0, 0.0),
                "drift_step_px": (0.0, 0.0),
                "vx_px_s": 0.0,
                "vy_px_s": 0.0,
                "speed_px_s": 0.0,
                "suggested_k": self.anchor_every_n,
            }

        assert self.prev_path is not None and self.prev_time is not None

        # STEP drift (prev -> current)
        dx_s, dy_s = self.calculate_translation(self.prev_path, current_path)

        dt = max(1e-6, t - self.prev_time)
        vx = dx_s / dt
        vy = dy_s / dt

        self.vx_hist.append(vx)
        self.vy_hist.append(vy)
        self.dt_hist.append(dt)
        if len(self.vx_hist) > self.vel_window:
            self.vx_hist.pop(0); self.vy_hist.pop(0); self.dt_hist.pop(0)

        # estimate velocity based on median

        vx_est = _median(self.vx_hist)
        vy_est = _median(self.vy_hist)
        dt_est = _median(self.dt_hist)
        speed = math.hypot(vx_est, vy_est)

        # accumulate drift vs ref by summing successive steps
        Dx, Dy = self.drift_accum_px
        Dx += dx_s
        Dy += dy_s
        self.drift_accum_px = (Dx, Dy)

        self.idx += 1

        anchored = False

        # keyframe safety cadence: every N frames, anchor using ref -> current (overwrites drift_accum)
        if self.anchor_every_n > 0 and (self.idx % self.anchor_every_n == 0):
            dx_a, dy_a = self.calculate_translation(self.ref_path, current_path)
            self.drift_accum_px = (dx_a, dy_a)
            anchored = True

        # compute suggested K based on current velocity
        # expected drift per frame ~ speed * dt_est; we want <= dmax_px
        if speed > 1e-9:
            drift_per_frame = speed * dt_est
            k_suggest = int(max(2, min(50, math.floor(self.dmax_px / max(1e-9, drift_per_frame)))))
        else:
            k_suggest = self.anchor_every_n

        # update previous pointers
        self.prev_path = current_path
        self.prev_time = t

        return {
            "idx": self.idx,
            "anchored": anchored,
            "drift_ref_px": self.drift_accum_px,
            "drift_step_px": (dx_s, dy_s),
            "vx_px_s": vx_est,
            "vy_px_s": vy_est,
            "speed_px_s": speed,
            "suggested_k": k_suggest,
        }
