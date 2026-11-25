# -*- coding: utf-8 -*-
"""Trajectories of sources and receivers: a polyline of timed waypoints, sampled
for auralis.engine.motion, plus the checks made along the whole path."""
from __future__ import annotations

import math

import numpy as np

ORIENTATIONS = ("fixed", "follow", "target")
INTERPOLATIONS = ("linear", "smooth")
MAX_WAYPOINTS = 200
MAX_DURATION_S = 600.0
SAMPLE_DT = 0.05           # s, engine trajectory sampling
FOLLOW_TAU_S = 0.25        # s, half-window of the heading estimate (smooths corners)
CHECK_STEP_M = 0.10        # m, spacing of the volume/surface checks along a path
CHECK_DT = 0.02            # s, time step of the source-receiver distance check


def _direction(az_deg, el_deg):
    az, el = math.radians(float(az_deg or 0.0)), math.radians(float(el_deg or 0.0))
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def clean_path(path, name):
    """Validated, normalised copy of a point's `path` (None: the point is static)."""
    if not path:
        return None
    wps = path.get("waypoints") or []
    if not 2 <= len(wps) <= MAX_WAYPOINTS:
        raise ValueError("%s: a trajectory needs 2 to %d waypoints" % (name, MAX_WAYPOINTS))
    out = []
    for k, w in enumerate(wps):
        p = w.get("position")
        if not (isinstance(p, (list, tuple)) and len(p) == 3):
            raise ValueError("%s: waypoint %d needs a position [x, y, z]" % (name, k + 1))
        p = [float(x) for x in p]
        t = float(w.get("t"))
        if not all(np.isfinite(p + [t])):
            raise ValueError("%s: waypoint %d is not finite" % (name, k + 1))
        out.append({"position": p, "t": t})
    ts = [w["t"] for w in out]
    if ts[0] < 0 or any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("%s: waypoint times must start at 0 s or later and increase" % name)
    if ts[-1] > MAX_DURATION_S:
        raise ValueError("%s: trajectories are limited to %.0f s" % (name, MAX_DURATION_S))
    orient = path.get("orientation") or "fixed"
    if orient not in ORIENTATIONS:
        raise ValueError("%s: orientation must be one of %s" % (name, ", ".join(ORIENTATIONS)))
    target = path.get("target_id") if orient == "target" else None
    if orient == "target" and not target:
        raise ValueError("%s: orientation 'target' needs a target" % name)
    interp = path.get("interpolation") or "linear"
    if interp not in INTERPOLATIONS:
        raise ValueError("%s: interpolation must be 'linear' or 'smooth'" % name)
    return {"waypoints": out, "orientation": orient, "target_id": target,
            "interpolation": interp}


def _tangents(t, p):
    """Catmull-Rom tangents in time, (p[k+1] - p[k-1]) / (t[k+1] - t[k-1]); one-sided at
    the ends, and zero at either end of a hold (two equal waypoints: the point stops)."""
    m = np.zeros_like(p)
    if len(t) > 2:
        m[1:-1] = (p[2:] - p[:-2]) / (t[2:] - t[:-2])[:, None]
    m[0] = (p[1] - p[0]) / (t[1] - t[0])
    m[-1] = (p[-1] - p[-2]) / (t[-1] - t[-2])
    still = np.all(np.abs(np.diff(p, axis=0)) < 1e-9, axis=1)
    m[:-1][still] = 0.0
    m[1:][still] = 0.0
    return m


def _smooth_at(ts, ps, t):
    """Cubic Hermite through the waypoints at their times (C1, corners rounded), held
    outside the path's time span."""
    m = _tangents(ts, ps)
    tc = np.clip(t, ts[0], ts[-1])
    k = np.clip(np.searchsorted(ts, tc, side="right") - 1, 0, len(ts) - 2)
    h = (ts[k + 1] - ts[k])[:, None]
    s = ((tc - ts[k]) / (ts[k + 1] - ts[k]))[:, None]
    return ((2 * s ** 3 - 3 * s ** 2 + 1) * ps[k] + (s ** 3 - 2 * s ** 2 + s) * h * m[k]
            + (-2 * s ** 3 + 3 * s ** 2) * ps[k + 1] + (s ** 3 - s ** 2) * h * m[k + 1])


def end_time(item):
    path = item.get("path")
    return float(path["waypoints"][-1]["t"]) if path else 0.0


def position_at(item, t):
    """(n, 3) positions at times t (held before the first and after the last waypoint)."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    path = item.get("path")
    if not path:
        return np.broadcast_to(np.asarray(item["position"], float), (len(t), 3)).copy()
    ts = np.array([w["t"] for w in path["waypoints"]])
    ps = np.array([w["position"] for w in path["waypoints"]], dtype=np.float64)
    if path.get("interpolation") == "smooth":
        return _smooth_at(ts, ps, t)
    return np.stack([np.interp(t, ts, ps[:, i]) for i in range(3)], axis=1)


def forward_at(item, t, items_by_id):
    """(n, 3) unit front axes: fixed (azimuth/elevation), follow (heading along the
    path, the point's elevation kept) or target (towards another point, which may move)."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    fixed = _direction(item.get("azimuth_deg", 0), item.get("elevation_deg", 0))
    path = item.get("path")
    orient = path["orientation"] if path else "fixed"
    out = np.broadcast_to(fixed, (len(t), 3)).copy()
    if orient == "follow":
        d = position_at(item, t + FOLLOW_TAU_S) - position_at(item, t - FOLLOW_TAU_S)
        h = d[:, :2]
        n = np.linalg.norm(h, axis=1)
        moving = n > 1e-6
        if moving.any():
            h = np.where(moving[:, None], h / np.maximum(n, 1e-12)[:, None], np.nan)
            idx = np.where(moving, np.arange(len(t)), -1)
            idx = np.maximum.accumulate(idx)                 # hold the last heading...
            first = int(np.argmax(moving))
            idx[idx < 0] = first                             # ...and the first one before it
            h = h[idx]
            el = math.radians(float(item.get("elevation_deg", 0) or 0.0))
            out = np.column_stack([math.cos(el) * h, np.full(len(t), math.sin(el))])
    elif orient == "target":
        tgt = items_by_id.get(path["target_id"])
        if tgt is not None:
            d = position_at(tgt, t) - position_at(item, t)
            n = np.linalg.norm(d, axis=1)
            ok = n > 1e-6
            out[ok] = d[ok] / n[ok, None]
    return out


def up_for(forward):
    """Up axes matching the runner's static receivers (z, or x when looking straight up/down)."""
    up = np.zeros_like(forward)
    vertical = np.abs(forward[:, 2]) >= 0.99
    up[~vertical, 2] = 1.0
    up[vertical, 0] = 1.0
    return up


def sample_times(items, t_end):
    """Shared sampling: a regular grid plus every waypoint time (kinks kept exact)."""
    ts = [np.arange(0.0, t_end + SAMPLE_DT, SAMPLE_DT)]
    for it in items:
        if it.get("path"):
            ts.append([w["t"] for w in it["path"]["waypoints"]])
    t = np.unique(np.concatenate([np.asarray(x, float) for x in ts]))
    return t[t <= t_end + SAMPLE_DT]


def engine_trajectory(item, items_by_id, t_end):
    """auralis.engine.motion.Trajectory for one point over [0, t_end]."""
    from auralis.engine.motion import Trajectory
    t = sample_times([item] + ([items_by_id[item["path"]["target_id"]]]
                               if item.get("path") and item["path"].get("target_id")
                               in items_by_id else []), t_end)
    f = forward_at(item, t, items_by_id)
    return Trajectory(times=t, positions=position_at(item, t), forwards=f, ups=up_for(f))


def _curve_times(item):
    """Times sampling a smooth path at most CHECK_STEP_M apart (refined until true)."""
    wps = item["path"]["waypoints"]
    ps = np.array([w["position"] for w in wps], dtype=np.float64)
    ts = np.array([w["t"] for w in wps])
    m = _tangents(ts, ps)
    out = [ts[:1]]
    for k in range(len(ts) - 1):
        h = ts[k + 1] - ts[k]
        est = np.linalg.norm(ps[k + 1] - ps[k]) + 0.5 * h * (np.linalg.norm(m[k]) + np.linalg.norm(m[k + 1]))
        n = max(1, int(math.ceil(est / CHECK_STEP_M)))
        for _ in range(6):
            tk = ts[k] + h * np.arange(0, n + 1) / n
            if np.linalg.norm(np.diff(_smooth_at(ts, ps, tk), axis=0), axis=1).max() <= CHECK_STEP_M:
                break
            n *= 2
        out.append(tk[1:])
    return np.concatenate(out)


def path_points(item):
    """Points spaced at most CHECK_STEP_M along the path (waypoints included)."""
    if item["path"].get("interpolation") == "smooth":
        return position_at(item, _curve_times(item))
    ps = np.array([w["position"] for w in item["path"]["waypoints"]], dtype=np.float64)
    pts = [ps[:1]]
    for a, b in zip(ps[:-1], ps[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / CHECK_STEP_M)))
        pts.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.concatenate(pts)


def path_info(item):
    wps = item["path"]["waypoints"]
    ps = np.array([w["position"] for w in wps])
    ts = np.array([w["t"] for w in wps])
    if item["path"].get("interpolation") == "smooth":
        tc = _curve_times(item)
        pc = position_at(item, tc)
        seg = np.linalg.norm(np.diff(pc, axis=0), axis=1)
        speed = seg / np.maximum(np.diff(tc), 1e-12)
        return {"length_m": round(float(seg.sum()), 3), "start_s": float(ts[0]),
                "end_s": float(ts[-1]), "max_speed_mps": round(float(speed.max()), 3),
                "curve": np.round(pc, 3).tolist()}
    seg = np.linalg.norm(np.diff(ps, axis=0), axis=1)
    speed = seg / np.diff(ts)
    return {"length_m": round(float(seg.sum()), 3), "start_s": float(ts[0]),
            "end_s": float(ts[-1]), "max_speed_mps": round(float(speed.max()), 3)}


def closest_approach(a, b, t_end):
    """(min distance, time) between two points over [0, t_end]."""
    t = np.unique(np.concatenate([np.arange(0.0, t_end + CHECK_DT, CHECK_DT),
                                  [w["t"] for it in (a, b) if it.get("path")
                                   for w in it["path"]["waypoints"]]]))
    t = t[t <= t_end + CHECK_DT]
    d = np.linalg.norm(position_at(a, t) - position_at(b, t), axis=1)
    k = int(np.argmin(d))
    return float(d[k]), float(t[k])
