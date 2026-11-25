# -*- coding: utf-8 -*-
"""ctypes bridge to the native C++ kernel (libauralis).
Library search order: AURALIS_NATIVE_LIB, <package>/lib/<platform-tag>/,
then <repo>/native/build/ and <repo>/native/.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from typing import Optional

import numpy as np

from .bands import CENTERS, N_BANDS
from .geometry import SceneData

_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_int32_p = ctypes.POINTER(ctypes.c_int32)
_c_double_p = ctypes.POINTER(ctypes.c_double)


class AurTraceParams(ctypes.Structure):
    _fields_ = [
        ("absorption", _c_float_p),
        ("scattering", _c_float_p),
        ("transmission", _c_float_p),
        ("n_mats", ctypes.c_int32),
        ("n_bands", ctypes.c_int32),
        ("src", ctypes.c_float * 3),
        ("rcv", ctypes.c_float * 3),
        ("rcv_radius", ctypes.c_float),
        ("n_rays", ctypes.c_int32),
        ("max_bounces", ctypes.c_int32),
        ("max_time", ctypes.c_float),
        ("bin_dt", ctypes.c_float),
        ("n_bins", ctypes.c_int32),
        ("speed_of_sound", ctypes.c_float),
        ("air_db_per_m", _c_float_p),
        ("spec_exclude_order", ctypes.c_int32),
        ("energy_cap", ctypes.c_float),
        ("seed", ctypes.c_uint64),
        ("n_threads", ctypes.c_int32),
        ("dir_table", _c_float_p),
        ("n_dir_samples", ctypes.c_int32),
        ("src_forward", ctypes.c_float * 3),
    ]


class AurRbs(ctypes.Structure):
    """Reflection-based scattering inputs of aur_trace2 (ABI 2.2)."""
    _fields_ = [
        ("tri_l", _c_float_p),
        ("tri_w", _c_float_p),
        ("band_hz", _c_float_p),
    ]


def platform_tag() -> str:
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if sys.platform.startswith("linux"):
        osname = "linux"
    elif sys.platform == "darwin":
        osname = "macos"
        if machine == "aarch64":
            machine = "arm64"
    elif sys.platform.startswith("win"):
        osname = "windows"
    else:
        osname = sys.platform
    return "%s-%s" % (osname, machine)


def _lib_filename() -> str:
    if sys.platform.startswith("win"):
        return "auralis.dll"
    if sys.platform == "darwin":
        return "libauralis.dylib"
    return "libauralis.so"


def _gpu_lib_filename() -> str:
    if sys.platform.startswith("win"):
        return "auralis_cuda.dll"
    if sys.platform == "darwin":
        return "libauralis_cuda.dylib"
    return "libauralis_cuda.so"


def _find(fname: str, env_var: str) -> Optional[str]:
    env = os.environ.get(env_var)
    if env and os.path.isfile(env):
        return env
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(pkg_root, "lib", platform_tag(), fname),
        os.path.join(os.path.dirname(pkg_root), "native", "build", fname),
        os.path.join(os.path.dirname(pkg_root), "native", fname),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def find_library() -> Optional[str]:
    return _find(_lib_filename(), "AURALIS_NATIVE_LIB")


def find_gpu_library() -> Optional[str]:
    return _find(_gpu_lib_filename(), "AURALIS_CUDA_LIB")


_LIBS = {}


def load_library(path: Optional[str] = None):
    p = path or find_library()
    if p is None:
        raise OSError("libauralis not found (see native/build.sh)")
    if p in _LIBS:
        return _LIBS[p]
    lib = ctypes.CDLL(p)
    lib.aur_scene_create.restype = ctypes.c_void_p
    lib.aur_scene_create.argtypes = [_c_float_p, ctypes.c_int32, _c_int32_p]
    lib.aur_scene_destroy.restype = None
    lib.aur_scene_destroy.argtypes = [ctypes.c_void_p]
    lib.aur_scene_node_count.restype = ctypes.c_int32
    lib.aur_scene_node_count.argtypes = [ctypes.c_void_p]
    lib.aur_trace.restype = ctypes.c_int32
    lib.aur_trace.argtypes = [ctypes.c_void_p, ctypes.POINTER(AurTraceParams),
                              _c_float_p, _c_float_p, _c_double_p]
    if hasattr(lib, "aur_trace2"):
        lib.aur_trace2.restype = ctypes.c_int32
        lib.aur_trace2.argtypes = [ctypes.c_void_p, ctypes.POINTER(AurTraceParams),
                                   ctypes.POINTER(AurRbs), _c_float_p, _c_float_p,
                                   _c_double_p]
    lib.aur_segment_transmission.restype = ctypes.c_int32
    lib.aur_segment_transmission.argtypes = [ctypes.c_void_p,
                                             ctypes.c_float * 3, ctypes.c_float * 3,
                                             _c_float_p, ctypes.c_int32,
                                             ctypes.c_int32, _c_float_p]
    lib.aur_raycast.restype = ctypes.c_int32
    lib.aur_raycast.argtypes = [ctypes.c_void_p, ctypes.c_float * 3,
                                ctypes.c_float * 3, ctypes.c_float,
                                _c_float_p, ctypes.c_float * 3]
    lib.aur_version.restype = ctypes.c_char_p
    lib.aur_version.argtypes = []
    if hasattr(lib, "aur_cuda_available"):
        lib.aur_cuda_available.restype = ctypes.c_int32
        lib.aur_cuda_available.argtypes = []
    _LIBS[p] = lib
    return lib


def is_available() -> bool:
    try:
        load_library()
        return True
    except OSError:
        return False


_GPU_OK = None


def gpu_available() -> bool:
    """True if a CUDA library is built AND a GPU answers (honors
    AURALIS_GPU=0 to disable)."""
    global _GPU_OK
    if os.environ.get("AURALIS_GPU", "") == "0":
        return False
    if _GPU_OK is not None:
        return _GPU_OK
    p = find_gpu_library()
    ok = False
    if p is not None:
        try:
            lib = load_library(p)
            ok = bool(getattr(lib, "aur_cuda_available", lambda: 1)())
        except OSError:
            ok = False
    _GPU_OK = ok
    return ok


def mark_gpu_dead(reason: str = ""):
    """Mark the GPU unavailable for the rest of the session after a runtime failure.

    gpu_available() then returns False for both the tracer and the FDTD."""
    global _GPU_OK
    if _GPU_OK is not False:
        print("[Auralis] GPU marked unavailable for this session"
              + (": %s" % reason if reason else "")
              + " - falling back to the native CPU. (nvidia-smi to see"
              " the process holding the card; AURALIS_GPU=0 to force"
              " the CPU.)")
    _GPU_OK = False


def _fp(a: np.ndarray) -> _c_float_p:
    return a.ctypes.data_as(_c_float_p)


class NativeEngine:
    """Owns a native scene handle; exposes trace / permeation / raycast."""

    def __init__(self, scene: SceneData, lib_path: Optional[str] = None):
        self.lib = load_library(lib_path)
        self.scene = scene
        self._verts = np.ascontiguousarray(scene.tri_verts.reshape(-1),
                                           dtype=np.float32)
        self._mats = np.ascontiguousarray(scene.tri_mat, dtype=np.int32)
        self._alpha = np.ascontiguousarray(scene.absorption, dtype=np.float32)
        self._scat = np.ascontiguousarray(scene.scattering, dtype=np.float32)
        self._tau = np.ascontiguousarray(scene.transmission, dtype=np.float32)
        self.handle = self.lib.aur_scene_create(
            _fp(self._verts), scene.n_tris,
            self._mats.ctypes.data_as(_c_int32_p))
        if not self.handle:
            raise RuntimeError("aur_scene_create failed")
        # numpy copy of the normals for batched callers (cast() also returns them)
        e1 = scene.tri_verts[:, 1] - scene.tri_verts[:, 0]
        e2 = scene.tri_verts[:, 2] - scene.tri_verts[:, 0]
        ng = np.cross(e1.astype(np.float64), e2.astype(np.float64))
        nl = np.linalg.norm(ng, axis=1, keepdims=True)
        nl[nl == 0] = 1.0
        self.normals = ng / nl
        self.tri_mat = scene.tri_mat

    def close(self):
        if getattr(self, "handle", None):
            self.lib.aur_scene_destroy(ctypes.c_void_p(self.handle))
            self.handle = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    @property
    def node_count(self) -> int:
        return int(self.lib.aur_scene_node_count(ctypes.c_void_p(self.handle)))

    @property
    def version(self) -> str:
        return self.lib.aur_version().decode()

    # ---- stochastic trace ------------------------------------------------
    def trace(self, src, rcv, n_rays, max_bounces, max_time, bin_dt, n_bins,
              speed_of_sound, air_db_per_m, spec_exclude_order=2,
              energy_cap=1e-6, seed=1234, n_threads=0, rcv_radius=0.3,
              src_forward=None, dir_table=None, rbs=None):
        """rbs = (tri_l, tri_w): reflection-based scattering (aur_trace2, ABI 2.2)."""
        p = AurTraceParams()
        p.absorption = _fp(self._alpha)
        p.scattering = _fp(self._scat)
        p.transmission = _fp(self._tau)
        p.n_mats = self.scene.n_mats
        p.n_bands = N_BANDS
        p.src = (ctypes.c_float * 3)(*[float(x) for x in src])
        p.rcv = (ctypes.c_float * 3)(*[float(x) for x in rcv])
        p.rcv_radius = float(rcv_radius)
        p.n_rays = int(n_rays)
        p.max_bounces = int(max_bounces)
        p.max_time = float(max_time)
        p.bin_dt = float(bin_dt)
        p.n_bins = int(n_bins)
        p.speed_of_sound = float(speed_of_sound)
        air = np.ascontiguousarray(air_db_per_m, dtype=np.float32)
        p.air_db_per_m = _fp(air)
        p.spec_exclude_order = int(spec_exclude_order)
        p.energy_cap = float(energy_cap)
        p.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        p.n_threads = int(n_threads)
        if dir_table is not None and src_forward is not None:
            dtab = np.ascontiguousarray(dir_table, dtype=np.float32)
            self._dir_keepalive = dtab          # ctypes: keep a reference
            p.dir_table = _fp(dtab)
            p.n_dir_samples = int(dtab.shape[1])
            p.src_forward = (ctypes.c_float * 3)(
                *[float(x) for x in src_forward])
        else:
            p.dir_table = None
            p.n_dir_samples = 0
            p.src_forward = (ctypes.c_float * 3)(0.0, 0.0, 0.0)

        hist = np.zeros(N_BANDS * n_bins, dtype=np.float32)
        dirs = np.zeros(n_bins * 3, dtype=np.float32)
        stats = np.zeros(4, dtype=np.float64)
        if rbs is None:
            ret = self.lib.aur_trace(ctypes.c_void_p(self.handle), ctypes.byref(p),
                                     _fp(hist), _fp(dirs),
                                     stats.ctypes.data_as(_c_double_p))
        else:
            if not hasattr(self.lib, "aur_trace2"):
                raise RuntimeError("reflection-based scattering needs the native library "
                                   "ABI 2.2 (aur_trace2): rebuild native/ and the CUDA libs")
            tl = np.ascontiguousarray(rbs[0], dtype=np.float32)
            tw = np.ascontiguousarray(rbs[1], dtype=np.float32)
            bands = np.ascontiguousarray(CENTERS, dtype=np.float32)
            r = AurRbs(_fp(tl), _fp(tw), _fp(bands))
            ret = self.lib.aur_trace2(ctypes.c_void_p(self.handle), ctypes.byref(p),
                                      ctypes.byref(r), _fp(hist), _fp(dirs),
                                      stats.ctypes.data_as(_c_double_p))
        if ret != 0:
            raise RuntimeError("aur_trace ret=%d" % ret)
        return (hist.reshape(N_BANDS, n_bins).astype(np.float64),
                dirs.reshape(n_bins, 3).astype(np.float64),
                {"rays": stats[0], "mean_bounces": stats[1],
                 "escaped": stats[2], "deposits": stats[3]})

    # ---- permeation / raycast --------------------------------------------
    def segment_transmission(self, a, b, transmission_table=None):
        tab = self._tau if transmission_table is None else \
            np.ascontiguousarray(transmission_table, dtype=np.float32)
        out = np.zeros(N_BANDS, dtype=np.float32)
        n = self.lib.aur_segment_transmission(
            ctypes.c_void_p(self.handle),
            (ctypes.c_float * 3)(*[float(x) for x in a]),
            (ctypes.c_float * 3)(*[float(x) for x in b]),
            _fp(tab), tab.shape[0], N_BANDS, _fp(out))
        return out.astype(np.float64), int(n)

    def cast(self, o, d, t_max=1e30):
        t = ctypes.c_float(0.0)
        n = (ctypes.c_float * 3)()
        tri = self.lib.aur_raycast(
            ctypes.c_void_p(self.handle),
            (ctypes.c_float * 3)(*[float(x) for x in o]),
            (ctypes.c_float * 3)(*[float(x) for x in d]),
            float(t_max), ctypes.byref(t), n)
        if tri < 0:
            return -1, np.inf, None
        return int(tri), float(t.value), np.array([n[0], n[1], n[2]], dtype=np.float64)

    def cast_batch(self, origins, dirs, t_max=None):
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        n = len(o)
        tm = np.full(n, 1e30) if t_max is None else \
            np.asarray(t_max, dtype=np.float64).reshape(-1)
        tri = np.full(n, -1, dtype=np.int32)
        tt = np.full(n, np.inf)
        for i in range(n):
            k, t, _ = self.cast(o[i], d[i], tm[i])
            tri[i] = k
            if k >= 0:
                tt[i] = t
        return tri, tt

    def occluded_batch(self, origins, dirs, t_max):
        tri, _t = self.cast_batch(origins, dirs, t_max)
        return tri >= 0
