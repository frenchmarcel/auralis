# -*- coding: utf-8 -*-
"""Material libraries: engine presets, pyroomacoustics (MIT), the local
Odeon library (research licence, never redistributed) and user materials."""
from __future__ import annotations

import json
import os
import re

import numpy as np

from auralis.engine import materials as mat_lib
from auralis.engine.bands import CENTERS
from auralis.engine import poromat

ALPHA_MAX = 0.95
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PRA_FILE = os.path.join(HERE, "data", "pyroomacoustics_materials.json")
ODEON_FILE = os.path.join(REPO, "data_odeon", "raw", "Material.Li8")

LIBRARIES = {
    "auralis": {"id": "auralis", "name": "Auralis presets", "redistributable": True,
                "license": "MIT"},
    "pyroomacoustics": {"id": "pyroomacoustics", "name": "pyroomacoustics (Vorländer tables)",
                        "redistributable": True, "license": "MIT (EPFL-LCAV)"},
    "odeon_local": {"id": "odeon_local", "name": "Odeon library (local, research licence)",
                    "redistributable": False,
                    "license": "Odeon A/S - research use only, do not redistribute"},
    "odeon_rooms": {"id": "odeon_rooms", "name": "Odeon room materials (local, research licence)",
                    "redistributable": False,
                    "license": "Odeon A/S - research use only, do not redistribute"},
    "bras": {"id": "bras", "name": "BRAS measured materials (local)", "redistributable": False,
             "license": "BRAS database (TU Berlin), CC BY-SA 4.0: credit, share-alike for "
                        "derived tables (DepositOnce record v3)"},
    "custom": {"id": "custom", "name": "My materials", "redistributable": True,
               "license": "user"},
}

_PRESET_CATEGORY = {
    "concrete": "Hard surfaces", "brick": "Hard surfaces", "marble": "Hard surfaces",
    "plaster": "Hard surfaces", "drywall": "Lightweight constructions",
    "wood_panel": "Wood", "wood_floor": "Floors", "parquet": "Floors",
    "glass_window": "Glazing", "glass_heavy": "Glazing", "carpet_thin": "Floors",
    "carpet_heavy": "Floors", "curtain_heavy": "Curtains", "acoustic_tile": "Ceiling absorbers",
    "acoustic_foam": "Absorbers", "audience": "Audience", "metal_panel": "Lightweight constructions",
    "door_wood": "Wood", "water_surface": "Outdoor", "grass": "Outdoor", "default": "Generic",
    "opening": "Openings",
}


def _material(mid, library, name, category, absorption, scattering, source,
              description="", a63=None, physical=None):
    a = np.asarray(absorption, dtype=float)
    clipped = bool((a > ALPHA_MAX).any() or (a63 is not None and a63 > ALPHA_MAX))
    return {"id": mid, "library": library, "name": name, "category": category,
            "description": description,
            "absorption": np.clip(a, 0.0, ALPHA_MAX).round(4).tolist(),
            "absorption_63": None if a63 is None else round(min(max(a63, 0.0), ALPHA_MAX), 4),
            "scattering": round(float(scattering), 4), "source": source,
            "license": LIBRARIES[library]["license"], "clipped": clipped,
            "physical": physical}


def _presets():
    out = []
    for name in mat_lib.PRESET_NAMES:
        p = mat_lib.PRESETS[name]
        out.append(_material("auralis:" + name, "auralis", name.replace("_", " ").capitalize(),
                             _PRESET_CATEGORY.get(name, "Generic"), p["absorption"],
                             float(np.mean(p["scattering"])), "Auralis engine preset"))
    return out


def _pyroomacoustics():
    if not os.path.exists(PRA_FILE):
        return []
    with open(PRA_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for cat, entries in data["absorption"].items():
        for key, e in entries.items():
            c = list(e["coeffs"])
            desc = e.get("description", key)
            if len(c) == 6:
                c.append(c[-1])
                desc += " [8 kHz value repeated from 4 kHz: not in the source table]"
            elif len(c) != 7:
                continue
            out.append(_material("pyroomacoustics:" + key, "pyroomacoustics", desc[:90],
                                 cat, c, 0.15,
                                 "Vorländer, Auralization (2008), via pyroomacoustics",
                                 description=desc))
    return out


def _odeon():
    if not os.path.exists(ODEON_FILE):
        return []
    lines = [l.rstrip("\r\n") for l in open(ODEON_FILE, encoding="utf-8-sig")]
    out = []
    for i, line in enumerate(lines[:-1]):
        m = re.match(r"^(\d+)\s+(.*)$", line.strip())
        if not m:
            continue
        fields = dict((k.strip().rstrip("*"), v.strip())
                      for k, v in re.findall(r"\{([^=}]+)=([^}]*)\}", m.group(2)))
        vals = lines[i + 1].split()
        if "/Area" in vals:
            vals = vals[:vals.index("/Area")]
        try:
            a = [float(x) for x in vals]
        except ValueError:
            continue
        name = fields.get("Product name", "Odeon %s" % m.group(1))
        if len(a) != 8 or name.lower().startswith("transparent"):
            continue    # 'Transparent' means "no surface" in Odeon: meaningless here
        layered = "LayerType" in m.group(2)
        out.append(_material(
            "odeon_local:%s" % m.group(1), "odeon_local", name,
            fields.get("Category", "Odeon"), a[1:], 0.15,
            "Odeon Material.Li8 #%s" % m.group(1),
            description=("Layer model available in the source (not used yet). " if layered else "")
            + "The 63 Hz value feeds the wave band.",
            a63=a[0]))
    return out


def _name_key(name):
    """Names compare without case and extra spaces."""
    return " ".join(str(name).split()).casefold()


class Library:
    """All materials, plus persistence of the user's own ones."""

    def __init__(self, custom_path):
        self.custom_path = custom_path
        self.reload()

    def reload(self):
        self.builtin = _presets() + _pyroomacoustics() + _odeon()
        self.custom = []
        if os.path.exists(self.custom_path):
            with open(self.custom_path, encoding="utf-8") as fh:
                self.custom = json.load(fh)
        self.local = []
        if os.path.exists(self.local_path):
            with open(self.local_path, encoding="utf-8") as fh:
                self.local = json.load(fh)

    @property
    def local_path(self):
        """Materials extracted from local data (Odeon room models, BRAS): never redistributed."""
        return os.path.join(os.path.dirname(self.custom_path), "materials_local.json")

    def add_local(self, library, key, name, category, absorption, scattering, source,
                  description="", a63=None):
        """Idempotent: the same `key` (e.g. 'musikverein:9') always maps to one material."""
        mid = "%s:%s" % (library, key)
        for m in self.local:
            if m["id"] == mid:
                return m
        m = _material(mid, library, name, category, absorption, scattering, source,
                      description=description, a63=a63)
        self.local.append(m)
        tmp = self.local_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.local, fh, indent=1)
        os.replace(tmp, self.local_path)
        return m

    def _save(self):
        tmp = self.custom_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.custom, fh, indent=1)
        os.replace(tmp, self.custom_path)

    def all(self):
        mats = self.builtin + self.local + self.custom
        present = sorted({m["library"] for m in mats} | {"custom"})
        return {"bands_hz": [int(c) for c in CENTERS],
                "libraries": [LIBRARIES[k] for k in LIBRARIES if k in present],
                "materials": mats}

    def get(self, mid):
        for m in self.builtin + self.local + self.custom:
            if m["id"] == mid:
                return m
        return None

    def add_custom(self, name, category, description, absorption, scattering, physical=None,
                   a63=None):
        """a63: optional 63 Hz alpha, used by the wave band only."""
        a = np.asarray(absorption, dtype=float)
        if a.shape != (mat_lib.N_BANDS,):
            raise ValueError("absorption needs %d octave values (125 Hz - 8 kHz)" % mat_lib.N_BANDS)
        if not np.all(np.isfinite(a)) or (a < 0).any():
            raise ValueError("absorption values must be finite and >= 0")
        if a63 is not None:
            a63 = float(a63)
            if not np.isfinite(a63) or a63 < 0:
                raise ValueError("the 63 Hz absorption must be finite and >= 0")
            if physical:
                raise ValueError("a physical material defines its own low-frequency curve: "
                                 "no separate 63 Hz value")
        if not 0.0 <= float(scattering) <= 1.0:
            raise ValueError("scattering must be within [0, 1]")
        n = self._next_number()
        m = _material("custom:%d" % n, "custom", name or "Material %d" % n,
                      category or "Custom", a, scattering, "User", description or "",
                      a63=a63, physical=physical)
        self.custom.append(m)
        self._save()
        return m

    def delete_custom(self, mid):
        if not mid.startswith("custom:"):
            raise ValueError("only your own materials can be deleted")
        before = len(self.custom)
        self.custom = [m for m in self.custom if m["id"] != mid]
        if len(self.custom) == before:
            raise KeyError(mid)
        self._mark_issued(int(mid.split(":")[1]))
        self._save()

    # -- ids, rename, duplicate ------------------------------------------------
    @property
    def _seq_path(self):
        return os.path.join(os.path.dirname(self.custom_path), "materials_custom_seq.json")

    def _last_issued(self):
        try:
            with open(self._seq_path, encoding="utf-8") as fh:
                return int(json.load(fh)["last"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def _mark_issued(self, n):
        n = max(n, self._last_issued())
        tmp = self._seq_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"last": n}, fh)
        os.replace(tmp, self._seq_path)

    def _next_number(self):
        """A custom id is never reused: a simulation still naming a deleted material must
        show it missing, not silently pick up the next material created."""
        n = 1 + max([int(m["id"].split(":")[1]) for m in self.custom] + [self._last_issued()])
        self._mark_issued(n)
        return n

    def _check_name(self, name, exclude=None):
        name = " ".join(str(name or "").split())
        if not name:
            raise ValueError("a material name cannot be empty")
        if len(name) > 200:
            raise ValueError("a material name is at most 200 characters")
        for m in self.custom:
            if m["id"] != exclude and _name_key(m["name"]) == _name_key(name):
                raise ValueError("one of your materials is already named '%s'" % m["name"])
        return name

    def rename_custom(self, mid, name):
        """Only the name changes: the id (what simulations store) and the data stay."""
        if not mid.startswith("custom:"):
            if self.get(mid) is None:
                raise KeyError(mid)
            raise ValueError("only your own materials can be renamed: duplicate this one first")
        m = next((x for x in self.custom if x["id"] == mid), None)
        if m is None:
            raise KeyError(mid)
        m["name"] = self._check_name(name, exclude=mid)
        self._save()
        return m

    def duplicate(self, mid, name=None):
        """A custom copy of any material (new id). A copy of a library material keeps its
        source, licence and redistributability; `derived_from` names the original."""
        src = self.get(mid)
        if src is None:
            raise KeyError(mid)
        if name is None or not str(name).strip():
            base = "%s (copy)" % src["name"]
            taken = {_name_key(m["name"]) for m in self.custom}
            name, k = base, 2
            while _name_key(name) in taken:
                name, k = "%s (copy %d)" % (src["name"], k), k + 1
        name = self._check_name(name)
        m = json.loads(json.dumps(src))
        if src["library"] != "custom":
            m["redistributable"] = LIBRARIES[src["library"]]["redistributable"]
        m.update(id="custom:%d" % self._next_number(), library="custom", name=name,
                 derived_from=mid)
        self.custom.append(m)
        self._save()
        return m


Z0 = 413.3      # rho0 c0 of air at 20 C (poromat.Z0 convention: normalised values)


def gamma_from_input(freqs, re, im, kind):
    """Normalised admittance gamma(f) from a complex normal-incidence quantity:
    kind "z" = surface impedance Z / (rho0 c0), "r" = reflection factor R."""
    f = np.asarray(freqs, dtype=float)
    q = np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float)
    if len(f) < 3 or len(f) != len(q) or np.any(np.diff(f) <= 0) or f[0] <= 0:
        raise ValueError("need >= 3 increasing frequencies with one complex value each")
    if not np.all(np.isfinite(q)):
        raise ValueError("values must be finite")
    if kind == "z":
        if np.any(q.real <= 0):
            raise ValueError("a passive surface has Re(Z) > 0")
        g = 1.0 / q
    elif kind == "r":
        if np.any(np.abs(q) >= 1.0):
            raise ValueError("a passive surface has |R| < 1")
        g = (1.0 - q) / (1.0 + q)
    else:
        raise ValueError("kind must be 'z' (impedance / rho c) or 'r' (reflection factor)")
    return f, g


def table_alpha(f, g):
    """Random-incidence alpha per octave of a gamma table (Paris integral, band mean)."""
    from auralis.engine.reflection import alpha_random
    out = []
    lg = np.log(f)
    for fc in CENTERS:
        fb = np.geomspace(fc / 2 ** 0.5, fc * 2 ** 0.5, 9)
        gb = np.interp(np.log(fb), lg, g.real) + 1j * np.interp(np.log(fb), lg, g.imag)
        out.append(float(np.mean(alpha_random(gb))))
    return np.clip(out, 0.0, ALPHA_MAX)


def porous_spec(model, sigma, thickness_mm, air_gap_mm):
    """Builder parameters -> poromat spec (porous layer + optional air gap, rigid backing)."""
    model = str(model or "miki").lower()
    if model not in ("miki", "jca"):
        raise ValueError("model must be 'miki' or 'jca'")
    if not (200.0 <= float(sigma) <= 5e6):
        raise ValueError("flow resistivity must be within 200 - 5e6 Pa.s/m2")
    if not (0.1 <= float(thickness_mm) <= 2000.0) or not (0.0 <= float(air_gap_mm) <= 5000.0):
        raise ValueError("thickness must be 0.1-2000 mm and air gap 0-5000 mm")
    return poromat.spec_from_params(model, float(sigma), float(thickness_mm), float(air_gap_mm))


PANEL_KINDS = ("perforated", "slotted", "mpp")


def perforated_spec(kind, size_mm, t_mm, open_pct, plenum_mm, wool_mm=0.0, wool_sigma=10000.0,
                    fleece_rayl=0.0):
    """Builder parameters -> poromat stack: perforated / slotted / micro-perforated sheet
    (hole diameter or slot width size_mm, thickness t_mm, open area %), optional fleece
    (flow resistance, Pa.s/m), optional wool layer behind it, air for the rest of the
    plenum, rigid backing."""
    kind = str(kind or "perforated").lower()
    if kind not in PANEL_KINDS:
        raise ValueError("kind must be one of %s" % ", ".join(PANEL_KINDS))
    size, t, op = float(size_mm), float(t_mm), float(open_pct)
    plenum, wool = float(plenum_mm), float(wool_mm or 0.0)
    if not (0.05 <= size <= 50.0) or not (0.1 <= t <= 50.0):
        raise ValueError("hole / slot size must be 0.05-50 mm and the sheet 0.1-50 mm thick")
    if not (0.1 <= op <= 60.0):
        raise ValueError("open area must be 0.1-60 %")
    if not (5.0 <= plenum <= 1500.0) or not (0.0 <= wool <= plenum):
        raise ValueError("plenum must be 5-1500 mm and the wool no thicker than the plenum")
    if not (0.0 <= float(fleece_rayl or 0.0) <= 5000.0):
        raise ValueError("fleece flow resistance must be 0-5000 Pa.s/m")
    sheet = {"model": kind, "t": t * 1e-3, "phi": op / 100.0}
    sheet["w_slot" if kind == "slotted" else "d"] = size * 1e-3
    layers = [sheet]
    if fleece_rayl:
        layers.append({"model": "screen", "r": float(fleece_rayl), "m": 0.0})
    if wool > 0.0:
        if not (1000.0 <= float(wool_sigma) <= 1e6):
            raise ValueError("wool flow resistivity must be 1000-1e6 Pa.s/m2")
        layers.append({"model": "miki", "d": wool * 1e-3, "sigma": float(wool_sigma)})
    if plenum - wool > 0.05:
        layers.append({"model": "air", "d": (plenum - wool) * 1e-3})
    return {"layers": layers}


def perforated_alpha(**params):
    return poromat.alpha_bands(perforated_spec(**params)).tolist()


def porous_alpha(model, sigma, thickness_mm, air_gap_mm):
    return poromat.alpha_bands(porous_spec(model, sigma, thickness_mm, air_gap_mm)).tolist()
