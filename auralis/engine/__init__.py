# -*- coding: utf-8 -*-
"""Auralis engine: hybrid wave and geometrical room-acoustics core.
Pure NumPy, with optional native C++ and CUDA kernels."""

from .air import air_attenuation_db_per_m, air_energy_factor, speed_of_sound
from .bands import CENTERS, LABELS, N_BANDS
from .convolve import fft_convolve, make_test_signal, normalize_peak, resample_linear
from .geometry import (NumpyRaycaster, SceneData, add_box, box_triangles,
                       extract_planes, make_shoebox)
from .ism import AcousticEvent, compute_direct, compute_image_sources
from .materials import PRESET_NAMES, PRESETS, build_tables, get_preset
from .metrics import compute_metrics, events_to_hist, format_report
from .diffraction import (DiffEdge, compute_diffraction, extract_edges,
                          fresnel_F, utd_coefficient)
from .fdtd import (FdtdInfo, design_crossover, fit_admittance,
                   fuse_low_high, make_sponge, run_low_band, voxelize)
from .level import (band_energies, dba_total, level_stats, spl1m_from_power,
                    spl_bands, strength_g)
from .reflection import (alpha_random, band_R_mag, build_admittance_table,
                         event_reflection_spectrum, fit_admittance_random,
                         invert_paris, paris_alpha_real, refl_coeff)
from .directivity import (Directivity, DIRECTIVITY_NAMES,
                          DIRECTIVITY_PRESETS, as_directivity, cone,
                          get_directivity)
from .motion import (MotionConfig, MovingReceiver, MovingSource,
                     Trajectory, render_moving, render_scene_moving)
from .pipeline import (BakeConfig, BakeResult, ReceiverDef, SourceDef, bake,
                       bake_pair, estimate_max_time, gpu_available,
                       native_available, precompute_low_bands, source_low_bands)
from .ir_synthesis import synthesize_binaural, synthesize_mono
from .stochastic import trace_python
from .wavio import read_wav, write_wav


__all__ = [
    "air_attenuation_db_per_m", "air_energy_factor", "speed_of_sound",
    "CENTERS", "LABELS", "N_BANDS",
    "fft_convolve", "make_test_signal", "normalize_peak", "resample_linear",
    "FdtdInfo", "design_crossover", "fit_admittance", "fuse_low_high",
    "make_sponge", "run_low_band",
    "band_energies", "dba_total", "level_stats", "spl1m_from_power",
    "spl_bands", "strength_g",
    "alpha_random", "band_R_mag", "build_admittance_table",
    "event_reflection_spectrum", "fit_admittance_random", "invert_paris",
    "paris_alpha_real", "refl_coeff",
    "voxelize",
    "DiffEdge", "compute_diffraction", "extract_edges", "fresnel_F",
    "utd_coefficient",
    "Directivity", "DIRECTIVITY_NAMES", "DIRECTIVITY_PRESETS",
    "as_directivity", "cone", "get_directivity",
    "MotionConfig", "MovingReceiver", "MovingSource", "Trajectory",
    "render_moving", "render_scene_moving",
    "NumpyRaycaster", "SceneData", "add_box", "box_triangles",
    "extract_planes", "make_shoebox",
    "AcousticEvent", "compute_direct", "compute_image_sources",
    "PRESET_NAMES", "PRESETS", "build_tables", "get_preset",
    "compute_metrics", "events_to_hist", "format_report",
    "BakeConfig", "BakeResult", "ReceiverDef", "SourceDef", "bake",
    "estimate_max_time", "native_available", "gpu_available",
    "precompute_low_bands", "source_low_bands", "bake_pair",
    "synthesize_binaural", "synthesize_mono",
    "trace_python", "read_wav", "write_wav",
]
