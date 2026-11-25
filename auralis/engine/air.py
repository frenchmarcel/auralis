# -*- coding: utf-8 -*-
"""Atmospheric absorption (ISO 9613-1): air attenuation coefficient in
dB/m per band, for a given temperature, relative humidity and static
pressure."""

from __future__ import annotations

import numpy as np

from .bands import CENTERS

_T0 = 293.15    # reference temperature (K)
_T01 = 273.16   # triple point of water (K)
_PR = 101.325   # reference static pressure (kPa)


def air_attenuation_db_per_m(
    frequencies=None,
    temperature_c: float = 20.0,
    humidity_percent: float = 50.0,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """ISO 9613-1 attenuation coefficient in dB/m at the given
    frequencies (Hz), band centres by default; humidity 0..100 %."""
    f = np.asarray(CENTERS if frequencies is None else frequencies, dtype=np.float64)
    T = float(temperature_c) + 273.15
    pa = float(pressure_kpa)
    hr = float(np.clip(humidity_percent, 0.0, 100.0))

    # Saturation vapour pressure ratio (ISO 9613-1, eq. B.3)
    C = -6.8346 * (_T01 / T) ** 1.261 + 4.6151
    psat_over_pr = 10.0 ** C
    # Molar concentration of water vapour (%)
    h = hr * psat_over_pr / (pa / _PR)

    # Relaxation frequencies of oxygen and nitrogen (eq. 3 & 4)
    fr_o = (pa / _PR) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = (pa / _PR) * (T / _T0) ** -0.5 * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((T / _T0) ** (-1.0 / 3.0) - 1.0))
    )

    f2 = f * f
    # Attenuation coefficient in dB/m (eq. 5)
    alpha = 8.686 * f2 * (
        1.84e-11 * (pa / _PR) ** -1.0 * (T / _T0) ** 0.5
        + (T / _T0) ** -2.5 * (
            0.01275 * np.exp(-2239.1 / T) / (fr_o + f2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / T) / (fr_n + f2 / fr_n)
        )
    )
    return alpha


def air_energy_factor(distance_m, air_db_per_m) -> np.ndarray:
    """Per-band ENERGY attenuation factor over a straight path:
    10 ** (-db_per_m * d / 10)."""
    d = np.asarray(distance_m, dtype=np.float64)
    k = (np.log(10.0) / 10.0) * np.asarray(air_db_per_m, dtype=np.float64)
    if d.ndim == 0:
        return np.exp(-k * float(d))
    return np.exp(-np.multiply.outer(d, k))


def speed_of_sound(temperature_c: float = 20.0) -> float:
    """Approximate speed of sound in air (m/s)."""
    return 331.3 + 0.606 * float(temperature_c)


_CRAMER = (331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782, -1.82e-7, 3.73e-8,
           -2.93e-10, -85.20931, -0.228525, 5.91e-5, -2.835149, -2.15e-13, 29.179762, 0.000486)


def speed_of_sound_cramer(temperature_c: float = 20.0, humidity_percent: float = 50.0,
                          pressure_kpa: float = 101.325, co2_fraction: float = 0.0004) -> float:
    """Speed of sound in humid air (m/s), Cramer (JASA 93, 1993), the water-vapour mole
    fraction from Davis's saturation pressure and enhancement factor (1992). BRAS RS6
    (19.9 C, 40.1 %): 343.80 against 343.36 for the dry formula."""
    a = _CRAMER
    T = float(temperature_c)
    TK = T + 273.15
    p = float(pressure_kpa) * 1e3
    psv = np.exp(1.2378847e-5 * TK ** 2 - 1.9121316e-2 * TK + 33.93711047 - 6.3431645e3 / TK)
    f = 1.00062 + 3.14e-8 * p + 5.6e-7 * T ** 2
    xw = float(np.clip(humidity_percent, 0.0, 100.0)) / 100.0 * f * psv / p
    xc = float(co2_fraction)
    return float(a[0] + a[1] * T + a[2] * T ** 2 + (a[3] + a[4] * T + a[5] * T ** 2) * xw
                 + (a[6] + a[7] * T + a[8] * T ** 2) * p + (a[9] + a[10] * T + a[11] * T ** 2) * xc
                 + a[12] * xw ** 2 + a[13] * p ** 2 + a[14] * xc ** 2 + a[15] * xw * p * xc)


def config_speed_of_sound(cfg) -> float:
    """The speed of sound of a BakeConfig: with cfg.air_cramer (the BakeConfig default),
    Cramer's humid-air value at its temperature and humidity, else the dry 331.3 + 0.606 T."""
    if getattr(cfg, "air_cramer", False):
        return speed_of_sound_cramer(cfg.temperature_c, getattr(cfg, "humidity_percent", 50.0))
    return speed_of_sound(cfg.temperature_c)
