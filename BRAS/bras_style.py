#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared visual identity for the BRAS figures (CR2, RS1, ...).

One palette and one typeface, whatever the plotting script."""

TEAL = "#0e9594"   # Auralis (simulation)
INK = "#222222"    # measurement (reference)
BAND = "#bbbbbb"   # +/-1 JND / uncertainty band
FAINT = "#999999"  # individual pairs / thin lines


def rc():
    """Matplotlib rcParams common to every BRAS figure."""
    return {
        "axes.edgecolor": "#888888",
        "axes.grid": True,
        "grid.color": "#e9e9e9",
        "font.size": 10,
    }


def color_for(label):
    """Canonical curve color from its label: Auralis in TEAL, everything
    else (measurement/reference) in INK."""
    return TEAL if "auralis" in label.lower() else INK
