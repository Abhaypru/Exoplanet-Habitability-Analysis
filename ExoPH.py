#!/usr/bin/env python3
# =============================================================================
#  EXOPLANET HABITABILITY ANALYSIS
#  Plot Generation Script
#  -----------------------------------------------------------------------------
#  Project:
#      A Comprehensive Analysis of Exoplanet Habitability using
#      the NASA Exoplanet Archive (May 2026 dataset)
#
#  Authors:
#      Abhay Kumar Prusty, Jeeban Jyoti, Koushik Singha
#
#  Description:
#      This script generates all figures used in the analysis of
#      exoplanet habitability, including population statistics,
#      habitable zone selection, and PHC identification.
#
#  Data Source:
#      NASA Exoplanet Archive
#      https://exoplanetarchive.ipac.caltech.edu
#      Retrieved: 1 May 2026
#
#  Input Files (place in DATA_DIR):
#      1. PSCompPars_*.csv        — Planetary Systems Composite Parameters
#      2. PS_*.csv               — Planetary Systems (all solutions)
#      3. TD_*.csv               — Transit Detections
#      4. STELLARHOSTS_*.csv     — Stellar Host Catalogue
#      5. ML_*.csv               — Microlensing Planets
#      6. directimaging_*.csv    — Direct Imaging Planets
#
#  Output:
#      Publication-style figures saved to OUTPUT_DIR
#
#  Usage:
#      python exoplanet_habitability_plots.py
#
#  Requirements:
#      Python >= 3.10
#      numpy, pandas, matplotlib, scipy
#
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import warnings
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# =============================================================================
#  USER CONFIGURATION
# =============================================================================
DATA_DIR   = Path(".")          # ← folder containing the six CSV files
OUTPUT_DIR = Path("figures")    # ← all PNGs will be written here
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI      = 180                  # figure resolution (180 = publication quality)
FMT      = "png"                # "png" | "pdf" | "svg"
FONTSIZE = 11                   # base font size

# =============================================================================
#  GLOBAL MATPLOTLIB STYLE
# =============================================================================
plt.rcParams.update({
    # ── Resolution ───────────────────────────────
    "figure.dpi":        300,
    "savefig.dpi":       300,

    # ── Fonts ───────────────────────────────────
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.titleweight":  "bold",

    # ── Lines & axes ────────────────────────────
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.8,

    # ── Ticks ───────────────────────────────────
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.minor.size":  2,
    "ytick.minor.size":  2,

    # ── Spines ──────────────────────────────────
    "axes.spines.top":   False,
    "axes.spines.right": False,

    # ── Grid (lighter, less distracting) ────────
    "axes.grid":         True,
    "grid.alpha":        0.15,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,

    # ── Legend ──────────────────────────────────
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "legend.fancybox":   False,
    "legend.edgecolor":  "0.8",

    # ── Layout ──────────────────────────────────
    "figure.constrained_layout.use": False,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})
# =============================================================================
#  COLOUR PALETTE  (consistent across all figures)
# =============================================================================
COLOURS = {
    "transit":   "#2196F3",
    "rv":        "#FF9800",
    "ml":        "#4CAF50",
    "imaging":   "#9C27B0",
    "ttv":       "#F44336",
    "etv":       "#00BCD4",
    "other":     "#607D8B",
    "rocky":     "#27AE60",
    "superearth":"#F39C12",
    "subneptune":"#2980B9",
    "neptune":   "#8E44AD",
    "giant":     "#C0392B",
    "phc":       "#E53935",
    "earth":     "#1565C0",
    "hz_fill":   "#A5D6A7",
    "gap_fill":  "#EF9A9A",
}

METHOD_COLOURS = {
    "Transit":                    COLOURS["transit"],
    "Radial Velocity":            COLOURS["rv"],
    "Microlensing":               COLOURS["ml"],
    "Imaging":                    COLOURS["imaging"],
    "Transit Timing Variations":  COLOURS["ttv"],
    "Eclipse Timing Variations":  COLOURS["etv"],
    "Astrometry":                 "#795548",
    "Pulsar Timing":              "#607D8B",
    "Orbital Brightness Modulation": "#E91E63",
}

# =============================================================================
#  CONSTANTS & HABITABILITY THRESHOLDS
# =============================================================================
#S_star      = # is the incident solar radiation (shortwave radiation) on an object, which can be measured at the top of an atmosphere or on the surface.
HZ_S_INNER  = 1.80   # S_Earth  — conservative inner edge (moist greenhouse)
HZ_S_OUTER  = 0.20   # S_Earth  — conservative outer edge (max greenhouse)
PHC_R_MIN   = 0.50   # R_Earth # Potentially Habitable candidate (PHC)
PHC_R_MAX   = 1.50   # R_Earth
LW_T_MIN    = 200    # K   — liquid-water equilibrium temperature lower bound
LW_T_MAX    = 370    # K   — liquid-water equilibrium temperature upper bound
EARTH_INSOL = 1.00   # S_Earth
EARTH_TEQ   = 255    # K
EARTH_RAD   = 1.00   # R_Earth
SOL_TEFF    = 5778   # K
MJUP_MEARTH = 317.83 # M_Earth per M_Jupiter
RJUP_REARTH = 11.21  # R_Earth per R_Jupiter
AU_SNOWLINE  = 2.0   # AU — approximate snow line for 0.5 M_Sun host

# =============================================================================
#  HELPER UTILITIES
# =============================================================================

def save(fig: plt.Figure, name: str) -> None:
    """Save figure to OUTPUT_DIR and close it."""
    path = OUTPUT_DIR / f"{name}.{FMT}"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved]  {path}")


def phc_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: conservative HZ + Earth-analogue radius."""
    return (
        (df["pl_insol"] >= HZ_S_OUTER)
        & (df["pl_insol"] <= HZ_S_INNER)
        & (df["pl_rade"]  >= PHC_R_MIN)
        & (df["pl_rade"]  <= PHC_R_MAX)
    )


def planet_class(r: float) -> str:
    """Classify planet by radius in R_Earth."""
    if   r < 1.5:  return "Rocky/Terrestrial"
    elif r < 2.5:  return "Super-Earth"
    elif r < 4.0:  return "Sub-Neptune"
    elif r < 8.0:  return "Neptune-class"
    else:          return "Gas Giant"


def kopparapu_hz(teff_arr: np.ndarray) -> dict:
    """
    Return Kopparapu et al. (2013) HZ insolation boundary curves.
    Returns dict with keys: 'runaway', 'moist_gh', 'max_gh', 'early_mars'
    evaluated at each T_eff in teff_arr.
    """
    T = teff_arr - 5780.0
    coeffs = {
        "runaway":    (1.1076,  1.0026e-4,  5.765e-9,  -1.5218e-11, -1.1986e-14),# runaway → inner edge (planet loses oceans quickly)
        "moist_gh":   (1.0140,  8.1774e-5,  1.7063e-9, -4.3241e-12, -6.6462e-16), #moist_gh → inner edge (water vapor dominates stratosphere)
        "max_gh":     (0.3438,  5.8942e-5,  1.6558e-9, -3.0045e-12, -5.2983e-16), #max_gh → outer edge (CO₂ greenhouse limit)
        "early_mars": (0.3179,  5.4513e-5,  1.5313e-9, -2.7786e-12, -4.8997e-16), #early_mars → optimistic outer edge
    }
    out = {}
    for name, (s0, a1, a2, a3, a4) in coeffs.items():
        out[name] = s0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4
    return out


# =============================================================================
#  DATA LOADING
# =============================================================================

def load_data(data_dir: Path) -> dict:
    files = {
        "comp":   "PSCompPars_2026.05.01_19.47.58.csv",
        "ps":     "PS_2026.05.01_19.38.07.csv",
        "td":     "TD_2026.05.01_21.21.51.csv",
        "stars":  "STELLARHOSTS_2026.05.01_21.21.19.csv",
        "ml":     "ML_2026.05.01_21.21.23.csv",
        "di":     "directimaging_2026.05.01_21.21.41.csv",
    }

    data = {}
    print("\n── Loading NASA Exoplanet Archive data ─────────────────────────────")

    for key, fname in files.items():
        path = data_dir / fname

        if not path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Missing file: {path}\n"
                f"Place all CSVs in: {data_dir.resolve()}\n"
            )

        df = pd.read_csv(
            path,
            comment="#",
            low_memory=False,
            na_values=["", " ", "null", "NULL"]
        )

        data[key] = df

        mem = df.memory_usage(deep=True).sum() / 1e6
        print(f"  {key:8s} → {path.name} ({len(df):,} rows × {df.shape[1]} cols")

    print()
    return data

# =============================================================================
# =============================================================================
#  FIGURE FUNCTIONS  (Fig 01 – Fig 25)
# =============================================================================
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 01  Discovery Method Breakdown + Annual Discovery Rate
# ─────────────────────────────────────────────────────────────────────────────
def fig01_discovery_overview(df: pd.DataFrame) -> None:
    """Pie chart of discovery methods and bar chart of annual discoveries."""
    fig, axes = plt.subplots(1, 2, figsize=(10,6))
    fig.suptitle("Exoplanet Discovery Overview  (N = 6,278)", fontsize=17, fontweight="bold")


    methods = df["discoverymethod"].value_counts()

    threshold = 0.01 * len(df)
    small = methods[methods < threshold]
    large = methods[methods >= threshold]

    methods_clean = large.copy()
    methods_clean["Other"] = small.sum()
    methods_clean = methods_clean.sort_values(ascending=False)

    wedges, _, autotexts = axes[0].pie(
        methods_clean.values,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
    )

    axes[0].legend(
        wedges,
        methods_clean.index,
        title="Detection Method",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=11
    )

    axes[0].set_title("Confirmed Planets by Detection Method", fontsize=13)



    # ── Annual bar chart ──
    year_counts = df.groupby("disc_year").size()
    axes[1].bar(
        year_counts.index, year_counts.values,
        color=COLOURS["transit"], edgecolor="navy", alpha=0.82, width=0.85,
    )
    axes[1].axvline(2009, color="#E53935", lw=2, ls="--", alpha=0.85, label="Kepler launch (2009)")
    axes[1].axvline(2018, color="#2E7D32", lw=2, ls="--", alpha=0.85, label="TESS launch (2018)")
    axes[1].axvline(2022, color="navy",    lw=2, ls=":",  alpha=0.75, label="JWST launch (2022)")
    axes[1].set_xlabel("Discovery Year", fontsize=12)
    axes[1].set_ylabel("Planets Discovered per Year", fontsize=15)
    axes[1].set_title("Annual Exoplanet Discovery Rate", fontsize=15)
    axes[1].legend(fontsize=9.5)
    axes[1].set_xlim(1991, 2027)

    plt.tight_layout()
    save(fig, "fig01_discovery_overview")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 02  Cumulative Discovery Curve by Detection Method
# ─────────────────────────────────────────────────────────────────────────────
def fig02_cumulative_discovery(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    years = np.arange(1992, 2027)

    for method, colour in METHOD_COLOURS.items():
        sub = df[df["discoverymethod"] == method]["disc_year"].dropna().astype(int)

        if len(sub) < 20:
            continue

        counts = sub.value_counts().sort_index()
        counts = counts.reindex(years, fill_value=0)

        cumul = counts.cumsum().replace(0, np.nan)

        ax.plot(
            years, cumul,
            color=colour,
            lw=2.5,
            label=f"{method} ({len(sub):,})",
            alpha=0.9
        )

    # Mission markers
    ax.axvline(2009, color="#E53935", lw=2, ls="--", alpha=0.85, label="Kepler launch (2009)")
    ax.axvline(2018, color="#2E7D32", lw=2, ls="--", alpha=0.85, label="TESS launch (2018)")
    ax.axvline(2022, color="navy",    lw=2, ls=":",  alpha=0.75, label="JWST launch (2022)")
    ax.set_yscale("log")
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel("Cumulative Confirmed Planets", fontsize=15)
    ax.set_title(
        "Cumulative Exoplanet Discovery by Detection Method (1992–2026)",
        fontsize=20, fontweight="bold"
    )

    ax.set_xlim(1992, 2026)
    ax.legend(loc="upper left", ncol=2, fontsize=13)

    plt.tight_layout()
    save(fig, "fig02_cumulative_discovery")

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 03  Mass–Radius Diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig03_mass_radius(df: pd.DataFrame) -> None:
    """Log–log mass–radius diagram colour-coded by detection method."""
    fig, ax = plt.subplots(figsize=(11, 8))

    v = df.dropna(subset=["pl_bmasse", "pl_rade"])
    v = v[(v["pl_bmasse"] > 0) & (v["pl_rade"] > 0) & (v["pl_rade"] < 26) & (v["pl_bmasse"] < 1.2e4)]

    for method, colour in METHOD_COLOURS.items():
        sub = v[v["discoverymethod"] == method]
        ax.scatter(
            sub["pl_bmasse"], sub["pl_rade"],
            c=colour, alpha=0.50, s=14,
            label=f"{method}  (n={len(sub)})", zorder=3,
        )

    # Regime shading
    ax.axhspan(PHC_R_MIN, PHC_R_MAX, alpha=0.08, color="green",  zorder=1, label="Earth-like Radius Zone")
    ax.axhspan(1.50, 2.00,           alpha=0.10, color="red",    zorder=1, label="Fulton Gap (1.5–2.0 R⊕)")

    # Reference lines
    for m, lbl in [(1.0, "1 M⊕"), (MJUP_MEARTH, "1 M$_{Jup}$"), (13 * MJUP_MEARTH, "13 M$_{Jup}$ (BD limit)")]:
        ax.axvline(m, color="gray", lw=1.2, ls=":", alpha=0.7)
        ax.text(m * 1.05, 0.38, lbl, fontsize=7.5, color="gray", va="bottom")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Planet Mass (M$_\\oplus$)", fontsize=15)
    ax.set_ylabel("Planet Radius (R$_\\oplus$)", fontsize=15)
    ax.set_title("Mass–Radius Diagram of Confirmed Exoplanets (log-log)", fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", ncol=2, fontsize=9.5)
    ax.set_xlim(0.02, 1.5e4); ax.set_ylim(0.3, 26)
    plt.tight_layout()
    save(fig, "fig03_mass_radius")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 04  Planet Classification Map (Radius vs Insolation)
# ─────────────────────────────────────────────────────────────────────────────
def fig04_classification_map(df: pd.DataFrame) -> None:
    """Radius vs insolation flux with improved clarity and physics."""

    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 8))

    # ─────────────────────────────────────────────
    # DATA CLEANING
    # ─────────────────────────────────────────────
    v = df.dropna(subset=["pl_rade", "pl_insol"])
    v = v[
        (v["pl_rade"] > 0.3) & (v["pl_rade"] < 26) &
        (v["pl_insol"] > 0) & (v["pl_insol"] < 1e5)
    ].copy()

    v["class"] = v["pl_rade"].apply(planet_class)

    # Slight jitter to remove horizontal banding
    v["rade_jitter"] = v["pl_rade"] * np.random.normal(1.0, 0.03, len(v))

    # ─────────────────────────────────────────────
    # CLASS COLORS
    # ─────────────────────────────────────────────
    class_colours = {
        "Rocky/Terrestrial": COLOURS["rocky"],
        "Super-Earth":       COLOURS["superearth"],
        "Sub-Neptune":       COLOURS["subneptune"],
        "Neptune-class":     COLOURS["neptune"],
        "Gas Giant":         COLOURS["giant"],
    }

    # ─────────────────────────────────────────────
    # SCATTER PLOT (cleaner)
    # ─────────────────────────────────────────────
    for cls, colour in class_colours.items():
        s = v[v["class"] == cls]

        # shuffle to avoid visual streaks
        s = s.sample(frac=1, random_state=42)

        ax.scatter(
            s["pl_insol"], s["rade_jitter"],
            c=colour,
            alpha=0.28,
            s=9,
            label=f"{cls}  (n={len(s):,})",
            zorder=3,
            rasterized=True
        )

    # ─────────────────────────────────────────────
    # HABITABLE ZONE (fixed)
    # ─────────────────────────────────────────────
    hz_min = min(HZ_S_OUTER, HZ_S_INNER)
    hz_max = max(HZ_S_OUTER, HZ_S_INNER)

    ax.axvspan(
        hz_min, hz_max,
        alpha=0.10,
        color="cyan",
        zorder=1,
        label="Conservative HZ"
    )

    # ─────────────────────────────────────────────
    # FULTON GAP
    # ─────────────────────────────────────────────
    ax.axhspan(
        1.5, 2.0,
        alpha=0.15,
        color="red",
        zorder=1,
        label="Fulton Gap"
    )

    # ─────────────────────────────────────────────
    # EARTH REFERENCE
    # ─────────────────────────────────────────────
    ax.axvline(
        EARTH_INSOL,
        color="gold",
        lw=2,
        ls="--",
        alpha=0.9,
        label="Earth Insolation"
    )

    ax.axhline(
        EARTH_RAD,
        color="green",
        lw=1.5,
        ls=":",
        alpha=0.8,
        label="Earth Radius"
    )

    # ─────────────────────────────────────────────
    # PHC OVERLAY
    # ─────────────────────────────────────────────
    phc = df[phc_mask(df)]

    ax.scatter(
        phc["pl_insol"],
        phc["pl_rade"],
        c="black",
        s=80,
        marker="*",
        label=f"PHC candidates (n={len(phc)})",
        zorder=7
    )

    # ─────────────────────────────────────────────
    # AXES & STYLE
    # ─────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Insolation Flux (S$_\\oplus$)", fontsize=15)
    ax.set_ylabel("Planet Radius (R$_\\oplus$)", fontsize=15)

    ax.set_title(
        "Planet Classification Map: Radius vs Insolation Flux",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlim(0.04, 3e4)
    ax.set_ylim(0.28, 26)

    ax.legend(loc="upper right", ncol=2, fontsize=9.5)

    plt.tight_layout()
    save(fig, "fig04_classification_map")
# ─────────────────────────────────────────────────────────────────────────────
#  FIG 05  Habitable Zone: Insolation vs Planet Radius
# ─────────────────────────────────────────────────────────────────────────────
def fig05_hz_insolation(df: pd.DataFrame) -> None:
    """Habitable Zone diagnostic: insolation vs radius with clean highlighting."""

    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    # ─────────────────────────────────────────────
    # CLEAN DATA
    # ─────────────────────────────────────────────
    v = df.dropna(subset=["pl_insol", "pl_rade"])
    v = v[
        (v["pl_insol"] > 0) &
        (v["pl_rade"] > 0) &
        (v["pl_rade"] < 16) &
        (v["pl_insol"] < 100)
    ].copy()

    # shuffle to avoid visual streaks
    v = v.sample(frac=1, random_state=42)

    # ─────────────────────────────────────────────
    # ALL PLANETS (background)
    # ─────────────────────────────────────────────
    ax.scatter(
        v["pl_insol"],
        v["pl_rade"],
        c="silver",
        alpha=0.25,
        s=7,
        zorder=2,
        label="All planets",
        rasterized=True
    )

    # ─────────────────────────────────────────────
    # PHC CANDIDATES
    # ─────────────────────────────────────────────
    phc = df[phc_mask(df)].dropna(subset=["pl_insol", "pl_rade"])

    # apply same limits
    phc = phc[
        (phc["pl_insol"] > 0) &
        (phc["pl_rade"] > 0) &
        (phc["pl_rade"] < 16) &
        (phc["pl_insol"] < 100)
    ]

    ax.scatter(
        phc["pl_insol"],
        phc["pl_rade"],
        c=COLOURS["phc"],
        s=80,
        edgecolors="darkred",
        linewidth=0.8,
        zorder=6,
        label=f"PHC candidates (n={len(phc)})"
    )

    # ─────────────────────────────────────────────
    # HABITABLE ZONE (fixed ordering)
    # ─────────────────────────────────────────────
    hz_min = min(HZ_S_OUTER, HZ_S_INNER)
    hz_max = max(HZ_S_OUTER, HZ_S_INNER)

    ax.axvspan(
        hz_min,
        hz_max,
        alpha=0.14,
        color="#66BB6A",
        zorder=1,
        label="Conservative HZ"
    )

    # ─────────────────────────────────────────────
    # EARTH-LIKE RADIUS BAND
    # ─────────────────────────────────────────────
    ax.axhspan(
        PHC_R_MIN,
        PHC_R_MAX,
        alpha=0.10,
        color="#42A5F5",
        zorder=1,
        label="Earth-like Radius"
    )

    # ─────────────────────────────────────────────
    # EARTH REFERENCE
    # ─────────────────────────────────────────────
    ax.axvline(
        EARTH_INSOL,
        color="gold",
        lw=2.0,
        ls="--",
        alpha=0.9,
        label="Earth Insolation"
    )

    # ─────────────────────────────────────────────
    # LABEL IMPORTANT PLANETS (clean offsets)
    # ─────────────────────────────────────────────
    labels = {
        "Proxima Cen b", "TRAPPIST-1 e", "TRAPPIST-1 f", "TRAPPIST-1 g",
        "TOI-700 d", "Wolf 1069 b", "Kepler-186 f",
        "Kepler-442 b", "Gliese 12 b",
    }

    for _, row in phc.iterrows():
        if row["pl_name"] in labels:
            ax.annotate(
                row["pl_name"],
                (row["pl_insol"], row["pl_rade"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                color="#B71C1C",
                fontweight="bold"
            )



    # ─────────────────────────────────────────────
    # AXES
    # ─────────────────────────────────────────────
    ax.set_xscale("log")

    ax.set_xlabel("Insolation Flux (S$_\\oplus$)", fontsize=15)
    ax.set_ylabel("Planet Radius (R$_\\oplus$)", fontsize=15)

    ax.set_title(
        "Habitable Zone Analysis: Insolation Flux vs Planet Radius",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlim(0.05, 80)
    ax.set_ylim(0, 12)

    ax.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    save(fig, "fig05_hz_insolation")
# ─────────────────────────────────────────────────────────────────────────────
#  FIG 06  Kopparapu HZ Boundaries (Teff vs Insolation)
# ─────────────────────────────────────────────────────────────────────────────
def fig06_kopparapu_hz(df: pd.DataFrame) -> None:
    """Kopparapu 2013 HZ curves overlaid on the confirmed planet population."""
    fig, ax = plt.subplots(figsize=(13, 8))

    teff_arr = np.linspace(2600, 7200, 400)
    hz = kopparapu_hz(teff_arr)

    ax.fill_between(teff_arr, hz["max_gh"],  hz["moist_gh"], alpha=0.22, color="#66BB6A", label="Conservative HZ")
    ax.fill_between(teff_arr, hz["moist_gh"],hz["runaway"],  alpha=0.10, color="#FFF176", label="Optimistic extension (inner)")
    ax.fill_between(teff_arr, hz["early_mars"], hz["max_gh"],alpha=0.10, color="#80CBC4", label="Optimistic extension (outer)")

    ax.plot(teff_arr, hz["moist_gh"],   color="#2E7D32", lw=2.2, ls="-",  label="Moist greenhouse limit")
    ax.plot(teff_arr, hz["max_gh"],     color="#1565C0", lw=2.2, ls="-",  label="Max greenhouse limit")
    ax.plot(teff_arr, hz["runaway"],    color="#E65100", lw=1.8, ls="--", label="Runaway greenhouse")
    ax.plot(teff_arr, hz["early_mars"], color="#6A1B9A", lw=1.8, ls="--", label="Early Mars limit")

    v = df.dropna(subset=["st_teff", "pl_insol"])
    v = v[(v["st_teff"] > 2600) & (v["st_teff"] < 7200) & (v["pl_insol"] > 0) & (v["pl_insol"] < 5)]
    ax.scatter(v["st_teff"], v["pl_insol"], c="silver", alpha=0.18, s=8, zorder=2)

    phc = df[phc_mask(df)].dropna(subset=["st_teff", "pl_insol"])
    ax.scatter(phc["st_teff"], phc["pl_insol"], c=COLOURS["phc"], s=90, zorder=7,
               marker="*", edgecolors="darkred", lw=0.8, label=f"PHC candidates  (n={len(phc)})")

    for _, row in phc[phc["hostname"] == "TRAPPIST-1"].iterrows():
        suffix = row["pl_name"].split("-1")[1] if "-1" in row["pl_name"] else row["pl_name"]
        ax.annotate(suffix, (row["st_teff"], row["pl_insol"]),
                    fontsize=7, color="#B71C1C", xytext=(4, 4), textcoords="offset points")

    ax.axvline(SOL_TEFF, color="gold", lw=2.0, ls="--", alpha=0.85, label="Sun T$_{eff}$ (5778 K)")
    ax.axhline(1.0, color="gold", lw=1.2, ls=":", alpha=0.6)

    ax.set_xlabel("Stellar Effective Temperature T$_{eff}$ (K)", fontsize=13)
    ax.set_ylabel("Insolation Flux (S$_\\oplus$)", fontsize=13)
    ax.set_title("Kopparapu (2013) HZ Boundaries with Confirmed Planet Overlay", fontsize=14, fontweight="bold")
    ax.set_xlim(2600, 7200); ax.set_ylim(0, 4.0)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    plt.tight_layout()
    save(fig, "fig06_kopparapu_hz")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 07  Equilibrium Temperature Distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig07_thermal_properties(df: pd.DataFrame) -> None:
    """Equilibrium temperature histogram and insolation distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Planetary Thermal Environment", fontsize=17, fontweight="bold")

    t = df["pl_eqt"].dropna()
    t = t[(t > 50) & (t < 4000)]
    axes[0].hist(t, bins=80, color=COLOURS["phc"], edgecolor="darkred", alpha=0.82)
    axes[0].axvspan(LW_T_MIN, LW_T_MAX, alpha=0.22, color="cyan", label=f"Liquid-water window ({LW_T_MIN}–{LW_T_MAX} K)")
    axes[0].axvline(EARTH_TEQ, color=COLOURS["earth"], lw=2.2, ls="--", label=f"Earth T$_{{eq}}$ ({EARTH_TEQ} K)")
    axes[0].set_xlabel("Equilibrium Temperature T$_{eq}$ (K)", fontsize=14)
    axes[0].set_ylabel("Number of Planets", fontsize=14)
    axes[0].set_title("Equilibrium Temperature Distribution", fontsize=14)
    axes[0].legend(fontsize=10)

    s = df["pl_insol"].dropna()
    s = s[(s > 0) & (s < 200)]
    axes[1].hist(np.log10(s), bins=80, color=COLOURS["rv"], edgecolor="darkorange", alpha=0.82)
    axes[1].axvspan(np.log10(HZ_S_OUTER), np.log10(HZ_S_INNER),
                    alpha=0.22, color="#66BB6A", label="Conservative HZ (0.2–1.8 S⊕)")
    axes[1].axvline(np.log10(EARTH_INSOL), color="gold", lw=2.2, ls="--", label="Earth insolation")
    axes[1].set_xlabel("log$_{10}$(Insolation Flux  /  S$_\\oplus$)", fontsize=14)
    axes[1].set_ylabel("Number of Planets", fontsize=14)
    axes[1].set_title("Insolation Flux Distribution", fontsize=14)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    save(fig, "fig07_thermal_properties")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 08  Equilibrium Temperature vs Orbital Period
# ─────────────────────────────────────────────────────────────────────────────
def fig08_teq_vs_period(df: pd.DataFrame) -> None:
    """T_eq vs orbital period, coloured by stellar T_eff."""
    fig, ax = plt.subplots(figsize=(12, 8))

    v = df.dropna(subset=["pl_orbper", "pl_eqt", "st_teff"])
    v = v[(v["pl_orbper"] > 0.5) & (v["pl_orbper"] < 1000)
          & (v["pl_eqt"] > 50) & (v["pl_eqt"] < 5000)]

    sc = ax.scatter(
        v["pl_orbper"], v["pl_eqt"], c=v["st_teff"],
        cmap="RdYlBu", vmin=3000, vmax=7000,
        alpha=0.40, s=12, zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="Stellar T$_{eff}$ (K)")

    ax.axhspan(LW_T_MIN, LW_T_MAX, alpha=0.15, color="cyan", label=f"Liquid-water T$_{{eq}}$ window ({LW_T_MIN}–{LW_T_MAX} K)")
    ax.axvline(365.25, color="gold", lw=2.2, ls="--", alpha=0.85, label="1-year period")
    ax.axhline(EARTH_TEQ, color=COLOURS["earth"], lw=1.5, ls=":", alpha=0.7, label=f"Earth T$_{{eq}}$ = {EARTH_TEQ} K")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Orbital Period (days)", fontsize=16)
    ax.set_ylabel("Equilibrium Temperature T$_{eq}$ (K)", fontsize=16)
    ax.set_title("Equilibrium Temperature vs Orbital Period\n(coloured by stellar T$_{eff}$)",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, "fig08_teq_vs_period")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 09  Stellar HR Diagram with Planet Radius Colour Coding
# ─────────────────────────────────────────────────────────────────────────────
def fig09_stellar_hr(df: pd.DataFrame) -> None:
    """HR diagram of planet hosts, colour = planet radius; PHC hosts highlighted."""
    fig, ax = plt.subplots(figsize=(12, 8))

    v = df.dropna(subset=["st_teff", "st_rad", "pl_rade"])
    v = v[(v["st_teff"] > 2000) & (v["st_teff"] < 12000)
          & (v["st_rad"] > 0)   & (v["st_rad"] < 10)
          & (v["pl_rade"] > 0)]

    sc = ax.scatter(
        v["st_teff"], v["st_rad"], c=v["pl_rade"],
        cmap="plasma", norm=LogNorm(vmin=0.5, vmax=20),
        s=v["pl_rade"] * 4 + 3, alpha=0.42, zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="Planet Radius (R$_\\oplus$)")

    phc = df[phc_mask(df)].dropna(subset=["st_teff", "st_rad"])
    ax.scatter(phc["st_teff"], phc["st_rad"], c="cyan", s=140, zorder=7,
               marker="D", edgecolors="navy", linewidth=1.4, label="PHC host stars")

    for teff, lbl, colour in [
        (2700, "M dwarfs\n(2700–4000 K)",  "#FFCDD2"),
        (4000, "K dwarfs\n(4000–5300 K)",  "#FFE0B2"),
        (5300, "G dwarfs\n(5300–6000 K)",  "#FFF9C4"),
        (6000, "F dwarfs\n(6000–7500 K)",  "#E3F2FD"),
    ]:
        pass  # bands already implied by scatter colouring

    ax.axvspan(2700, 4000, alpha=0.07, color="red")
    ax.axvspan(4000, 5300, alpha=0.07, color="orange")
    ax.axvspan(5300, 6000, alpha=0.07, color="yellow")
    ax.axvspan(6000, 7500, alpha=0.06, color="lightblue")
    ax.axvline(SOL_TEFF, color="orange", lw=2.0, ls="--", alpha=0.85, label="Sun T$_{eff}$ (5778 K)")

    ax.set_xlim(8000, 2000)   # reverse: hot → cool
    ax.set_ylim(0, 8)
    ax.set_xlabel("Stellar Effective Temperature T$_{eff}$ (K)", fontsize=15)
    ax.set_ylabel("Stellar Radius (R$_\\odot$)", fontsize=15)
    ax.set_title("HR Diagram of Exoplanet Host Stars\n(colour = planet radius)", fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    save(fig, "fig09_stellar_hr")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 10  Host Star Temperature and Mass Distributions
# ─────────────────────────────────────────────────────────────────────────────
def fig10_stellar_properties(df: pd.DataFrame) -> None:
    """Stellar T_eff and mass histograms for the confirmed-planet host sample."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Host Star Properties", fontsize=15, fontweight="bold")

    t = df["st_teff"].dropna()
    t = t[(t > 1000) & (t < 12000)]
    axes[0].hist(t, bins=70, color="#9C27B0", edgecolor="#4A148C", alpha=0.82)
    for lo, hi, colour, lbl in [
        (2700, 4000, "#EF9A9A", "M dwarf"), (4000, 5300, "#FFCC80", "K dwarf"),
        (5300, 6000, "#FFF59D", "G dwarf"), (6000, 7500, "#B3E5FC", "F dwarf"),
    ]:
        axes[0].axvspan(lo, hi, alpha=0.20, color=colour, label=f"{lbl} ({lo}–{hi} K)")
    axes[0].axvline(SOL_TEFF, color="orange", lw=2.2, ls="--", label=f"Sun ({SOL_TEFF} K)")
    axes[0].set_xlabel("Stellar Effective Temperature T$_{eff}$ (K)", fontsize=12)
    axes[0].set_ylabel("Number of Systems", fontsize=12)
    axes[0].set_title("Host Star Temperature Distribution", fontsize=12)
    axes[0].legend(fontsize=8, ncol=2)

    m = df["st_mass"].dropna()
    m = m[(m > 0) & (m < 3)]
    axes[1].hist(m, bins=65, color=COLOURS["transit"], edgecolor="navy", alpha=0.82)
    axes[1].axvline(1.0, color="orange", lw=2.2, ls="--", label="1.0 M$_\\odot$ (Solar)")
    axes[1].set_xlabel("Stellar Mass (M$_\\odot$)", fontsize=12)
    axes[1].set_ylabel("Number of Systems", fontsize=12)
    axes[1].set_title("Host Star Mass Distribution", fontsize=12)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save(fig, "fig10_stellar_properties")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 11  Planet Radius Distribution — Fulton Gap
# ─────────────────────────────────────────────────────────────────────────────
def fig11_fulton_gap(df: pd.DataFrame) -> None:
    """Radius histogram showing the Fulton bimodality."""
    fig, ax = plt.subplots(figsize=(11, 6))

    r = df["pl_rade"].dropna()
    r = r[(r > 0.3) & (r < 10)]
    ax.hist(r, bins=100, color="#00BCD4", edgecolor="#006064", alpha=0.85)

    ax.axvspan(1.5, 2.0, alpha=0.30, color=COLOURS["phc"], label="Fulton Radius Gap (1.5–2.0 R$_\\oplus$)")
    ax.axvline(EARTH_RAD, color="#2E7D32", lw=2.2, ls="--", alpha=0.85, label="Earth radius (1.0 R$_\\oplus$)")
    ax.axvline(3.865,      color="navy",    lw=2.0, ls="--", alpha=0.75, label="Neptune radius (3.87 R$_\\oplus$)")

    ax.set_xlabel("Planet Radius (R$_\\oplus$)", fontsize=13)
    ax.set_ylabel("Number of Planets", fontsize=13)
    ax.set_title("Planet Radius Distribution — Fulton Radius Gap", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0.3, 10)
    plt.tight_layout()
    save(fig, "fig11_fulton_gap")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 12  Orbital Period vs Planet Radius
# ─────────────────────────────────────────────────────────────────────────────
def fig12_period_radius(df: pd.DataFrame) -> None:
    """Period–radius diagram coloured by log(radius)."""
    fig, ax = plt.subplots(figsize=(11, 8))

    v = df.dropna(subset=["pl_orbper", "pl_rade"])
    v = v[(v["pl_orbper"] > 0) & (v["pl_orbper"] < 1000) & (v["pl_rade"] > 0) & (v["pl_rade"] < 20)]

    sc = ax.scatter(
        v["pl_orbper"], v["pl_rade"],
        c=np.log10(v["pl_rade"]), cmap="viridis",
        alpha=0.40, s=10, zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="log$_{10}$(R$_\\oplus$)")

    ax.axvline(365.25, color="gold", lw=2.2, ls="--", alpha=0.85, label="1-year period")
    ax.axhspan(PHC_R_MIN, PHC_R_MAX, alpha=0.12, color="cyan", label="Earth-like Radius Zone (0.5–1.5 R$_\\oplus$)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Orbital Period (days)", fontsize=13)
    ax.set_ylabel("Planet Radius (R$_\\oplus$)", fontsize=13)
    ax.set_title("Orbital Period vs Planet Radius", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, "fig12_period_radius")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 13  Metallicity — Giant vs Terrestrial + Planet Multiplicity
# ─────────────────────────────────────────────────────────────────────────────
def fig13_metallicity_multiplicity(df: pd.DataFrame) -> None:
    """Metallicity distributions by planet class and system multiplicity histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Stellar Metallicity & Planet System Architecture", fontsize=15, fontweight="bold")

    v = df.dropna(subset=["st_met", "pl_rade"])
    v = v[(v["pl_rade"] > 0) & (v["pl_rade"] < 22)]
    giants = v[v["pl_rade"] >= 6]
    terres = v[v["pl_rade"] <  2]

    axes[0].hist(terres["st_met"], bins=45, alpha=0.72, color=COLOURS["rocky"],
                 label=f"Terrestrial (R < 2 R$_\\oplus$, n={len(terres):,})", density=True)
    axes[0].hist(giants["st_met"], bins=45, alpha=0.72, color=COLOURS["transit"],
                 label=f"Giant (R ≥ 6 R$_\\oplus$, n={len(giants):,})", density=True)
    axes[0].axvline(0.0, color=COLOURS["phc"], lw=2.2, ls="--", label="Solar metallicity ([Fe/H] = 0)")
    axes[0].set_xlabel("[Fe/H] (dex)", fontsize=12)
    axes[0].set_ylabel("Normalised Count (density)", fontsize=12)
    axes[0].set_title("Stellar Metallicity vs Planet Type", fontsize=12)
    axes[0].legend(fontsize=9)

    sys_counts = df.groupby("hostname").size().value_counts().sort_index()
    axes[1].bar(sys_counts.index, sys_counts.values, color=COLOURS["rv"],
                edgecolor="darkorange", alpha=0.85)
    for x, y in zip(sys_counts.index[:10], sys_counts.values[:10]):
        axes[1].text(x, y + 40, str(y), ha="center", fontsize=8.5, fontweight="bold")
    axes[1].set_xlabel("Number of Confirmed Planets in System", fontsize=12)
    axes[1].set_ylabel("Number of Systems", fontsize=12)
    axes[1].set_title("Planet Multiplicity Distribution", fontsize=12)
    axes[1].set_xlim(0, 11)

    plt.tight_layout()
    save(fig, "fig13_metallicity_multiplicity")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 14  Transit vs Radial Velocity Parameter Spaces
# ─────────────────────────────────────────────────────────────────────────────
def fig14_transit_vs_rv(df: pd.DataFrame) -> None:
    """Mass–period (RV) vs radius–period (transit) parameter space comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Transit vs Radial Velocity: Complementary Detection Spaces", fontsize=14, fontweight="bold")

    rv = df[df["discoverymethod"] == "Radial Velocity"].dropna(subset=["pl_orbper", "pl_bmasse"])
    rv = rv[(rv["pl_orbper"] > 0) & (rv["pl_orbper"] < 1e4) & (rv["pl_bmasse"] > 0)]

    sc1 = axes[0].scatter(
        rv["pl_orbper"], rv["pl_bmasse"],
        c=rv["pl_bmasse"], cmap="coolwarm",
        norm=LogNorm(vmin=0.1, vmax=5000),
        alpha=0.50, s=15,
    )
    plt.colorbar(sc1, ax=axes[0], label="Planet Mass (M$_\\oplus$)")
    axes[0].axhline(EARTH_RAD,           color="#2E7D32", lw=1.5, ls="--", label="1 M$_\\oplus$")
    axes[0].axhline(MJUP_MEARTH,         color="navy",    lw=1.5, ls="--", label="1 M$_{Jup}$")
    axes[0].axhline(13 * MJUP_MEARTH,    color="gray",    lw=1.2, ls=":",  label="BD limit (13 M$_{Jup}$)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Orbital Period (days)", fontsize=12)
    axes[0].set_ylabel("Planet Mass (M$_\\oplus$)", fontsize=12)
    axes[0].set_title(f"Radial Velocity  (n = {len(rv):,})", fontsize=12)
    axes[0].legend(fontsize=8)

    tr = df[df["discoverymethod"] == "Transit"].dropna(subset=["pl_orbper", "pl_rade"])
    tr = tr[(tr["pl_orbper"] > 0) & (tr["pl_orbper"] < 1000) & (tr["pl_rade"] > 0)]

    sc2 = axes[1].scatter(
        tr["pl_orbper"], tr["pl_rade"],
        c=tr["pl_rade"], cmap="viridis",
        norm=LogNorm(vmin=0.3, vmax=25),
        alpha=0.38, s=10,
    )
    plt.colorbar(sc2, ax=axes[1], label="Planet Radius (R$_\\oplus$)")
    axes[1].axhline(EARTH_RAD,   color="#2E7D32", lw=1.5, ls="--", label="1 R$_\\oplus$")
    axes[1].axhline(RJUP_REARTH, color="navy",    lw=1.5, ls="--", label="1 R$_{Jup}$")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Orbital Period (days)", fontsize=12)
    axes[1].set_ylabel("Planet Radius (R$_\\oplus$)", fontsize=12)
    axes[1].set_title(f"Transit  (n = {len(tr):,})", fontsize=12)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    save(fig, "fig14_transit_vs_rv")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 15  Eccentricity vs Semi-major Axis
# ─────────────────────────────────────────────────────────────────────────────
def fig15_eccentricity(df: pd.DataFrame) -> None:
    """Eccentricity vs semi-major axis coloured by detection method."""
    fig, ax = plt.subplots(figsize=(12, 7))

    v = df.dropna(subset=["pl_orbsmax", "pl_orbeccen"])
    v = v[(v["pl_orbsmax"] > 0) & (v["pl_orbsmax"] < 100)
          & (v["pl_orbeccen"] >= 0) & (v["pl_orbeccen"] <= 1)]

    for method, colour in METHOD_COLOURS.items():
        s = v[v["discoverymethod"] == method]
        if len(s) == 0:
            continue
        ax.scatter(s["pl_orbsmax"], s["pl_orbeccen"], c=colour, alpha=0.48,
                   s=16, label=f"{method}  ({len(s)})", zorder=3)

    ax.axhline(0.017, color="#2E7D32", lw=2.0, ls="--", alpha=0.8, label="Earth  e = 0.017")
    ax.axhline(0.000, color="gray",    lw=1.0, ls=":",  alpha=0.5)
    ax.axvspan(0.7, 1.6, alpha=0.10, color="cyan", label="Approx. HZ (G-star)")

    ax.set_xscale("log")
    ax.set_xlabel("Semi-major Axis (AU)", fontsize=13)
    ax.set_ylabel("Orbital Eccentricity", fontsize=13)
    ax.set_title("Eccentricity vs Semi-major Axis by Detection Method", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.set_xlim(0.005, 100); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    save(fig, "fig15_eccentricity")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 16  Semi-major Axis by Detection Method (4-panel)
# ─────────────────────────────────────────────────────────────────────────────
def fig16_sma_by_method(df: pd.DataFrame, ml: pd.DataFrame) -> None:
    """Four-panel semi-major axis histograms for transit, RV, ML, and imaging."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Semi-major Axis Distribution by Detection Method\n(Complementary Orbital Regimes)",
                 fontsize=15, fontweight="bold")

    panels = [
        ("Transit",        df[df["discoverymethod"] == "Transit"],         COLOURS["transit"], "navy",
         [(np.log10(0.04), "r:", "0.04 AU"), (np.log10(1.0), "gold--", "1 AU")]),
        ("Radial Velocity", df[df["discoverymethod"] == "Radial Velocity"], COLOURS["rv"],     "darkorange",
         [(np.log10(1.0), "gold--", "1 AU"), (np.log10(5.2), "navy:", "Jupiter (5.2 AU)")]),
        ("Microlensing",   ml,                                              COLOURS["ml"],      "darkgreen",
         [(np.log10(AU_SNOWLINE), "r--", f"Snow line (~{AU_SNOWLINE} AU)"), (np.log10(5.2), "navy:", "Jupiter")]),
        ("Direct Imaging", df[df["discoverymethod"] == "Imaging"],         COLOURS["imaging"], "purple",
         [(np.log10(10), "gold--", "10 AU"), (np.log10(100), "r:", "100 AU")]),
    ]

    for (title, sub, colour, edgecolour, vlines), ax in zip(panels, axes.flat):
        col = "pl_orbsmax"
        sv = sub.dropna(subset=[col])
        sv = sv[(sv[col] > 0) & (sv[col] < 500)]
        ax.hist(np.log10(sv[col] + 1e-5), bins=55, color=colour,
                edgecolor=edgecolour, alpha=0.82, linewidth=0.5)
        for xval, style, lbl in vlines:
            ls = "--" if "--" in style else ":"
            c2 = style.replace("--", "").replace(":", "")
            ax.axvline(xval, color=c2, lw=1.8, ls=ls, label=lbl)
        ax.set_title(f"{title}  (n = {len(sv):,})", fontsize=12)
        ax.set_xlabel("log$_{10}$(Semi-major Axis / AU)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save(fig, "fig16_sma_by_method")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 17  Planet Mass Distribution by Class (4-panel)
# ─────────────────────────────────────────────────────────────────────────────
def fig17_mass_by_class(df: pd.DataFrame) -> None:
    """Four-panel mass histograms split by planet class."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Planet Mass Distribution by Class", fontsize=15, fontweight="bold")

    v = df.dropna(subset=["pl_bmasse"])
    v = v[v["pl_bmasse"] > 0]

    panels = [
        ("Rocky Planets  (< 5 M$_\\oplus$)",        v[v["pl_bmasse"] < 5],                       COLOURS["rocky"],    (0, 5),     False),
        ("Super-Earths  (5–20 M$_\\oplus$)",         v[(v["pl_bmasse"] >= 5)  & (v["pl_bmasse"] < 20)],  COLOURS["superearth"],(5, 20),    False),
        ("Neptune-class  (20–150 M$_\\oplus$)",      v[(v["pl_bmasse"] >= 20) & (v["pl_bmasse"] < 150)], COLOURS["neptune"],  (0, 150),   False),
        ("Giant Planets  (> 150 M$_\\oplus$)",       v[v["pl_bmasse"] >= 150],                    COLOURS["giant"],    (100, 1.2e4), True),
    ]
    for (title, sub, colour, xrng, log_x), ax in zip(panels, axes.flat):
        sub2 = sub[(sub["pl_bmasse"] >= xrng[0]) & (sub["pl_bmasse"] <= xrng[1])]
        data = np.log10(sub2["pl_bmasse"]) if log_x else sub2["pl_bmasse"]
        ax.hist(data, bins=50, color=colour, edgecolor="k", alpha=0.82, linewidth=0.4)
        ax.set_title(f"{title}\n(n = {len(sub2):,})", fontsize=11, fontweight="bold", color=colour)
        xlabel = "log$_{10}$(Mass / M$_\\oplus$)" if log_x else "Mass (M$_\\oplus$)"
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        if log_x:
            ax.axvline(np.log10(MJUP_MEARTH),      color="navy",  lw=1.8, ls="--", label="1 M$_{Jup}$")
            ax.axvline(np.log10(13*MJUP_MEARTH),   color="gray",  lw=1.5, ls=":",  label="13 M$_{Jup}$ (BD)")
            ax.legend(fontsize=8)

    plt.tight_layout()
    save(fig, "fig17_mass_by_class")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 18  Stellar Type vs Planet Occurrence Heat Map
# ─────────────────────────────────────────────────────────────────────────────
def fig18_occurrence_heatmap(df: pd.DataFrame) -> None:
    """2-D histogram of stellar T_eff vs planet radius as a heat map."""
    fig, ax = plt.subplots(figsize=(13, 7))

    v = df.dropna(subset=["st_teff", "pl_rade"])
    v = v[(v["st_teff"] > 2500) & (v["st_teff"] < 8000)
          & (v["pl_rade"] > 0.3) & (v["pl_rade"] < 20)]

    teff_bins = np.arange(2500, 8050, 250)
    rad_bins  = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0, 12.0, 20.0]

    H, xe, ye = np.histogram2d(v["st_teff"], v["pl_rade"], bins=[teff_bins, rad_bins])
    im = ax.pcolormesh(xe, ye, H.T, cmap="inferno", shading="flat")
    plt.colorbar(im, ax=ax, label="Number of Planets per Bin")

    ax.axhline(1.5, color="cyan",  lw=2.0, ls="--", alpha=0.85, label="Fulton gap lower bound (1.5 R$_\\oplus$)")
    ax.axhline(2.0, color="cyan",  lw=2.0, ls=":",  alpha=0.85, label="Fulton gap upper bound (2.0 R$_\\oplus$)")
    ax.axvline(SOL_TEFF, color="gold", lw=2.0, ls="--", alpha=0.85, label="Solar T$_{eff}$ (5778 K)")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Stellar Effective Temperature T$_{eff}$ (K)", fontsize=15)
    ax.set_ylabel("Planet Radius (R$_\\oplus$)", fontsize=15)
    ax.set_title("Planet Occurrence Heat Map:\nStellar Temperature vs Planet Radius",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, "fig18_occurrence_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 19  PHC Distance Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def fig19_phc_distances(df: pd.DataFrame) -> None:
    """Horizontal bar chart of distances to all 29 PHCs."""
    phc = df[phc_mask(df)].copy().sort_values("sy_dist").dropna(subset=["sy_dist"])

    fig, ax = plt.subplots(figsize=(12, max(8, len(phc) * 0.42)))
    y_pos = range(len(phc))

    bars = ax.barh(y_pos, phc["sy_dist"].values, color=COLOURS["phc"],
                   edgecolor="darkred", alpha=0.82)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(phc["pl_name"].values, fontsize=9.5)

    ax.axvline(10,  color=COLOURS["earth"], lw=2.0, ls="--", alpha=0.85, label="10 pc (JWST limit)")
    ax.axvline(50,  color="#2E7D32",        lw=2.0, ls="--", alpha=0.75, label="50 pc (HWO limit)")
    ax.axvline(100, color="gray",           lw=1.5, ls=":",  alpha=0.65, label="100 pc")

    for i, (d, name) in enumerate(zip(phc["sy_dist"].values, phc["pl_name"].values)):
        ax.text(d + 2, i, f"{d:.1f} pc", va="center", fontsize=8.5)

    ax.set_xlabel("Distance (parsecs)", fontsize=15)
    ax.set_title("Distances to Potentially Habitable Planet Candidates (PHCs)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9.5)
    plt.tight_layout()
    save(fig, "fig19_phc_distances")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 20  PHC: Equilibrium Temperature vs Radius
# ─────────────────────────────────────────────────────────────────────────────
def fig20_phc_teq_radius(df: pd.DataFrame) -> None:
    """PHC scatter plot T_eq vs radius, colour = distance; TRAPPIST-1 highlighted."""
    fig, ax = plt.subplots(figsize=(13, 8))

    hz_all = df[(df["pl_insol"] >= HZ_S_OUTER) & (df["pl_insol"] <= HZ_S_INNER)]\
               .dropna(subset=["pl_rade", "pl_eqt"])
    ax.scatter(hz_all["pl_eqt"], hz_all["pl_rade"], c="lightgray", s=30, alpha=0.60,
               zorder=2, label="HZ planets (all radii)")

    phc = df[phc_mask(df)].dropna(subset=["pl_rade", "pl_eqt"])
    sc = ax.scatter(
        phc["pl_eqt"], phc["pl_rade"], c=phc["sy_dist"],
        cmap="plasma_r", vmin=0, vmax=400,
        s=160, zorder=5, edgecolors="k", linewidth=0.9,
        label="PHC candidates",
    )
    plt.colorbar(sc, ax=ax, label="Distance (pc)")

    trap = phc[phc["hostname"] == "TRAPPIST-1"]
    ax.scatter(trap["pl_eqt"], trap["pl_rade"], c="cyan", s=220, zorder=7,
               marker="D", edgecolors="navy", lw=1.6, label="TRAPPIST-1 PHCs")

    for _, row in phc.iterrows():
        ax.annotate(
            row["pl_name"],
            (row["pl_eqt"], row["pl_rade"]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=6.5, color="#1A237E", fontweight="bold",
        )

    ax.scatter([EARTH_TEQ], [EARTH_RAD], c=COLOURS["earth"], s=260,
               marker=r"$\oplus$", zorder=9, label="Earth", linewidths=0)

    ax.axhspan(PHC_R_MIN, PHC_R_MAX, alpha=0.10, color="green", label="Earth-like Radius Zone")
    ax.axvspan(LW_T_MIN,  LW_T_MAX,  alpha=0.10, color="cyan",  label="Liquid-water T$_{eq}$ Zone")
    ax.axvline(EARTH_TEQ, color=COLOURS["earth"], lw=1.5, ls="--", alpha=0.65)
    ax.axhline(EARTH_RAD, color="green",           lw=1.5, ls="--", alpha=0.65)

    ax.set_xlabel("Equilibrium Temperature T$_{eq}$ (K)", fontsize=15)
    ax.set_ylabel("Planet Radius (R$_\\oplus$)", fontsize=15)
    ax.set_title("PHC Candidates: Equilibrium Temperature vs Radius\n(colour = distance; diamond = TRAPPIST-1)",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9.5, ncol=2)
    ax.set_xlim(120, 410); ax.set_ylim(0.48, 1.72)
    plt.tight_layout()
    save(fig, "fig20_phc_teq_radius")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 21  HZ Planets: Temperature vs Radius by Method
# ─────────────────────────────────────────────────────────────────────────────
def fig21_hz_temp_radius(df: pd.DataFrame) -> None:
    """T_eq vs radius for all HZ planets, colour-coded by detection method."""
    fig, ax = plt.subplots(figsize=(12, 8))

    hz = df[(df["pl_insol"] >= HZ_S_OUTER) & (df["pl_insol"] <= HZ_S_INNER)]\
           .dropna(subset=["pl_eqt", "pl_rade"])

    for method, colour in METHOD_COLOURS.items():
        s = hz[hz["discoverymethod"] == method]
        if len(s) == 0:
            continue
        ax.scatter(s["pl_eqt"], s["pl_rade"], c=colour, alpha=0.70, s=55,
                   edgecolors="k", linewidth=0.5, label=f"{method}  ({len(s)})", zorder=4)

    ax.axhspan(PHC_R_MIN, PHC_R_MAX, alpha=0.10, color="green",  label="Earth-like Radius Zone")
    ax.axvspan(LW_T_MIN,  LW_T_MAX,  alpha=0.14, color="cyan",   label="Liquid-water T$_{eq}$ Window")
    ax.axvline(EARTH_TEQ, color=COLOURS["earth"], lw=2.0, ls="--", alpha=0.80, label=f"Earth T$_{{eq}}$ = {EARTH_TEQ} K")

    ax.set_xlabel("Equilibrium Temperature T$_{eq}$ (K)", fontsize=13)
    ax.set_ylabel("Planet Radius (R$_\\oplus$)", fontsize=13)
    ax.set_title("Habitable Zone Planets: T$_{eq}$ vs Radius by Detection Method",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(100, 700); ax.set_ylim(0, 7)
    plt.tight_layout()
    save(fig, "fig21_hz_temp_radius")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 22  Microlensing Mass & Orbital Separation Statistics
# ─────────────────────────────────────────────────────────────────────────────
def fig22_microlensing(ml: pd.DataFrame) -> None:
    """Mass and semi-major axis distributions for the microlensing planet sample."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Microlensing Planet Population Statistics  (N = 883)", fontsize=15, fontweight="bold")

    m = ml.dropna(subset=["pl_masse"])
    m = m[m["pl_masse"] > 0]
    axes[0].hist(np.log10(m["pl_masse"]), bins=55, color=COLOURS["ml"],
                 edgecolor="darkgreen", alpha=0.85)
    axes[0].axvline(0.0,                     color=COLOURS["earth"], lw=2.2, ls="--", label="1 M$_\\oplus$")
    axes[0].axvline(np.log10(MJUP_MEARTH),   color="navy",           lw=2.0, ls="--", label="1 M$_{Jup}$")
    axes[0].axvline(np.log10(13*MJUP_MEARTH),color="gray",           lw=1.5, ls=":",  label="13 M$_{Jup}$ (BD)")
    axes[0].set_xlabel("log$_{10}$(Planet Mass / M$_\\oplus$)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Microlensing Planet Mass Distribution", fontsize=12)
    axes[0].legend(fontsize=9)

    a = ml.dropna(subset=["pl_orbsmax"])
    a = a[(a["pl_orbsmax"] > 0) & (a["pl_orbsmax"] < 100)]
    axes[1].hist(np.log10(a["pl_orbsmax"]), bins=55, color=COLOURS["rv"],
                 edgecolor="darkorange", alpha=0.85)
    axes[1].axvline(np.log10(AU_SNOWLINE), color=COLOURS["phc"], lw=2.2, ls="--",
                    label=f"Snow line (~{AU_SNOWLINE} AU)")
    axes[1].axvline(np.log10(5.2),         color="navy",          lw=2.0, ls="--", label="Jupiter orbit (5.2 AU)")
    axes[1].axvline(0.0,                   color="gold",          lw=1.5, ls=":",  label="1 AU")
    axes[1].set_xlabel("log$_{10}$(Semi-major Axis / AU)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Microlensing Orbital Separation Distribution", fontsize=12)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save(fig, "fig22_microlensing")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 23  PHC Habitability Metric Profiles (12-panel)
# ─────────────────────────────────────────────────────────────────────────────
def fig23_phc_profiles(df: pd.DataFrame) -> None:
    """12-panel comparison of radius, T_eq, insolation, distance for nearest PHCs."""
    phc = df[phc_mask(df)].copy()
    phc = phc.dropna(subset=["pl_rade", "pl_eqt", "pl_insol", "sy_dist"]).sort_values("sy_dist").head(12)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(
        "Individual PHC Habitability Profiles — 12 Nearest Candidates\n"
        "(vs Earth reference values: R=1.0 R⊕, T=255 K, S=1.0 S⊕)",
        fontsize=14, fontweight="bold",
    )

    metrics    = ["pl_rade",  "pl_eqt",  "pl_insol", "sy_dist"]
    bar_cols   = [COLOURS["transit"], COLOURS["phc"], COLOURS["rv"], COLOURS["ml"]]
    bar_labels = ["R (R⊕)",   "T (K)",   "S (S⊕)",   "d (pc)"]
    earth_vals = [EARTH_RAD,  EARTH_TEQ, EARTH_INSOL, 0]
    x = np.arange(len(metrics))

    for idx, (_, row) in enumerate(phc.iterrows()):
        ax = axes[idx // 4][idx % 4]
        vals  = [row[m] if not np.isnan(row[m]) else 0 for m in metrics]
        bars1 = ax.bar(x - 0.2, vals,       0.38, color=bar_cols, alpha=0.85, label="Planet")
        bars2 = ax.bar(x + 0.2, earth_vals, 0.38, color="silver",  alpha=0.65, label="Earth")
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_title(row["pl_name"], fontsize=9, fontweight="bold")
        spec = row.get("st_spectype", "?")
        spec = spec[:4] if isinstance(spec, str) else "?"
        ax.text(0.97, 0.96, spec, transform=ax.transAxes,
                fontsize=7, ha="right", va="top", color="purple", fontweight="bold")
        if idx == 0:
            ax.legend(fontsize=6.5, loc="upper left")

    plt.tight_layout()
    save(fig, "fig23_phc_profiles")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 24  All-Sky Mollweide Map
# ─────────────────────────────────────────────────────────────────────────────
def fig24_skymap(df: pd.DataFrame) -> None:
    """Mollweide projection of all exoplanets and PHCs (equatorial coordinates)."""
    fig = plt.figure(figsize=(16, 8))
    ax  = fig.add_subplot(111, projection="mollweide")

    v = df.dropna(subset=["ra", "dec"])
    ra_all  = np.radians(v["ra"].values  - 180)
    dec_all = np.radians(v["dec"].values)
    ax.scatter(ra_all, dec_all, c="#90CAF9", alpha=0.15, s=3, zorder=2, label="All exoplanets")

    phc = df[phc_mask(df)].dropna(subset=["ra", "dec"])
    ra_phc  = np.radians(phc["ra"].values  - 180)
    dec_phc = np.radians(phc["dec"].values)
    ax.scatter(ra_phc, dec_phc, c=COLOURS["phc"], s=90, zorder=6, marker="*",
               edgecolors="darkred", lw=0.8, label=f"PHC candidates  (n={len(phc)})")

    for _, row in phc[phc["sy_dist"] < 15].dropna(subset=["sy_dist"]).iterrows():
        ax.annotate(
            row["pl_name"],
            (np.radians(row["ra"] - 180), np.radians(row["dec"])),
            fontsize=6.5, color="darkred",
            xytext=(3, 3), textcoords="offset points",
        )

    ax.grid(True, alpha=0.30, color="gray")
    ax.set_title(
        "All-Sky Distribution of Confirmed Exoplanets  (Mollweide Projection)",
        fontsize=14, fontweight="bold", pad=22,
    )
    ax.set_xlabel("Right Ascension (°)", fontsize=11)
    ax.set_ylabel("Declination (°)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout()
    save(fig, "fig24_skymap")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 25  PHC Radius and Insolation Summary (2-panel horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig25_phc_summary(df: pd.DataFrame) -> None:
    """Horizontal bar charts of radius and insolation for all 29 PHCs."""
    phc = df[phc_mask(df)].copy().sort_values("sy_dist")

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle("Potentially Habitable Planet Candidates (PHCs) — Summary",
                 fontsize=15, fontweight="bold")

    # Radius panel
    r_valid = phc.dropna(subset=["pl_rade"])
    y_r = range(len(r_valid))
    axes[0].barh(y_r, r_valid["pl_rade"].values, color=COLOURS["transit"],
                 edgecolor="navy", alpha=0.85)
    axes[0].set_yticks(y_r)
    axes[0].set_yticklabels(r_valid["pl_name"].values, fontsize=8.5)
    axes[0].axvline(EARTH_RAD, color=COLOURS["phc"], lw=2.2, ls="--", label="Earth (1.0 R⊕)")
    axes[0].axvline(PHC_R_MIN, color="gray", lw=1.2, ls=":", alpha=0.7)
    axes[0].axvline(PHC_R_MAX, color="gray", lw=1.2, ls=":", alpha=0.7, label="PHC radius bounds")
    axes[0].set_xlabel("Planet Radius (R$_\\oplus$)", fontsize=12)
    axes[0].set_title("PHC Radii vs Earth", fontsize=12)
    axes[0].legend(fontsize=9.5)

    # Insolation panel
    s_valid = phc.dropna(subset=["pl_insol"])
    y_s = range(len(s_valid))
    axes[1].barh(y_s, s_valid["pl_insol"].values, color=COLOURS["rv"],
                 edgecolor="darkorange", alpha=0.85)
    axes[1].set_yticks(y_s)
    axes[1].set_yticklabels(s_valid["pl_name"].values, fontsize=8.5)
    axes[1].axvline(EARTH_INSOL, color=COLOURS["earth"], lw=2.2, ls="--", label="Earth (1.0 S⊕)")
    axes[1].axvline(HZ_S_OUTER,  color="#2E7D32", lw=1.5, ls=":", alpha=0.8)
    axes[1].axvline(HZ_S_INNER,  color="#2E7D32", lw=1.5, ls=":", alpha=0.8, label="Conservative HZ bounds")
    axes[1].set_xlabel("Insolation Flux (S$_\\oplus$)", fontsize=12)
    axes[1].set_title("PHC Insolation Fluxes vs Earth", fontsize=12)
    axes[1].legend(fontsize=9.5)

    plt.tight_layout()
    save(fig, "fig25_phc_summary")


# =============================================================================
#  MAIN EXECUTION PIPELINE
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("  EXOPLANET HABITABILITY ANALYSIS — PLOT GENERATION PIPELINE")
    print("  Sharma et al. 2026  |  ApJS (submitted)")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data  = load_data(DATA_DIR)
    df    = data["comp"]    # PSCompPars — primary analysis table
    ps    = data["ps"]      # Planetary Systems (multi-solution)
    td    = data["td"]      # Transit Detections
    stars = data["stars"]   # Stellar Hosts
    ml    = data["ml"]      # Microlensing
    di    = data["di"]      # Direct Imaging

    n_phc = phc_mask(df).sum()
    print(f"  Total confirmed planets : {len(df):,}")
    print(f"  PHC candidates          : {n_phc}")
    print(f"  Output directory        : {OUTPUT_DIR.resolve()}")
    print(f"  Resolution              : {DPI} DPI  |  Format: {FMT.upper()}")
    print()

    # ── 2. Generate figures ───────────────────────────────────────────────────
    figures = [
        ("Fig 01 — Discovery Overview",             lambda: fig01_discovery_overview(df)),
        ("Fig 02 — Cumulative Discovery",            lambda: fig02_cumulative_discovery(df)),
        ("Fig 03 — Mass–Radius Diagram",             lambda: fig03_mass_radius(df)),
        ("Fig 04 — Classification Map",              lambda: fig04_classification_map(df)),
        ("Fig 05 — HZ Insolation Diagram",           lambda: fig05_hz_insolation(df)),
        ("Fig 06 — Kopparapu HZ Boundaries",         lambda: fig06_kopparapu_hz(df)),
        ("Fig 07 — Thermal Properties",              lambda: fig07_thermal_properties(df)),
        ("Fig 08 — T_eq vs Orbital Period",          lambda: fig08_teq_vs_period(df)),
        ("Fig 09 — Stellar HR Diagram",              lambda: fig09_stellar_hr(df)),
        ("Fig 10 — Stellar Properties",              lambda: fig10_stellar_properties(df)),
        ("Fig 11 — Fulton Radius Gap",               lambda: fig11_fulton_gap(df)),
        ("Fig 12 — Period–Radius Diagram",           lambda: fig12_period_radius(df)),
        ("Fig 13 — Metallicity & Multiplicity",      lambda: fig13_metallicity_multiplicity(df)),
        ("Fig 14 — Transit vs RV Parameter Spaces",  lambda: fig14_transit_vs_rv(df)),
        ("Fig 15 — Eccentricity vs SMA",             lambda: fig15_eccentricity(df)),
        ("Fig 16 — SMA by Detection Method",         lambda: fig16_sma_by_method(df, ml)),
        ("Fig 17 — Mass by Planet Class",            lambda: fig17_mass_by_class(df)),
        ("Fig 18 — Occurrence Heat Map",             lambda: fig18_occurrence_heatmap(df)),
        ("Fig 19 — PHC Distances",                   lambda: fig19_phc_distances(df)),
        ("Fig 20 — PHC T_eq vs Radius",              lambda: fig20_phc_teq_radius(df)),
        ("Fig 21 — HZ T_eq vs Radius by Method",     lambda: fig21_hz_temp_radius(df)),
        ("Fig 22 — Microlensing Statistics",         lambda: fig22_microlensing(ml)),
        ("Fig 23 — PHC Habitability Profiles",       lambda: fig23_phc_profiles(df)),
        ("Fig 24 — All-Sky Mollweide Map",           lambda: fig24_skymap(df)),
        ("Fig 25 — PHC Summary Bars",                lambda: fig25_phc_summary(df)),
    ]

    total = len(figures)
    for i, (label, func) in enumerate(figures, start=1):
        print(f"  [{i:02d}/{total}]  Generating {label} …", end="  ", flush=True)
        try:
            func()
        except Exception as exc:
            print(f"\n  [WARNING] {label} failed: {exc}")

    print()
    print("─" * 70)
    print(f"  ✓  {total} figures written to: {OUTPUT_DIR.resolve()}")
    print("─" * 70)


# =============================================================================
if __name__ == "__main__":
    main()
# =============================================================================
