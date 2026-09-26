"""Shared colors and limits for the GUI."""

MAX_DOF = 8

# One color per mass, used consistently in the animation, time plots and FRF.
MASS_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]

# Mode shapes use a separate palette so they are not confused with masses.
MODE_COLORS = [
    "#264653",
    "#2a9d8f",
    "#b08900",
    "#e76f51",
    "#6a4c93",
    "#1982c4",
    "#8ac926",
    "#ff595e",
]

# Substructures A, B on the Substructuring page (bands, springs, and labels).
SUB_COLORS = ["#3a6ea5", "#d17a00", "#4f8f3a"]

FORCE_COLOR = "#c1121f"
STRUCTURE_COLOR = "#333333"

# Energy bars: stored (kinetic, potential), and the ledger of energy in and out.
ENERGY_COLORS = {
    "kinetic": "#1982c4",
    "potential": "#2a9d8f",
    "stored": "#8d99ae",
    "added": "#6a4c93",
    "work": FORCE_COLOR,
    "dissipated": "#e76f51",
}

# Component mode synthesis methods, on the Substructuring page's comparison plot.
METHOD_COLORS = {"craig-bampton": "#333333", "rubin": "#2a9d8f", "macneal": "#e76f51"}
