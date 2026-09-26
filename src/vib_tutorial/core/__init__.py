"""Physics core: model assembly, modal analysis, forcing, and time integration.

Nothing in this package depends on Qt, so it can be used from scripts,
notebooks, and tests.
"""

from .forcing import ForceController, ForceKind, ForceSettings
from .modal import DampedPole, ModalResult, Mode, frf, modal_analysis
from .model import ChainSystem, assemble_chain, state_space
from .simulator import Simulator, foh_discretize

__all__ = [
    "ChainSystem",
    "DampedPole",
    "ForceController",
    "ForceKind",
    "ForceSettings",
    "ModalResult",
    "Mode",
    "Simulator",
    "assemble_chain",
    "foh_discretize",
    "frf",
    "modal_analysis",
    "state_space",
]
