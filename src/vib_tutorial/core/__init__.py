"""Physics core: model assembly, modal analysis, forcing, and time integration.

Nothing in this package depends on Qt, so it can be used from scripts,
notebooks, and tests.
"""

from .forcing import ForceController, ForceKind, ForceSettings
from .modal import ComplexMode, ModalResult, Mode, frf, modal_analysis, receptance
from .model import ChainSystem, assemble_chain, state_space
from .simulator import Simulator, foh_discretize
from .substructure import (
    CraigBamptonModel,
    ModeComparison,
    Substructure,
    compare_modes,
    craig_bampton,
    damped_poles,
    interior_counts,
    reduced_frf,
    substructure_damping,
)

__all__ = [
    "ChainSystem",
    "ComplexMode",
    "CraigBamptonModel",
    "ForceController",
    "ForceKind",
    "ForceSettings",
    "ModalResult",
    "Mode",
    "ModeComparison",
    "Simulator",
    "Substructure",
    "assemble_chain",
    "compare_modes",
    "craig_bampton",
    "damped_poles",
    "foh_discretize",
    "frf",
    "interior_counts",
    "modal_analysis",
    "receptance",
    "reduced_frf",
    "state_space",
    "substructure_damping",
]
