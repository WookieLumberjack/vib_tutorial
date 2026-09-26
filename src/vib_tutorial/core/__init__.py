"""Physics core: model assembly, modal analysis, forcing, and time integration.

Nothing in this package depends on Qt, so it can be used from scripts,
notebooks, and tests.
"""

from .forcing import ForceController, ForceKind, ForceSettings
from .frf_matrix import ModalTerm, frf_matrix, modal_frf_terms
from .modal import ComplexMode, ModalResult, Mode, frf, modal_analysis, modal_coordinate_map, receptance
from .model import ChainSystem, assemble_chain, state_space
from .simulator import Simulator, foh_discretize
from .substructure import (
    ComponentMode,
    CraigBamptonModel,
    ModeComparison,
    Substructure,
    compare_modes,
    component_modes,
    craig_bampton,
    damped_poles,
    interior_counts,
    reduced_frf,
    substructure_damping,
)

__all__ = [
    "ChainSystem",
    "ComplexMode",
    "ComponentMode",
    "CraigBamptonModel",
    "ForceController",
    "ForceKind",
    "ForceSettings",
    "ModalResult",
    "ModalTerm",
    "Mode",
    "ModeComparison",
    "Simulator",
    "Substructure",
    "assemble_chain",
    "compare_modes",
    "component_modes",
    "craig_bampton",
    "damped_poles",
    "foh_discretize",
    "frf",
    "frf_matrix",
    "interior_counts",
    "modal_analysis",
    "modal_coordinate_map",
    "modal_frf_terms",
    "receptance",
    "reduced_frf",
    "state_space",
    "substructure_damping",
]
