"""Physics core: model assembly, modal analysis, forcing, and time integration.

Nothing in this package depends on Qt, so it can be used from scripts,
notebooks, and tests.
"""

from .energy import kinetic_energy, modal_energies, potential_energy, stored_energy
from .forcing import ForceController, ForceKind, ForceSettings
from .frf_matrix import ModalTerm, frf_matrix, modal_frf_terms
from .modal import ComplexMode, ModalResult, Mode, frf, modal_analysis, modal_coordinate_map, receptance
from .model import ChainSystem, assemble_chain, state_space
from .simulator import Simulator, foh_discretize, foh_quadratic_integrals
from .substructure import (
    METHOD_NAMES,
    METHODS,
    CMSModel,
    ComponentMode,
    CraigBamptonModel,
    ModeComparison,
    Substructure,
    compare_modes,
    component_mode_synthesis,
    component_modes,
    craig_bampton,
    damped_poles,
    free_interface,
    interior_counts,
    kept_ranges,
    reduced_frf,
    substructure_damping,
)

__all__ = [
    "METHODS",
    "METHOD_NAMES",
    "CMSModel",
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
    "component_mode_synthesis",
    "component_modes",
    "craig_bampton",
    "damped_poles",
    "foh_discretize",
    "foh_quadratic_integrals",
    "frf",
    "free_interface",
    "frf_matrix",
    "interior_counts",
    "kept_ranges",
    "kinetic_energy",
    "modal_analysis",
    "modal_coordinate_map",
    "modal_energies",
    "potential_energy",
    "modal_frf_terms",
    "receptance",
    "reduced_frf",
    "state_space",
    "stored_energy",
    "substructure_damping",
]
