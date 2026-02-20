"""
.. include:: ../../README.md
"""

from .Mesh import Mesh, MeshLattice
from .marching_cubes import marching_cubes
from .C1_triplets import (
    get_all_triplets,
    make_C1_eqs,
    # make_dirichlet_eqs,
    # make_outside_image_eqs,
    # compute_C_from_eqs,
)
from .membrane_stiffness import make_membrane_stifness
from .image_utils import find_fg_bg, otsu_threshold, hist, find_sigma_hat
from .virtual_image_correlation_energy import (
    VirtualImageCorrelationEnergyElem,
    make_image_energies,
)
from .virtual_image import g_slide
from .Problem import Problem
