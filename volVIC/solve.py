from typing import Iterable
import numpy as np
import scipy.sparse as sps
from volVIC.VirtualImageCorrelationEnergyElem import (
    VirtualImageCorrelationEnergyElem,
    plot_last_profile,
    compute_image_energy_operators,
)
from volVIC.Mesh import Mesh
from sksparse.cholmod import cholesky


def iteration(
    u_field: np.ndarray[np.floating],
    rho: float,
    mesh: Mesh,
    image_energies: Iterable[VirtualImageCorrelationEnergyElem],
    image: np.ndarray,
    membrane_K: sps.spmatrix,
    membrane_weight: float,
    C: sps.spmatrix,
    verbose: bool = True,
    disable_parallel: bool = False,
):
    """
    Perform one iteration of the Virtual Image Correlation (VIC) energy minimization algorithm.

    This function updates the displacement field and the virtual image parameter by solving
    a linearized system derived from the total energy, which combines the image correlation energy
    and an intrapatch membrane regularization. Dirichlet and interpatch C¹ constraints are imposed
    through the constraint matrix `C`.

    Parameters
    ----------
    u_field : np.ndarray[np.floating]
        Current displacement field defined at the mesh control points.
    rho : float
        Current virtual image parameter controlling the gray-level transformation.
    mesh : Mesh
        Geometric mesh to be deformed during the optimization.
    image_energies : Iterable[VirtualImageCorrelationEnergyElem]
        Collection of patchwise virtual image correlation energy elements.
    image : np.ndarray
        Real image (experimental or reference) to which the virtual image is fitted.
    membrane_K : scipy.sparse.spmatrix
        Sparse matrix representing the intrapatch membrane stiffness operator.
    membrane_weight : float
        Weight factor applied to the membrane regularization term.
    C : scipy.sparse.spmatrix
        Constraint matrix enforcing Dirichlet and C¹ coupling conditions.
    verbose : bool, optional
        If True, display intermediate energy values and diagnostic plots. Default is True.
    disable_parallel : bool, optional
        If True, disable parallel computation during energy operator evaluation. Default is False.

    Returns
    -------
    du_field : np.ndarray
        Increment of the displacement field obtained from the linearized system.
    drho : float
        Increment of the virtual image parameter corresponding to intensity adaptation.

    Notes
    -----
    The function assembles and solves the following linearized system:

        H_tot * Δu = -grad_tot

    where
        grad_tot = Cᵀ (grad + w_mem * K_mem * u)
        H_tot    = Cᵀ (H + w_mem * K_mem) C

    The scalar parameter increment Δρ is computed independently as:

        Δρ = - (∂E/∂ρ) / (∂²E/∂ρ²)
    """
    E, grad, H, dE_drho, d2E_drho2 = compute_image_energy_operators(
        image_energies,
        mesh,
        image,
        u_field,
        rho,
        verbose=verbose,
        disable_parallel=disable_parallel,
    )
    u = u_field.ravel()
    if verbose:
        plot_last_profile(image_energies)
        print(f"E_vic = {E:.3E}; E_mem = {0.5*(membrane_K@u)@u:.3E}")
    grad_tot = C.T @ (grad + membrane_weight * membrane_K @ u)
    if C.data.size / np.prod(C.shape) > 0.1:
        H_tot = C.T.A @ (H + membrane_weight * membrane_K) @ C.A
        dof = np.linalg.solve(H_tot, -grad_tot)
    else:
        H_tot = C.T @ H @ C + membrane_weight * (C.T @ membrane_K @ C)
        factor = cholesky(H_tot)
        dof = factor(-grad_tot)
    factor = cholesky(H_tot)
    dof = factor(-grad_tot)
    du_field = (C @ dof).reshape(u_field.shape)
    drho = -dE_drho / d2E_drho2
    return du_field, drho
