from typing import Iterable
import numpy as np
import scipy.sparse as sps

# from scipy.stats import chi2

from bsplyne import BSpline, parallel_blocks

from volVIC.Mesh import Mesh, MeshLattice
from volVIC.virtual_image_correlation_energy import VirtualImageCorrelationEnergyElem


def make_coordinates_systems(
    ctrl_pts: np.ndarray[np.floating], dN_dxi: sps.spmatrix, dN_deta: sps.spmatrix
) -> tuple[
    np.ndarray[np.floating],
    np.ndarray[np.floating],
    np.ndarray[np.floating],
    np.ndarray[np.floating],
    np.ndarray[np.floating],
]:
    """
    Compute the covariant basis vectors and transformation components between covariant and contravariant
    bases.

    Given `ctrl_pts` (control points in physical space) and the derivatives of the basis functions with
    respect to the isoparametric coordinates (`dN_dxi`, `dN_deta`), this function returns the covariant basis
    vectors (`A1`, `A2`) and the transformation components (`A11`, `A22`, `A12`) for each evaluation point.

    Parameters
    ----------
    ctrl_pts : np.ndarray[np.floating]
        Array of control points in physical space, shaped as (`3`, `n_nodes_xi`, `n_nodes_eta`).
    dN_dxi : sps.spmatrix
        Sparse matrix of derivatives of basis functions with respect to the first isoparametric coordinate
        (`xi`). Shape: (`n_gauss`, `n_nodes`).
    dN_deta : sps.spmatrix
        Sparse matrix of derivatives of basis functions with respect to the second isoparametric coordinate
        (`eta`). Shape: (`n_gauss`, `n_nodes`).

    Returns
    -------
    tuple[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating]]
        Tuple containing:
        - `A1` (`np.ndarray[np.floating]`): Covariant basis vector 1 at each evaluation point.
        - `A2` (`np.ndarray[np.floating]`): Covariant basis vector 2 at each evaluation point.
        - `A11` (`np.ndarray[np.floating]`): First component of the covariant-to-contravariant transformation.
        - `A22` (`np.ndarray[np.floating]`): Second component of the covariant-to-contravariant transformation.
        - `A12` (`np.ndarray[np.floating]`): Third component of the covariant-to-contravariant transformation.

    Notes
    -----
    - The transformation components are computed pointwise for each evaluation point in the isoparametric space.
    - The returned tuple has length 5, with each entry corresponding to an array of values at all evaluation
    points.
    """
    # Covariant coordinates system
    A1 = ctrl_pts.reshape((3, -1)) @ dN_dxi.T
    A2 = ctrl_pts.reshape((3, -1)) @ dN_deta.T
    # Covariant to contravariant transformation
    A11cov = (A1 * A1).sum(axis=0)
    A22cov = (A2 * A2).sum(axis=0)
    A12cov = (A1 * A2).sum(axis=0)
    det_inv = 1 / (A11cov * A22cov - A12cov * A12cov)
    A11 = A22cov * det_inv
    A22 = A11cov * det_inv
    A12 = -A12cov * det_inv

    return A1, A2, A11, A22, A12


def make_H(
    A11: np.ndarray[np.floating],
    A22: np.ndarray[np.floating],
    A12: np.ndarray[np.floating],
) -> sps.spmatrix:
    """
    Assemble Hooke's tensor in contravariant basis Voigt notation for a membrane material with Poisson's ratio
    set to 0 and Young's modulus set to 1, using B-spline basis components.

    Parameters
    ----------
    A11 : np.ndarray[np.floating]
        Covariant-to-contravariant basis component array for the (1,1) direction in isoparametric space.
    A22 : np.ndarray[np.floating]
        Covariant-to-contravariant basis component array for the (2,2) direction in isoparametric space.
    A12 : np.ndarray[np.floating]
        Covariant-to-contravariant basis component array for the (1,2) direction in isoparametric space.

    Returns
    -------
    sps.spmatrix
        Hooke's tensor as a sparse matrix in contravariant basis Voigt notation, with shape
        (`3 * n_gauss`, `3 * n_gauss`), where `n_gauss` is the number of isoparametric points.

    Notes
    -----
    - The tensor is block-assembled using `scipy.sparse.bmat` and diagonal matrices from the input arrays.
    - The material law is restricted to Poisson's ratio = 0 and Young's modulus = 1.
    - The resulting tensor is suitable for use in isogeometric Kirchhoff-Love membrane formulations.
    """
    H11 = sps.diags(A11 * A11)
    H22 = sps.diags(A22 * A22)
    H33 = sps.diags(0.5 * (A11 * A22 + A12 * A12))
    H12 = sps.diags(A12 * A12)
    H13 = sps.diags(A11 * A12)
    H23 = sps.diags(A22 * A12)
    H = sps.bmat([[H11, H12, H13], [H12, H22, H23], [H13, H23, H33]])
    return H


def make_Bm(
    dN_dxi: sps.spmatrix,
    dN_deta: sps.spmatrix,
    A1: np.ndarray[np.floating],
    A2: np.ndarray[np.floating],
) -> sps.spmatrix:
    """
    Assemble the Jacobian matrix `Bm` for membrane deformation in Voight notation in isogeometric analysis.

    This function constructs the sparse matrix `Bm` by combining the derivatives of the shape functions in the
    isoparametric (`xi`, `eta`) directions with the corresponding covariant tangent vectors `A1` and `A2`. The
    resulting matrix maps nodal displacements to membrane strains in Voight notation, and is used in
    isogeometric membrane finite element formulations.

    Parameters
    ----------
    dN_dxi : sps.spmatrix
        Sparse matrix of derivatives of the shape functions with respect to the `xi` coordinate in
        isoparametric space. Shape: (`n_gauss`, `n_nodes`).
    dN_deta : sps.spmatrix
        Sparse matrix of derivatives of the shape functions with respect to the `eta` coordinate in
        isoparametric space. Shape: (`n_gauss`, `n_nodes`).
    A1 : np.ndarray[np.floating]
        Covariant tangent vector in the `xi` direction. Shape: (`3`, `n_gauss`).
    A2 : np.ndarray[np.floating]
        Covariant tangent vector in the `eta` direction. Shape: (`3`, `n_gauss`).

    Returns
    -------
    sps.spmatrix
        Sparse matrix `Bm` representing the Jacobian of the membrane deformation transformation in Voight notation. Shape: (`3 * n_gauss`, `3 * n_nodes`).

    Notes
    -----
    - The output matrix `Bm` has shape (`3 * n_gauss`, `3 * n_nodes`), where `n_gauss` is the number of
    quadrature points and `n_nodes` is the number of control points (basis functions).
    - The isoparametric space refers to the parametric domain of the B-spline basis.
    """
    de_xi_du = sps.hstack(
        (
            dN_dxi.multiply(A1[0, :, None]),
            dN_dxi.multiply(A1[1, :, None]),
            dN_dxi.multiply(A1[2, :, None]),
        )
    )
    de_eta_du = sps.hstack(
        (
            dN_deta.multiply(A2[0, :, None]),
            dN_deta.multiply(A2[1, :, None]),
            dN_deta.multiply(A2[2, :, None]),
        )
    )
    de_xi_eta_2_du = sps.hstack(
        (
            dN_deta.multiply(A1[0, :, None]) + dN_dxi.multiply(A2[0, :, None]),
            dN_deta.multiply(A1[1, :, None]) + dN_dxi.multiply(A2[1, :, None]),
            dN_deta.multiply(A1[2, :, None]) + dN_dxi.multiply(A2[2, :, None]),
        )
    )
    Bm = sps.vstack((de_xi_du, de_eta_du, de_xi_eta_2_du))
    return Bm


def make_membrane_stiffness_operator(
    spline: BSpline, ctrl_pts: np.ndarray[np.floating]
) -> sps.spmatrix:
    """
    Assemble the global membrane stiffness matrix for a B-spline surface patch.

    This function computes the global stiffness operator `K` for a B-spline surface patch, considering only
    in-plane (membrane) strain energy contributions and omitting out-of-plane (bending) effects. The assembly
    is performed using Gauss-Legendre quadrature over the isoparametric domain.

    Parameters
    ----------
    spline : BSpline
        B-spline surface patch object defining the geometry and basis functions.
    ctrl_pts : np.ndarray[np.floating]
        Array of control points defining the physical geometry of the surface.
        Shape should be (`3`, `n_nodes_xi`, `n_nodes_eta`).

    Returns
    -------
    K : sps.spmatrix
        Global stiffness matrix (`sps.spmatrix`) containing only membrane (in-plane) contributions.

    Notes
    -----
    - The integration is performed using quadrature points determined by the spline degrees in each
    isoparametric direction.
    - The returned matrix `K` does not include bending or out-of-plane stiffness terms.
    """
    # Integration space
    p, q = spline.getDegrees()
    (xi, eta), (dxi, deta) = spline.gauss_legendre_for_integration([p + 1, q + 1])
    # Local covariant and contravariant bases
    dN_dxi, dN_deta = spline.DN([xi, eta], 1)  # type: ignore
    A1, A2, A11, A22, A12 = make_coordinates_systems(ctrl_pts, dN_dxi, dN_deta)
    A3 = np.cross(A1, A2, axis=0)
    detJ = np.linalg.norm(A3, axis=0)
    A3 = A3 / detJ[None]
    WdetJ = detJ * np.outer(dxi, deta).ravel()
    # Hooke tensor in the contravariant basis in Voigt notation
    H = make_H(A11, A22, A12)
    # Membrane strain tensor in the contravariant basis in Voigt notation
    Bm = make_Bm(dN_dxi, dN_deta, A1, A2)
    # Linear elasticity operator
    K = (Bm.multiply(np.hstack([WdetJ] * 3)[:, None])).T @ H @ Bm
    return K


def make_membrane_stifness(
    mesh: Mesh, verbose: bool = True, disable_parallel: bool = False
) -> sps.spmatrix:
    """
    Assemble the global membrane stiffness matrix for a full multipatch mesh.

    This function computes the global membrane stiffness matrix by summing the contributions
    from all patches of the mesh. Each patch is processed separately using the B-spline geometry
    and control points of each patch.

    Parameters
    ----------
    mesh : Mesh
        Mesh object containing B-spline patches.
    verbose : bool, optional
        If True, displays a progress bar during assembly. Default is True.
    disable_parallel : bool, optional
        If True, disables parallel computation. Default is False.

    Returns
    -------
    sps.spmatrix
        Global membrane stiffness matrix, including all patches, ready for use in
        isogeometric membrane finite element analysis. The size corresponds to `3 * n_dofs`,
        where `n_dofs` is the total number of control points in the mesh.

    Notes
    -----
    - The function internally calls `make_membrane_stiffness_operator` for each patch.
    - Uses parallel computation if optimal when `disable_parallel` is False.
    - The resulting matrix includes only in-plane (membrane) stiffness contributions.
    """
    if isinstance(mesh, MeshLattice):
        cell_mesh = mesh.get_mesh_cell_one()
        separated_ctrl_pts = cell_mesh.get_separated_ctrl_pts()
        args_list = [
            (cell_mesh.splines[patch], separated_ctrl_pts[patch])
            for patch in range(cell_mesh.connectivity.nb_patchs)
        ]
        K_elem_cell = parallel_blocks(
            make_membrane_stiffness_operator,
            args_list,
            verbose=verbose,
            pbar_title="Make membrane stiffness",
            disable_parallel=disable_parallel,
        )
        K_elem = K_elem_cell * (mesh.l * mesh.m * mesh.n)
    else:
        separated_ctrl_pts = mesh.get_separated_ctrl_pts()
        args_list = [
            (mesh.splines[patch], separated_ctrl_pts[patch])
            for patch in range(mesh.connectivity.nb_patchs)
        ]
        K_elem = parallel_blocks(
            make_membrane_stiffness_operator,
            args_list,
            verbose=verbose,
            pbar_title="Make membrane stiffness",
            disable_parallel=disable_parallel,
        )
    K = mesh.assemble_hessians(K_elem, 3)
    return K


# def make_membrane_weight_old(
#     mesh: Mesh,
#     image_energies: Iterable[VirtualImageCorrelationEnergyElem],
#     membrane_K: sps.spmatrix,
#     image_std: float = 5_000,
#     expected_mean_dist: float = 5,
#     monte_carlo_size: int = 100,
# ) -> float:
#     """
#     Infer an appropriate membrane regularization weight for virtual image correlation (VIC) fitting.

#     The membrane weight is determined by comparing the expected virtual image correlation (VIC)
#     energy to the average membrane energy. The goal is to scale the membrane contribution
#     such that, at convergence, the image energy approximates the membrane energy:

#         E_VIC(U_CV) ≈ W_MEM * E_MEM(U_CV)

#     The procedure is as follows:
#     1. Simulate random displacements of the B-spline control points using Gaussian noise
#        with amplitude proportional to the expected mean distance between the converged mesh
#        and the real geometry.
#     2. Compute the membrane energy for each Monte Carlo sample using the provided membrane stiffness matrix.
#     3. Average the membrane energy over all samples to obtain the expected membrane energy.
#     4. Estimate the expected VIC energy based on the image standard deviation, expected
#        mean displacement, voxel volume, and Chi-squared statistics.
#     5. Set the membrane weight W_MEM such that the expected VIC energy roughly matches
#        the average membrane energy.

#     Parameters
#     ----------
#     mesh : Mesh
#         The initial mesh to be deformed. Each patch contains B-spline splines with methods
#         for discretization, interpolation, and energy computation.
#     image_energies : Iterable[VirtualImageCorrelationEnergyElem]
#         VIC energies for each patch. Each element provides access to voxel weights, Jacobian determinants,
#         and patch size.
#     membrane_K : sps.spmatrix
#         Sparse, symmetric membrane stiffness operator (pure membrane behavior, no bending).
#         Used to compute the membrane regularization energy.
#     image_std : float, optional
#         Standard deviation of the image intensity, used to estimate the VIC energy.
#         Default is 5_000.
#     expected_mean_dist : float, optional
#         Expected mean displacement between the converged mesh and the real geometry.
#         Default is 5.
#     monte_carlo_size : int, optional
#         Number of Monte Carlo samples for estimating the average membrane energy.
#         Default is 100.

#     Returns
#     -------
#     membrane_weight : float
#         The inferred membrane regularization weight ensuring that, at convergence, the
#         VIC energy roughly matches the membrane energy.
#     """
#     weights = (
#         expected_mean_dist
#         * np.sqrt(8 / np.pi)
#         * mesh.separated_to_unique(
#             [s.DN(s.greville_abscissa()).diagonal() for s in mesh.splines]
#         )
#     )
#     cutoff_u = (
#         weights[None, None, :]
#         * np.random.randn(monte_carlo_size, *mesh.unique_ctrl_pts.shape)
#     ).reshape((monte_carlo_size, -1))
#     E_mem = (0.5 * ((cutoff_u @ membrane_K) * cutoff_u).sum(axis=1)).mean()
#     volume_of_voxels = sum(
#         [image_energy.wdetJ.sum() for image_energy in image_energies]
#     )
#     E_vic_cv = (
#         (expected_mean_dist * image_std) ** 2
#         / (4 * image_energies[0].h)
#         * chi2.mean(volume_of_voxels)
#     )
#     membrane_weight = 10 * E_vic_cv / E_mem
#     return membrane_weight


def make_membrane_weight(
    mesh: Mesh,
    image_energies: Iterable[VirtualImageCorrelationEnergyElem],
    membrane_K: sps.spmatrix,
    rho: float = 1.5,
    image_std: float = 5_000,
    expected_mean_dist: float = 5,
    n_intg: int = 100,
) -> float:
    """
    Compute a membrane regularization weight by matching the expected membrane
    energy to the expected virtual image correlation (VIC) energy.

    The membrane energy is evaluated analytically from a probabilistic model
    of the B-spline control point displacements, while the VIC energy is
    estimated from finite differences of the virtual image response.

    Parameters
    ----------
    mesh : Mesh
        B-spline mesh defining the control points and basis functions.
    image_energies : iterable of VirtualImageCorrelationEnergyElem
        VIC energy objects associated with each patch.
    membrane_K : scipy.sparse.spmatrix
        Membrane stiffness matrix of size (3 * n_bf, 3 * n_bf).
    rho : float, optional
        Image correlation parameter passed to the virtual image operator. By default 1.5.
    image_std : float, optional
        Standard deviation of the image noise. By default 5_000.
    expected_mean_dist : float, optional
        Expected mean distance between the converged B-spline geometry and the
        target image. By default 5.
    n_intg : int, optional
        Number of integration points used for estimating the VIC energy.
        By default 100.

    Returns
    -------
    membrane_weight : float
        Regularization weight such that the expected membrane energy matches
        the expected VIC energy.

    Notes
    -----
    **Expected membrane energy**

    Each B-spline control point ``a`` is assigned a random displacement

        u_a = d_a * w_a * n_a   ∈ R^3

    where:

    - ``d_a = expected_mean_dist * (1 + ε_a)``, with ``ε_a ~ N(0, 1)``
    - ``w_a = B_a(ξ_a)`` is the B-spline basis evaluated at the Greville abscissa
    - ``n_a`` is an isotropic random unit vector in R^3

    The membrane stiffness matrix ``K ∈ R^{3 n_bf × 3 n_bf}`` is decomposed into
    3×3 blocks ``K_{a,b}`` such that

        [K_{a,b}]_{c,d} = K_{c * n_bf + a, d * n_bf + b}

    The discrete membrane energy reads

        E_mem(U) = 1/2 * Σ_{a,b} w_a w_b d_a d_b n_aᵀ K_{a,b} n_b

    Taking the expectation and assuming independence between radial amplitudes
    and directions yields

        E[E_mem] = 1/2 * Σ_{a,b} w_a w_b E[d_a d_b] E[n_aᵀ K_{a,b} n_b]

    For a ≠ b, the isotropy of ``n_b`` implies

        E[n_aᵀ K_{a,b} n_b] = 0

    Hence only diagonal terms remain:

        E[E_mem] = 1/2 * Σ_a w_a² E[d_a²] E[n_aᵀ K_{a,a} n_a]

    Using isotropy:

        E[n_a n_aᵀ] = 1/3 * I_3
        ⇒ E[n_aᵀ K_{a,a} n_a] = 1/3 * tr(K_{a,a})

    Moreover:

        E[d_a²] = expected_mean_dist² * E[(1 + ε_a)²] = 2 * expected_mean_dist²

    Finally:

        E[E_mem] = expected_mean_dist² / 3 * Σ_a w_a² * tr(K_{a,a})

    Introducing:

        W = (w_1, ..., w_{n_bf}, w_1, ..., w_{n_bf}, w_1, ..., w_{n_bf})ᵀ

    this can be written compactly as:

        E[E_mem] = expected_mean_dist² / 3 * (W² · diag(K))

    **Expected VIC energy**

    The observed image near the converged surface is modeled as:

        f(γ) = g(γ + d) + σ ε

    where ε ~ N(0,1), σ = image_std, and d is a random normal offset along
    the surface normal such that E(|d|) = expected_mean_dist.

    The VIC energy is defined as

        E_vic = 1/2 ∫_Ω 1/(2h) ∫_{-h}^{h} (f(γ) - g(γ))² dγ dΩ

    Substituting the model:

        f(γ) - g(γ) = g(γ + d) - g(γ) + σ ε

    and taking the expectation:

        E[E_vic] = 1/2 ∫_Ω 1/(2h) ∫_{-h}^{h} E[(g(γ + d) - g(γ))²] dγ dΩ
                    + 1/2 ∫_Ω 1/(2h) ∫_{-h}^{h} E[σ² ε²] dγ dΩ

    The cross term vanishes because E[ε] = 0. The noise term evaluates to:

        1/2 ∫_Ω 1/(2h) ∫_{-h}^{h} σ² dγ dΩ = |Ω| σ² / 2

    Assuming the virtual image profile g is locally antisymmetric around the
    surface (antisymmetry of the transition), the difference

        g(γ + d) - g(γ)

    is symmetric with respect to -d/2, which implies the squared difference
    depends only on |d|. As the law of d is otherwise unknown, the expectation
    is approximated by evaluating the energy for a representative offset

        d ≈ expected_mean_dist

    giving the final estimate:

        E[E_vic] ≈ |Ω|/2 * ( σ² + 1/(2h) ∫_{-h}^{h} (g(γ + expected_mean_dist) - g(γ))² dγ )

    This is exactly what is implemented in the code: the integral is computed
    by discrete summation on each patch.
    """
    weights = mesh.separated_to_unique(
        [s.DN(s.greville_abscissa()).diagonal() for s in mesh.splines]
    )
    weights = np.hstack([weights] * 3)

    E_mem = expected_mean_dist**2 / 3 * np.dot(weights * weights, membrane_K.diagonal())

    z = np.zeros(1)
    E_vic = 0
    for e in image_energies:
        gamma = np.linspace(-e.h, e.h, n_intg)
        f = e.virtual_image(z, z, gamma + expected_mean_dist, rho)[0]
        g = e.virtual_image(z, z, gamma, rho)[0]
        mean_patch_error = np.mean((f - g) ** 2)
        patch_area = e.wdetJs.sum()
        E_vic += patch_area / 2 * (image_std**2 + mean_patch_error)

    return E_vic / E_mem
