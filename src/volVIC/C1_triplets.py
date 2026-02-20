from typing import Union, Literal
import numpy as np
import numba as nb
import scipy.sparse as sps
from volVIC.Mesh import Mesh


@nb.njit(cache=True)
def recover_nodes_couples_from_unique_inds(
    unique_nodes_inds: np.ndarray[np.integer],
) -> np.ndarray[np.integer]:
    """
    Recover all unordered pairs of node indices sharing the same unique node ID.

    This function is intended for multipatch B-spline lattices, where multiple
    nodes may correspond to the same unique node (shared across patches). It
    returns all combinations of indices `(i, j)` such that both nodes correspond
    to the same unique node ID.

    Parameters
    ----------
    unique_nodes_inds : np.ndarray of int
        Array of length `n_nodes` mapping each mesh node to its unique node ID.

    Returns
    -------
    np.ndarray of int, shape (n_couples, 2)
        Array of node index pairs `(i, j)` with `i < j`, for all nodes sharing the
        same unique node ID.

    Notes
    -----
    - The output pairs are sorted by group but not globally by node index.
    - Each unique unordered pair is returned exactly once.
    """
    # Sort the indices to group identical unique node ids together
    sorted_inds = np.argsort(unique_nodes_inds)
    sorted_ids = unique_nodes_inds[sorted_inds]
    nb_nodes = unique_nodes_inds.size

    # Count the size of each group of identical unique node ids
    counts = []
    starts = []
    start = 0
    for i in range(1, nb_nodes):
        if sorted_ids[i] != sorted_ids[i - 1]:
            starts.append(start)
            counts.append(i - start)
            start = i
    starts.append(start)
    counts.append(nb_nodes - start)

    # Compute total number of node pairs (i, j) within each group (combinatorics: n choose 2)
    total = 0
    for c in counts:
        total += c * (c - 1) // 2
    nodes_couples = np.empty((total, 2), dtype="int")

    # Generate all unique unordered pairs of indices (i, j) from each group
    pos = 0
    for s, c in zip(starts, counts):
        if c < 2:
            continue
        for i in range(c - 1):
            a = sorted_inds[s + i]
            for j in range(i + 1, c):
                b = sorted_inds[s + j]
                nodes_couples[pos, 0] = a
                nodes_couples[pos, 1] = b
                pos += 1

    return nodes_couples


def get_all_triplets(
    unique_nodes_inds: np.ndarray[np.integer], shape_by_patch: np.ndarray[np.integer]
) -> np.ndarray[np.integer]:
    """
    Compute triplets of nodes `(A, B, C)` for C1 continuity constraints.

    For lattice or multipatch meshes, each triplet corresponds to three nodes
    that form a C1 constraint (smoothness across patches). This function handles
    corner cases and boundary nodes, ensuring only valid triplets are returned.

    Parameters
    ----------
    unique_nodes_inds : np.ndarray of int
        Array mapping each mesh node to its unique node ID.
    shape_by_patch : np.ndarray of int, shape (n_patches, 2)
        Number of nodes along each parametric direction per patch.

    Returns
    -------
    np.ndarray of int, shape (3, n_triplets)
        Array of triplets of unique node indices. Each column corresponds to
        a triplet (A, B, C) forming a candidate C1 continuity constraint.

    Notes
    -----
    - Handles corner and edge nodes correctly, discarding invalid triplets.
    - Ensures that each triplet respects mesh topology and lattice boundaries.
    """
    # Recover node index pairs sharing the same unique node id
    B1, B2 = recover_nodes_couples_from_unique_inds(unique_nodes_inds).T
    nb_couples = B1.size

    # Stack all involved indices (B1 and B2) for easier processing
    unpacked_ind = np.hstack((B1, B2))

    # Compute patch indices and local coordinates of each node
    patch_limits_global = np.concatenate(
        ([0], np.cumsum(np.prod(shape_by_patch, axis=1)))
    )
    patch = np.searchsorted(patch_limits_global, unpacked_ind, side="right") - 1
    patch_limits = patch_limits_global[patch]
    local_index = unpacked_ind - patch_limits
    nrows, ncols = shape_by_patch[patch].T
    i, j = np.divmod(local_index, ncols)

    # Detect whether each point lies on a boundary edge
    on_top = i == 0
    on_bottom = i == (nrows - 1)
    on_left = j == 0
    on_right = j == (ncols - 1)

    # Prepare output and validity mask
    inside_nodes = np.zeros(unpacked_ind.size, dtype=unpacked_ind.dtype)
    validity_mask = np.zeros(nb_couples, dtype="bool")

    # Identify corner points (nodes on two boundaries)
    corner = (on_bottom | on_top) & (on_left | on_right)
    invalid_couples = np.stack(
        [np.logical_xor(corner[:nb_couples], corner[nb_couples:])] * 2
    ).ravel()
    corner[invalid_couples] = False
    i_c, j_c = i[corner], j[corner]

    # Find neighboring indices depending on corner position
    i_neighbor = np.stack((i_c, np.where(on_top[corner], i_c + 1, i_c - 1)))
    j_neighbor = np.stack((np.where(on_left[corner], j_c + 1, j_c - 1), j_c))
    neighbors = i_neighbor * ncols[corner] + j_neighbor + patch_limits[corner]

    # Reshape neighbor indices to distinguish B1 and B2 contributions
    B1_neighbors, B2_neighbors = neighbors.reshape((2, 2, -1)).transpose(1, 2, 0)

    # Compare the unique node ids of opposite sides
    B1_unique_neighbors = unique_nodes_inds[B1_neighbors]
    B2_unique_neighbors = unique_nodes_inds[B2_neighbors]
    neighbors_mask = (
        B1_unique_neighbors[:, :, None] == B2_unique_neighbors[:, None, :]
    )  # check for matches

    # A corner is "facing" if any of its B1 neighbors match a B2 neighbor
    is_facing = neighbors_mask.reshape((-1, 4)).sum(axis=1) == 1

    # Mark these pairs as valid
    validity_mask[corner.reshape((2, -1))[0]] = is_facing

    # Retrieve the "inside" node for corner cases (opposite to the matched ones)
    B1_inside_corners = B1_neighbors[is_facing][~neighbors_mask[is_facing].any(axis=2)]
    B2_inside_corners = B2_neighbors[is_facing][~neighbors_mask[is_facing].any(axis=1)]
    inside_nodes[np.tile(validity_mask, 2)] = np.hstack(
        (B1_inside_corners, B2_inside_corners)
    )

    # Handle non-corner boundary cases
    side = ~corner
    i_s, j_s = i[side], j[side]
    i_inside = np.where(on_top[side], i_s + 1, np.where(on_bottom[side], i_s - 1, i_s))
    j_inside = np.where(on_left[side], j_s + 1, np.where(on_right[side], j_s - 1, j_s))
    inside_nodes[side] = i_inside * ncols[side] + j_inside + patch_limits[side]
    validity_mask[side.reshape((2, -1))[0]] = True

    # Construct triplets (A, B, C) from (inside, mid, inside) nodes
    unique_A, unique_C = np.sort(
        unique_nodes_inds[inside_nodes.reshape((2, nb_couples))], axis=0
    )[:, validity_mask]
    unique_B = unique_nodes_inds[B1][validity_mask]
    unique_ABC = np.stack((unique_A, unique_B, unique_C))

    # Remove duplicate triplets
    unique_ABC = np.unique(unique_ABC, axis=1)

    return unique_ABC


def make_C1_eqs(
    mesh: Mesh,
    C1_inds: Union[None, Literal["auto", "none", "all"], np.ndarray[np.integer]] = None,
    threshold: float = 1e-1,  # 10%
    field_size: int = 3,
    verbose: bool = True,
) -> sps.spmatrix:
    """
    Generate sparse C1 continuity equations for a multipatch or lattice mesh.

    Constructs a sparse matrix encoding C1 continuity constraints between triplets
    of nodes in the mesh. Each row corresponds to one triplet `(A, B, C)` with
    the equation `A - 2*B + C = 0` for enforcing smoothness.

    Parameters
    ----------
    mesh : Mesh
        Multipatch B-spline mesh providing:
        - unique control points (`mesh.unique_ctrl_pts`)
        - patch topology (`mesh.connectivity`)
    C1_inds : None, {"auto", "none", "all"} or ndarray of int, optional
        Selection mode for C1 triplets:
        - `None` or `"auto"`: automatically select triplets based on geometry.
          Triplets already close to C1 continuity are preserved.
        - `"none"`: do not generate any C1 constraint.
        - `"all"`: enforce C1 continuity on all valid triplets.
        - `ndarray`:
            * shape `(n_nodes,)`: select triplets whose middle node `B`
              belongs to the given unique node indices.
            * shape `(3, n_triplets)`: explicit `(A, B, C)` triplets.
    threshold : float, optional
        Relative geometric tolerance used in `"auto"` mode.

        For each triplet `(A, B, C)`, the quantity::

            rhs = ||A - 2 B + C||

        is compared to the upper bound::

            up_bound = ||B - A|| + ||C - B||

        The triplet is kept if::

            rhs <= threshold * up_bound

        Default is `0.1` (10%).
    field_size : int, optional
        Size of the field on which C1 continuity is enforced:
        - `1` for scalar fields
        - `3` for vector fields
        - etc.

        The constraint matrix is expanded block-diagonally accordingly.
        Default is `3`.
    verbose : bool, optional
        If `True`, print information about selected triplets and modes.
        Default is `True`.

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse constraint matrix of shape::

            (n_triplets * field_size,
             n_unique_nodes * field_size)

        Each block row encodes the equation `A - 2 B + C = 0`.

    Raises
    ------
    ValueError
        If `C1_inds` has an unsupported shape or an unknown mode is provided.

    Notes
    -----
    - Constraints are built on **unique control points**, not per-patch DOFs.
    - Vector-valued fields are handled by block-diagonal duplication.
    - Intended for C1 regularization in volume-based virtual image correlation
      (volVIC) and related inverse problems.
    """
    if isinstance(C1_inds, np.ndarray):
        mode = "manual"
    else:
        mode = C1_inds

    if mode is None:
        mode = "auto"

    if mode == "none":
        A = np.empty(0, dtype="int")
        B = np.empty(0, dtype="int")
        C = np.empty(0, dtype="int")
        if verbose:
            print("[VIC][C1] mode=none → no C1 constraints")
    elif mode == "all":
        A, B, C = get_all_triplets(
            mesh.connectivity.unique_nodes_inds, mesh.connectivity.shape_by_patch
        )
        if verbose:
            print(f"[VIC][C1] mode=all → enforcing C1 on all triplets ({B.size})")
    elif mode == "auto":
        ABC = get_all_triplets(
            mesh.connectivity.unique_nodes_inds, mesh.connectivity.shape_by_patch
        )
        if ABC.size == 0:
            A, B, C = ABC
        else:
            a, b, c = mesh.unique_ctrl_pts[:, ABC].transpose(1, 0, 2)
            indicator = np.linalg.norm(a + c - 2 * b, axis=0)
            indicator_sup = np.linalg.norm(b - a, axis=0) + np.linalg.norm(
                c - b, axis=0
            )
            to_keep = indicator <= threshold * indicator_sup
            A, B, C = ABC[:, to_keep]
            # import matplotlib.pyplot as plt
            # tmp = np.zeros(mesh.connectivity.nb_unique_nodes, dtype=bool)
            # tmp[B] = True
            # tmp = mesh.unique_to_separated(tmp)
            # tmp = [t if i % 2 == 0 else t[::-1] for i, t in enumerate(tmp)]
            # c = plt.imshow(np.mean(np.array(tmp, dtype=float), axis=0))
            # plt.colorbar(c)
            # plt.show()
        if verbose:
            print(
                f"[VIC][C1] mode=auto → kept {to_keep.sum()}/{indicator.size} triplets "
                f"(threshold={threshold:.2g})"
            )
    elif mode == "manual":
        if C1_inds.ndim == 1:
            ABC = get_all_triplets(
                mesh.connectivity.unique_nodes_inds, mesh.connectivity.shape_by_patch
            )
            if ABC.size == 0:
                A, B, C = ABC
            else:
                A, B, C = ABC[:, np.isin(ABC[1], C1_inds)]
            if verbose:
                print(
                    f"[VIC][C1] mode=manual (by node) → " f"{B.size} triplets selected"
                )
        elif C1_inds.ndim == 2:
            A, B, C = C1_inds
            if verbose:
                print(
                    f"[VIC][C1] mode=manual (explicit triplets) → " f"{B.size} triplets"
                )
        else:
            raise ValueError(
                f"C1 triplets error as `C1_inds` given has shape {C1_inds.shape} (dim!=1 and dim!=2) !"
            )
    else:
        raise ValueError(f"C1 triplets error as mode '{mode}' is unknown !")

    m = B.size
    rows = np.hstack([np.arange(m)] * 3)
    cols = np.hstack((A, B, C))
    data = np.hstack(
        (
            np.ones(m, dtype="float"),
            -2 * np.ones(m, dtype="float"),
            np.ones(m, dtype="float"),
        )
    )
    eqs = sps.coo_matrix(
        (data, (rows, cols)), shape=(m, mesh.connectivity.nb_unique_nodes)
    )
    eqs = sps.block_diag([eqs] * field_size)

    return eqs


# def make_dirichlet_eqs(
#     mesh: Mesh, unique_inds: np.ndarray[np.integer], field_size: int = 3
# ) -> sps.spmatrix:

#     m = unique_inds.size
#     rows = np.arange(m)
#     cols = unique_inds
#     data = np.ones(m, dtype="float")
#     eqs = sps.coo_matrix(
#         (data, (rows, cols)), shape=(m, field_size * mesh.connectivity.nb_unique_nodes)
#     )

#     return eqs


# def make_outside_image_eqs(
#     mesh: Mesh, image_shape: tuple[int, ...], field_size: int = 3
# ) -> sps.spmatrix:
#     print(
#         "Warning : dirty method to lock control points outside the image with a constant inward padding."
#     )
#     separated_ctrl_pts = mesh.get_separated_ctrl_pts()
#     separated_greville_eval = [
#         s(pts, s.greville_abscissa())
#         for s, pts in zip(mesh.splines, separated_ctrl_pts)
#     ]
#     unique_greville_eval = mesh.separated_to_unique(separated_greville_eval)
#     (unique_inds,) = np.where(
#         (
#             (unique_greville_eval < (np.zeros(len(image_shape))[:, None] + 15))
#             | (unique_greville_eval > (np.array(image_shape)[:, None] - 1 - 15))
#         ).any(axis=0)
#     )
#     unique_inds = np.hstack(
#         [unique_inds + i * mesh.connectivity.nb_unique_nodes for i in range(field_size)]
#     )
#     return make_dirichlet_eqs(mesh, unique_inds, field_size=field_size)


# def compute_C_from_eqs(
#     eqs: Union[sps.spmatrix, Iterable[sps.spmatrix]],
# ) -> sps.spmatrix:
#     """
#     Sparse + PySPQR-based nullspace extraction for general constraints:
#       - u[i] = 0 for i in inds_0
#       - u[i1] - 2u[i2] + u[i3] = 0 for each column in inds_ABC
#     """
#     # 1) Stack constraint matrices if necessary
#     if not sps.issparse(eqs):
#         eqs = sps.vstack(eqs)

#     # 2) Apply PySPQR QR factorization to eqs.T:
#     #    eqs.T = Q R P.T
#     #    Q: (n×n) sparse, R: (n×p) sparse, E: permutation indices of P, rank: int
#     Q, R, E, rank = qr(eqs.T, economy=False)

#     # 3) Extract nullspace basis: columns of Q beyond the numerical rank
#     C = Q.tocsc()[:, rank:]  # shape (n, d) with d = n - rank

#     # 4) Return sparse nullspace basis
#     return C
