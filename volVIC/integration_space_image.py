# %%
from typing import Union
import numpy as np
import numba as nb
from bsplyne import BSpline
from bsplyne.b_spline_basis import _DN

@nb.njit(nb.float64[:](nb.float64, nb.float64, nb.int64, nb.float64[:, :], 
                       nb.int64, nb.int64, nb.float64[:], nb.float64[:], nb.int64, nb.int64, nb.int64, nb.int64), 
         cache=True)
def get_tangent(xi, eta, axis, ctrl_pts, 
                p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta):
    xi = np.array([xi])
    eta = np.array([eta])
    if axis==0:
        data_xi, _, ind_xi = _DN(p_xi, m_xi, n_xi, knot_xi, xi, 1)
        data_eta, _, ind_eta = _DN(p_eta, m_eta, n_eta, knot_eta, eta, 0)
    elif axis==1:
        data_xi, _, ind_xi = _DN(p_xi, m_xi, n_xi, knot_xi, xi, 0)
        data_eta, _, ind_eta = _DN(p_eta, m_eta, n_eta, knot_eta, eta, 1)
    else:
        raise ValueError("Can only derive w.r.t. xi (axis=0) or eta (axis=1).")
    
    tangent = np.zeros(ctrl_pts.shape[0], dtype='float')
    for i in range(len(data_xi)):
        for j in range(len(data_eta)):
            xyz = ctrl_pts[:, ind_xi[i]*(n_eta + 1) + ind_eta[j]]
            tangent += xyz*data_xi[i]*data_eta[j]
    return tangent

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64[:, :], 
                       nb.int64, nb.int64, nb.float64[:], nb.float64[:], nb.int64, nb.int64, nb.int64, nb.int64), 
         cache=True)
def get_dxi(xi, eta, dist, ctrl_pts, 
            p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta):
    dX_dxi = get_tangent(xi, eta, 0, ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
    dxi = dist/np.sqrt(np.sum(dX_dxi*dX_dxi))
    return dxi

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64[:, :], 
                       nb.int64, nb.int64, nb.float64[:], nb.float64[:], nb.int64, nb.int64, nb.int64, nb.int64), 
         cache=True)
def get_deta(xi, eta, dist, ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta):
    dX_deta = get_tangent(xi, eta, 1, ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
    deta = dist/np.sqrt(np.sum(dX_deta*dX_deta))
    return deta

@nb.njit(nb.types.Tuple((nb.types.UniTuple(nb.float64[:], 2), nb.types.UniTuple(nb.float64[:], 2)))(
    nb.types.UniTuple(nb.types.UniTuple(nb.float64, 2), 2), 
    nb.int64, 
    nb.int64, 
    nb.float64[:], 
    nb.float64[:], 
    nb.types.UniTuple(nb.float64, 2), 
    nb.types.UniTuple(nb.float64, 2), 
    nb.float64[:, :], 
    nb.float64
    ), cache=True)
def linspace_for_VIC_elem_numba(alpha, p_xi, p_eta, knot_xi, knot_eta, span_xi, span_eta, ctrl_pts, dx):
    
    # p_xi, p_eta = spline_degrees
    # knot_xi, knot_eta = spline_knots
    
    m_xi = knot_xi.size - 1
    m_eta = knot_eta.size - 1
    n_xi = m_xi - p_xi - 1
    n_eta = m_eta - p_eta - 1
    
    # span_xi, span_eta = spline_spans
    xi_mid = sum(span_xi)/2
    eta_mid = sum(span_eta)/2
    
    xi_first = span_xi[0] + get_dxi(span_xi[0], eta_mid, alpha[0][0], ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
    xi = [xi_mid]
    dxi_left = []
    pas_fini = True
    while pas_fini:
        new_dxi = get_dxi(xi[-1], eta_mid, dx, ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
        new_xi = xi[-1] - new_dxi
        if new_xi < xi_first:
            dxi_left.append(xi[-1] - xi_first)
            xi.append(xi_first)
            dxi_left.append(xi_first - span_xi[0])
            pas_fini = False
        else:
            xi.append(new_xi)
            dxi_left.append(new_dxi)
    xi = xi[::-1]
    xi_last = span_xi[-1] - get_dxi(span_xi[-1], eta_mid, alpha[0][1], ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
    dxi_right = []
    pas_fini = True
    while pas_fini:
        new_dxi = get_dxi(xi[-1], eta_mid, dx, ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
        new_xi = xi[-1] + new_dxi
        if new_xi > xi_last:
            dxi_right.append(xi_last - xi[-1])
            xi.append(xi_last)
            dxi_right.append(span_xi[1] - xi_last)
            pas_fini = False
        else:
            xi.append(new_xi)
            dxi_right.append(new_dxi)
    xi = np.array(xi)
    dxi = np.array(dxi_left[:0:-1] + [dxi_left[0] + dxi_right[0]] + dxi_right[1:])
    
    eta_first = span_eta[0] + get_deta(xi_mid, span_eta[0], alpha[1][0], ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
    eta = [eta_mid]
    deta_left = []
    pas_fini = True
    while pas_fini:
        new_deta = get_deta(xi_mid, eta[-1], dx, ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
        new_eta = eta[-1] - new_deta
        if new_eta < eta_first:
            deta_left.append(eta[-1] - eta_first)
            eta.append(eta_first)
            deta_left.append(eta_first - span_eta[0])
            pas_fini = False
        else:
            eta.append(new_eta)
            deta_left.append(new_deta)
    eta = eta[::-1]
    eta_last = span_eta[-1] - get_deta(xi_mid, span_eta[-1], alpha[1][1], ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
    deta_right = []
    pas_fini = True
    while pas_fini:
        new_deta = get_deta(xi_mid, eta[-1], dx, ctrl_pts, p_xi, p_eta, knot_xi, knot_eta, m_xi, m_eta, n_xi, n_eta)
        new_eta = eta[-1] + new_deta
        if new_eta > eta_last:
            deta_right.append(eta_last - eta[-1])
            eta.append(eta_last)
            deta_right.append(span_eta[1] - eta_last)
            pas_fini = False
        else:
            eta.append(new_eta)
            deta_right.append(new_deta)
    eta = np.array(eta)
    deta = np.array(deta_left[:0:-1] + [deta_left[0] + deta_right[0]] + deta_right[1:])
    
    return (xi, eta), (dxi, deta)

def linspace_for_VIC_elem(
    spline: BSpline, 
    ctrl_pts: np.ndarray[np.floating], 
    dist: float=1.,
    alpha: Union[float, tuple[tuple[float, float], tuple[float, float]]]=0., 
    ) -> tuple[tuple[np.ndarray[np.floating], np.ndarray[np.floating]], tuple[np.ndarray[np.floating], np.ndarray[np.floating]]]:
    """
    Compute integration points and weights in the isoparametric space of a `BSpline` 
    surface, such that the mapped distance between points is close to `dist`.

    This function generates arrays of integration points and their corresponding 
    weights (step sizes) in both isoparametric directions (`xi` and `eta`) of a 
    `BSpline` surface. The integration points are distributed so that the mapped 
    distance between them, after transformation by the `BSpline`, is approximately 
    `dist`. The ignored border distance can be set independently for each boundary 
    using `alpha`. The computation is performed at the center of the other 
    isoparametric span for each direction.

    Parameters
    ----------
    spline : BSpline
        The `BSpline` surface object defining the isoparametric space and mapping.

    ctrl_pts : np.ndarray[np.floating]
        The control points of the `BSpline` surface, as a `numpy` array of shape 
        (3, N_xi, N_eta).

    dist : float, optional
        The target distance between integration points after mapping through the 
        `BSpline`.
        By default, 1.

    alpha : Union[float, tuple[tuple[float, float], tuple[float, float]]], optional
        The distance to ignore on the border of the patch in each isoparametric 
        direction.
        If a `float`, the same value is used for all boundaries.
        If a tuple of tuples, 
        ((`dist_xi_0`, `dist_xi_1`), (`dist_eta_0`, `dist_eta_1`)),
        where each value specifies the ignored distance at the corresponding boundary.
        By default, 0.

    Returns
    -------
    (xi, eta) : tuple[np.ndarray[np.floating], np.ndarray[np.floating]]
        Tuple of `numpy` arrays containing the integration points in the `xi` and 
        `eta` isoparametric directions.

    (dxi, deta) : tuple[np.ndarray[np.floating], np.ndarray[np.floating]]
        Tuple of `numpy` arrays containing the integration weights (step sizes) in 
        the `xi` and `eta` directions.

    Notes
    -----
    - The integration points are distributed so that the mapped distance between 
    them is approximately `dist`.
    - The ignored border distance can be set independently for each boundary using 
    `alpha`.
    - The function internally calls a `numba`-accelerated implementation for 
    performance.
    - For a surface (2D), returns 
    ((`xi` points, `eta` points), (`xi` weights, `eta` weights)).
    """
    if np.isscalar(alpha):
        alpha = ((alpha, alpha), (alpha, alpha)) # type: ignore
    else:
        alpha = ((alpha[0][0], alpha[0][1]), (alpha[1][0], alpha[1][1]))
    
    ctrl_pts = ctrl_pts.reshape((3, -1))
    p_xi, p_eta = spline.getDegrees()
    knot_xi, knot_eta = spline.getKnots()
    span_xi, span_eta = spline.getSpans()
    
    (xi, eta), (dxi, deta) = linspace_for_VIC_elem_numba(alpha, p_xi, p_eta, knot_xi, knot_eta, span_xi, span_eta, ctrl_pts, dist)
    return (xi, eta), (dxi, deta)

# %%
if __name__=='__main__':
    # Degrés
    p_xi, p_eta = 2, 2

    # Noeuds uniformes (clamped)
    knot_xi = np.array([0, 0, 0, 1, 1, 1], dtype='float')
    knot_eta = np.array([0, 0, 0, 1, 1, 1], dtype='float')

    # Points de contrôle 3D
    ctrl_pts = np.array([
        [  # x-coord
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0]
        ],
        [  # y-coord
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0]
        ],
        [  # z-coord
            [0.0, 0.2, 0.0],
            [0.2, 0.6, 0.2],
            [0.0, 0.2, 0.0]
        ]
    ]).reshape((3, -1))

    spline = BSpline([p_xi, p_eta], [knot_xi, knot_eta])

    (xi, eta), (dxi, deta) = linspace_for_VIC_elem(spline, ctrl_pts, dist=0.01, alpha=0.)
    
    dists = np.linalg.norm(np.diff(spline(ctrl_pts, (np.array([0.5]), eta)).reshape((3, -1)), axis=1), axis=0)
    dists.mean(), dists.std()
# %%
