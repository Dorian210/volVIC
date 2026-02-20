# %%
import numpy as np
import scipy.sparse as sps
from bsplyne import BSpline, MultiPatchBSplineConnectivity
from IGA_for_bsplyne import solve_sparse

import pickle as pkl
import matplotlib.pyplot as plt
import pyvista as pv


# %%
def ellipse(phi):  # x² - xy + y² = 3/2
    ellipse_x = np.sqrt(3 / 2) * np.cos(phi) - 1 / np.sqrt(2) * np.sin(phi)
    ellipse_y = np.sqrt(3 / 2) * np.cos(phi) + 1 / np.sqrt(2) * np.sin(phi)
    return np.array([ellipse_x, ellipse_y])


xi = np.linspace(0, 1, 1000)
phi = -np.pi / 3 + xi * np.pi / 3
x_hat, y_hat = ellipse(phi)


fig, ax = plt.subplots()
ax.plot([0, x_hat[0]], [0, y_hat[0]], color="grey")
ax.plot([0, x_hat[-1]], [0, y_hat[-1]], color="grey")
ax.plot(x_hat, y_hat, color="red", label="ellipse")
ax.set_xlim(0, 1.7)
ax.set_ylim(0)
ax.set_aspect("equal")
plt.legend()
plt.show()


# %%
p = 2
knot = np.array([0, 0, 0, 1, 1, 1])
spline_curve = BSpline([p], [knot])
spline_curve.orderElevation(None, [1])
spline_curve.knotInsertion(None, [2])
N = spline_curve.DN([xi], k=0)

weights = np.hstack(([1e10], np.ones(N.shape[0] - 2), [1e10]))
W = sps.diags(weights)

x = solve_sparse(N.T @ W @ N, N.T @ W @ x_hat)
y = solve_sparse(N.T @ W @ N, N.T @ W @ y_hat)


fig, ax = plt.subplots()
ax.plot(x_hat, y_hat, color="red", label="ellipse objectif")
spline_curve.plot(np.array([x, y]), plotter=ax, show=False)
ax.set_xlim(0, 1.7)
ax.set_ylim(0)
ax.set_aspect("equal")
plt.show()


# %%
l = 2
r = 0.7 / 2

degrees = [spline_curve.getDegrees()[0], 1, 1]
knot_lin = np.array([0, 0, 1, 1])
knots = [spline_curve.getKnots()[0], knot_lin, knot_lin]
spline = BSpline(degrees, knots)

X, Y, Z = np.zeros((3, *spline.getCtrlShape()))

o = np.zeros_like(x)
i = np.ones_like(x)
X[:, 0, 0] = o
Y[:, 0, 0] = o
Z[:, 0, 0] = o
X[:, 0, 1] = l * i
Y[:, 0, 1] = l * i
Z[:, 0, 1] = l * i
X[:, 1, 0] = r * x
Y[:, 1, 0] = r * y
Z[:, 1, 0] = o
X[:, 1, 1] = l * i
Y[:, 1, 1] = l * i - r * y[::-1]
Z[:, 1, 1] = l * i - r * x[::-1]


ctrl_pts = np.array([X, Y, Z])
pv_plotter = pv.Plotter()
spline.plot(ctrl_pts, plotter=pv_plotter, show=False)
pv_plotter.view_vector((6, -2, 1.5))
pv_plotter.show_axes()
pv_plotter.show()


# %%
ctrl_pts = np.array([X, Y, Z])

ctrl_pts = spline.orderElevation(ctrl_pts, [0, 2, 2])

ctrl_pts = spline.knotInsertion(ctrl_pts, [0, 3, 10])

pv_plotter = pv.Plotter()
spline.plot(ctrl_pts, plotter=pv_plotter, show=False)
pv_plotter.view_vector((6, -2, 1.5))
pv_plotter.show_axes()
pv_plotter.show()


# %%
start = np.sqrt(2) * r / np.sqrt(3) + 1e-2
greville = spline.bases[2].greville_abscissa()[1:-1]
a, b = greville[0], greville[-1]
coords = (l - 2 * start) * (greville - a) / (b - a) + start
# coords = np.linspace(start, l - start, greville.size)
centers = coords[None] * np.ones((3, 1))
disp = centers[:, None, None, :] - ctrl_pts[:, :, :, 1:-1]
disp_needed = np.ones((3, 1, 1, 1)) * disp.sum(axis=0)[None] / 3
ctrl_pts[:, :, :, 1:-1] += disp_needed
ctrl_pts[1:, 0, :, :] = np.mean(ctrl_pts[1:, 0, :, :], axis=0)[None]
ctrl_pts[:-1, -1, :, :] = np.mean(ctrl_pts[:-1, -1, :, :], axis=0)[None]

pv_plotter = pv.Plotter()
spline.plot(ctrl_pts, plotter=pv_plotter, show=False)
pv_plotter.view_vector((6, -2, 1))
pv_plotter.show_axes()
pv_plotter.show()


# %%
X, Y, Z = ctrl_pts
beam_pts = [[X, Y, Z], [X, Z, Y], [Y, X, Z], [Y, Z, X], [Z, X, Y], [Z, Y, X]]

separated_beam_ctrl_pts = [np.array([x, y, z]) for x, y, z in beam_pts]
beam_connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(
    separated_beam_ctrl_pts, eps=1e-5
)
beam_splines = [spline] * len(separated_beam_ctrl_pts)

beam_connectivity.save_paraview(
    beam_splines,
    separated_beam_ctrl_pts,
    "out_geo",
    "beam",
    n_eval_per_elem=10,
    disable_parallel=False,
)

# %%
cell_pts = (
    [[x, y, z] for x, y, z in beam_pts]
    + [[-x, y, z] for x, y, z in beam_pts]
    + [[x, -y, z] for x, y, z in beam_pts]
    + [[-x, -y, z] for x, y, z in beam_pts]
    + [[x, y, -z] for x, y, z in beam_pts]
    + [[-x, y, -z] for x, y, z in beam_pts]
    + [[x, -y, -z] for x, y, z in beam_pts]
    + [[-x, -y, -z] for x, y, z in beam_pts]
)

separated_ctrl_pts = [np.array([x, y, z]) for x, y, z in cell_pts]
connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(
    separated_ctrl_pts, eps=1e-5
)
splines = [spline] * len(separated_ctrl_pts)

connectivity.save_paraview(
    splines,
    separated_ctrl_pts,
    "out_geo",
    "cell",
    n_eval_per_elem=10,
    disable_parallel=False,
)


# %%
with open("BCC_cell.pkl", "wb") as file:
    pkl.dump((splines, separated_ctrl_pts, connectivity), file)

# %%
