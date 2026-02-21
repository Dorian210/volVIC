# %%
import numpy as np
from bsplyne import new_quarter_pipe, MultiPatchBSplineConnectivity
from volVIC import Mesh, Problem

# %%
# ============================ Synthetic image =============================
# This block builds a simple 3D synthetic image directly in voxel space.
# The geometry is a radially-modulated pipe used only for demonstration.
# -------------------------------------------------------------------------

size = 50
binning = 5

# Radial modulation (local thickening of the pipe by a sinusoidal bump)
r_in, r_out = 1.0, 1.5
rho = 1.0
a = 2 * np.pi / 8

# Voxel grid (before binning)
X, Y, Z = np.meshgrid(
    *([np.linspace(-2, 2, size * binning)] * 3),
    indexing="ij",
)

Phi = np.arctan2(Y, X)
Norm = np.sqrt(X**2 + Y**2)

mask_r = (np.abs(Z) < rho) & (Phi > (np.pi / 4 - a)) & (Phi < (np.pi / 4 + a))
R = (
    r_in
    + (r_out - r_in)
    * (
        0.5
        * (np.cos(Z * np.pi / rho) + 1)
        * 0.5
        * (np.cos((Phi - np.pi / 4 + a) / (2 * a) * 2 * np.pi - np.pi) + 1)
    )
    * mask_r
)

# Binary object (before binning)
not_binned = (Norm <= R) & (np.abs(Z) <= 1.75)
not_binned = not_binned.astype("float") * np.iinfo(np.uint16).max

# Voxel binning (mean intensity)
windows = np.lib.stride_tricks.sliding_window_view(not_binned, [binning] * 3)[
    ::binning, ::binning, ::binning
]

image = windows.reshape((*windows.shape[:3], -1)).mean(axis=-1)
image = image.astype(np.uint16)

# Foreground / background levels for volVIC
fg = float(image.max())
bg = float(image.min())

# %%
# ============================ Surface definition ===========================
# A spline surface approximating a quarter pipe, embedded in voxel space.
# -------------------------------------------------------------------------

spline, ctrl_pts = new_quarter_pipe(
    [0, 0, -1.5],  # pipe start (physical coordinates)
    [0, 0, 1.0],  # pipe end
    1.0,  # inner radius
    3.0,  # outer radius
)

# Increase spline resolution
ctrl_pts = spline.orderElevation(ctrl_pts, [1, 2])
ctrl_pts = spline.knotInsertion(ctrl_pts, [1, 10])

# Multi-patch connectivity
connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts([ctrl_pts])

# Map physical coordinates to voxel coordinates
ctrl_pts = (ctrl_pts + 2) * size / 4

# Build volVIC mesh
spline_mesh = Mesh([spline], [ctrl_pts], connectivity)

# Visual check: surface embedded in the image
spline_mesh.plot_in_image(image, mode="voxels")

# %%
# ============================ Problem definition ===========================
# Definition of the volVIC problem on the spline surface.
# -------------------------------------------------------------------------

pb = Problem(
    spline_mesh,
    image,
    ICP_init=False,
    fg_bg=(fg, bg),
    width_dx=0.5,
    surf_dx=0.25,
)

# %%
# ================================ Solve ===================================
# Compute the displacement field on the spline surface.
# -------------------------------------------------------------------------

u_field, rho = pb.solve()

# %%
# ============================== Visualization ==============================
# Display distance map on original and displaced surface.
# -------------------------------------------------------------------------

pb.plot_results()
pb.plot_results(u_field=u_field)

# %%
