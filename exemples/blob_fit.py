# %%
import numpy as np
from scipy.ndimage import gaussian_filter
from bsplyne import new_cube
from volVIC import Mesh, Problem

# %% Define the geometry as a B-spline volume

l = 1
spline, ctrl_pts = new_cube([0, 0, 0], [0, 0, 1], 1)
ctrl_pts = spline.orderElevation(ctrl_pts, [2, 2, 2])
ctrl_pts = spline.knotInsertion(ctrl_pts, [4, 4, 4])

# Rescale control points to match the desired physical size
ctrl_pts = ctrl_pts * l / (2 * np.linalg.norm(ctrl_pts, axis=0)[None])

# Smooth control points near edges and corners to avoid sharp geometric artifacts
ctrl_pts[:, 0, 0, 1:-1] = 0.5 * (ctrl_pts[:, 0, 1, 1:-1] + ctrl_pts[:, 1, 0, 1:-1])
ctrl_pts[:, 0, -1, 1:-1] = 0.5 * (ctrl_pts[:, 0, -2, 1:-1] + ctrl_pts[:, 1, -1, 1:-1])
ctrl_pts[:, -1, 0, 1:-1] = 0.5 * (ctrl_pts[:, -1, 1, 1:-1] + ctrl_pts[:, -2, 0, 1:-1])
ctrl_pts[:, -1, -1, 1:-1] = 0.5 * (
    ctrl_pts[:, -1, -2, 1:-1] + ctrl_pts[:, -2, -1, 1:-1]
)

ctrl_pts[:, 0, 1:-1, 0] = 0.5 * (ctrl_pts[:, 0, 1:-1, 1] + ctrl_pts[:, 1, 1:-1, 0])
ctrl_pts[:, 0, 1:-1, -1] = 0.5 * (ctrl_pts[:, 0, 1:-1, -2] + ctrl_pts[:, 1, 1:-1, -1])
ctrl_pts[:, -1, 1:-1, 0] = 0.5 * (ctrl_pts[:, -1, 1:-1, 1] + ctrl_pts[:, -2, 1:-1, 0])
ctrl_pts[:, -1, 1:-1, -1] = 0.5 * (
    ctrl_pts[:, -1, 1:-1, -2] + ctrl_pts[:, -2, 1:-1, -1]
)

ctrl_pts[:, 1:-1, 0, 0] = 0.5 * (ctrl_pts[:, 1:-1, 0, 1] + ctrl_pts[:, 1:-1, 1, 0])
ctrl_pts[:, 1:-1, 0, -1] = 0.5 * (ctrl_pts[:, 1:-1, 0, -2] + ctrl_pts[:, 1:-1, 1, -1])
ctrl_pts[:, 1:-1, -1, 0] = 0.5 * (ctrl_pts[:, 1:-1, -1, 1] + ctrl_pts[:, 1:-1, -2, 0])
ctrl_pts[:, 1:-1, -1, -1] = 0.5 * (
    ctrl_pts[:, 1:-1, -1, -2] + ctrl_pts[:, 1:-1, -2, -1]
)

# %% Create a mesh from this B-spline patch
mesh_vol = Mesh([spline], [ctrl_pts])

# Check element orientation to ensure right-handed outward normals
mesh_vol.plot_orientation()

# Uncomment to automatically correct orientation if needed
# mesh_vol.correct_orientation()

# %% Extract the exterior surface of the volumetric mesh
mesh_surf = mesh_vol.extract_border()
# Optionally select specific surface patches using `mesh_surf.subset`

mesh_surf.plot()

# %% Create the image to fit (typically provided as input data)

n = 100
fg, bg = 0.75, 0.25

# Create a binary image (`True`: inside object, `False`: outside)
X, Y, Z = np.meshgrid(*[np.linspace(-l, l, n)] * 3, indexing="ij")

# Simple spherical object
mask = (X**2 + Y**2 + Z**2) < (l / 2) ** 2

# More complex geometry: main sphere with three smaller embedded spheres
# mask = (
#     ((X**2 + Y**2 + Z**2) < (l / 2) ** 2)
#     | (((X - l / 2) ** 2 + Y**2 + Z**2) < (l / 4) ** 2)
#     | ((X**2 + (Y - l / 2) ** 2 + Z**2) < (l / 4) ** 2)
#     | ((X**2 + Y**2 + (Z - l / 2) ** 2) < (l / 4) ** 2)
# )

# Assign foreground and background gray levels
image = np.where(mask, fg, bg)

# Add small random noise
image += 1e-2 * (2 * np.random.rand(*mask.shape) - 1)

# Apply Gaussian smoothing to mimic imaging blur
image = gaussian_filter(image.astype("float"), sigma=1)

# Convert the image to a CT-like uint16 representation
gl_scaling = np.iinfo(np.uint16).max
image = (image * gl_scaling).astype(np.uint16)
fg_bg = (gl_scaling * fg, gl_scaling * bg)

mesh_surf.plot_in_image(image)

# %% Rescale the surface mesh to image (voxel) lengths

# Voxel size corresponding to the image resolution
voxel_size = 2 * l / n

# Convert surface control points from physical to voxel lengths
mesh_surf.unique_ctrl_pts /= voxel_size

mesh_surf.plot_in_image(image)

# %% Initialize the volVIC problem
# Define the problem using the surface mesh, image, gray levels,
# discretization parameters, and C¹ continuity enforcement
pb = Problem(
    mesh_surf,
    image,
    fg_bg=fg_bg,
    width_dx=0.5,
    surf_dx=1.0,
    C1_mode="all",
    verbose=True,
)

# %% Solve the problem (default initial rho = 1 voxel)
u_field, rho = pb.solve()

# %% Plot distance maps before and after optimization
pb.plot_results()
pb.plot_results(u_field=u_field)

# %% Visualize the deformed surface mesh
from copy import deepcopy

tmp = deepcopy(mesh_surf)
tmp.unique_ctrl_pts += u_field
tmp.plot()

# %% Propagate the surface displacement to the volumetric mesh
u_vol_field = pb.propagate_displacement_to_volume_mesh(
    u_field, mesh_vol, voxel_size=voxel_size
)

# Apply the volumetric displacement and visualize the deformed volume
tmp = deepcopy(mesh_vol)
tmp.unique_ctrl_pts += u_vol_field
tmp.plot()

# %%
