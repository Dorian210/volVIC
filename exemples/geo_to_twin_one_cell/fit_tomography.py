# %%
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle as pkl
from PIL import Image, ImageSequence
from copy import deepcopy

from volVIC import Problem, Mesh, find_fg_bg


# %%
with open("BCC_cell.pkl", "rb") as file:
    splines, separated_ctrl_pts, connectivity = pkl.load(file)

mesh = Mesh(splines, separated_ctrl_pts, connectivity)

voxel_size = 0.0108725
mesh.unique_ctrl_pts /= voxel_size

mesh.plot_orientation()

# %%
mesh.correct_orientation(axis=0)

mesh.plot_orientation()


# %%
surf_mesh = mesh.extract_border()

surf_mesh.plot()

subset_mesh = surf_mesh.subset(np.arange(0, surf_mesh.connectivity.nb_patchs, 2))

subset_mesh.plot()

with open("out_fitting/mesh_before_icp.pkl", "wb") as file:
    pkl.dump(subset_mesh, file)


# %%
def load_tiff_stack(path):
    with Image.open(path) as img:
        return np.array([np.array(f) for f in ImageSequence.Iterator(img)])


image_inside = load_tiff_stack("cropped_CT_scan.tiff")
fg, bg = find_fg_bg(image_inside)
n = 200
image = np.full([s + n for s in image_inside.shape], bg, dtype="uint16")
image[tuple(slice(n // 2, -n // 2, 1) for _ in range(3))] = image_inside

image = gaussian_filter(image, sigma=1)

np.save("out_fitting/image.npy", image)

subset_mesh.plot_in_image(image)

# %%
pb = Problem(
    deepcopy(subset_mesh),
    image,
    fg_bg_method="interp",
    width_dx=2.0,
    surf_dx=5.0,
    C1_mode="auto",
)

with open("out_fitting/mesh_after_icp.pkl", "wb") as file:
    pkl.dump(pb.mesh, file)

# %%
pts = subset_mesh.unique_ctrl_pts
(pts_to_lock,) = np.where(
    np.isclose(pts[0], pts[0].min())
    | np.isclose(pts[0], pts[0].max())
    | np.isclose(pts[1], pts[1].min())
    | np.isclose(pts[1], pts[1].max())
    | np.isclose(pts[2], pts[2].min())
    | np.isclose(pts[2], pts[2].max())
)
inds = np.hstack(
    (
        pts_to_lock + 0 * subset_mesh.connectivity.nb_unique_nodes,
        pts_to_lock + 1 * subset_mesh.connectivity.nb_unique_nodes,
        pts_to_lock + 2 * subset_mesh.connectivity.nb_unique_nodes,
    )
)
vals = np.zeros(inds.size)

pb.constraints.add_eqs_from_inds_vals(inds, vals)
pb.make_dirichlet()

# %%
u_field, rho = pb.solve(disable_parallel=False, verbose=True)

np.save("out_fitting/u_field.npy", u_field)

with open("out_fitting/problem.pkl", "wb") as file:
    pkl.dump(pb, file)

# %%
pb.plot_results()
pb.plot_results(u_field=u_field)

# %%
pb.save_paraview(
    u_field,
    folder="out_fitting",
    name="volvic_quantities",
    disable_parallel=False,
    verbose=True,
)

pb.mesh.save_paraview(
    "out_fitting",
    "fitted_surface",
    unique_fields={"u": u_field[None]},
    disable_parallel=False,
    verbose=True,
)

# %%
u_vol_field = pb.propagate_displacement_to_volume_mesh(
    u_field, mesh, disable_parallel=False
)

pts = mesh.unique_ctrl_pts
(pts_to_lock,) = np.where(
    np.isclose(pts[0], pts[0].min())
    | np.isclose(pts[0], pts[0].max())
    | np.isclose(pts[1], pts[1].min())
    | np.isclose(pts[1], pts[1].max())
    | np.isclose(pts[2], pts[2].min())
    | np.isclose(pts[2], pts[2].max())
)
u_vol_field[:, pts_to_lock] = 0.0

vol_mesh = deepcopy(mesh)

vol_mesh.unique_ctrl_pts += u_vol_field
vol_mesh.unique_ctrl_pts *= voxel_size

vol_mesh.save_paraview(
    "out_fitting",
    "fitted_volume",
    n_eval_per_elem=5,
    disable_parallel=False,
    verbose=True,
)
# %%
with open("BCC_cell_fitted.pkl", "wb") as file:
    pkl.dump(
        (vol_mesh.splines, vol_mesh.get_separated_ctrl_pts(), vol_mesh.connectivity),
        file,
    )

# %%
