# %% imports

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from PIL import Image, ImageSequence
import pyvista as pv

pv.set_jupyter_backend("static")
palette = np.vstack([plt.get_cmap(c).colors for c in ["tab20", "tab20b", "tab20c"]])
np.random.seed(42)
np.random.shuffle(palette)
colors = [to_hex(c) for c in palette]

pv.global_theme.window_size = [960, 1080]
pv.global_theme.anti_aliasing = "ssaa"
pv.global_theme.multi_samples = 32
pv.global_theme.render_lines_as_tubes = True
pv.global_theme.line_width = 5
pv.global_theme.render_points_as_spheres = True
pv.global_theme.point_size = 30
pv.global_theme.show_edges = False


def setup_camera(pv_plotter):
    pv_plotter.view_vector((6, -2, 1.5))
    pv_plotter.camera.zoom(1.2)


# %% geo: beam

interior_mesh = pv.read("out_geo/beam_interior_0_0.vtu")
sep_mesh = pv.read("out_geo/beam_elements_borders_0_0.vtu")
ctrl_mesh = pv.read("out_geo/beam_control_points_0_0.vtu")

pv_plotter = pv.Plotter()
pv_plotter.add_mesh(
    interior_mesh,
    scalars="patch_id",
    style="surface",
    cmap=colors,
    show_scalar_bar=False,
    specular=1.0,
    specular_power=20,
)
pv_plotter.add_mesh(
    sep_mesh,
    style="wireframe",
    color="black",
)
pv_plotter.add_mesh(
    ctrl_mesh,
    scalars="patch_id",
    style="points",
    cmap=colors,
    show_scalar_bar=False,
    specular=1.0,
    specular_power=20,
)
setup_camera(pv_plotter)
pv_plotter.save_graphic("figs/beam_multipatch.pdf")
pv_plotter.show()

# %% geo: cell

interior_mesh = pv.read("out_geo/cell_interior_0_0.vtu")
sep_mesh = pv.read("out_geo/cell_elements_borders_0_0.vtu")
# ctrl_mesh = pv.read("out_geo/cell_control_points_0_0.vtu")

pv_plotter = pv.Plotter()
pv_plotter.add_mesh(
    interior_mesh,
    scalars="patch_id",
    style="surface",
    cmap=colors,
    show_scalar_bar=False,
    specular=1.0,
    specular_power=20,
)
pv_plotter.add_mesh(
    sep_mesh,
    style="wireframe",
    color="black",
)
setup_camera(pv_plotter)
pv_plotter.save_graphic("figs/cell_multipatch.pdf")
pv_plotter.show()

# %% fitting: before and after icp

with open("out_fitting/mesh_before_icp.pkl", "rb") as file:
    mesh_before = pkl.load(file)

with open("out_fitting/mesh_after_icp.pkl", "rb") as file:
    mesh_after = pkl.load(file)

image = np.load("out_fitting/image.npy")

meshes = [mesh_before, mesh_after]
saves = ["figs/before_ICP.pdf", "figs/after_ICP.pdf"]
for mesh, save in zip(meshes, saves):
    pv_plotter = pv.Plotter()
    mesh.plot_in_image(
        image,
        mode="voxels",
        interior_only=False,
        pv_plotter=pv_plotter,
        show=False,
        image_show_edges=False,
        image_opacity=1.0,
        image_interior_color="white",
        color="white",
        style="surface",
        specular=1.0,
        specular_power=20,
    )
    setup_camera(pv_plotter)
    pv_plotter.save_graphic(save)
    pv_plotter.show()

# %% fitting: distance before and after fitting

u_field = np.load("out_fitting/u_field.npy")

with open("out_fitting/problem.pkl", "rb") as file:
    pb = pkl.load(file)

u_fields = [None, u_field]
saves = ["figs/distance_init.pdf", "figs/distance_final.pdf"]
for u, save in zip(u_fields, saves):
    pv_plotter = pv.Plotter()
    pb.plot_results(
        u_field=u,
        interior_only=False,
        pv_plotter=pv_plotter,
        show=False,
        scalar_bar_args={"position_y": 0.85},
        smooth_shading=True,
    )
    setup_camera(pv_plotter)
    pv_plotter.save_graphic(save)
    pv_plotter.show()


# %% simu: plot boundary conditions

interior_mesh = pv.read("out_simu/results_interior_0_0.vtu")
elem_mesh = pv.read("out_simu/results_elements_borders_0_0.vtu")

with open("out_simu/saved_values.pkl", "rb") as file:
    l, ref_point, t, theta = pkl.load(file)

rigid_plate_bot = pv.Plane(
    center=[0, 0, -l],
    direction=[0, 0, 1],
    i_size=2 * l,
    j_size=2 * l,
)
rigid_plate_top = pv.Plane(
    center=ref_point + np.array([0, 0, 3e-3]),
    direction=[0, 0, 1],
    i_size=2 * l,
    j_size=2 * l,
)
rp_point = pv.PolyData(ref_point)
rp_point["dir_u"] = 3 * t[None]
glyph_t = rp_point.glyph(orient="dir_u", geom=pv.Arrow())
r = 0.5
angle = np.linspace(0, 1.7 * np.pi, 25)  # - np.pi / 2
x, y = r * np.cos(angle), r * np.sin(angle)
z = np.zeros_like(x)
body_pts = np.stack((x, y, z)).T
body = pv.MultipleLines(body_pts).tube(radius=(r / 10), n_sides=50)
dhead = body_pts[-1] - body_pts[-2]
dhead /= np.linalg.norm(dhead)
head = pv.Cone(
    center=body_pts[-1] + dhead * (r / 4),
    direction=dhead,
    height=r / 2,
    radius=3 * r / 20,
    resolution=50,
)
glyph_theta = body.merge(head, merge_points=False).translate(
    ref_point + 2 * t, inplace=False
)

pv_plotter = pv.Plotter()
pv_plotter.add_mesh(
    interior_mesh,
    color="white",
    specular=1.0,
    specular_power=20,
    smooth_shading=True,
)
pv_plotter.add_mesh(
    elem_mesh,
    color="black",
    style="wireframe",
)
pv_plotter.add_mesh(rigid_plate_bot, color="grey")
pv_plotter.add_mesh(rigid_plate_top, color="grey")
pv_plotter.add_mesh(
    rp_point,
    color="#d95f02",
    point_size=100,
    smooth_shading=True,
)
pv_plotter.add_mesh(glyph_t, color="#1b9e77", smooth_shading=True)
pv_plotter.add_mesh(glyph_theta, color="#1b9e77", smooth_shading=True)
setup_camera(pv_plotter)
pv_plotter.save_graphic("figs/boundary_conditions.pdf")
pv_plotter.show()

# %% simu: plot von mises on deformed

interior_mesh = pv.read("out_simu/results_interior_0_0.vtu")
elem_mesh = pv.read("out_simu/results_elements_borders_0_0.vtu")

pv_plotter = pv.Plotter()
n_colors = 15
sargs = dict(
    title="Von Mises Stress [MPa]\n",
    vertical=False,
    position_x=0.1,
    position_y=0.85,
    height=0.05,
    width=0.8,
    fmt="%.1e",
    title_font_size=25,
    label_font_size=25,
    n_colors=n_colors,
)
cmap = plt.colormaps["jet"].resampled(n_colors)
warped_interior = interior_mesh.warp_by_vector(vectors="U")
pv_plotter.add_mesh(
    warped_interior,
    scalars="von_mises",
    cmap=cmap,
    log_scale=True,
    scalar_bar_args=sargs,
    label="Deformed State",
)
pv_plotter.add_mesh(
    elem_mesh,
    color="#e6ab02",
    style="wireframe",
    line_width=3,
    label="Reference Configuration",
)
legend_actor = pv_plotter.add_legend(
    loc="lower center",
    bcolor="gray",
    size=(0.4, 0.1),
)
setup_camera(pv_plotter)
pv_plotter.save_graphic("figs/simu_von_mises.pdf")
pv_plotter.show()

# %% schema twin: CAO

interior_mesh = pv.read("out_geo/cell_interior_0_0.vtu")
# sep_mesh = pv.read("out_geo/cell_elements_borders_0_0.vtu")
# ctrl_mesh = pv.read("out_geo/cell_control_points_0_0.vtu")

pv_plotter = pv.Plotter()
pv_plotter.add_mesh(
    interior_mesh,
    color="white",
    style="surface",
    specular=1.0,
    specular_power=20,
)
setup_camera(pv_plotter)
pv_plotter.save_graphic("figs/twin_schema/CAO.pdf")
pv_plotter.show()

# %% schema twin: tomo


def load_tiff_stack(path):
    with Image.open(path) as img:
        return np.array([np.array(f) for f in ImageSequence.Iterator(img)])


image = load_tiff_stack("cropped_CT_scan.tiff")

pv_plotter = pv.Plotter()
pv_plotter.add_volume(
    image,
    cmap="gray_r",
    clim=[20119, 22776],
    opacity="sigmoid_5",
    show_scalar_bar=False,
)
setup_camera(pv_plotter)
pv_plotter.disable_anti_aliasing()
pv_plotter.save_graphic("figs/twin_schema/tomo.pdf")
pv_plotter.show()

# %% schema twin: simu

interior_mesh = pv.read("out_simu/results_interior_0_0.vtu")
elem_mesh = pv.read("out_simu/results_elements_borders_0_0.vtu")

pv_plotter = pv.Plotter()
n_colors = 15
sargs = dict(
    title="Von Mises Stress [MPa]\n",
    vertical=False,
    position_x=0.1,
    position_y=0.85,
    height=0.05,
    width=0.8,
    fmt="%.1e",
    title_font_size=25,
    label_font_size=25,
    n_colors=n_colors,
)
cmap = plt.colormaps["jet"].resampled(n_colors)
warped_interior = interior_mesh.warp_by_vector(vectors="U")
pv_plotter.add_mesh(
    warped_interior,
    scalars="von_mises",
    cmap=cmap,
    log_scale=True,
    scalar_bar_args=sargs,
)
setup_camera(pv_plotter)
pv_plotter.save_graphic("figs/twin_schema/simu.pdf")
pv_plotter.show()
