# %%
import numpy as np
import matplotlib.pyplot as plt

# %% Define the geometry as a B-spline volume
from bsplyne import new_cube
l = 1
spline, ctrl_pts = new_cube([0, 0, 0], [0, 0, 1], 1)
ctrl_pts = spline.orderElevation(ctrl_pts, [2, 2, 2])
ctrl_pts = spline.knotInsertion(ctrl_pts, [4, 4, 4])
ctrl_pts = ctrl_pts*l/(2*np.linalg.norm(ctrl_pts, axis=0)[None])

ctrl_pts[:,  0,  0, 1:-1] = 0.5*(ctrl_pts[:,  0,  1, 1:-1] + ctrl_pts[:,  1,  0, 1:-1])
ctrl_pts[:,  0, -1, 1:-1] = 0.5*(ctrl_pts[:,  0, -2, 1:-1] + ctrl_pts[:,  1, -1, 1:-1])
ctrl_pts[:, -1,  0, 1:-1] = 0.5*(ctrl_pts[:, -1,  1, 1:-1] + ctrl_pts[:, -2,  0, 1:-1])
ctrl_pts[:, -1, -1, 1:-1] = 0.5*(ctrl_pts[:, -1, -2, 1:-1] + ctrl_pts[:, -2, -1, 1:-1])

ctrl_pts[:,  0, 1:-1,  0] = 0.5*(ctrl_pts[:,  0, 1:-1,  1] + ctrl_pts[:,  1, 1:-1,  0])
ctrl_pts[:,  0, 1:-1, -1] = 0.5*(ctrl_pts[:,  0, 1:-1, -2] + ctrl_pts[:,  1, 1:-1, -1])
ctrl_pts[:, -1, 1:-1,  0] = 0.5*(ctrl_pts[:, -1, 1:-1,  1] + ctrl_pts[:, -2, 1:-1,  0])
ctrl_pts[:, -1, 1:-1, -1] = 0.5*(ctrl_pts[:, -1, 1:-1, -2] + ctrl_pts[:, -2, 1:-1, -1])

ctrl_pts[:, 1:-1,  0,  0] = 0.5*(ctrl_pts[:, 1:-1,  0,  1] + ctrl_pts[:, 1:-1,  1,  0])
ctrl_pts[:, 1:-1,  0, -1] = 0.5*(ctrl_pts[:, 1:-1,  0, -2] + ctrl_pts[:, 1:-1,  1, -1])
ctrl_pts[:, 1:-1, -1,  0] = 0.5*(ctrl_pts[:, 1:-1, -1,  1] + ctrl_pts[:, 1:-1, -2,  0])
ctrl_pts[:, 1:-1, -1, -1] = 0.5*(ctrl_pts[:, 1:-1, -1, -2] + ctrl_pts[:, 1:-1, -2, -1])

# %% Create a mesh from this B-spline patch
from volVIC.Mesh import Mesh
mesh_vol = Mesh([spline], [ctrl_pts])
# mesh_vol.plot()

# %% Extract the exterior borders of the mesh
mesh_surf = mesh_vol.extract_border()
# mesh_surf.plot()

# %% Create the image to fit : this is usually an input to the method
n = 50
fg, bg = 0.75, 0.25
X, Y, Z = np.meshgrid(*[np.linspace(-l, l, n)]*3, indexing='ij')
mask = (X**2 + Y**2 + Z**2) < (l/2)**2
mask = (  ((X**2 + Y**2 + Z**2) < (l/2)**2) 
        | (((X - l/2)**2 + Y**2 + Z**2) < (l/4)**2) 
        | ((X**2 + (Y - l/2)**2 + Z**2) < (l/4)**2) 
        | ((X**2 + Y**2 + (Z - l/2)**2) < (l/4)**2))
image = np.where(mask, fg, bg)
from scipy.ndimage import gaussian_filter
image = gaussian_filter(image.astype('float'), sigma=1)
# image = (image*np.iinfo(np.uint16).max).astype(np.uint16)

c = plt.imshow(image[n//2])
plt.colorbar(c)

# %% Create a marching cubes mesh from the image to roughly initiallize the mesh in the image
from volVIC.marching_cubes import marching_cubes
threshold = 0.5*(fg + bg)
stl_mc = marching_cubes(image, threshold)

import pyvista as pv
pv_plotter = pv.Plotter()
pv_plotter.add_mesh(stl_mc, color="#7570b3", opacity=0.5, label="Real geometry")
pv_plotter.add_legend(bcolor="lightgray")
pv_plotter.show()
# %% Place the mesh in the image
voxel_size = 2*l/n
mesh_surf.unique_ctrl_pts /= voxel_size
R, t, d_max = mesh_surf.ICP_rigid_body_transform(stl_mc, plot_after=False)

# pv_plotter = mesh_surf.plot(show=False, color="white", label="B-spline")
# pv_plotter.add_mesh(stl_mc, color="#7570b3", opacity=0.5, label="Real geometry")
# pv_plotter.add_legend(bcolor="lightgray")
# pv_plotter.show()

# %% Define the virtual image : the target graylevel profile
from volVIC.virtual_image import g_slide
def virtual_image(xi, eta, gamma, rho):
    return g_slide(xi, eta, gamma, rho, fg, bg)

# %% Define the integration space and operators of the partially linearized VIC
from volVIC.VirtualImageCorrelationEnergyElem import make_image_energies
h = d_max + 1
image_energies = make_image_energies(mesh_surf, 
                                     h, 
                                     width_dx=0.25, 
                                     surf_dx=1., 
                                     alpha=0, 
                                     virtual_image=virtual_image, 
                                     verbose=True)

# %% Define the interpatch C1 constraints change of variable and lock a point
from volVIC.C1_triplets import make_C1_eqs, make_dirichlet_eqs, compute_C_from_eqs
C1_eqs = make_C1_eqs(mesh_surf, 'all', field_size=3)
nn = mesh_surf.connectivity.nb_unique_nodes
dir_eqs = make_dirichlet_eqs(mesh_surf, np.array([0*nn, 1*nn, 2*nn], dtype='int') + (nn - 1))
C = compute_C_from_eqs([dir_eqs, C1_eqs])

# %% Define the by-patch intrapatch membrane strain operator
from volVIC.membrane_stiffness import make_membrane_stifness
membrane_K = make_membrane_stifness(mesh_surf, verbose=True)

# %%
from volVIC.VirtualImageCorrelationEnergyElem import plot_last_profile, compute_image_energy_operators
from sksparse.cholmod import cholesky
def iter(u_field, rho, mesh, image_energies, image, membrane_K, membrane_weight, C, verbose=True):
    E, grad, H, dE_drho, d2E_drho2 = compute_image_energy_operators(
        image_energies, mesh, image, u_field, rho, verbose=verbose)
    if verbose: plot_last_profile(image_energies)
    u = u_field.ravel()
    print(f"E_vic = {E:.3E}")
    E_mem = 0.5*(membrane_K@u)@u
    print(f"E_mem = {E_mem:.3E}")
    global H_tot, grad_tot
    grad_tot = C.T@(grad + membrane_weight*membrane_K@u)
    H_tot = C.T@(H + membrane_weight*membrane_K)@C
    factor = cholesky(H_tot)
    dof = factor(-grad_tot)
    du_field = (C@dof).reshape(u_field.shape)
    drho = -dE_drho/d2E_drho2
    return du_field, drho

membrane_weight = 1e-1
u_field = np.zeros((3, mesh_surf.connectivity.nb_unique_nodes))
rho = h/4

du_field, drho = iter(u_field, rho, mesh_surf, image_energies, image, membrane_K, membrane_weight, C)


# %%
for i in range(20):
    du_field, drho = iter(u_field, rho, mesh_surf, image_energies, image, membrane_K, membrane_weight, C)
    u_field += du_field
    rho += drho
    print(f"new rho = {rho}")
    print(f"|drho|/|rho| = {abs(drho)/abs(rho)}")
    print(f"||du||/||u|| = {np.linalg.norm(du_field)/np.linalg.norm(u_field)}")
    print(f"iter {i} done.")
    if max(abs(drho)/abs(rho), np.linalg.norm(du_field)/np.linalg.norm(u_field))<1e-2:
        break
    
# %%
from volVIC.VirtualImageCorrelationEnergyElem import compute_distance_field
from copy import deepcopy
tmp_mesh = deepcopy(mesh_surf)
tmp_mesh.unique_ctrl_pts += u_field
tmp_mesh.plot_in_image(image, 0.5*(fg + bg), mode='marching cubes')

# %%
tmp_mesh = deepcopy(mesh_surf)
tmp_mesh.unique_ctrl_pts += u_field
XI_list = [(e.xi, e.eta) for e in image_energies]
d = compute_distance_field(mesh_surf, image_energies)
# tmp_mesh.plot_in_image(image, 0.5*(fg + bg), separated_field=d, XI_list=XI_list, mode='marching cubes')
tmp_mesh.plot(separated_field=d, XI_list=XI_list)

# %%
image_mesh = marching_cubes(image, threshold)
from copy import deepcopy
tmp_mesh = deepcopy(mesh_surf)
tmp_mesh.unique_ctrl_pts += u_field
pv_plotter = tmp_mesh.plot(show=False)
pv_plotter.add_mesh(
    image_mesh, 
    show_edges=True, 
    edge_color="black", 
    line_width=0.1, 
    opacity=0.75, 
    color="white")
# image_mesh.points -= 0.5
# pv_plotter.add_mesh(
#     image_mesh, 
#     show_edges=True, 
#     edge_color="black", 
#     line_width=0.1, 
#     opacity=0.75, 
#     color="white")
# image_mesh.points += 1
# pv_plotter.add_mesh(
#     image_mesh, 
#     show_edges=True, 
#     edge_color="black", 
#     line_width=0.1, 
#     opacity=0.75, 
#     color="white")
pv_plotter.show()

# %%
import pyvista as pv
pv.wrap(stl_mc).bounds
from copy import deepcopy
tmp_mesh = deepcopy(mesh_surf)
tmp_mesh.unique_ctrl_pts += u_field
stl_mesh = tmp_mesh.make_stl_mesh()
print(f"mesh {np.hstack((stl_mesh.min_, stl_mesh.max_))}")
print(f"slt  {np.array(pv.wrap(stl_mc).bounds)[:6]}")
# %%
