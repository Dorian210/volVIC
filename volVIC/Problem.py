# %%
from typing import Union, Literal, Iterable, Callable
import numpy as np
import scipy.sparse as sps
from tqdm import trange
import meshio as io
from sksparse.cholmod import cholesky
import pyvista as pv
from volVIC.Mesh import Mesh, MeshLattice
from volVIC.marching_cubes import marching_cubes
from volVIC.C1_triplets import get_C1_triplets, make_C1_C
from volVIC.membrane_stiffness import make_membrane_stifness
from volVIC.find_fg_bg import find_fg_bg
from volVIC.VirtualImageCorrelationEnergyElem import VirtualImageCorrelationEnergyElem, make_image_energies
from volVIC.virtual_image import g_slide

class Problem:
    
    mesh: Mesh
    image: np.ndarray[np.uint16]
    C1_ABC: tuple[np.ndarray[np.integer], np.ndarray[np.integer], np.ndarray[np.integer]]
    image_energies: list[VirtualImageCorrelationEnergyElem]
    
    def __init__(
        self, 
        mesh: Mesh, 
        image: np.ndarray[np.uint16], 
        ICP_init: bool=True, 
        fg_bg: Union[tuple[float, float], None]=None, 
        virtual_image: Callable[[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating], float], tuple[np.ndarray[np.floating], np.ndarray[np.floating]]]=g_slide, 
        h: Union[float, None]=None, 
        width_dx: float=2., 
        surf_dx: float=1., 
        alpha: Union[float, tuple[tuple[float, float], tuple[float, float]]]=0., 
        C1_mode: Union[None, Literal['auto', 'none', 'all'], np.ndarray[np.integer]]=None, 
        membrane_weight: Union[None, Literal['auto', 'none', 'all'], float]=None, 
        ):
        
        self.mesh = mesh
        self.image = image
        
        if fg_bg is None:
            self.fg, self.bg = self.find_fg_bg()
        else:
            self.fg, self.bg = fg_bg
        self.Rmat, self.tvec, h = self.initialize_h_mesh_ICP(ICP_init, h)
            
        self.image_energies = make_image_energies(self.mesh, 
                                                  h, 
                                                  width_dx=width_dx, 
                                                  surf_dx=surf_dx, 
                                                  alpha=alpha, 
                                                  virtual_image=virtual_image, 
                                                  verbose=True)
        self.preprocess_image_energies()
        
        self.C1_ABC = get_C1_triplets(self.mesh, C1_mode)
        self.C1_C = make_C1_C(self.mesh, self.C1_ABC, 3)
        
        self.membrane_K = make_membrane_stifness(self.mesh, verbose=True)
        
        self.membrane_weight = self.make_membrane_weight(membrane_weight)
        
    
    def find_fg_bg(self):
        return find_fg_bg(self.image)
    
    def initialize_h_mesh_ICP(self, ICP_init: bool=True, h: Union[float, None]=None):
        compute_h = h is None
        if ICP_init or compute_h:
            stl_mc = marching_cubes(self.image, 0.5*(self.fg + self.bg))
            if ICP_init:
                Rmat, tvec, max_dist = self.mesh.ICP_rigid_body_transform(stl_mc)
                if compute_h:
                    h = max_dist
            else:
                Rmat = np.eye(3)
                tvec = np.zeros(3)
                if compute_h:
                    distance_mesh = self.mesh.distance_to_meshio(stl_mc)
                    h = np.abs(distance_mesh.point_data['implicit_distance']).max()
        return Rmat, tvec, h
    
    def preprocess_image_energies(self):
        # TODO
        return
        ur = np.zeros(self.mesh.connectivity.nb_unique_nodes + 1, dtype='float')
        E = 0.
        dE_dur = np.zeros_like(ur)
        d2E_dur2_data = []
        d2E_dur2_rows = []
        d2E_dur2_cols = []
        separated_inds = self.connectivity.unique_field_indices((self.energies[0].ctrl_pts.shape[0],))
        for patch in range(self.connectivity.nb_patchs):
            inds_elem = np.append(separated_inds[patch].ravel(), [ur.size - 1])
            ur_elem = ur[inds_elem]
            E_elem, dE_elem_dur_elem, d2E_elem_dur_elem2 = self.energies[patch].E_dE_dur_d2E_dur2(ur_elem, image, binning=binning, save=save)
            E += E_elem
            np.add.at(dE_dur, inds_elem, dE_elem_dur_elem)
            d2E_elem_dur_elem2 = d2E_elem_dur_elem2.tocoo()
            d2E_dur2_data.append(d2E_elem_dur_elem2.data)
            d2E_dur2_rows.append(inds_elem[d2E_elem_dur_elem2.row])
            d2E_dur2_cols.append(inds_elem[d2E_elem_dur_elem2.col])
        E = float(E)
        d2E_dur2 = sps.coo_matrix((np.hstack(d2E_dur2_data), (np.hstack(d2E_dur2_rows), np.hstack(d2E_dur2_cols))), shape=(ur.size, ur.size))
        if save:
            self.save_paraview()
        return E, dE_dur, d2E_dur2
    
    def make_membrane_weight(self, membrane_weight: Union[None, Literal['auto', 'none', 'all'], float]) -> float:
        if isinstance(membrane_weight, float):
            return membrane_weight
        raise NotImplementedError("membrane_weight must be specified for now.")
    
    def make_image_energy_operators(
        self, 
        u_field: np.ndarray[np.floating], 
        rho: float
        ) -> tuple[float, np.ndarray[np.floating], sps.spmatrix, float, float]:
        separated_u_field = self.mesh.unique_to_separated(u_field)
        E = 0
        grad = []
        H = []
        dE_drho = 0
        d2E_drho2 = 0
        for patch in trange(self.mesh.connectivity.nb_patchs, desc="Make image energy operators"):
            E_elem, grad_elem, H_elem, dE_drho_elem, d2E_drho2_elem = self.image_energies[patch].E_dE_du_d2E_du2_dE_drho_d2E_drho2(
                separated_u_field[patch].ravel(), rho, self.image)
            E += E_elem
            grad.append(grad_elem)
            H.append(H_elem)
            dE_drho += dE_drho_elem
            d2E_drho2 += d2E_drho2_elem
        grad = self.mesh.assemble_grads(grad, 3)
        H = self.mesh.assemble_hessians(H, 3)
        return E, grad, H, dE_drho, d2E_drho2
    
    def one_gauss_newton_iter(
        self, 
        u_field: np.ndarray[np.floating], 
        rho: float
        ) -> tuple[float, np.ndarray[np.floating], sps.spmatrix, float, float]:
        E, grad, H, dE_drho, d2E_drho2 = self.make_image_energy_operators(u_field, rho)
        
        u = u_field.ravel()
        E += self.membrane_weight*0.5*(self.membrane_K@u)@u
        grad += self.membrane_weight*self.membrane_K@u
        grad = self.C1_C.T@grad
        H += self.membrane_weight*self.membrane_K
        H = self.C1_C.T@H@self.C1_C
        factor = cholesky(H)
        dof = factor(-grad)
        du_field = (self.C1_C@dof).reshape(u_field.shape)
        
        drho = -dE_drho/d2E_drho2
        
        return du_field, drho


if __name__=='__main__':
    import sys
    sys.path.append("/home-local/dbichet/Documents/These/code/VolVIC")
    from BCC_cell import BCC_grid
    l, m, n = 9, 4, 4
    cyl_radius = 0.7/2 * 1.3
    cell_half_length = 2
    order_elevation = [0, 0, 1]
    knot_insertion = [0, 0, 3]
    splines, separated_ctrl_pts, connectivity, _ = BCC_grid(l, m, n, cyl_radius, cell_half_length, 
                                                            order_elevation=order_elevation, 
                                                            knot_insertion=knot_insertion)
    nb_patchs_cell = connectivity.nb_patchs//(l*m*n)
    vol_mesh = MeshLattice(l, m, n, splines[:nb_patchs_cell], separated_ctrl_pts, connectivity)
    border_mesh = vol_mesh.extract_border()
    to_remove = (  np.logical_or.reduce(np.isclose(border_mesh.unique_ctrl_pts, np.array([0, 0, 0])[:, None]), axis=0) 
                 | np.logical_or.reduce(np.isclose(border_mesh.unique_ctrl_pts,  2*cell_half_length*np.array([l, m, n])[:, None]), axis=0))
    # to_remove = np.isclose(border_mesh.unique_ctrl_pts[0], 0) | np.isclose(border_mesh.unique_ctrl_pts[0],  2*cell_half_length*l)
    patches_to_keep, = np.where([not arr.all() for arr in border_mesh.unique_to_separated(to_remove)])
    surf_mesh = border_mesh.subset(patches_to_keep)
    surf_mesh_lattice = MeshLattice.from_mesh(surf_mesh, l, m, n)
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(np.deg2rad(90)*np.array([0, -1, 0]))
    surf_mesh_lattice.unique_ctrl_pts = rot.apply(surf_mesh_lattice.unique_ctrl_pts.T).T
    surf_mesh_lattice.unique_ctrl_pts /= 0.0108725*8
    
    from skimage.io import imread
    image = imread("/home-local/dbichet/Documents/These/code/VolVIC/exemples/metalic_BCC_traction/SlicesY-Lattice_BCC_traction_binned.tiff")
    
    pb = Problem(surf_mesh_lattice, image, membrane_weight=1e7)
    
    # du_field, drho = pb.one_gauss_newton_iter(np.zeros((3, surf_mesh_lattice.connectivity.nb_unique_nodes)), 0.5)
    
    # surf_mesh_lattice.unique_ctrl_pts += du_field
    
    # surf_mesh_lattice.plot()
    
# %%
