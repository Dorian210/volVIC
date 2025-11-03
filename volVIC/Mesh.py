# %%
from typing import Callable, Iterable, Literal, Union
import numpy as np
import scipy.sparse as sps
import pyvista as pv
import meshio as io
import pickle
from bsplyne import BSpline, MultiPatchBSplineConnectivity, parallel_blocks
from treeIDW import treeIDW

from volVIC.find_fg_bg import find_fg_bg
from volVIC.marching_cubes import marching_cubes

def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook ou lab
        elif shell == 'TerminalInteractiveShell':
            return False  # IPython en terminal
        else:
            return False  # autre type inconnu
    except (NameError, ImportError):
        return False      # En script Python standard

class Mesh:
    
    connectivity: MultiPatchBSplineConnectivity
    splines: list[BSpline]
    unique_ctrl_pts: np.ndarray[np.floating]
    ref_inds: np.ndarray[np.integer]
    
    def __init__(self, 
                 splines: list[BSpline], 
                 ctrl_pts: Union[list[np.ndarray[np.floating]], np.ndarray[np.floating]], 
                 connectivity: Union[MultiPatchBSplineConnectivity, None]=None, 
                 ref_inds: Union[np.ndarray[np.integer], None]=None):
        """
        Initialise la structure de mesh multi-patch.

        Parameters
        ----------
        separated_ctrl_pts : list[np.ndarray]
            Liste des points de contrôle par patch (forme : (dim_phys, n1, ..., nd)).
        splines : list[BSpline]
            Liste des objets BSpline associés à chaque patch.
        connectivity : MultiPatchBSplineConnectivity, optional
            Objet de connectivité multi-patch. Si None, il est reconstruit automatiquement.
        """
        if connectivity is None:
            if isinstance(ctrl_pts, np.ndarray):
                raise TypeError("Can only infer connectivity from separated control points list.")
            self.connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(ctrl_pts)
        else:
            self.connectivity = connectivity
        self.splines = splines
        if isinstance(ctrl_pts, np.ndarray): # unique or unpacked state
            if ctrl_pts.shape[1]==self.connectivity.nb_unique_nodes: # unique state
                self.unique_ctrl_pts = ctrl_pts
            else:
                self.unique_ctrl_pts = self.connectivity.pack(ctrl_pts)
        else:
            self.set_separated_ctrl_pts(ctrl_pts)
        if ref_inds is None:
            self.ref_inds = np.arange(self.connectivity.nb_unique_nodes)
        else:
            self.ref_inds = ref_inds
    
    def _repr_png_(self):
        self.show()
    
    def show(self):
        if hasattr(self, '_screenshot'):
            import matplotlib.pyplot as plt
            plt.imshow(self._screenshot)
            plt.axis('off')
            plt.title("Screenshot of last plot")
            plt.show()
        else:
            print(self)
    
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def separated_to_unique(self, field, method=None):
        if method is None:
            return self.connectivity.pack(self.connectivity.agglomerate(field))
        else:
            return self.connectivity.pack(self.connectivity.agglomerate(field), method=method)
    
    def unique_to_separated(self, field):
        return self.connectivity.separate(self.connectivity.unpack(field))
    
    def get_separated_ctrl_pts(self) -> list[np.ndarray[np.floating]]:
        return self.unique_to_separated(self.unique_ctrl_pts)
    
    def set_separated_ctrl_pts(self, separated_ctrl_pts: list[np.ndarray[np.floating]]):
        self.unique_ctrl_pts = self.separated_to_unique(separated_ctrl_pts)
    
    def __call__(
        self, 
        XI: Iterable[tuple[np.ndarray[np.floating], ...]], 
        k: Union[Iterable[Union[int, tuple[int, ...]]], int]=0, 
        data: Union[np.ndarray, Iterable[np.ndarray]]=None
        ) -> list:
        if isinstance(k, int):
            k = [k]*self.connectivity.nb_patchs
        elif isinstance(k, Iterable):
            if len(k)!=self.connectivity.nb_patchs:
                raise ValueError(f"Expected {self.connectivity.nb_patchs} entries in 'k', got {len(k)}.")
        else:
            raise TypeError(f"'k' must be an int or an iterable of ints/tuples, got {type(k).__name__}.")
        if data is None:
            data = self.get_separated_ctrl_pts()
        elif isinstance(data, np.ndarray):
            data = self.unique_to_separated(data)
        elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            if len(data) != self.connectivity.nb_patchs:
                raise ValueError(f"Expected {self.connectivity.nb_patchs} data arrays, got {len(data)}.")
        else:
            raise TypeError(
                "'data' must be None, a NumPy array, or an iterable of NumPy arrays."
                f" Got object of type {type(data).__name__}."
            )
        all_args = list(zip(data, XI, k))
        results = parallel_blocks(self.splines, all_args, pbar_title="Evaluate mesh")
        return results

    def assemble_grads(self, terms: list[np.ndarray], field_size: int) -> np.ndarray:
        size = self.connectivity.nb_unique_nodes*field_size
        total = np.zeros(size, dtype=terms[0].dtype)
        separated_inds = self.connectivity.unique_field_indices((field_size,), representation="separated")
        for patch, term in enumerate(terms):
            inds = separated_inds[patch].ravel()
            np.add.at(total, inds, term)
        return total

    def assemble_hessians(self, terms: list[sps.spmatrix], field_size: int) -> sps.spmatrix:
        rows = []
        cols = []
        data = []
        separated_inds = self.connectivity.unique_field_indices((field_size,), representation="separated")
        for patch, mat in enumerate(terms):
            inds = separated_inds[patch].ravel()
            mat = mat.tocoo()
            rows.append(inds[mat.row])
            cols.append(inds[mat.col])
            data.append(mat.data)
        size = self.connectivity.nb_unique_nodes*field_size
        return sps.coo_matrix((np.hstack(data), (np.hstack(rows), np.hstack(cols))), shape=(size, size))

    def extract_border(self) -> "Mesh":
        """
        Extrait les bords extérieurs du maillage B-spline multipatch.

        Returns
        -------
        Mesh
            Un nouvel objet Mesh représentant les bords extérieurs.
        """
        border_connectivity, border_splines, full_to_border = self.connectivity.extract_exterior_borders(self.splines)
        border_ctrl_pts = self.unique_ctrl_pts[..., full_to_border]
        border_ref_inds = self.ref_inds[full_to_border]
        return Mesh(list(border_splines), border_ctrl_pts, border_connectivity, border_ref_inds)
    
    def subset(self, patchs_to_keep: np.ndarray[np.integer]) -> "Mesh":
        splines = np.empty(self.connectivity.nb_patchs, dtype=object)
        splines[:] = self.splines
        subset_connectivity, subset_splines, full_to_subset = self.connectivity.subset(splines, patchs_to_keep)
        subset_ctrl_pts = self.unique_ctrl_pts[..., full_to_subset]
        subset_ref_inds = self.ref_inds[full_to_subset]
        return Mesh(list(subset_splines), subset_ctrl_pts, subset_connectivity, subset_ref_inds)
    
    def propagate_field(self, field_values, indices):
        shape = field_values[None].shape[:-1]
        field_values = field_values.reshape((np.prod(shape), -1))
        unknown_mask = np.ones(self.connectivity.nb_unique_nodes, dtype='bool')
        unknown_mask[indices] = False
        known_points = self.unique_ctrl_pts[:, indices].T
        unknown_points = self.unique_ctrl_pts[:, unknown_mask].T
        field = np.empty((field_values.shape[0], self.connectivity.nb_unique_nodes), dtype='float')
        field[:, indices] = field_values
        field[:, unknown_mask] = treeIDW(known_points, field_values.T, unknown_points, parallel=True).T
        return field.reshape((*shape[1:], -1))
    
    def propagate_field_from_submesh(self, submesh, field_values):
        shape = field_values[None].shape[:-1]
        field_values = field_values.reshape((np.prod(shape), -1))
        _, idx_mesh, idx_submesh = np.intersect1d(self.ref_inds, submesh.ref_inds, return_indices=True)
        field = self.propagate_field(field_values[:, idx_submesh], idx_mesh)
        return field.reshape((*shape[1:], -1))

    def make_stl_mesh(self, n_eval_per_elem=10, remove_empty_areas=True):
        from stl import mesh
        if self.connectivity.npa==2:
            border_mesh = self
        elif self.connectivity.npa==3:
            border_mesh = self.extract_border()
        else:
            raise NotImplementedError(f"Making stl mesh not implemented for {self.connectivity.npa}-D shapes.")
        grids = border_mesh([spline.linspace(n_eval_per_elem=n_eval_per_elem) for spline in border_mesh.splines])
        tri = []
        for grid in grids:
            A = grid[:,  :-1,  :-1].reshape((3, -1)).T[:, None, :]
            B = grid[:,  :-1, 1:  ].reshape((3, -1)).T[:, None, :]
            C = grid[:, 1:  ,  :-1].reshape((3, -1)).T[:, None, :]
            D = grid[:, 1:  , 1:  ].reshape((3, -1)).T[:, None, :]
            tri1 = np.concatenate((A, B, C), axis=1)
            tri2 = np.concatenate((D, C, B), axis=1)
            tri.append(np.concatenate((tri1, tri2), axis=0))
        tri = np.concatenate(tri, axis=0)
        data = np.empty(tri.shape[0], dtype=mesh.Mesh.dtype)
        data['vectors'] = tri
        m = mesh.Mesh(data, remove_empty_areas=remove_empty_areas)
        return m
    
    def plot(
        self, 
        n_eval_per_elem: Union[int, Iterable[int]]=10, 
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]]=None, 
        unique_field: Union[np.ndarray, None]=None, 
        separated_field: Union[list[np.ndarray], list[callable], None]=None, 
        interior_only: bool=True, 
        plt_ctrl_mesh: bool=False, 
        pv_plotter: Union[pv.Plotter, None]=None, 
        show: bool=True, 
        elem_sep_color: str="black", 
        ctrl_poly_color: str="green", 
        **pv_add_mesh_kwargs):
        
        if unique_field is not None:
            unique_fields = {"": unique_field[None, None] if unique_field.ndim==1 else unique_field[None]}
            separated_fields = None
        elif separated_field is not None:
            unique_fields = {}
            if isinstance(separated_field[0], Callable):
                separated_fields = [{"": lambda *args: field(*args)[None]} for field in separated_field]
            else:
                separated_fields = [{"": field[None]} for field in separated_field]
        else:
            unique_fields = {}
            separated_fields = None
        
        if pv_plotter is None:
            pv_plotter = pv.Plotter()
        
        elem_int_mesh, = self.connectivity.make_elements_interior_meshes(
            self.splines,
            self.get_separated_ctrl_pts(),
            n_step=1,
            n_eval_per_elem=n_eval_per_elem,
            unique_fields=unique_fields,
            separated_fields=separated_fields,
            XI_list=XI_list,
            verbose=True
        )
        pv_plotter.add_mesh(elem_int_mesh, **pv_add_mesh_kwargs)
        if not interior_only:
            elem_sep_mesh, = self.connectivity.make_elem_separator_meshes(
                self.splines,
                self.get_separated_ctrl_pts(),
                n_step=1,
                n_eval_per_elem=n_eval_per_elem,
                XI_list=XI_list
            )
            if "label" in pv_add_mesh_kwargs: del pv_add_mesh_kwargs["label"]
            pv_add_mesh_kwargs["style"] = "wireframe"
            pv_add_mesh_kwargs["color"] = elem_sep_color
            pv_plotter.add_mesh(elem_sep_mesh, **pv_add_mesh_kwargs)
            if plt_ctrl_mesh:
                ctrl_poly_mesh, = self.connectivity.make_control_poly_meshes(
                    self.splines,
                    self.get_separated_ctrl_pts(),
                    n_step=1,
                    n_eval_per_elem=n_eval_per_elem,
                    XI_list=XI_list
                )
                pv_add_mesh_kwargs["color"] = ctrl_poly_color
                pv_plotter.add_mesh(ctrl_poly_mesh, **pv_add_mesh_kwargs)
        
        if show:
            pv_plotter.show()
            if in_notebook():
                self._screenshot = pv_plotter.screenshot(return_img=True)
        
        return pv_plotter
    
    def plot_in_image(
        self, 
        image, 
        threshold: Union[float, None]=None, 
        mode: Literal['marching cubes', 'voxels']='marching cubes', 
        n_eval_per_elem: Union[int, Iterable[int]]=10, 
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]]=None, 
        interior_only: bool=True, 
        pv_plotter: Union[pv.Plotter, None]=None, 
        show: bool=True, 
        **pv_add_mesh_kwargs):
        
        if threshold is None:
            fg, bg = find_fg_bg(image, save_file_base=None)
            threshold = 0.5*(fg + bg)
        
        if mode=='marching cubes':
            image_mesh = marching_cubes(image, threshold)
        elif mode=='voxels':
            image_mesh = pv.wrap(image).threshold(threshold)
        else:
            raise KeyError(f"Unrecognized mode '{mode}'.")

        pv_plotter = self.plot(
            n_eval_per_elem=n_eval_per_elem, 
            XI_list=XI_list, 
            interior_only=interior_only, 
            pv_plotter=pv_plotter, 
            show=False, 
            **pv_add_mesh_kwargs)
        
        pv_plotter.add_mesh(
            image_mesh, 
            show_edges=True, 
            edge_color="black", 
            line_width=0.1, 
            opacity=0.75, 
            color="white")
        
        if show:
            pv_plotter.show()
        
        return pv_plotter    
    
    def ICP_rigid_body_transform(
        self, 
        tgt_meshio: io.Mesh, 
        n_eval_per_elem: Union[int, Iterable[int]]=5, 
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]]=None, 
        plot_after: bool=True
        ) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating], float]:
        """Compute the rigid body parameters (rx, ry, rz, tx, ty, tz) 
        to displace `self` as close as possible to `mesh` and actually displace it.

        Parameters
        ----------
        tgt_meshio : io.Mesh
            Target mesh.
        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each isoparametric dimension.
            By default, 10.
            - If an `int` is provided, the same number is used for all dimensions.
            - If an `Iterable` is provided, each value corresponds to a different dimension.
        XI : tuple[np.ndarray[np.floating], ...], optional
            Isoparametric coordinates at which to evaluate the B-spline.
            If not `None`, overrides the `n_eval_per_elem` parameter.
            If `None`, a regular grid is generated according to `n_eval_per_elem`.

        Returns
        -------
        Rmat, tvec, max_dist : tuple[np.ndarray[np.floating], np.ndarray[np.floating], float]
            Rotation matrix, translation vector and maximum distance.
            Minimizes the distance between the meshes when applied to `self`.
            The rotations are applied before the translation (`Rmat@xyz + tvec`).
        """

        # Create source and target PolyData
        if self.connectivity.npa==2:
            border_mesh = self
        elif self.connectivity.npa==3:
            border_mesh = self.extract_border()
        else:
            raise NotImplementedError(f"Making finding ICP rigid body transform not implemented for {self.connectivity.npa}-D shapes.")
        src_meshio = border_mesh.connectivity.make_elements_interior_meshes(
            border_mesh.splines, 
            border_mesh.get_separated_ctrl_pts(), 
            n_eval_per_elem=n_eval_per_elem, 
            XI_list=XI_list, 
            verbose=False
        )[0]
        src = pv.from_meshio(src_meshio)
        tgt = pv.from_meshio(tgt_meshio)

        # Perform ICP alignment
        aligned, matrix = src.align(
            tgt,
            max_landmarks=200,
            max_mean_distance=1e-6,
            max_iterations=100,
            start_by_matching_centroids=True,
            return_matrix=True
        )

        # Decompose transformation matrix
        Rmat = matrix[:3, :3]
        tvec = matrix[:3, 3]
        
        # Apply transformation to self
        self.unique_ctrl_pts = Rmat@self.unique_ctrl_pts + tvec[:, None]
        
        if plot_after:
            plotter = pv.Plotter()
            plotter.add_mesh(aligned, color="blue", opacity=0.8, label="Aligned surface mesh")
            plotter.add_mesh(tgt, color="red", opacity=0.5, label="STL Marching Cubes")
            plotter.add_legend()
            plotter.show()
        
        # Compute max distance between aligned and target meshes
        aligned.compute_implicit_distance(tgt.extract_surface(), inplace=True)
        dist_array = aligned["implicit_distance"]
        max_dist = float(np.max(np.abs(dist_array)))
        
        return Rmat, tvec, max_dist
    def distance_to_meshio(
        self, 
        tgt_meshio: io.Mesh, 
        n_eval_per_elem: Union[int, Iterable[int]]=5, 
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]]=None
        ) -> io.Mesh:
        if self.connectivity.npa==2:
            border_mesh = self
        elif self.connectivity.npa==3:
            border_mesh = self.extract_border()
        else:
            raise NotImplementedError(f"Distance field not implemented for {self.connectivity.npa}-D shapes.")
        src_meshio = border_mesh.connectivity.make_elements_interior_meshes(
            border_mesh.splines, 
            border_mesh.get_separated_ctrl_pts(), 
            n_eval_per_elem=n_eval_per_elem, 
            XI_list=XI_list, 
            verbose=False
        )[0]
        src = pv.from_meshio(src_meshio)
        tgt = pv.from_meshio(tgt_meshio)
        src.compute_implicit_distance(tgt.extract_surface(), inplace=True)
        return pv.to_meshio(src)
    
    def save_paraview(self, 
                      path: str, 
                      name: str, 
                      n_step: int=1, 
                      n_eval_per_elem: Union[int, Iterable[int]]=10, 
                      unique_fields: dict={}, 
                      separated_fields: Union[list[dict], None]=None, 
                      XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]]=None, 
                      groups: Union[dict[str, dict[str, Union[str, int]]], None]=None, 
                      make_pvd: bool=True, 
                      verbose: bool=True, 
                      fields_on_interior_only: Union[bool, Literal['auto'], list[str]]='auto'):
        return self.connectivity.save_paraview(self.splines, 
                                               self.get_separated_ctrl_pts(), 
                                               path, 
                                               name, 
                                               n_step, 
                                               n_eval_per_elem, 
                                               unique_fields, 
                                               separated_fields, 
                                               XI_list, 
                                               groups, 
                                               make_pvd, 
                                               verbose, 
                                               fields_on_interior_only)

if __name__=='__main__':
    import os, sys
    sys.path.append("/home-local/dbichet/Documents/These/code/VolVIC")
    from BCC_cell import BCC_grid
    l, m, n = 11, 4, 4
    cyl_radius = 0.7/2 * 1.3
    cell_half_length = 2
    order_elevation = [0, 0, 1]
    knot_insertion = [0, 0, 3]
    splines, separated_ctrl_pts, connectivity, _ = BCC_grid(l, m, n, cyl_radius, cell_half_length, 
                                                            order_elevation=order_elevation, 
                                                            knot_insertion=knot_insertion)
    mesh = Mesh(splines, separated_ctrl_pts, connectivity)
    surf = mesh.extract_border()
    # surf.plot()
    
    from marching_cubes import marching_cubes
    from skimage.io import imread
    volume = imread("/home-local/dbichet/Documents/These/code/VolVIC/exemples/metalic_BCC_traction/SlicesY-Lattice_BCC_traction_binned.tiff") # 
    threshold = 21472.413169028445
    verts, faces = marching_cubes(volume, threshold)
    stl_mc = io.Mesh(points=verts, cells={"triangle": faces})
    surf.unique_ctrl_pts /= 0.0108725*8
    
    # plotter = surf.plot(show=False, color="blue", opacity=0.8)
    # plotter.add_mesh(stl_mc, color="red", opacity=0.5)
    # plotter.show()
    
    Rmat, tvec, max_dist = surf.ICP_rigid_body_transform(stl_mc)
    
    plotter = surf.plot(show=False, color="blue", opacity=0.8)
    plotter.add_mesh(stl_mc, color="red", opacity=0.5)
    plotter.show()

# %%

class MeshLattice(Mesh):
    l: int
    m: int
    n: int
    
    def __init__(self, 
                 l: int, m: int, n: int, 
                 splines_cell: list[BSpline], 
                 ctrl_pts: Union[list[np.ndarray[np.floating]], np.ndarray[np.floating]], 
                 connectivity: Union[MultiPatchBSplineConnectivity, None]=None, 
                 ref_inds: Union[np.ndarray[np.integer], None]=None):
        splines = splines_cell*(l*m*n)
        super().__init__(splines, ctrl_pts, connectivity=connectivity, ref_inds=ref_inds)
        self.l = l
        self.m = m
        self.n = n
    
    @classmethod
    def from_mesh(cls, mesh: Mesh, l: int, m: int, n: int) -> "MeshLattice":
        splines_cell = mesh.splines[:mesh.connectivity.nb_patchs//(l*m*n)]
        return cls(l, m, n, splines_cell, mesh.unique_ctrl_pts, mesh.connectivity, mesh.ref_inds)
    
    def get_mesh_cell_one(self):
        splines = np.empty(self.connectivity.nb_patchs, dtype=object)
        splines[:] = self.splines
        nb_patchs_cell = len(self.splines)//(self.l*self.m*self.n)
        connectivity_cell, splines_cell, full_to_cell = self.connectivity.subset(splines, np.arange(nb_patchs_cell))
        mesh_cell = Mesh(list(splines_cell), self.unique_ctrl_pts[..., full_to_cell], connectivity=connectivity_cell, ref_inds=self.ref_inds[full_to_cell])
        return mesh_cell
    
    def get_nb_patchs_cell(self):
        return self.connectivity.nb_patchs//(self.l*self.m*self.n)
        