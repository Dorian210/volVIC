# %%
from typing import Callable, Iterable, Literal, Union
import numpy as np
import scipy.sparse as sps
import pyvista as pv
import meshio as io
import pickle
from bsplyne import BSpline, MultiPatchBSplineConnectivity, parallel_blocks
from treeIDW import treeIDW

from volVIC.image_utils import find_fg_bg
from volVIC.marching_cubes import marching_cubes


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook ou lab
        elif shell == "TerminalInteractiveShell":
            return False  # IPython en terminal
        else:
            return False  # autre type inconnu
    except (NameError, ImportError):
        return False  # En script Python standard


class Mesh:
    """
    Represents a multi-patch B-spline mesh.

    This class manages a multi-patch B-spline mesh, including control points,
    connectivity, evaluation, field propagation, and visualization. It supports
    both 2D and 3D meshes.

    Attributes
    ----------
    connectivity : MultiPatchBSplineConnectivity
        Multi-patch connectivity structure.
    splines : list[BSpline]
        List of B-spline objects for each patch.
    unique_ctrl_pts : np.ndarray
        Array of unique control points for the entire mesh.
    ref_inds : np.ndarray
        Reference indices used for linking to submeshes.
    """

    connectivity: MultiPatchBSplineConnectivity
    splines: list[BSpline]
    unique_ctrl_pts: np.ndarray[np.floating]
    ref_inds: np.ndarray[np.integer]

    def __init__(
        self,
        splines: list[BSpline],
        ctrl_pts: Union[list[np.ndarray[np.floating]], np.ndarray[np.floating]],
        connectivity: Union[MultiPatchBSplineConnectivity, None] = None,
        ref_inds: Union[np.ndarray[np.integer], None] = None,
    ):
        """
        Initialize a multi-patch B-spline mesh.

        This constructor sets up a multi-patch mesh with B-spline patches,
        control points, and optional connectivity and reference indices. The mesh
        can be initialized either with separated control points per patch or with
        a packed array of unique control points. If connectivity is not provided,
        it is inferred automatically from separated control points.

        Parameters
        ----------
        splines : list[BSpline]
            List of `BSpline` objects for each patch.

        ctrl_pts : list[np.ndarray] or np.ndarray
            Control points of the mesh.
            - If a list of arrays is provided, each array corresponds to a patch
            (shape: `(dim_phys, n1, ..., nd)`).
            - If a single array is provided, it should contain either the unique
            control points or the full concatenated points for all patches.

        connectivity : MultiPatchBSplineConnectivity, optional
            Multi-patch connectivity object. If `None`, it is automatically inferred
            from separated control points.

        ref_inds : np.ndarray, optional
            Reference indices mapping control points to submeshes or external data.
            If `None`, defaults to `np.arange(nb_unique_nodes)`.

        Notes
        -----
        - The mesh stores control points in a "unique" packed format (`self.unique_ctrl_pts`)
        internally, regardless of input format.
        - If `ctrl_pts` is a single array but does not match the number of unique nodes,
        it is automatically packed using the provided or inferred connectivity.
        """
        if connectivity is None:
            if isinstance(ctrl_pts, np.ndarray):
                raise TypeError(
                    "Can only infer connectivity from separated control points list."
                )
            self.connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(
                ctrl_pts
            )
        else:
            self.connectivity = connectivity
        self.splines = splines
        if isinstance(ctrl_pts, np.ndarray):  # unique or unpacked state
            if ctrl_pts.shape[1] == self.connectivity.nb_unique_nodes:  # unique state
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
        """
        IPython/Jupyter display hook for the mesh.

        Shows a screenshot of the last plot if available when
        the object is displayed in a notebook.
        """
        self.show()

    def show(self):
        """
        Display the mesh or the last plot screenshot.

        - If a screenshot from the last `PyVista` plot exists, it is displayed
        via `Matplotlib`.
        - Otherwise, prints a simple textual representation of the mesh.
        """
        if hasattr(self, "_screenshot"):
            import matplotlib.pyplot as plt

            plt.imshow(self._screenshot)
            plt.axis("off")
            plt.title("Screenshot of last plot")
            plt.show()
        else:
            print(self)

    def save(self, filename: str):
        """
        Serialize and save the mesh object to a file using pickle.

        Parameters
        ----------
        filename : str
            Path to the file where the mesh will be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> "Mesh":
        """
        Load a mesh object previously saved with :meth:`save`.

        Parameters
        ----------
        filename : str
            Path to the file containing the pickled mesh.

        Returns
        -------
        Mesh
            The loaded mesh object.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def separated_to_unique(self, field, method=None):
        """
        Convert a field given per patch to the unique/global control points representation.

        Parameters
        ----------
        field : list[np.ndarray]
            Field separated per patch.
        method : optional
            Agglomeration method passed to :meth:`MultiPatchBSplineConnectivity.agglomerate`.

        Returns
        -------
        np.ndarray
            Field in the unique/global representation.
        """
        if method is None:
            return self.connectivity.pack(self.connectivity.agglomerate(field))
        else:
            return self.connectivity.pack(
                self.connectivity.agglomerate(field), method=method
            )

    def unique_to_separated(self, field):
        """
        Convert a field in the unique/global representation to a per-patch separated form.

        Parameters
        ----------
        field : np.ndarray
            Field in unique/global representation.

        Returns
        -------
        list[np.ndarray]
            Field separated per patch.
        """
        return self.connectivity.separate(self.connectivity.unpack(field))

    def get_separated_ctrl_pts(self) -> list[np.ndarray[np.floating]]:
        """
        Return the control points as a list of arrays, one per patch.

        Returns
        -------
        list[np.ndarray]
            Control points separated per patch.
        """
        return self.unique_to_separated(self.unique_ctrl_pts)

    def set_separated_ctrl_pts(self, separated_ctrl_pts: list[np.ndarray[np.floating]]):
        """
        Set the mesh control points from a list of arrays (one per patch).

        Parameters
        ----------
        separated_ctrl_pts : list[np.ndarray]
            Control points separated per patch.
        """
        self.unique_ctrl_pts = self.separated_to_unique(separated_ctrl_pts)

    def __call__(
        self,
        XI: Iterable[tuple[np.ndarray[np.floating], ...]],
        k: Union[Iterable[Union[int, tuple[int, ...]]], int] = 0,
        data: Union[np.ndarray, Iterable[np.ndarray]] = None,
    ) -> list:
        """
        Evaluate the mesh (B-spline patches) at given parametric points.

        This method evaluates each B-spline patch of the mesh at the provided
        isoparametric coordinates `XI`. The evaluation can be performed on either
        the current control points of the mesh or a provided field.

        Internally, this method calls :meth:`BSpline.__call__` for each patch.

        Parameters
        ----------
        XI : iterable of tuple of np.ndarray
            Isoparametric coordinates for evaluation, provided per patch.
        k : int or iterable of int/tuple, optional
            Degree of derivative to compute at each evaluation point. Can be a single
            integer applied to all patches or an iterable specifying values per patch.
            Default is 0 (function value).
        data : np.ndarray or iterable of np.ndarray, optional
            Field values to use for evaluation. If `None` (default), uses the current
            control points. Can be a single array in unique/global representation
            or a list of arrays per patch.

        Returns
        -------
        list
            List of evaluation results, one per patch, in the order of `XI`.
        """
        if isinstance(k, int):
            k = [k] * self.connectivity.nb_patchs
        elif isinstance(k, Iterable):
            if len(k) != self.connectivity.nb_patchs:
                raise ValueError(
                    f"Expected {self.connectivity.nb_patchs} entries in 'k', got {len(k)}."
                )
        else:
            raise TypeError(
                f"'k' must be an int or an iterable of ints/tuples, got {type(k).__name__}."
            )
        if data is None:
            data = self.get_separated_ctrl_pts()
        elif isinstance(data, np.ndarray):
            data = self.unique_to_separated(data)
        elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            if len(data) != self.connectivity.nb_patchs:
                raise ValueError(
                    f"Expected {self.connectivity.nb_patchs} data arrays, got {len(data)}."
                )
        else:
            raise TypeError(
                "'data' must be None, a NumPy array, or an iterable of NumPy arrays."
                f" Got object of type {type(data).__name__}."
            )
        all_args = list(zip(data, XI, k))
        results = parallel_blocks(self.splines, all_args, pbar_title="Evaluate mesh")
        return results

    def assemble_grads(self, terms: list[np.ndarray], field_size: int) -> np.ndarray:
        """
        Assemble patch-wise gradient vectors into a global gradient vector.

        Each entry in `terms` corresponds to a gradient vector computed on a single
        patch. This method uses the mesh connectivity to map patch-local indices
        to global unique node indices and sums contributions from overlapping nodes.

        Important: Assumes that the degrees of freedom are arranged per field,
        i.e., `[x_0, ..., x_n, y_0, ..., y_n, z_0, ..., z_n]`. It does not
        work if the DOFs are interleaved as `[x_0, y_0, z_0, x_1, y_1, z_1, ...]`.

        Parameters
        ----------
        terms : list of np.ndarray
            Gradient vectors for each patch, typically computed from patch-local
            operations.
        field_size : int
            Number of degrees of freedom per mesh node (e.g., 3 for 3D displacements).

        Returns
        -------
        np.ndarray
            Global gradient vector of size `field_size * nb_unique_nodes`.

        Notes
        -----
        - Uses :meth:`MultiPatchBSplineConnectivity.unique_field_indices` to map
        patch-local degrees of freedom to global indices.
        """
        size = self.connectivity.nb_unique_nodes * field_size
        total = np.zeros(size, dtype=terms[0].dtype)
        separated_inds = self.connectivity.unique_field_indices(
            (field_size,), representation="separated"
        )
        for patch, term in enumerate(terms):
            inds = separated_inds[patch].ravel()
            np.add.at(total, inds, term)
        return total

    def assemble_hessians(
        self, terms: list[sps.spmatrix], field_size: int
    ) -> sps.spmatrix:
        """
        Assemble patch-wise Hessian matrices into a global sparse Hessian.

        Each matrix in `terms` corresponds to the Hessian on a single patch. This
        method maps the local patch indices to global unique node indices using the
        mesh connectivity and constructs a global COO sparse matrix by summing
        overlapping contributions.

        Important: Assumes that the degrees of freedom are arranged per field,
        i.e., `[x_0, ..., x_n, y_0, ..., y_n, z_0, ..., z_n]`. It does not
        work if the DOFs are interleaved as `[x_0, y_0, z_0, x_1, y_1, z_1, ...]`.

        Parameters
        ----------
        terms : list of scipy.sparse.spmatrix
            Hessian matrices for each patch, typically in CSR or CSC format.
        field_size : int
            Number of degrees of freedom per mesh node (e.g., 3 for 3D displacements).

        Returns
        -------
        scipy.sparse.spmatrix
            Global sparse Hessian matrix of size `(field_size * nb_unique_nodes, field_size * nb_unique_nodes)`.

        Notes
        -----
        - Uses :meth:`MultiPatchBSplineConnectivity.unique_field_indices` to map
        patch-local degrees of freedom to global indices.
        - The returned matrix is in COO format, suitable for further assembly or
        conversion to CSR/CSC.
        """
        rows = []
        cols = []
        data = []
        separated_inds = self.connectivity.unique_field_indices(
            (field_size,), representation="separated"
        )
        for patch, mat in enumerate(terms):
            inds = separated_inds[patch].ravel()
            mat = mat.tocoo()
            rows.append(inds[mat.row])
            cols.append(inds[mat.col])
            data.append(mat.data)
        size = self.connectivity.nb_unique_nodes * field_size
        return sps.coo_matrix(
            (np.hstack(data), (np.hstack(rows), np.hstack(cols))), shape=(size, size)
        )

    def extract_border(self) -> "Mesh":
        """
        Extract the exterior boundary mesh from a multi-patch B-spline mesh.

        This method identifies the nodes and patches lying on the exterior of the
        mesh and constructs a new `Mesh` object representing only these boundary
        elements. The reference indices are also updated to correspond to the
        subset of unique nodes retained.

        Returns
        -------
        Mesh
            A new `Mesh` instance representing the exterior boundaries of the mesh.

        Notes
        -----
        - The `ref_inds` of the returned mesh correspond to the indices of the
        original mesh's unique nodes included in the boundary.
        - The global layout of degrees of freedom remains the same as the parent
        mesh.
        """
        border_connectivity, border_splines, full_to_border = (
            self.connectivity.extract_exterior_borders(self.splines)
        )
        border_ctrl_pts = self.unique_ctrl_pts[..., full_to_border]
        border_ref_inds = self.ref_inds[full_to_border]
        return Mesh(
            list(border_splines), border_ctrl_pts, border_connectivity, border_ref_inds
        )

    def subset(self, patchs_to_keep: np.ndarray[np.integer]) -> "Mesh":
        """
        Create a submesh containing only the specified patches.

        The method constructs a new `Mesh` object consisting of the patches
        indexed by `patchs_to_keep`. The unique control points and reference
        indices are reduced accordingly to match the retained patches.

        Parameters
        ----------
        patchs_to_keep : np.ndarray of int
            Array of patch indices to retain in the submesh.

        Returns
        -------
        Mesh
            A new `Mesh` instance containing only the specified patches and the
            associated control points.

        Notes
        -----
        - The `ref_inds` of the returned submesh correspond to the subset of the
        original mesh's reference indices included in the retained patches.
        - The global layout of degrees of freedom is preserved.
        - Useful for operations on a specific region or cell of a larger multipatch
        mesh.
        """
        splines = np.empty(self.connectivity.nb_patchs, dtype=object)
        splines[:] = self.splines
        subset_connectivity, subset_splines, full_to_subset = self.connectivity.subset(
            splines, patchs_to_keep
        )
        subset_ctrl_pts = self.unique_ctrl_pts[..., full_to_subset]
        subset_ref_inds = self.ref_inds[full_to_subset]
        return Mesh(
            list(subset_splines), subset_ctrl_pts, subset_connectivity, subset_ref_inds
        )

    def propagate_field(
        self,
        field_values: np.ndarray[np.floating],
        indices: np.ndarray[np.integer],
        disable_parallel: bool = False,
    ) -> np.ndarray[np.floating]:
        """
        Propagate a field defined on a subset of nodes to the entire mesh using IDW.

        This method uses inverse distance weighting (IDW) interpolation to estimate
        the values of a field at unknown nodes from known nodes. The known nodes are
        specified by `indices` and the field values at those nodes are provided
        through `field_values`.

        Parameters
        ----------
        field_values : np.ndarray of float
            Field values at known nodes, shape `(dof, n_known)` or `(n_known,)`.
            Must be of `float` type for IDW interpolation.
        indices : np.ndarray of int
            Indices of the known nodes in `self.unique_ctrl_pts`.
        disable_parallel : bool, optional
            If `True`, disables parallel computation in the IDW interpolation. Default is `False`.

        Returns
        -------
        np.ndarray of float
            Field values propagated to all nodes of the mesh, same dtype as input.

        Notes
        -----
        - `self.unique_ctrl_pts` must be float type for IDW to work correctly.
        - Interpolation is performed only for nodes not in `indices`; known node values are preserved.
        """
        shape = field_values[None].shape[:-1]
        field_values = field_values.reshape((np.prod(shape), -1))
        unknown_mask = np.ones(self.connectivity.nb_unique_nodes, dtype="bool")
        unknown_mask[indices] = False
        known_points = self.unique_ctrl_pts[:, indices].T
        unknown_points = self.unique_ctrl_pts[:, unknown_mask].T
        field = np.empty(
            (field_values.shape[0], self.connectivity.nb_unique_nodes), dtype="float"
        )
        field[:, indices] = field_values
        field[:, unknown_mask] = treeIDW(
            known_points, field_values.T, unknown_points, parallel=not disable_parallel
        ).T
        return field.reshape((*shape[1:], -1))

    def propagate_field_from_submesh(
        self,
        submesh: "Mesh",
        field_values: np.ndarray[np.floating],
        disable_parallel: bool = False,
    ) -> np.ndarray[np.floating]:
        """
        Propagate a field from a submesh to the full mesh using IDW interpolation.

        This method first identifies the nodes in the submesh that correspond to the
        full mesh, then propagates the field to all nodes of the full mesh using
        inverse distance weighting (IDW). The values at known nodes are preserved.

        Parameters
        ----------
        submesh : Mesh
            Submesh containing the nodes where the field is defined.
        field_values : np.ndarray of float
            Field values defined on the submesh nodes. Must be of `float` type for IDW.
        disable_parallel : bool, optional
            If `True`, disables parallel computation in the IDW interpolation. Default is `False`.

        Returns
        -------
        np.ndarray of float
            Field values propagated to all nodes of the full mesh.

        Notes
        -----
        - Both `field_values` and `self.unique_ctrl_pts` must be of float type.
        - Interpolation is done only for nodes not in the submesh; known node values are preserved.
        """
        shape = field_values[None].shape[:-1]
        field_values = field_values.reshape((np.prod(shape), -1))
        _, idx_mesh, idx_submesh = np.intersect1d(
            self.ref_inds, submesh.ref_inds, return_indices=True
        )
        field = self.propagate_field(
            field_values[:, idx_submesh], idx_mesh, disable_parallel=disable_parallel
        )
        return field.reshape((*shape[1:], -1))

    def get_orientation_field(
        self,
        n_eval_per_elem: int = 5,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        verbose: bool = True,
        disable_parallel: bool = False,
    ) -> list[np.ndarray[np.floating]]:
        """
        Compute the orientation field of a 3D multipatch B-spline mesh.

        The orientation field is calculated as the scalar triple product of the
        local tangent vectors of each element. It indicates the local handedness
        of the mesh elements (positive for right-handed orientation, negative for left-handed).

        Parameters
        ----------
        n_eval_per_elem : int, optional
            Number of evaluation points per element along each parametric direction.
            Default is `5`.
        XI_list : iterable of tuple of np.ndarray, optional
            Predefined parametric coordinates for each patch. If `None`, a regular grid
            is generated using `n_eval_per_elem`.
        verbose : bool, optional
            If `True`, prints progress messages. Default is `True`.
        disable_parallel : bool, optional
            If `True`, disables parallel computation. Default is `False`.

        Returns
        -------
        list of np.ndarray of float
            A `list` (one entry per patch) containing the orientation scalar field
            for at each evaluation point.

        Notes
        -----
        - Only implemented for 3D meshes (`self.connectivity.npa == 3`).
        - Uses parallel evaluation via `parallel_blocks`.
        """

        assert (
            self.connectivity.npa == 3
        ), "Orientation field is only implemented for 3D meshes."

        def fct_elem(spline, ctrl_pts, XI):
            ds_dxi, ds_deta, ds_dzeta = spline(ctrl_pts, XI, k=1)
            cross_prod = np.cross(ds_dxi, ds_deta, axis=0)
            mixt_prod = np.sum(cross_prod * ds_dzeta, axis=0)
            return mixt_prod.astype(float)

        if XI_list is None:
            XI_list = [
                s.linspace(n_eval_per_elem=n_eval_per_elem) for s in self.splines
            ]

        args = list(zip(self.splines, self.get_separated_ctrl_pts(), XI_list))

        orientation_field = parallel_blocks(
            fct_elem,
            args,
            pbar_title="Computing orientation field",
            disable_parallel=disable_parallel,
            verbose=verbose,
        )

        return orientation_field

    def correct_orientation(
        self,
        separated_fields: list = [],
        axis: int = 2,
        verbose: bool = True,
        disable_parallel: bool = False,
    ):
        """
        Ensure consistent orientation of all patches in a 3D multipatch B-spline mesh.

        This method detects and corrects inverted patch orientations by evaluating
        the integrated orientation field over each patch. Patches with a negative
        signed orientation are flipped along the specified parametric axis.

        The orientation of each patch is assessed by integrating the scalar triple
        product of the parametric tangent vectors using Gauss-Legendre quadrature.
        A negative integral indicates an inverted (left-handed) parametrization.

        Parameters
        ----------
        separated_fields : list, optional
            List of fields defined in separated (per-patch) representation that must
            be flipped consistently with the control points. Each entry is modified
            before returning it. Default is an empty list.
        axis : int, optional
            Parametric axis along which inverted patches are flipped
            (`0 -> xi`, `1 -> eta`, `2 -> zeta`). Default is `2`.
        verbose : bool, optional
            If `True`, prints diagnostic information, including the number of flipped
            patches. Default is `True`.
        disable_parallel : bool, optional
            If `True`, disables parallel computation. Default is `False`.

        Returns
        -------
        list
            Flipped version, consistently with the control points, of `separated_fields`.

        Notes
        -----
        - Only applicable to 3D meshes (`self.connectivity.npa == 3`).
        - Patch orientation is determined by integrating the orientation field
          returned by :meth:`get_orientation_field`.
        - Flipping a patch involves:
            * Reversing the knot vector of the selected parametric axis.
            * Reversing the corresponding axis of the separated control points.
            * Reversing the same axis for all fields listed in `separated_fields`.
        - The mesh connectivity (`unique_nodes_inds`) and `unique_ctrl_pts` are
          modified to reflect the updated orientation.
        - This operation modifies the mesh in place.
        """
        nb_gauss = [[(3 * p + 1) // 2 for p in s.getDegrees()] for s in self.splines]
        XI_dXI_list = [
            s.gauss_legendre_for_integration(n_eval_per_elem=ng)
            for s, ng in zip(self.splines, nb_gauss)
        ]
        XI_list, dXI_list = zip(*XI_dXI_list)
        orientation_field = self.get_orientation_field(
            XI_list=XI_list, verbose=verbose, disable_parallel=disable_parallel
        )
        patchs_orientation = []
        for f, (dxi, deta, dzeta) in zip(orientation_field, dXI_list):
            dXI = np.outer(dxi, np.outer(deta, dzeta)).ravel()
            patchs_orientation.append(f.ravel() @ dXI)
        patchs_to_flip = np.where(np.array(patchs_orientation) < 0)[0]
        if verbose:
            print(
                f"Flipping {len(patchs_to_flip)} out of {self.connectivity.nb_patchs} patchs."
            )
        separated_ctrl_pts = self.get_separated_ctrl_pts()
        inds = self.connectivity.unique_field_indices(())
        flip = tuple(
            [...]
            + [
                slice(None, None, -1) if a == axis else slice(None)
                for a in range(self.connectivity.npa)
            ]
        )
        for i in patchs_to_flip:
            axis_basis = self.splines[i].bases[axis]
            a, b = axis_basis.span
            axis_basis.knot = (a + b) - axis_basis.knot[::-1]
            separated_ctrl_pts[i] = separated_ctrl_pts[i][flip]
            inds[i] = inds[i][flip]
            for separated_field in separated_fields:
                separated_field[i] = separated_field[i][flip]
        self.connectivity.unique_nodes_inds = self.connectivity.agglomerate(inds)
        self.unique_ctrl_pts = self.separated_to_unique(separated_ctrl_pts)

        return separated_fields

    def plot_orientation(
        self,
        n_eval_per_elem: Union[int, Iterable[int]] = 5,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        verbose: bool = True,
        disable_parallel: bool = False,
        **kwargs,
    ) -> pv.Plotter:
        """
        Visualize the orientation (handedness) of the mesh elements.

        This method computes the orientation field of the mesh and visualizes it
        as a boolean field indicating whether each evaluation point corresponds
        to a right-handed (`True`) or left-handed (`False`) parametrization.

        Internally, the orientation field is evaluated using
        :meth:`get_orientation_field`, then thresholded as
        `orientation > 0` and reshaped to comply with plotting requirements.

        Parameters
        ----------
        n_eval_per_elem : int or iterable of int, optional
            Number of evaluation points per element along each parametric direction.
            If an iterable is provided, it must match the parametric dimension.
            Default is `5`.
        XI_list : iterable of tuple of ndarray, optional
            Parametric coordinates at which the orientation field is evaluated,
            one entry per patch. If `None` (default), a uniform sampling is used
            for each patch.
        verbose : bool, optional
            If `True`, displays progress information. Default is `True`.
        disable_parallel : bool, optional
            If `True`, disables parallel computation. Default is `False`.
        **kwargs : dict, optional
            Additional keyword arguments forwarded to :meth:`Mesh.plot`, such as
            `cmap`, `scalar_bar_args`, `show_scalar_bar`, etc.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object used for visualization.

        Notes
        -----
        - The orientation field is visualized as a boolean scalar field, where
          `True` indicates a right-handed parametrization and `False` a
          left-handed one.
        - The default colormap uses red for left-handed and green for right-handed
          orientations.
        - This method is intended as a diagnostic tool, typically used before or
          after calling :meth:`correct_orientation`.
        """
        separated_field = self.get_orientation_field(
            n_eval_per_elem=n_eval_per_elem,
            XI_list=XI_list,
            disable_parallel=disable_parallel,
            verbose=verbose,
        )
        separated_field = [field[None] > 0 for field in separated_field]

        kwargs.setdefault("cmap", ["#ff0000", "#00ff00"])
        kwargs.setdefault("show_scalar_bar", True)
        kwargs.setdefault("annotations", {0: "False", 1: "True"})
        scalar_bar_args = kwargs.get("scalar_bar_args", {})
        scalar_bar_args.setdefault("title", "Is Right Handed")
        scalar_bar_args.setdefault("n_labels", 0)
        kwargs["scalar_bar_args"] = scalar_bar_args

        return self.plot(
            n_eval_per_elem=n_eval_per_elem,
            XI_list=XI_list,
            separated_field=separated_field,
            disable_parallel=disable_parallel,
            verbose=verbose,
            **kwargs,
        )

    def plot(
        self,
        n_eval_per_elem: Union[int, Iterable[int]] = 10,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        unique_field: Union[np.ndarray, None] = None,
        separated_field: Union[list[np.ndarray], list[callable], None] = None,
        interior_only: bool = True,
        plt_ctrl_mesh: bool = False,
        verbose: bool = True,
        disable_parallel: bool = False,
        pv_plotter: Union[pv.Plotter, None] = None,
        show: bool = True,
        elem_sep_color: str = "black",
        ctrl_poly_color: str = "green",
        **pv_add_mesh_kwargs,
    ):
        """
        Visualize the B-spline multipatch mesh using `PyVista`.

        This method evaluates the mesh geometry and optional scalar fields at
        parametric sampling points and renders them using `PyVista`. Fields can be
        provided either in a unique (global) or separated (per-patch) representation.

        Parameters
        ----------
        n_eval_per_elem : int or iterable of int, optional
            Number of evaluation points per element along each parametric direction.
            If an iterable is provided, it must match the parametric dimension.
            Default is `10`.
        XI_list : iterable of tuple of np.ndarray, optional
            Explicit parametric coordinates for evaluation, provided per patch.
            If `None`, a regular grid based on `n_eval_per_elem` is used.
        unique_field : np.ndarray, optional
            Field defined on the unique control points of the mesh. Mutually exclusive
            with `separated_field`.
        separated_field : list of np.ndarray or list of callable, optional
            Field defined per patch, either as arrays evaluated on parametric grids
            or as callables evaluated at runtime. Mutually exclusive with `unique_field`.
        interior_only : bool, optional
            If `True`, only interior elements are displayed. Default is `True`.
        plt_ctrl_mesh : bool, optional
            If `True`, overlays the control polygon mesh (only if `interior_only=False`).
            Default is `False`.
        verbose : bool, optional
            If `True`, prints progress messages. Default is `True`.
        disable_parallel : bool, optional
            If `True`, disables parallel computation. Default is `False`.
        pv_plotter : pv.Plotter or None, optional
            Existing `PyVista` plotter to use. If `None`, a new one is created.
        show : bool, optional
            If `True`, renders the scene immediately. Default is `True`.
        elem_sep_color : str, optional
            Color used for element separators. Default is `'black'`.
        ctrl_poly_color : str, optional
            Color used for the control polygon mesh. Default is `'green'`.
        **pv_add_mesh_kwargs : dict, optional
            Additional keyword arguments forwarded to `pv.Plotter.add_mesh`
            (e.g. `cmap`, `clim`, `scalar_bar_args`, etc.).

        Returns
        -------
        pv.Plotter
            The `PyVista` plotter used for rendering.

        Notes
        -----
        - Exactly one of `unique_field` or `separated_field` may be provided.
        - Fields passed as NumPy arrays for plotting must have the same number of
          dimensions (`ndim`) as the control points array. Scalar fields must therefore
          include a leading dimension of size 1.
        - Fields are evaluated on-the-fly at visualization points, not at control points.
        - If executed in a Jupyter notebook, a screenshot of the last plot is stored
          for display via `_repr_png_`.
        """
        if unique_field is not None:
            unique_fields = {
                "": (
                    unique_field[None, None]
                    if unique_field.ndim == 1
                    else unique_field[None]
                )
            }
            separated_fields = None
        elif separated_field is not None:
            unique_fields = {}
            if isinstance(separated_field[0], Callable):
                separated_fields = [
                    {"": lambda *args: field(*args)[None]} for field in separated_field
                ]
            else:
                separated_fields = [{"": field[None]} for field in separated_field]
        else:
            unique_fields = {}
            separated_fields = None

        if pv_plotter is None:
            pv_plotter = pv.Plotter()

        (elem_int_mesh,) = self.connectivity.make_elements_interior_meshes(
            self.splines,
            self.get_separated_ctrl_pts(),
            n_step=1,
            n_eval_per_elem=n_eval_per_elem,
            unique_fields=unique_fields,
            separated_fields=separated_fields,
            XI_list=XI_list,
            disable_parallel=disable_parallel,
            verbose=verbose,
        )
        pv_plotter.add_mesh(elem_int_mesh, **pv_add_mesh_kwargs)
        if not interior_only:
            (elem_sep_mesh,) = self.connectivity.make_elem_separator_meshes(
                self.splines,
                self.get_separated_ctrl_pts(),
                n_step=1,
                n_eval_per_elem=n_eval_per_elem,
                XI_list=XI_list,
                disable_parallel=disable_parallel,
                verbose=verbose,
            )
            if "label" in pv_add_mesh_kwargs:
                del pv_add_mesh_kwargs["label"]
            pv_add_mesh_kwargs["style"] = "wireframe"
            pv_add_mesh_kwargs["color"] = elem_sep_color
            pv_plotter.add_mesh(elem_sep_mesh, **pv_add_mesh_kwargs)
            if plt_ctrl_mesh:
                (ctrl_poly_mesh,) = self.connectivity.make_control_poly_meshes(
                    self.splines,
                    self.get_separated_ctrl_pts(),
                    n_step=1,
                    n_eval_per_elem=n_eval_per_elem,
                    XI_list=XI_list,
                    disable_parallel=disable_parallel,
                    verbose=verbose,
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
        image: np.ndarray,
        threshold: Union[float, None] = None,
        threshold_method: Literal["otsu", "interp"] = "otsu",
        mode: Literal["marching cubes", "voxels"] = "marching cubes",
        n_eval_per_elem: Union[int, Iterable[int]] = 10,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        interior_only: bool = True,
        plt_ctrl_mesh: bool = False,
        verbose: bool = True,
        disable_parallel: bool = False,
        pv_plotter: Union[pv.Plotter, None] = None,
        show: bool = True,
        image_show_edges: bool = False,
        image_line_width: float = 0.05,
        image_opacity: float = 0.85,
        image_edge_color: str = "black",
        image_interior_color: str = "white",
        **pv_add_mesh_kwargs,
    ):
        """
        Visualize the mesh together with an image-derived surface.

        This method overlays the B-spline mesh visualization with a surface extracted
        from a volumetric image, using either marching cubes or voxel thresholding.
        It is typically used to assess geometric alignment between the mesh and image
        features.

        Parameters
        ----------
        image : np.ndarray
            Input volumetric image of shape (n1, n2, n3).
        threshold : float or None, optional
            Threshold value used to extract the image surface. If `None`, it is
            automatically estimated using `threshold_method`.
        threshold_method : {"otsu", "interp"}, optional
            Method used to estimate the threshold when `threshold` is `None`.
            Default is `"otsu"`.
        mode : {"marching cubes", "voxels"}, optional
            Method used to extract the image surface. If `"marching cubes"`,
            the image must be of dtype `uint16`. Default is `"marching cubes"`.
        n_eval_per_elem : int or iterable of int, optional
            Number of evaluation points per element along each parametric direction.
            Default is `10`.
        XI_list : iterable of tuple of np.ndarray, optional
            Explicit parametric coordinates for mesh evaluation.
        interior_only : bool, optional
            If `True`, only interior elements of the mesh are displayed. Default is `True`.
        plt_ctrl_mesh : bool, optional
            If `True`, overlays the control polygon mesh. Default is `False`.
        verbose : bool, optional
            If `True`, prints progress messages. Default is `True`.
        disable_parallel : bool, optional
            If `True`, disables parallel computation. Default is `False`.
        pv_plotter : pv.Plotter or None, optional
            Existing PyVista plotter to use. If `None`, a new one is created.
        show : bool, optional
            If `True`, renders the scene immediately. Default is `True`.
        image_show_edges : bool, optional
            Whether to display edges of the image-derived surface. Default is `False`.
        image_line_width : float, optional
            Line width used when displaying image edges. Default is `0.05`.
        image_opacity : float, optional
            Opacity of the image-derived surface. Default is `0.85`.
        image_edge_color : str, optional
            Color of image surface edges. Default is `'black'`.
        image_interior_color : str, optional
            Color of the image-derived surface interior. Default is `'white'`.
        **pv_add_mesh_kwargs : dict, optional
            Additional keyword arguments forwarded to :meth:`plot`.

        Returns
        -------
        pv.Plotter
            The PyVista plotter used for visualization.

        Notes
        -----
        - The mesh is rendered first, then the image-derived surface is overlaid.
        - This method does not modify the mesh or image data.
        """
        if threshold is None:
            fg, bg = find_fg_bg(image, method=threshold_method, verbose=False)
            threshold = 0.5 * (fg + bg)

        if mode == "marching cubes":
            image_mesh = marching_cubes(image, threshold)
        elif mode == "voxels":
            image_mesh = pv.wrap(image).threshold(threshold)
        else:
            raise KeyError(f"Unrecognized mode '{mode}'.")

        pv_plotter = self.plot(
            n_eval_per_elem=n_eval_per_elem,
            XI_list=XI_list,
            interior_only=interior_only,
            plt_ctrl_mesh=plt_ctrl_mesh,
            pv_plotter=pv_plotter,
            show=False,
            disable_parallel=disable_parallel,
            verbose=verbose,
            **pv_add_mesh_kwargs,
        )

        pv_plotter.add_mesh(
            image_mesh,
            show_edges=image_show_edges,
            edge_color=image_edge_color,
            line_width=image_line_width,
            opacity=image_opacity,
            color=image_interior_color,
            specular=1.0,
        )

        if show:
            pv_plotter.show()

        return pv_plotter

    def ICP_rigid_body_transform(
        self,
        tgt_meshio: io.Mesh,
        n_eval_per_elem: Union[int, Iterable[int]] = 5,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        plot_after: bool = True,
        disable_parallel: bool = False,
    ) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating], float]:
        """
        Compute and apply a rigid body transformation aligning the mesh to a target mesh.

        This method computes the rigid body transformation (rotation and translation)
        that best aligns the current mesh to a target mesh using an Iterative Closest
        Point (ICP) algorithm. The transformation is then directly applied to the
        control points of the mesh.

        Parameters
        ----------
        tgt_meshio : io.Mesh
            Target mesh used as reference for the alignment.
        n_eval_per_elem : int or iterable of int, optional
            Number of evaluation points per element for each parametric dimension.
            If an integer is provided, the same value is used for all dimensions.
            If an iterable is provided, each value corresponds to one parametric
            direction. Default is `5`.
        XI_list : iterable of tuple of ndarray, optional
            Parametric coordinates at which the geometry is evaluated to generate
            the source surface mesh. If provided, this overrides `n_eval_per_elem`.
        plot_after : bool, optional
            If `True`, displays a visualization of the aligned source mesh and
            the target mesh after the ICP procedure. Default is `True`.
        disable_parallel : bool, optional
            If `True`, disables parallel evaluation. Default is `False`.

        Returns
        -------
        Rmat : np.ndarray of float, shape (3, 3)
            Rotation matrix of the rigid body transformation.
        tvec : np.ndarray of float, shape (3,)
            Translation vector of the rigid body transformation.
        max_dist : float
            Maximum absolute distance between the aligned source mesh and the
            target mesh after transformation.

        Notes
        -----
        - For 3D meshes, the ICP is performed on the extracted boundary surface.
        - The transformation is applied as `Rmat @ x + tvec`.
        - This method modifies the mesh in place by updating `self.unique_ctrl_pts`.
        """
        # Create source and target PolyData
        if self.connectivity.npa == 2:
            border_mesh = self
        elif self.connectivity.npa == 3:
            border_mesh = self.extract_border()
        else:
            raise NotImplementedError(
                f"Making finding ICP rigid body transform not implemented for {self.connectivity.npa}-D shapes."
            )
        src_meshio = border_mesh.connectivity.make_elements_interior_meshes(
            border_mesh.splines,
            border_mesh.get_separated_ctrl_pts(),
            n_eval_per_elem=n_eval_per_elem,
            XI_list=XI_list,
            verbose=False,
            disable_parallel=disable_parallel,
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
            return_matrix=True,
        )

        # Decompose transformation matrix
        Rmat = matrix[:3, :3]
        tvec = matrix[:3, 3]

        # Apply transformation to self
        self.unique_ctrl_pts = Rmat @ self.unique_ctrl_pts + tvec[:, None]

        if plot_after:
            plotter = pv.Plotter()
            plotter.add_mesh(
                aligned, color="blue", opacity=0.8, label="Aligned surface mesh"
            )
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
        n_eval_per_elem: Union[int, Iterable[int]] = 5,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
    ) -> io.Mesh:
        """
        Compute the signed distance field to a target mesh.

        This method evaluates the geometry of the mesh (or its boundary in 3D)
        and computes the implicit distance to the surface of a target mesh.
        The resulting distance field is returned as a `MeshIO` mesh.

        Parameters
        ----------
        tgt_meshio : io.Mesh
            Target mesh defining the reference surface.
        n_eval_per_elem : int or iterable of int, optional
            Number of evaluation points per element for each parametric dimension.
            Default is `5`.
        XI_list : iterable of tuple of ndarray, optional
            Parametric coordinates at which the geometry is evaluated.
            If provided, this overrides `n_eval_per_elem`.

        Returns
        -------
        io.Mesh
            `MeshIO` mesh containing the evaluated geometry and the associated
            implicit distance field.
        """
        if self.connectivity.npa == 2:
            border_mesh = self
        elif self.connectivity.npa == 3:
            border_mesh = self.extract_border()
        else:
            raise NotImplementedError(
                f"Distance field not implemented for {self.connectivity.npa}-D shapes."
            )
        src_meshio = border_mesh.connectivity.make_elements_interior_meshes(
            border_mesh.splines,
            border_mesh.get_separated_ctrl_pts(),
            n_eval_per_elem=n_eval_per_elem,
            XI_list=XI_list,
            verbose=False,
        )[0]
        src = pv.from_meshio(src_meshio)
        tgt = pv.from_meshio(tgt_meshio)
        src.compute_implicit_distance(tgt.extract_surface(), inplace=True)
        return pv.to_meshio(src)

    def save_paraview(
        self,
        path: str,
        name: str,
        n_step: int = 1,
        n_eval_per_elem: Union[int, Iterable[int]] = 10,
        unique_fields: dict = {},
        separated_fields: Union[list[dict], None] = None,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        groups: Union[dict[str, dict[str, Union[str, int]]], None] = None,
        make_pvd: bool = True,
        verbose: bool = True,
        fields_on_interior_only: Union[bool, Literal["auto"], list[str]] = "auto",
        disable_parallel: bool = False,
    ):
        """
        Save the mesh geometry and associated fields as ParaView-compatible files.

        This method is a convenience wrapper around
        `MultiPatchBSplineConnectivity.save_paraview`.
        It evaluates the multipatch B-spline geometry represented by the current
        mesh and exports several VTU files suitable for visualization in ParaView,
        with an optional PVD file for grouping time steps and mesh components.

        The following visualization meshes are generated:
        - Interior mesh representing the evaluated B-spline geometry
        - Element borders mesh showing the patch discretization
        - Control points mesh showing the control structure

        Parameters
        ----------
        path : str
            Directory where the ParaView files will be written.
        name : str
            Base name used for all output files.
        n_step : int, optional
            Number of time steps to export. Default is `1`.
        n_eval_per_elem : int or iterable of int, optional
            Number of evaluation points per element for each parametric dimension.
            If an integer is provided, the same value is used for all dimensions.
            If an iterable is provided, each value corresponds to one parametric
            direction. Default is `10`.
        unique_fields : dict, optional
            Fields defined on unique control points to be exported.
            Keys are field names and values are numpy arrays.
            Callables are not supported here. Default is an empty dictionary.
        separated_fields : list of dict, optional
            Fields defined in separated (per-patch) representation.
            Each list entry corresponds to one patch and contains a dictionary
            mapping field names to field values. See
            `MultiPatchBSplineConnectivity.save_paraview` for supported
            formats.
        XI_list : iterable of tuple of ndarray, optional
            Parametric coordinates at which the geometry and fields are evaluated.
            If provided, this overrides `n_eval_per_elem`.
        groups : dict, optional
            Dictionary describing ParaView file groups for PVD organization.
            If `None`, groups are created automatically.
        make_pvd : bool, optional
            If `True`, generates a PVD file grouping all VTU files.
            Default is `True`.
        verbose : bool, optional
            If `True`, prints progress information. Default is `True`.
        fields_on_interior_only : bool, "auto", or list of str, optional
            Controls on which meshes fields are written:
            - `True`: interior mesh only
            - `False`: all meshes
            - `"auto"`: displacement-like fields are written on all meshes,
              others only on the interior mesh
            - list of str: names of fields to include on all meshes
              Default is `"auto"`.
        disable_parallel : bool, optional
            If `True`, disables parallel evaluation. Default is `False`.

        Returns
        -------
        dict[str, dict[str, Union[str, int]]]
            Updated groups dictionary describing the generated ParaView files.

        Notes
        -----
        - This method don't modifies the internal state; it only performs evaluation and
          export.
        - Geometry and fields are evaluated using the current control points and splines.
        - For full details on supported field formats and file organization, see
          `MultiPatchBSplineConnectivity.save_paraview`.
        """
        return self.connectivity.save_paraview(
            self.splines,
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
            fields_on_interior_only,
            disable_parallel,
        )


# %%


class MeshLattice(Mesh):
    """
    Structured lattice mesh with an explicit reference B-spline cell.

    This class represents a *fully instantiated* multipatch B-spline mesh that
    exhibits a regular lattice structure obtained by repeating a reference cell
    along three directions. The lattice structure is described by the repetition
    counts `(l, m, n)`.

    The control points and connectivity describe the *entire lattice geometry*,
    exactly as in a standard :class:`Mesh`. In contrast, the `BSpline` instances
    correspond to a single reference cell and are reused for all lattice cells.

    `MeshLattice` therefore behaves as a standard :class:`Mesh`, while additionally
    storing lattice metadata and providing utilities to extract or operate on the
    reference cell only.

    This is a utility class, notably used to allow
    :func:`VirtualImageCorrelationEnergyElem.make_image_energies` and
    :func:`membrane_stiffness.make_membrane_stifness` to assemble
    operators on a single cell while exploiting lattice periodicity at the algebraic
    level.

    Attributes
    ----------
    l : int
        Number of repetitions of the reference cell along the first lattice
        direction.
    m : int
        Number of repetitions of the reference cell along the second lattice
        direction.
    n : int
        Number of repetitions of the reference cell along the third lattice
        direction.

    Notes
    -----
    - The lattice geometry is assumed to be fully defined at construction time.
    - No geometric replication is performed internally.
    - All lattice cells share identical B-spline definitions.
    - The total number of patches is `l * m * n * nb_patchs_cell`.
    """

    l: int
    m: int
    n: int

    def __init__(
        self,
        l: int,
        m: int,
        n: int,
        splines_cell: list[BSpline],
        ctrl_pts: Union[list[np.ndarray[np.floating]], np.ndarray[np.floating]],
        connectivity: Union[MultiPatchBSplineConnectivity, None] = None,
        ref_inds: Union[np.ndarray[np.integer], None] = None,
    ):
        """
        Initialize a lattice mesh from a reference B-spline cell.
        See :meth:`Mesh.__init__` for more details.

        Parameters
        ----------
        l : int
            Number of repetitions along the first lattice direction.
        m : int
            Number of repetitions along the second lattice direction.
        n : int
            Number of repetitions along the third lattice direction.
        splines_cell : list of BSpline
            B-spline patches defining a single reference cell.
        ctrl_pts : list of np.ndarray or np.ndarray
            Control points of the full lattice structure, in separated or unique
            representation.
        connectivity : MultiPatchBSplineConnectivity or None, optional
            Connectivity describing the multipatch topology of the full lattice
            structure. If `None` (default), it is inferred automatically.
        ref_inds : np.ndarray of int or None, optional
            Optional reference indices associated with the control points.
        """
        splines = splines_cell * (l * m * n)
        super().__init__(
            splines, ctrl_pts, connectivity=connectivity, ref_inds=ref_inds
        )
        self.l = l
        self.m = m
        self.n = n

    @classmethod
    def from_mesh(cls, mesh: Mesh, l: int, m: int, n: int) -> "MeshLattice":
        """
        Construct a lattice mesh from an existing mesh.

        This method interprets the input `mesh` as a lattice composed of
        `l * m * n` identical cells and reconstructs a :class:`MeshLattice`
        instance using the corresponding reference cell splines.

        Parameters
        ----------
        mesh : Mesh
            Existing mesh representing a lattice geometry.
        l : int
            Number of repetitions along the first lattice direction.
        m : int
            Number of repetitions along the second lattice direction.
        n : int
            Number of repetitions along the third lattice direction.

        Returns
        -------
        MeshLattice
            A lattice mesh sharing the control points, connectivity and
            reference indices of `mesh`.

        Notes
        -----
        - The reference cell is inferred from the first
          `mesh.connectivity.nb_patchs // (l * m * n)` patches.
        - No geometry or connectivity modification is performed; the operation
          is purely structural.
        """
        splines_cell = mesh.splines[: mesh.connectivity.nb_patchs // (l * m * n)]
        return cls(
            l,
            m,
            n,
            splines_cell,
            mesh.unique_ctrl_pts,
            mesh.connectivity,
            mesh.ref_inds,
        )

    def get_mesh_cell_one(self) -> Mesh:
        """
        Extract the reference cell mesh corresponding to a single lattice cell.

        This method returns a :class:`Mesh` representing one reference cell of
        the lattice, including:
        - the corresponding subset of splines,
        - the associated control points,
        - a reduced connectivity.

        Returns
        -------
        Mesh
            Mesh corresponding to a single lattice cell.

        Notes
        -----
        - The returned mesh is independent from the lattice structure but
          shares the same control point values.
        - This is typically used for analysis or visualization of the unit cell.
        """
        splines = np.empty(self.connectivity.nb_patchs, dtype=object)
        splines[:] = self.splines
        nb_patchs_cell = len(self.splines) // (self.l * self.m * self.n)
        connectivity_cell, splines_cell, full_to_cell = self.connectivity.subset(
            splines, np.arange(nb_patchs_cell)
        )
        mesh_cell = Mesh(
            list(splines_cell),
            self.unique_ctrl_pts[..., full_to_cell],
            connectivity=connectivity_cell,
            ref_inds=self.ref_inds[full_to_cell],
        )
        return mesh_cell

    def get_nb_patchs_cell(self) -> int:
        """
        Return the number of patches in a single lattice cell.

        Returns
        -------
        int
            Number of B-spline patches defining the reference cell.
        """
        return self.connectivity.nb_patchs // (self.l * self.m * self.n)
