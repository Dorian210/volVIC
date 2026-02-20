from copy import deepcopy
from functools import partial
from typing import Union, Literal, Callable
import numpy as np
import scipy.sparse as sps
import pyvista as pv
import matplotlib.pyplot as plt
from IGA_for_bsplyne.Dirichlet import Dirichlet, DirichletConstraintHandler

from volVIC.Mesh import Mesh
from volVIC.marching_cubes import marching_cubes
from volVIC.C1_triplets import make_C1_eqs
from volVIC.membrane_stiffness import make_membrane_stifness, make_membrane_weight
from volVIC.image_utils import find_fg_bg, find_sigma_hat
from volVIC.virtual_image_correlation_energy import (
    VirtualImageCorrelationEnergyElem,
    make_image_energies,
    compute_distance_field,
)
from volVIC.virtual_image import g_slide
from volVIC.solve import iteration


class Problem:
    """
    Virtual Image Correlation (VIC) problem definition and solver.

    This class encapsulates the full setup of a surface-based VIC problem:
    geometry, image model, regularization, constraints, and nonlinear solver.
    A typical workflow is:
        1. Instantiate a Problem from a mesh and an image
        2. Call `solve` to estimate the displacement field
        3. Export results using `save_paraview` or propagate them to a volume mesh

    Attributes
    ----------
    mesh : Mesh
        Surface mesh used in the VIC problem.

    image : np.ndarray
        Image converted to floating-point format for computations.

    fg, bg : float
        Estimated or user-defined foreground and background gray levels.

    Rmat : np.ndarray
        Rotation matrix resulting from ICP initialization.

    tvec : np.ndarray
        Translation vector resulting from ICP initialization.

    image_energies : list[VirtualImageCorrelationEnergyElem]
        Image energy elements used to assemble the VIC objective.

    constraints : DirichletConstraintHandler
        Handler storing linear equality constraints (e.g. C¹ continuity).

    dirichlet : Dirichlet
        Dirichlet constraint object derived from the constraint handler.

    membrane_K : scipy.sparse.spmatrix
        Membrane stiffness matrix used for regularization.

    membrane_weight : float
        Weight associated with the membrane regularization term.

    initial_rho : float
        Initial value of the distance scaling parameter.

    saved_data_0 : list[dict[str, np.ndarray]]
        Cached image energy data from the first iteration, used for
        post-processing and visualization.
    """

    mesh: Mesh
    image: np.ndarray[np.uint16]
    fg: float
    bg: float
    Rmat: np.ndarray[np.floating]
    tvec: np.ndarray[np.floating]
    image_energies: list[VirtualImageCorrelationEnergyElem]
    constraints: DirichletConstraintHandler
    dirichlet: Dirichlet
    membrane_K: sps.spmatrix
    membrane_weight: float
    initial_rho: float
    saved_data_0: list[dict[str, np.ndarray[np.floating]]]

    def __init__(
        self,
        mesh: Mesh,
        image: np.ndarray[np.uint16],
        ICP_init: bool = True,
        reversed_normals: bool = False,
        fg_bg: Union[tuple[float, float], None] = None,
        fg_bg_method: Literal["otsu", "interp"] = "otsu",
        virtual_image: Callable[
            [
                np.ndarray[np.floating],
                np.ndarray[np.floating],
                np.ndarray[np.floating],
                float,
            ],
            tuple[np.ndarray[np.floating], np.ndarray[np.floating]],
        ] = g_slide,
        h: Union[float, None] = None,
        width_dx: float = 2.0,
        surf_dx: float = 1.0,
        alpha: Union[float, tuple[tuple[float, float], tuple[float, float]]] = 0.0,
        C1_mode: Union[
            None, Literal["auto", "none", "all"], np.ndarray[np.integer]
        ] = None,
        membrane_weight: Union[None, float] = None,
        initial_rho: float = 5.0,
        expected_mean_dist: float = 5.0,
        n_intg_membrane_weight_comput: int = 100,
        disable_parallel: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize a Virtual Image Correlation (VIC) problem.

        This constructor performs the full problem setup, including:
        - foreground/background gray-level estimation,
        - optional rigid-body ICP alignment between mesh and image,
        - construction of virtual image energies,
        - assembly of C¹ continuity constraints,
        - assembly of membrane regularization,
        - estimation of the coresponding regularization weight,
        - initialization of all data structures required by the nonlinear solver.

        After initialization, the problem is ready to be solved using
        :meth:`solve`, which performs a Gauss–Newton optimization to estimate
        the displacement field and the distance scaling parameter.

        The mesh coordinates are assumed to be expressed in image voxel units.
        No unit conversion or rescaling is performed internally.

        Parameters
        ----------
        mesh : Mesh
            Surface mesh defining the geometry to be matched to the image.
            The mesh coordinates must be expressed in image voxel units.
            Any conversion from physical units (e.g. mm or m) to voxel coordinates
            must be performed beforehand by the user.
            If `ICP_init=True`, a rigid-body alignment is performed, but this
            does not handle unit conversion or rescaling. ICP only corrects
            for rotation and translation, assuming consistent units.

        image : np.ndarray[np.uint16]
            3D volumetric image (e.g. CT scan). The gray-level distribution is
            assumed to be bimodal (background / foreground).

        ICP_init : bool, optional
            If `True` (default), a rigid-body ICP is performed to align the mesh
            with the image prior to solving the VIC problem.
            Disable only if the mesh is already well aligned.

        reversed_normals : bool, optional
            If `True`, swaps the foreground/background convention in the virtual
            image model. Useful if the mesh normals are oriented inward instead
            of outward. By default, `False`.

        fg_bg : tuple[float, float] or None, optional
            Tuple `(fg, bg)` specifying the foreground and background gray levels.
            If `None` (default), these values are automatically estimated from the
            image.

        fg_bg_method : Literal["otsu", "interp"], optional
            Method to compute the foreground and background gray levels. `'otsu'`
            uses :func:`image_utils.otsu_threshold` to compute the gray levels based
            on Otsu's method. `'interp'` uses :func:`image_utils.interp_fg_bg` to
            compute the gray levels using interpolation of histogram peaks.
            Default is `'otsu'`.

        virtual_image : Callable, optional
            Virtual image model mapping signed distance values to gray levels.
            The default (`g_slide`) is suitable for sharp material/background
            transitions. See :func:`virtual_image.g_slide` for the function
            signature.

        h : float or None, optional
            Half-width of the search domain along surface normals.
            If `None` (default), `h` is automatically estimated from the image
            during initialization.

        width_dx : float, optional
            Integration step along the normal direction for image energies.

        surf_dx : float, optional
            Integration step along the surface for image energies.

        alpha : float or tuple, optional
            Tangential regularization weight used in the image energies.
            A value of 0.0 (default) disables tangential smoothing.

        C1_mode : {"auto", "none", "all"} or np.ndarray or None, optional
            Definition of C¹ continuity constraints between patches:
                - "auto": automatically selected constraints (recommended)
                - "none": no C¹ constraints
                - "all": enforce C¹ everywhere
                - array: user-defined triplets of constrained control points

        membrane_weight : float or None, optional
            Weight of the membrane (elastic) regularization term.
            If None (default), the weight is automatically calibrated.

        initial_rho : float, optional
            Initial value of the transition distance parameter used in the VIC model.

        expected_mean_dist : float, optional
            Expected mean distance (in image units) used for automatic membrane
            weight calibration.

        n_intg_membrane_weight_comput : int, optional
            Number of integration points used to compute the membrane weight
            when it is determined automatically.

        disable_parallel : bool, optional
            If True, disables parallel execution (useful for debugging or
            reproducibility).

        verbose : bool, optional
            If True, prints detailed information during initialization and
            subsequent computations.
        """

        if verbose:
            print("[VIC] Initializing VIC problem")

        self.mesh = mesh
        self.image = image

        if verbose:
            print(f"[VIC] Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"[VIC] Mesh: {mesh}")

        # --- Foreground / Background -------------------------------------------------
        if fg_bg is None:
            if verbose:
                print("[VIC] Estimating foreground/background levels from image")
            self.fg, self.bg = self.find_fg_bg(method=fg_bg_method, verbose=verbose)
        else:
            self.fg, self.bg = fg_bg

        if verbose:
            print(f"[VIC] fg = {self.fg}, bg = {self.bg}")

        # --- ICP + h initialization --------------------------------------------------
        if verbose:
            print("[VIC] Initializing placement and research half width h")

        self.Rmat, self.tvec, h = self.initialize_h_mesh_ICP(
            ICP_init, h, disable_parallel=disable_parallel, verbose=verbose
        )

        if verbose:
            print(f"[VIC] h = {h}")
            print(f"[VIC] Rmat:\n{self.Rmat}")
            print(f"[VIC] tvec: {self.tvec}")

        # --- Virtual image ------------------------------------------------------------
        if reversed_normals:
            if verbose:
                print("[VIC] Using reversed normals for virtual image")
            g = partial(virtual_image, bg=self.bg, fg=self.fg)
        else:
            g = partial(virtual_image, bg=self.fg, fg=self.bg)

        # --- Image energies -----------------------------------------------------------
        if verbose:
            print("[VIC] Building image energies")
            print(
                f"[VIC] Integration params: surf_dx={surf_dx}, width_dx={width_dx}, alpha={alpha}"
            )

        self.image_energies = make_image_energies(
            self.mesh,
            h,
            width_dx=width_dx,
            surf_dx=surf_dx,
            alpha=alpha,
            virtual_image=g,
            verbose=verbose,
            disable_parallel=disable_parallel,
        )

        # --- C1 constraints -----------------------------------------------------------
        if verbose:
            print("[VIC] Building C1 constraints")

        C1_eqs = make_C1_eqs(self.mesh, C1_mode)

        self.constraints = DirichletConstraintHandler(
            self.mesh.connectivity.nb_unique_nodes * 3
        )
        self.constraints.add_eqs(C1_eqs, np.zeros(C1_eqs.shape[0], dtype="float"))

        if verbose:
            print(f"[VIC] Number of C1 equations: {C1_eqs.shape[0]}")

        self.dirichlet = self.make_dirichlet()

        # --- Membrane stiffness -------------------------------------------------------
        if verbose:
            print("[VIC] Assembling membrane stiffness matrix")

        self.membrane_K = make_membrane_stifness(
            self.mesh, verbose=verbose, disable_parallel=disable_parallel
        )

        # --- Membrane weight ----------------------------------------------------------
        if membrane_weight is None:
            if verbose:
                print("[VIC] Computing membrane weight automatically")
                print(
                    f"[VIC] initial_rho={initial_rho}, expected_mean_dist={expected_mean_dist}"
                )

            self.make_membrane_weight(
                initial_rho,
                expected_mean_dist=expected_mean_dist,
                n_intg=n_intg_membrane_weight_comput,
            )
            if verbose:
                print(f"[VIC] Computed membrane weight : {self.membrane_weight}")
        else:
            self.membrane_weight = membrane_weight
            if verbose:
                print(f"[VIC] Using user-defined membrane_weight = {membrane_weight}")

        self.initial_rho = initial_rho

        self.image = self.image.astype(
            float
        )  # TODO remove casting by adapting interpolation

        if verbose:
            print("[VIC] Initialization done ✔")

    def find_fg_bg(
        self, method: Literal["otsu", "interp"] = "otsu", verbose: bool = True
    ):
        """
        Estimate foreground and background gray levels from the input image.

        This method analyzes the gray-level distribution of the volumetric image
        and returns representative foreground (`fg`) and background (`bg`) values.
        These values are used by the virtual image model to map signed distances
        to gray levels in the VIC formulation.

        Parameters
        ----------
        method : Literal["otsu", "interp"], optional
            Strategy used to estimate foreground and background levels:
                - "otsu": threshold-based estimation using Otsu's method.
                - "interp": estimation based on interpolation of histogram peaks.
            Default is "otsu".

        verbose : bool, optional
            If True, prints diagnostic information during the estimation process.

        Returns
        -------
        fg : float
            Estimated foreground gray level.

        bg : float
            Estimated background gray level.
        """
        if verbose:
            print(f"[VIC][FG/BG] method={method}")
        fg, bg = find_fg_bg(self.image, method=method, verbose=verbose, save_file=None)
        if verbose:
            print(f"[VIC][FG/BG] fg={fg:.1f}, bg={bg:.1f}")
        return fg, bg

    def initialize_h_mesh_ICP(
        self,
        ICP_init: bool = True,
        h: Union[float, None] = None,
        disable_parallel: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the mesh placement and the normal search half-width `h`.

        This method optionally performs a rigid-body Iterative Closest Point (ICP)
        alignment between the surface mesh and an isosurface extracted from the
        image. It also initializes the normal search half-width `h`, which
        defines the integration domain along surface normals for the image
        correlation energies.

        The isosurface is obtained using a marching-cubes extraction at the
        mid-gray level between the estimated foreground and background values.

        Behavior depends on the input parameters:
            - If `ICP_init=True`, a rigid-body ICP alignment is performed and
              the resulting rotation matrix and translation vector are returned.
            - If `h` is `None`, the value of `h` is automatically estimated
              from the maximum distance between the mesh and the extracted
              isosurface.
            - If `ICP_init=False` and `h` is provided, no alignment is
              performed and `h` is left unchanged.

        Parameters
        ----------
        ICP_init : bool, optional
            If `True` (default), performs a rigid-body ICP alignment between the
            mesh and the image-derived isosurface.

        h : float or None, optional
            Normal search half-width used in image energy integration.
            If `None` (default), `h` is automatically estimated from the maximum
            mesh-to-image distance.

        disable_parallel : bool, optional
            If `True`, disables parallel execution during ICP and distance-field
            computations. By default, `False`.

        verbose : bool, optional
            If `True`, prints diagnostic information during initialization.
            By default, `True`.

        Returns
        -------
        Rmat : np.ndarray
            3x3 rotation matrix resulting from the ICP alignment.
            Identity if no ICP is performed.

        tvec : np.ndarray
            Translation vector resulting from the ICP alignment.
            Zero vector if no ICP is performed.

        h : float
            Initialized normal search half-width.
        """
        compute_h = h is None
        if ICP_init or compute_h:
            stl_mc = marching_cubes(self.image, 0.5 * (self.fg + self.bg))
            if ICP_init:
                Rmat, tvec, max_dist = self.mesh.ICP_rigid_body_transform(
                    stl_mc, disable_parallel=disable_parallel, plot_after=verbose
                )
                if compute_h:
                    h = max_dist + 1.0
                if verbose:
                    print(f"[VIC] ICP done | max_dist={max_dist:.3g}, h={h:.3g}")
            else:
                Rmat = np.eye(3)
                tvec = np.zeros(3)
                if compute_h:
                    distance_mesh = self.mesh.distance_to_meshio(stl_mc)
                    max_dist = np.abs(
                        distance_mesh.point_data["implicit_distance"]
                    ).max()
                    h = max_dist + 1.0
                    if verbose:
                        print(
                            f"[VIC] h from distance field | max_dist={max_dist:.3g}, h={h:.3g}"
                        )
        else:
            Rmat = np.eye(3)
            tvec = np.zeros(3)
        return Rmat, tvec, h

    def make_membrane_weight(
        self,
        rho: Union[float, None] = None,
        expected_mean_dist: float = 5.0,
        n_intg: int = 100,
    ) -> float:
        """
        Compute and set the membrane regularization weight.

        This method computes the weight associated with the membrane (elastic)
        regularization term based on the current problem configuration and image
        statistics. The weight is calibrated such that the regularization term
        is consistent with the expected image-to-surface distance scale.

        The actual computation is delegated to
        :func:`volVIC.membrane_stiffness.make_membrane_weight`.

        Parameters
        ----------
        rho : float or None, optional
            Value of the transition distance parameter used in the VIC
            model. If `None` (default), `self.initial_rho` is used.

        expected_mean_dist : float, optional
            Expected mean signed distance (in image units) between the surface
            and the image features. This value is used to calibrate the membrane
            weight. Default is `5.0`.

        n_intg : int, optional
            Number of integration points used to estimate the membrane weight.
            Higher values improve accuracy at the cost of additional computation.
            Default is `100`.

        Returns
        -------
        float
            Computed membrane regularization weight.
        """
        if rho is None:
            rho = self.initial_rho
        image_std = find_sigma_hat(self.image, self.fg, self.bg)
        self.membrane_weight = make_membrane_weight(
            self.mesh,
            self.image_energies,
            self.membrane_K,
            rho=rho,
            image_std=image_std,
            expected_mean_dist=expected_mean_dist,
            n_intg=n_intg,
        )

    def make_dirichlet(self):
        """
        Build the reduced-DOF Dirichlet representation from linear constraints.

        The object `self.constraints` stores linear equality constraints of
        the form `D @ u = c`. This method converts this representation into a
        reduced-DOF Dirichlet mapping of the form

            u = C @ dof + k ,

        where `C` is a basis of the null space of `D` (`D @ C = 0`) and
        `k` is a particular solution satisfying `D @ k = c`.

        The resulting mapping is stored in `self.dirichlet` and is used to
        eliminate constrained degrees of freedom in the solver.
        """
        self.dirichlet = self.constraints.create_dirichlet()

    def one_gauss_newton_iter(
        self,
        u_field: np.ndarray[np.floating],
        rho: float,
        verbose: bool = True,
        disable_parallel: bool = False,
    ) -> tuple[float, np.ndarray[np.floating], sps.spmatrix, float, float]:
        """
        Perform a single Gauss-Newton iteration for the VIC problem.

        This method updates the displacement field and the transition distance
        parameter `rho` by computing the incremental corrections using the
        Gauss-Newton scheme. The actual computation is performed by
        :func:`volVIC.solve.iteration`.

        Parameters
        ----------
        u_field : np.ndarray
            Current displacement field of the mesh nodes, shape (3, N).

        rho : float
            Current value of the transition distance parameter used in the VIC model.

        verbose : bool, optional
            If `True`, prints diagnostic information during the iteration.
            By default, `True`.

        disable_parallel : bool, optional
            If `True`, disables parallel execution during the iteration.
            By default, `False`.

        Returns
        -------
        du_field : np.ndarray
            Incremental displacement field computed during this iteration.

        drho : float
            Incremental update of the transition distance parameter `rho`.
        """
        du_field, drho = iteration(
            u_field,
            rho,
            self.mesh,
            self.image_energies,
            self.image,
            self.membrane_K,
            self.membrane_weight,
            self.dirichlet.C,
            verbose=verbose,
            disable_parallel=disable_parallel,
        )
        return du_field, drho

    def solve(
        self,
        u_field: np.ndarray[np.floating] = None,
        rho: float = None,
        eps: float = 5e-2,
        max_iter: int = 20,
        verbose: bool = True,
        disable_parallel: bool = False,
    ) -> tuple[np.ndarray[np.floating], float]:
        """
        Solve the VIC problem using a Gauss-Newton optimization.

        This method iteratively updates the displacement field and the transition
        distance parameter `rho` to minimize the VIC energy. Each iteration is
        performed using :meth:`one_gauss_newton_iter`.

        The iteration stops when either the maximum number of iterations is
        reached or the relative update of the displacement field is below
        `eps`.

        The displacement field is updated in-place and stored in `u_field`.
        The first iteration's data from all image energies are cached in
        `self.saved_data_0` for post-processing and visualization.

        Parameters
        ----------
        u_field : np.ndarray, optional
            Initial displacement field of shape (3, N). If `None` (default),
            initializes to zero.

        rho : float, optional
            Initial value of the distance scaling parameter. If `None` (default),
            `self.initial_rho` is used.

        eps : float, optional
            Convergence tolerance. The relative norm of the displacement update
            is compared to `eps` to decide convergence. Default is `5e-2`.

        max_iter : int, optional
            Maximum number of Gauss-Newton iterations. Default is `20`.

        verbose : bool, optional
            If `True`, prints iteration diagnostics including `rho` and
            relative displacement updates. Default is `True`.

        disable_parallel : bool, optional
            If `True`, disables parallel execution during the iterations.
            Default is `False`.

        Returns
        -------
        u_field : np.ndarray
            Final displacement field after convergence or reaching `max_iter`.

        rho : float
            Final value of the transition distance parameter after convergence.

        Notes
        -----
        This method relies on :meth:`one_gauss_newton_iter` to compute the
        incremental updates for each iteration.
        """
        if u_field is None:
            u_field = np.zeros((3, self.mesh.connectivity.nb_unique_nodes))
        if rho is None:
            rho = self.initial_rho
        for i in range(max_iter):
            du_field, drho = self.one_gauss_newton_iter(
                u_field, rho, verbose=verbose, disable_parallel=disable_parallel
            )
            u_field += du_field
            rho += drho
            rho = np.clip(rho, 0.1, 0.95 * self.image_energies[0].h)
            if i == 0:
                self.saved_data_0 = [
                    deepcopy(e.saved_data) for e in self.image_energies
                ]
            du_u = np.linalg.norm(du_field) / np.linalg.norm(u_field)
            if verbose:
                print(f"iter = {i}, rho = {rho:.4f}, du/u = {du_u:.2E}")
            if du_u < eps:
                break
        return u_field, rho

    def save_paraview(
        self,
        u_field: np.ndarray[np.floating],
        folder: str,
        name: str,
        disable_parallel: bool = False,
        verbose: bool = True,
    ):
        """
        Export the current VIC results to Paraview-compatible VTK files.

        This method computes the signed distance fields before and after the
        displacement update, associates them with the mesh nodes, and calls
        the mesh's :meth:`Mesh.save_paraview` method to save all fields for
        visualization in Paraview.

        The exported fields include:
            - `u`: displacement field provided in `u_field`.
            - `d0`: signed distance fields at the first iteration (cached in
              `self.saved_data_0`).
            - `d`: signed distance fields at the current iteration (not computed
              from `u_field`).

        Parameters
        ----------
        u_field : np.ndarray
            Displacement field of shape (3, N) to be saved.

        folder : str
            Destination folder where the VTK files will be written.

        name : str
            Base name for the exported VTK files.

        disable_parallel : bool, optional
            If `True`, disables parallel execution for distance field computation
            and mesh export. Default is `False`.

        verbose : bool, optional
            If `True` (default), prints diagnostic messages during computation
            and export.

        Notes
        -----
        The method automatically constructs the `XI_list` from the parametric
        coordinates of each image energy element and sets
        `fields_on_interior_only="auto"` for the mesh export.
        """
        d0 = compute_distance_field(
            self.image_energies,
            saved_data=self.saved_data_0,
            disable_parallel=disable_parallel,
            verbose=verbose,
        )
        d = compute_distance_field(
            self.image_energies,
            disable_parallel=disable_parallel,
            verbose=verbose,
        )
        separated_fields = [{"d": di[None], "d0": d0i[None]} for di, d0i in zip(d, d0)]
        unique_fields = {"u": u_field[None]}
        XI_list = [(e.xi, e.eta) for e in self.image_energies]
        self.mesh.save_paraview(
            folder,
            name,
            unique_fields=unique_fields,
            separated_fields=separated_fields,
            XI_list=XI_list,
            fields_on_interior_only="auto",
            disable_parallel=disable_parallel,
            verbose=verbose,
        )

    def plot_results(
        self,
        u_field: Union[None, np.ndarray[np.floating]] = None,
        disable_parallel: bool = False,
        verbose: bool = True,
        n_colors: int = 15,
        interior_only: bool = True,
        plt_ctrl_mesh: bool = False,
        pv_plotter: Union[pv.Plotter, None] = None,
        show: bool = True,
        elem_sep_color: str = "black",
        ctrl_poly_color: str = "green",
        **pv_add_mesh_kwargs,
    ):
        """
        Visualize VIC results with PyVista, optionally applying a displacement field.

        This method plots the signed distance fields of the mesh with respect to
        the image features. If a displacement field `u_field` is provided, it
        is applied to a copy of the mesh for visualization; otherwise, the
        distances from the first iteration are displayed. It is recommended
        to pass the displacement field obtained from `solve` to make sure to match
        the distance map to the correct displacement field.

        The visualization can be customized, including the number of colors,
        control mesh display, scalar bar, and interior-only rendering. Any
        additional keyword arguments are passed to :meth:`Mesh.plot`, which
        ultimately forwards them to :meth:`pv.Plotter.add_mesh`.

        Parameters
        ----------
        u_field : np.ndarray or None, optional
            Displacement field to apply to the mesh for visualization. If `None`
            (default), the distances from the first iteration (`self.saved_data_0`)
            are displayed. If provided, the mesh is warped according to this
            displacement. To correctly display the most recent VIC results,
            pass the `u_field` output from the `solve` method, as the distance
            map displayed corresponds to the last iteration computed by the
            image energy terms.

        disable_parallel : bool, optional
            If `True`, disables parallel computation of distance fields.
            Default is `False`.

        verbose : bool, optional
            If `True`, prints diagnostic messages. Default is `True`.

        n_colors : int, optional
            Number of discrete colors in the colormap. Default is `15`.

        interior_only : bool, optional
            If `True`, only interior elements of the mesh are plotted.
            Default is `True`.

        plt_ctrl_mesh : bool, optional
            If `True`, overlays the control mesh on the plot.
            Default is `False`.

        pv_plotter : pv.Plotter or None, optional
            PyVista plotter object to use. If `None`, a new plotter is created.

        show : bool, optional
            If `True`, displays the plot immediately. Default is `True`.

        elem_sep_color : str, optional
            Color used for separating elements. Only applied if `interior_only`
            is `False`. Default is `'black'`.

        ctrl_poly_color : str, optional
            Color of the control mesh. Only applied if `plt_ctrl_mesh` is `True`
            and `interior_only` is `False`. Default is `'green'`.

        **pv_add_mesh_kwargs : dict, optional
            Additional keyword arguments passed to :meth:`Mesh.plot`, such as
            `cmap`, `clim`, `scalar_bar_args`, etc.

        Returns
        -------
        pv.Plotter or None
            The PyVista plotter object used for the visualization.

        Notes
        -----
        - The signed distance field `d` is computed from the image energies via
          :func:`volVIC.VirtualImageCorrelationEnergyElem.compute_distance_field`.
        - The default colormap is a resampled `"RdBu"` with `n_colors`.
        - Scalar bar properties are automatically set but can be overridden via
          `pv_add_mesh_kwargs["scalar_bar_args"]`.
        - To make sure the deformation matches the distance field, always pass the
          displacement field returned by :meth:`solve`.
        """
        if u_field is None:
            d = compute_distance_field(
                self.image_energies,
                saved_data=self.saved_data_0,
                disable_parallel=disable_parallel,
                verbose=verbose,
            )
            mesh = self.mesh
        else:
            if verbose:
                print("Assume last iteration state.")
            d = compute_distance_field(
                self.image_energies,
                disable_parallel=disable_parallel,
                verbose=verbose,
            )
            mesh = deepcopy(self.mesh)
            mesh.unique_ctrl_pts.flat += u_field.ravel()

        cmap = plt.colormaps["RdBu"].resampled(n_colors)
        pv_add_mesh_kwargs.setdefault("cmap", cmap)
        if "clim" not in pv_add_mesh_kwargs:
            lim = self.image_energies[0].h - 1e-3
            pv_add_mesh_kwargs["clim"] = [-lim, lim]
        sargs = pv_add_mesh_kwargs.get("scalar_bar_args", {})
        defaults_sargs = dict(
            title="Signed distance to the image features [px]\n",
            vertical=False,
            position_x=0.1,
            position_y=0.05,
            height=0.1,
            width=0.8,
            n_colors=n_colors,
            title_font_size=25,
            label_font_size=25,
        )
        for key, value in defaults_sargs.items():
            sargs.setdefault(key, value)
        pv_add_mesh_kwargs["scalar_bar_args"] = sargs

        XI_list = [(e.xi, e.eta) for e in self.image_energies]
        return mesh.plot(
            XI_list=XI_list,
            separated_field=d,
            interior_only=interior_only,
            plt_ctrl_mesh=plt_ctrl_mesh,
            pv_plotter=pv_plotter,
            show=show,
            elem_sep_color=elem_sep_color,
            ctrl_poly_color=ctrl_poly_color,
            **pv_add_mesh_kwargs,
        )

    def propagate_displacement_to_volume_mesh(
        self,
        u_field: np.ndarray[np.floating],
        volume_mesh: Mesh,
        disable_parallel: bool = False,
    ) -> np.ndarray[np.floating]:
        """
        Propagate the surface displacement field to a volumetric mesh.

        This method maps the displacement field computed on the surface mesh
        (`self.mesh`) to a target volume mesh (`volume_mesh`). The mapping uses
        the surface-to-volume interpolation implemented in
        :meth:`Mesh.propagate_field_from_submesh`. The resulting volumetric
        displacement field is returned in the same coordinate system as the
        original surface mesh (before ICP).

        Parameters
        ----------
        u_field : np.ndarray[np.floating]
            Surface displacement field to propagate. Typically, this is the
            `u_field` obtained from the :meth:`solve` method.

        volume_mesh : Mesh
            Target volumetric mesh on which to propagate the displacement.

        disable_parallel : bool, optional
            If `True`, disables parallel computation. Default is `False`.

        Returns
        -------
        np.ndarray[np.floating]
            Displacement field defined on the volume mesh, in the same
            coordinate system as the input surface displacement.

        Notes
        -----
        - The propagation is performed via
          :meth:`Mesh.propagate_field_from_submesh`, which interpolates the
          surface displacement onto the volume nodes.
        - The method internally applies and then removes the rigid-body
          rotation `self.Rmat` used during VIC initialization to ensure
          consistency between surface and volume coordinate frames.
        """

        u_vol_field = volume_mesh.propagate_field_from_submesh(
            self.mesh,
            self.Rmat @ u_field,
            disable_parallel=disable_parallel,
        )
        u_vol_field = self.Rmat.T @ u_vol_field
        return u_vol_field
