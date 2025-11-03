# %%
from typing import Callable, Union
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from interpylate import TriLinearRegularGridInterpolator
from bsplyne import BSpline, parallel_blocks

if __name__=='__main__':
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from volVIC.Mesh import Mesh, MeshLattice
from volVIC.integration_space_image import linspace_for_VIC_elem
from volVIC.virtual_image import g_slide

class VirtualImageCorrelationEnergyElem:
    """
    Class representing the elementary VIC energy computation over a B-spline patch in 
    isoparametric space.

    This class encapsulates the setup and evaluation of the Virtual Image Correlation (VIC) 
    energy, including the construction of integration points and operators for a normal 
    neighborhood of a B-spline surface. It provides methods to compute the energy, its gradient, 
    and Hessian with respect to both the B-spline control point displacements and the virtual 
    image parameter, as well as utilities for interpolation and residual computation.

    Attributes
    ----------
    spline : BSpline
        The `BSpline` surface object defining the isoparametric space and mapping.
    ctrl_pts : np.ndarray[np.floating]
        The control points of the `BSpline` surface, as a `numpy` array of shape (3, N_xi, N_eta).
    virtual_image : Callable[[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating], float], tuple[np.ndarray[np.floating], np.ndarray[np.floating]]]
        Function to compute the virtual image and its derivative with respect to `rho`.
    h : float
        Half width of the normal neighborhood.
    xi : np.ndarray[np.floating]
        Integration points in the `xi` isoparametric direction.
    dxi : np.ndarray[np.floating]
        Weights of the integration points in the `xi` isoparametric direction.
    eta : np.ndarray[np.floating]
        Integration points in the `eta` isoparametric direction.
    deta : np.ndarray[np.floating]
        Weights of the integration points in the `eta` isoparametric direction.
    gamma : np.ndarray[np.floating]
        Integration points along the normal direction of the B-spline.
    dgamma : np.ndarray[np.floating]
        Weights of the integration points along the normal direction.
    Xv0 : np.ndarray[np.floating]
        Flattened coordinates of the reference points in the normal neighborhood.
    Uv_p : sps.spmatrix
        Sparse linear operator mapping control point displacements to displacements of points in 
        the normal neighborhood.
    wdetJs : np.ndarray[np.floating]
        xi and eta quadrature weights scaled by the surface Jacobian determinant.
    wdetJ : np.ndarray[np.floating]
        xi, eta and gamma quadrature weights scaled by the surface Jacobian determinant.

    Notes
    -----
    - Integration points and weights in the isoparametric space and along the normal direction 
    are computed automatically.
    - The `virtual_image` function must return a tuple (`g`, `g_prime`), where `g` is the virtual 
    image and `g_prime` its derivative with respect to `rho`.
    - Methods are provided for generating integration grids, constructing displacement operators, 
    interpolating image values, evaluating the virtual image, computing residuals, and assembling 
    the VIC energy and its derivatives.
    """
    spline: BSpline
    ctrl_pts: np.ndarray[np.floating]
    virtual_image: Callable[[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating], float], tuple[np.ndarray[np.floating], np.ndarray[np.floating]]]
    h: float
    xi: np.ndarray[np.floating]
    dxi: np.ndarray[np.floating]
    eta: np.ndarray[np.floating]
    deta: np.ndarray[np.floating]
    gamma: np.ndarray[np.floating]
    dgamma: np.ndarray[np.floating]
    Xv0: np.ndarray[np.floating]
    Uv_p: sps.spmatrix
    wdetJs: np.ndarray[np.floating]
    wdetJ: np.ndarray[np.floating]
    saved_data: dict[str, np.ndarray[np.floating]]
        
    def __init__(self, 
                 spline: BSpline, 
                 ctrl_pts: np.ndarray[np.floating], 
                 surf_dx: float, 
                 alpha: Union[float, tuple[tuple[float, float], tuple[float, float]]], 
                 width_dx: float, 
                 h: float, 
                 virtual_image: Callable[[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating], float], tuple[np.ndarray[np.floating], np.ndarray[np.floating]]]=g_slide):
        """
        Initialize a `VirtualImageCorrelationEnergyElem` object for VIC energy computation over a 
        B-spline surface.

        This constructor sets up the isoparametric space, integration points, and operators 
        required for evaluating the VIC energy in a normal neighborhood of the B-spline surface. 
        It allows customization of the integration grid and the virtual image model.

        Parameters
        ----------
        spline : BSpline
            The `BSpline` surface object defining the isoparametric space and mapping.
        ctrl_pts : np.ndarray[np.floating]
            The control points of the `BSpline` surface, as a `numpy` array of shape 
            (3, N_xi, N_eta).
        surf_dx : float
            Target mapped distance between integration points in the isoparametric directions.
        alpha : Union[float, tuple[tuple[float, float], tuple[float, float]]]
            Distance to ignore on the border of the patch in each isoparametric direction.
            If a `float`, the same value is used for all boundaries.
            If a tuple of tuples, ((`dist_xi_0`, `dist_xi_1`), (`dist_eta_0`, `dist_eta_1`)), 
            each value specifies the ignored distance at the corresponding boundary.
        width_dx : float
            Target mapped distance between integration points along the normal direction.
        h : float
            Half-width of the neighborhood along the normal direction.
        virtual_image : Callable[[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating], float], tuple[np.ndarray[np.floating], np.ndarray[np.floating]]], optional
            Function to compute the virtual image and its derivative with respect to `rho`. 
            By default, `g_slide_g_prime`.

        Notes
        -----
        - The integration points and weights in the isoparametric space and along the normal 
        direction are computed automatically.
        - The `virtual_image` function must return a tuple (`g`, `g_prime`), where `g` is the 
        virtual image and `g_prime` its derivative with respect to `rho`.
        """
        self.spline = spline
        self.ctrl_pts = ctrl_pts
        self.virtual_image = virtual_image
        self.h = h
        intg_points, intg_weights = self.make_intg_space(surf_dx, alpha, width_dx, h)
        self.xi, self.eta, self.gamma = intg_points
        self.dxi, self.deta, self.dgamma = intg_weights
        self.Xv0, self.Uv_p, self.wdetJs, self.wdetJ = self.make_operators()
        self.saved_data = {}
    
    def rigid_body_copy(self, new_ctrl_pts: np.ndarray[np.floating]):
        new = self.__class__.__new__(self.__class__)  # Bypass __init__
        new.spline = self.spline
        new.ctrl_pts = new_ctrl_pts
        new.virtual_image = self.virtual_image
        new.h = self.h
        new.xi = self.xi
        new.eta = self.eta
        new.gamma = self.gamma
        new.dxi = self.dxi
        new.deta = self.deta
        new.dgamma = self.dgamma
        new.Xv0 = self.Xv0
        new.Uv_p = self.Uv_p
        new.wdetJs = self.wdetJs
        new.wdetJ = self.wdetJ
        new.saved_data = self.saved_data
        return new
    
    def make_intg_space(
            self, 
            surf_dx: float, 
            padding: Union[float, tuple[tuple[float, float], tuple[float, float]]], 
            width_dx: float, 
            h: float
        ) -> tuple[tuple[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating]], tuple[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating]]]:
        """
        Generate integration points and weights in the isoparametric space and along the normal direction for the VIC element.

        This method computes arrays of integration points and their corresponding weights in the isoparametric directions (`xi`, `eta`) 
        of the `BSpline` surface, as well as along the normal direction (`gamma`). The integration points in the isoparametric space are 
        distributed so that the mapped distance between them is close to `surf_dx`, with optional ignored border distances specified by 
        `padding`. The normal direction is discretized over the interval `[-h, h]` with spacing close to `width_dx`.

        Parameters
        ----------
        surf_dx : float
            Target mapped distance between integration points in the isoparametric directions.
        padding : Union[float, tuple[tuple[float, float], tuple[float, float]]]
            Distance to ignore on the border of the patch in each isoparametric direction.
            If a `float`, the same value is used for all boundaries.
            If a tuple of tuples, ((`dist_xi_0`, `dist_xi_1`), (`dist_eta_0`, `dist_eta_1`)), 
            each value specifies the ignored distance at the corresponding boundary.
        width_dx : float
            Target mapped distance between integration points along the normal direction.
        h : float
            Half-width of the neighborhood along the normal direction.

        Returns
        -------
        (xi, eta, gamma) : tuple[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating]]
            Tuple of arrays containing the integration points in the `xi`, `eta` isoparametric directions and the normal direction `gamma`.
        (dxi, deta, dgamma) : tuple[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating]]
            Tuple of arrays containing the integration weights (step sizes) in the `xi`, `eta` isoparametric directions and the normal direction `gamma`.

        Notes
        -----
        - The integration points in the isoparametric space are computed using `linspace_for_VIC_elem`.
        - The normal direction is discretized uniformly over `[-h, h]`.
        - The number of integration points along the normal is chosen so that the spacing is as close as possible to `width_dx`.
        """
        (xi, eta), (dxi, deta) = linspace_for_VIC_elem(self.spline, self.ctrl_pts, dist=surf_dx, alpha=padding)
        n = int(np.rint(2*h/width_dx))
        true_dx = 2*h/n
        gamma = np.linspace(-h + 0.5*true_dx, h - 0.5*true_dx, n)
        dgamma = true_dx*np.ones_like(gamma)
        return (xi, eta, gamma), (dxi, deta, dgamma)
        
    def make_operators(self) -> tuple[np.ndarray[np.floating], sps.spmatrix, np.ndarray[np.floating], np.ndarray[np.floating]]:
        """
        Construct a linearized displacement operator and compute the initial position 
        of a set of points in a normal neighborhood of the B-spline surface.

        This operator allows exploring a tubular neighborhood of the surface by 
        linearly extrapolating points along the local normal direction.

        The displacement of a point at a signed normal distance γ from the surface is given by:
        
            u_v(ξ, η, γ) = u(ξ, η) + γ * (Φ(ξ, η) × a₃(ξ, η))
            
            Φ = φ₁ a₁ + φ₂ a₂
            
            φ₁ =  du/dη · a₃ / ||a₁ × a₂||
            φ₂ = -du/dξ · a₃ / ||a₁ × a₂||

        where:
            - u is the displacement of the surface,
            - a₁, a₂ are the tangents to the surface,
            - a₃ = (a₁ × a₂) / ||a₁ × a₂|| is the surface normal (normalized),
            - γ is the signed distance along the normal.

        The operator computed here linearly maps control point displacements to displacements 
        of γ-sampled points along the normal.

        Returns
        -------
        Xv0 : np.ndarray[np.floating]
            Flattened coordinates of the points in the reference configuration, 
            located along the normal at a signed distance γ from the surface.
            Flattened array of shape (3 * n_points,)
        Uv_p : sps.spmatrix
            Sparse linear operator that maps displacements at control points 
            to displacements of the corresponding points in the normal neighborhood.
            Sparse matrix of shape (3 * n_points, 3 * n_ctrl_pts).
        wdetJs : np.ndarray[np.floating]
            xi and eta quadrature weights scaled by the surface Jacobian determinant.
            Array of shape (n_surf_points,)
        wdetJ : np.ndarray[np.floating]
            xi, eta and gamma quadrature weights scaled by the surface Jacobian determinant.
            Array of shape (n_points,)
        """
        # First derivatives of basis functions at (xi, eta)
        dN_dxi, dN_deta = self.spline.DN([self.xi, self.eta], 1)  # type: ignore

        # Tangent vectors a1, a2 at evaluation points (shape: (3, n_points))
        A1 = self.ctrl_pts.reshape((3, -1)) @ dN_dxi.T
        A2 = self.ctrl_pts.reshape((3, -1)) @ dN_deta.T

        # Cross product a3 = a1 × a2, not yet normalized (shape: (3, n_points))
        A3 = np.cross(A1, A2, axis=0)

        # Normalize a3 to get unit normals (shape: (3, n_points))
        Se = np.linalg.norm(A3, axis=0)           # Surface element (area scaling)
        bad_points = Se==0
        A3[:, bad_points] = 0
        Se[bad_points] = 1
        A3 /= Se[None, :]                         # Unit normal vectors

        # Derivatives of φ₁ and φ₂
        A3_over_Se = A3/Se[None, :]
        phi1_p = sps.hstack((dN_deta.multiply(A3_over_Se[0, :, None]),
                             dN_deta.multiply(A3_over_Se[1, :, None]),
                             dN_deta.multiply(A3_over_Se[2, :, None])))
        phi2_p = sps.hstack((-dN_dxi.multiply(A3_over_Se[0, :, None]),
                             -dN_dxi.multiply(A3_over_Se[1, :, None]),
                             -dN_dxi.multiply(A3_over_Se[2, :, None])))

        # Compute derivatives of Φ
        psix_p = phi1_p.multiply(A1[0, :, None]) + phi2_p.multiply(A2[0, :, None])
        psiy_p = phi1_p.multiply(A1[1, :, None]) + phi2_p.multiply(A2[1, :, None])
        psiz_p = phi1_p.multiply(A1[2, :, None]) + phi2_p.multiply(A2[2, :, None])

        # Compute skew-symmetric product: Φ × a₃
        nx_p = psiy_p.multiply(A3[2, :, None]) - psiz_p.multiply(A3[1, :, None])
        ny_p = psiz_p.multiply(A3[0, :, None]) - psix_p.multiply(A3[2, :, None])
        nz_p = psix_p.multiply(A3[1, :, None]) - psiy_p.multiply(A3[0, :, None])
        n_p = sps.vstack((nx_p, ny_p, nz_p))      # Shape: (3*n_points, n_ctrl_pts)

        # Standard block-diagonal displacement operator
        N = self.spline.DN([self.xi, self.eta], 0)
        U_p = sps.block_diag([N] * 3)        # Shape: (3*n_points, 3*n_ctrl_pts)

        # Final displacement operator: u + γ * (Φ × a₃)
        ones = np.ones_like(self.gamma)           # Shape: (n_points,)
        Uv_p = sps.kron(U_p, ones[:, None]) + sps.kron(n_p, self.gamma[:, None])
        # Shape: (3*n_points, 3*n_ctrl_pts)

        # Initial positions of the γ-displaced points in the reference config
        X0s = self.ctrl_pts.reshape((3, -1)) @ N.T         # Shape: (3, n_points)
        Xv0 = (np.kron(X0s, ones) + np.kron(A3, self.gamma)).ravel()
        # Shape: (3*n_points,)

        # Weighted Jacobian for integration over the surface
        intg_weights_surf = np.kron(self.dxi, self.deta)        # Shape: (n_points,)
        wdetJs = Se * intg_weights_surf                         # Shape: (n_points,)
        
        detJ = np.kron(Se, np.ones_like(self.gamma))            # Shape: (n_points,)
        intg_weights = np.kron(np.kron(self.dxi, self.deta), 
                               self.dgamma)                     # Shape: (n_points,)
        wdetJ = detJ * intg_weights                             # Shape: (n_points,)

        return Xv0, Uv_p, wdetJs, wdetJ
    
    def f_df_dX(self, X: np.ndarray[np.floating], image: np.ndarray) -> tuple[np.ndarray[np.floating], sps.spmatrix]:
        """
        Interpolate the grey level values of a voxel-based `image` at specified coordinates `X` and compute the derivative 
        of the interpolation with respect to `X`.

        Parameters
        ----------
        X : np.ndarray[np.floating]
            Coordinates of the evaluation points as continuous pixel indices, flattened as
            `[x_0, ..., x_n, y_0, ..., y_n, z_0, ..., z_n]`.
        image : np.ndarray
            Voxel-based image (volume) in which the transition area is searched.

        Returns
        -------
        f : np.ndarray[np.floating]
            Array of grey levels of the `image` interpolated at the coordinates specified by `X`.
        df_dX : sps.spmatrix
            Sparse matrix representing the derivative of the grey level interpolation with respect to
            the integration point coordinates `X`.

        Notes
        -----
        - The interpolation and its gradient are computed using the internal `TriLinearRegularGridInterpolator`.
        - The returned `df_dX` is a sparse matrix with block-diagonal structure, where each block corresponds to the partial 
        derivatives with respect to `x`, `y`, and `z` coordinates.
        """
        # move half a voxel in each dimension towards (0, 0, 0) 
        # to simulate voxels spanning [i-0.5, i+0.5]x... instead of [i, i+1]x...
        (df_dx, df_dy, df_dz), f = TriLinearRegularGridInterpolator().grad(image, X.reshape((3, -1)), evaluate_too=True)
        df_dX = sps.hstack((sps.diags(df_dx), sps.diags(df_dy), sps.diags(df_dz)))
        return f, df_dX
    
    def g_dg_drho(self, rho: float) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]]:
        """
        Evaluate the virtual image and its derivative with respect to `rho` at the 
        integration points in the isoparametric space and along the normal direction.

        Parameters
        ----------
        rho : float
            Parameter of the virtual image.

        Returns
        -------
        g : np.array of float
            Virtual image at the integration points.
        dg_drho : np.array of float
            Derivative of the virtual image with respect to `rho` at the integration points.

        Notes
        -----
        - The integration points are defined by the arrays `xi`, `eta`, and `gamma` in the 
        isoparametric space and along the normal direction.
        - The `virtual_image` function must return a tuple (`g`, `g_prime`), where `g` is 
        the virtual image and `g_prime` its derivative with respect to `rho`.
        """
        g, g_prime = self.virtual_image(self.xi, self.eta, self.gamma, rho)
        g = g.ravel()
        dg_drho = g_prime.ravel()
        return g, dg_drho
    
    def r_dr_dg_dr_df(self, g: np.ndarray[np.floating], f: np.ndarray[np.floating]) -> tuple[np.ndarray[np.floating], sps.spmatrix, sps.spmatrix]:
        """
        Compute the residual of the VIC problem and its derivatives with respect to the virtual image and 
        the image at the integration points in the isoparametric space and along the normal direction.

        Parameters
        ----------
        g : np.ndarray[np.floating]
            Virtual image evaluated at the integration points (in the isoparametric space and along the 
            normal direction).
        f : np.ndarray[np.floating]
            Image evaluated at the integration points (in the isoparametric space and along the normal 
            direction).

        Returns
        -------
        r : np.ndarray[np.floating]
            Residual of the VIC problem, i.e., `g - f`, flattened as a 1D array.
        dr_dg : sps.spmatrix
            Sparse identity matrix representing the derivative of the residual with respect to the virtual 
            image evaluation (`g`).
        dr_df : sps.spmatrix
            Sparse negative identity matrix representing the derivative of the residual with respect to the 
            image evaluation (`f`).

        Notes
        -----
        - The input arrays `g` and `f` are reshaped internally to match the integration grid defined by (`xi`, `eta`, `gamma`).
        - The returned derivatives are sparse matrices of shape (`n_points`, `n_points`), where `n_points` is the total number of integration points.
        """
        g = g.reshape((self.xi.size*self.eta.size, self.gamma.size))
        f = f.reshape((self.xi.size*self.eta.size, self.gamma.size))
        # average_op = sps.block_diag([sps.csr_matrix(self.dgamma/(2*self.h))]*(self.xi.size*self.eta.size))
        # r = np.sqrt(average_op @ ((g - f).ravel()**2))
        # dr_dg = sps.diags(1/(2*r)) @ average_op @ sps.diags(2*(g - f).ravel())
        # dr_df = -dr_dg
        r = (g - f).ravel()
        dr_dg = sps.eye(g.size)
        dr_df = -sps.eye(f.size)
        return r, dr_dg, dr_df
    
    def E_dE_du_d2E_du2_dE_drho_d2E_drho2(
        self, 
        u: np.ndarray[np.floating], 
        rho: float, 
        image: np.ndarray
        ) -> tuple[float, np.ndarray[np.floating], sps.spmatrix, float, float]:
        """
        Compute the VIC energy, its gradient, and Hessian with respect to the B-spline control point displacements (`u`)
        and the virtual image parameter (`rho`).

        This method evaluates the energy functional by integrating the squared residual between the virtual image and the
        interpolated voxel-based `image` over the B-spline surface space and along the normal direction.
        It also computes the first and second derivatives of the energy with respect to both `u` and `rho`.

        Parameters
        ----------
        u : np.ndarray[np.floating]
            Displacements of the B-spline control points in the isoparametric space.
            Should be a 1D array of length `3 * n_ctrl_pts`.
        rho : float
            Virtual image parameter.
        image : np.ndarray
            Voxel-based image (volume) on which the surface fitting is performed.

        Returns
        -------
        E : float
            Value of the VIC energy for the current parameters.
        dE_du : np.ndarray[np.floating]
            Gradient of the energy with respect to the control point displacements (`u`).
            1D array of length `3 * n_ctrl_pts`.
        d2E_du2 : sps.spmatrix
            Sparse Hessian matrix of the energy with respect to the control point displacements (`u`).
            Shape: (`3 * n_ctrl_pts`, `3 * n_ctrl_pts`).
        dE_drho : float
            First derivative (scalar) of the energy with respect to the virtual image parameter (`rho`).
        d2E_drho2 : float
            Second derivative (scalar) of the energy with respect to the virtual image parameter (`rho`).

        Notes
        -----
        - The energy is computed as `0.5 * sum(r**2 * wdetJ)`, where `r` is the residual between the virtual image and the interpolated image,
        and `wdetJ` are the quadrature weights scaled by the Jacobian determinant to ensure integrating on the surface.
        - The gradient and Hessian are assembled using the chain rule, leveraging the derivatives of the residual with respect to both
        the B-spline control point displacements and the virtual image parameter.
        - Intermediate results (`r`, `f`, `g`, `rho`) are stored as attributes (`last_saved_r`, `last_saved_f`, `last_saved_g`, `last_saved_rho`)
        for potential later use.
        """
        
        Xv = self.Xv0 + self.Uv_p@u
        f, df_dXv = self.f_df_dX(Xv, image)
        del Xv
        df_du = df_dXv @ self.Uv_p
        del df_dXv
        g, dg_drho = self.g_dg_drho(rho)
        r, dr_dg, dr_df = self.r_dr_dg_dr_df(g, f)
        dr_du = dr_df @ df_du
        del df_du, dr_df
        dr_drho = dr_dg @ dg_drho
        del dg_drho, dr_dg
        
        E = float(0.5*np.sum(r*r*self.wdetJ))
        
        dE_du = dr_du.multiply((r*self.wdetJ)[:, None]).sum(axis=0).A.ravel()
        d2E_du2 = dr_du.multiply(self.wdetJ[:, None]).T@dr_du
        del dr_du
        dE_drho = (r*dr_drho*self.wdetJ).sum()
        d2E_drho2 = (dr_drho*dr_drho*self.wdetJ).sum()
        del dr_drho
        
        self.saved_data['last_saved_r'] = r
        self.saved_data['last_saved_f'] = f
        self.saved_data['last_saved_g'] = g
        self.saved_data['last_saved_rho'] = rho
        
        return E, dE_du, d2E_du2, dE_drho, d2E_drho2
    
    def taylor_test(self,
                    u: np.ndarray,
                    rho: float,
                    image: np.ndarray,
                    eps: float=1e-5,
                    random_state: int=None):
        """
        Taylor test for all derivative stages:
        - f_df_dX
        - g_dg_drho
        - r_dr_dg_dr_df (with respect to g and f)
        - E_dE_du_d2E_du2_dE_drho_d2E_drho2

        Parameters
        ----------
        u : np.ndarray, shape (3*n_ctrl_pts,)
            Control point displacement vector.
        rho : float
            Virtual image parameter.
        image : np.ndarray
            3D volume image used for interpolation.
        eps : float, optional
            Perturbation step for Taylor expansion (default=1e-5).
        random_state : int, optional
            Seed for reproducibility.
        """
        rng = np.random.default_rng(random_state)

        # 1) f_df_dX
        X0 = self.Xv0 + self.Uv_p @ u
        dX = rng.standard_normal(size=X0.shape)
        f0, df0 = self.f_df_dX(X0, image)
        f1, _ = self.f_df_dX(X0 + eps * dX, image)
        f_lin = f0 + eps * (df0 @ dX)
        err_f = np.abs(f1 - f_lin).mean()
        err_f_norm = err_f / (eps**2)
        print(f"[f_df_dX] mean|err|={err_f:.3e}, mean|err|/eps^2={err_f_norm:.3e}")

        # 2) g_dg_drho
        drho = float(rng.standard_normal())
        g0, dg0 = self.g_dg_drho(rho)
        g1, _ = self.g_dg_drho(rho + eps * drho)
        g_lin = g0 + eps * (dg0 * drho)
        err_g = np.abs(g1 - g_lin).mean()
        err_g_norm = err_g / (eps**2)
        print(f"[g_dg_drho] mean|err|={err_g:.3e}, mean|err|/eps^2={err_g_norm:.3e}")

        # 3a) r with respect to g (f fixed)
        dg_dir = rng.standard_normal(size=g0.shape)
        r0, drdg, _ = self.r_dr_dg_dr_df(g0, f0)
        r1_g, _, _ = self.r_dr_dg_dr_df(g0 + eps * dg_dir, f0)
        r_lin_g = r0 + eps * (drdg @ dg_dir)
        err_rg = np.abs(r1_g - r_lin_g).mean()
        err_rg_norm = err_rg / (eps**2)
        print(f"[r wrt g] mean|err|={err_rg:.3e}, mean|err|/eps^2={err_rg_norm:.3e}")

        # 3b) r with respect to f (g fixed)
        df_dir = rng.standard_normal(size=f0.shape)
        r1_f, _, drdf = self.r_dr_dg_dr_df(g0, f0 + eps * df_dir)
        r_lin_f = r0 + eps * (drdf @ df_dir)
        err_rf = np.abs(r1_f - r_lin_f).mean()
        err_rf_norm = err_rf / (eps**2)
        print(f"[r wrt f] mean|err|={err_rf:.3e}, mean|err|/eps^2={err_rf_norm:.3e}")

        # 4) Total energy E
        dx_u = rng.standard_normal(size=u.shape)
        dx_rho = float(rng.standard_normal())
        E0, dE_du, _, dE_drho, _ = self.E_dE_du_d2E_du2_dE_drho_d2E_drho2(u, rho, image)
        E1, _, _, _, _ = self.E_dE_du_d2E_du2_dE_drho_d2E_drho2(
            u + eps * dx_u,
            rho + eps * dx_rho,
            image
        )
        gradE = np.concatenate([dE_du, np.array([dE_drho])])
        dx = np.concatenate([dx_u, np.array([dx_rho])])
        E_lin = E0 + eps * (gradE @ dx)
        err_E = np.abs(E1 - E_lin).mean()
        err_E_norm = err_E / (eps**2)
        print(f"[E_total] mean|err|={err_E:.3e}, mean|err|/eps^2={err_E_norm:.3e}")



if __name__=='__main__':
    from bsplyne import BSpline
    size = 100
    ctrl_pts = np.stack((*np.meshgrid(np.linspace(size/4, 3*size/4, 3), np.linspace(size/4, 3*size/4, 3), indexing='ij'), 
                         size/2*np.ones((3, 3))))
    spline = BSpline([2]*2, [np.array([0, 0, 0, 1, 1, 1], dtype='float')]*2)
    image = np.zeros([size]*3, dtype='float')
    image[:, :, (6*size//10):] = 1.
    image += 0.01*np.random.randn(*image.shape)
    h = size/8
    energy = VirtualImageCorrelationEnergyElem(spline, ctrl_pts, surf_dx=1., alpha=0., width_dx=1., h=h)
    energy.taylor_test(np.zeros_like(ctrl_pts).ravel(), h/4, image)
    
    u = np.random.randn(ctrl_pts.size)*0.5
    Xv = energy.Xv0 + energy.Uv_p@u
    import pyvista as pv
    plotter = pv.Plotter()
    plotter.add_mesh(pv.PolyData(energy.Xv0.reshape((3, -1)).T), color='green', point_size=5, render_points_as_spheres=True)
    plotter.add_mesh(pv.PolyData(Xv.reshape((3, -1)).T), color='blue', point_size=5, render_points_as_spheres=True)
    plotter.show()


# %%
def make_image_energies(mesh, h, width_dx=0.5, surf_dx=1, alpha=0., virtual_image=g_slide, verbose=True):
    if isinstance(mesh, MeshLattice):
        cell_mesh = mesh.get_mesh_cell_one()
        separated_ctrl_pts = cell_mesh.get_separated_ctrl_pts()
        args_list = [(cell_mesh.splines[patch], 
                      separated_ctrl_pts[patch], 
                      surf_dx, 
                      alpha, 
                      width_dx, 
                      h, 
                      virtual_image) for patch in range(cell_mesh.connectivity.nb_patchs)]
        image_energies_cell = parallel_blocks(VirtualImageCorrelationEnergyElem, args_list, verbose=verbose, pbar_title="Init image energies")
        separated_ctrl_pts = mesh.get_separated_ctrl_pts()
        image_energies = []
        for i in range(mesh.l):
            for j in range(mesh.m):
                for k in range(mesh.n):
                    for cell_patch in range(cell_mesh.connectivity.nb_patchs):
                        patch = np.ravel_multi_index((i, j, k, cell_patch), (mesh.l, mesh.m, mesh.n, cell_mesh.connectivity.nb_patchs))
                        image_energies.append(image_energies_cell[cell_patch].rigid_body_copy(separated_ctrl_pts[patch]))
    else:
        separated_ctrl_pts = mesh.get_separated_ctrl_pts()
        args_list = [(mesh.splines[patch], 
                      separated_ctrl_pts[patch], 
                      surf_dx, 
                      alpha, 
                      width_dx, 
                      h, 
                      virtual_image) for patch in range(mesh.connectivity.nb_patchs)]
        image_energies = parallel_blocks(VirtualImageCorrelationEnergyElem, args_list, verbose=verbose, pbar_title="Init image energies")
    return image_energies

# %%
def plot_last_profile(image_energies, name="last_profile.svg"): # computes ANOVA values and plot them
    gamma = image_energies[0].gamma
    areas = np.array([e.wdetJs.sum() for e in image_energies])
    weights = areas/areas.sum()
    mean_inner = lambda e, field: (field.reshape((e.xi.size*e.eta.size, e.gamma.size))*e.wdetJs.reshape((-1, 1))).sum(axis=0)
    compute_surf_means = lambda field_name: np.array([mean_inner(e, e.saved_data[field_name]) for e in image_energies])/areas[:, None]
    var_inner = lambda e, field, mean: ((field.reshape((e.xi.size*e.eta.size, e.gamma.size)) - mean[None, :])**2*e.wdetJs.reshape((-1, 1))).sum(axis=0)
    compute_surf_vars = lambda field_name, means: np.array([var_inner(e, e.saved_data[field_name], m) for e, m in zip(image_energies, means)])/areas[:, None]
    g = (compute_surf_means("last_saved_g")*weights[:, None]).sum(axis=0)
    
    r_means = compute_surf_means("last_saved_r")
    f_means = compute_surf_means("last_saved_f")
    r_mean = (r_means*weights[:, None]).sum(axis=0)
    f_mean = (f_means*weights[:, None]).sum(axis=0)
    
    r_vars = compute_surf_vars("last_saved_r", r_means)
    f_vars = compute_surf_vars("last_saved_f", f_means)
    r_std_intra = np.sqrt((r_vars*weights[:, None]).sum(axis=0))
    f_std_intra = np.sqrt((f_vars*weights[:, None]).sum(axis=0))
    r_std_inter = np.sqrt(((r_means - r_mean[None, :])**2/areas[:, None]).sum(axis=0))
    f_std_inter = np.sqrt(((f_means - f_mean[None, :])**2/areas[:, None]).sum(axis=0))
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    ax15 = ax1.twinx()
    ax15.plot(gamma, r_mean, label=r"$r \pm\sigma_{inter-patch}$", color="#7570b3")
    ax15.fill_between(gamma, r_mean - r_std_inter, r_mean + r_std_inter, alpha=.3, color="#7570b3")
    ax15.set_ylabel("$\Delta$ Graylevel")
    ax15.legend(loc='upper right')
    ax1.plot(gamma, f_mean, label=r"$f \pm\sigma_{inter-patch}$", color="#d95f02")
    ax1.fill_between(gamma, f_mean - f_std_inter, f_mean + f_std_inter, alpha=.3, color="#d95f02")
    ax1.plot(gamma, g, label=r"$g$", color="#1b9e77")
    ax1.set_ylabel("Graylevel")
    ax1.legend(loc='upper left')

    ax25 = ax2.twinx()
    ax25.plot(gamma, r_mean, label=r"$r \pm\sigma_{intra-patch}$", color="#7570b3")
    ax25.fill_between(gamma, r_mean - r_std_intra, r_mean + r_std_intra, alpha=.3, color="#7570b3")
    ax25.set_ylabel("$\Delta$ Graylevel")
    ax25.legend(loc='upper right')
    ax2.plot(gamma, f_mean, label=r"$f \pm\sigma_{intra-patch}$", color="#d95f02")
    ax2.fill_between(gamma, f_mean - f_std_intra, f_mean + f_std_intra, alpha=.3, color="#d95f02")
    ax2.plot(gamma, g, label=r"$g$", color="#1b9e77")
    ax2.set_xlabel("Normal neighborhood (voxel)")
    ax2.set_ylabel("Graylevel")
    ax2.legend(loc='upper left')
    
    fig.suptitle("Inter/Intra-patch standard deviation comparison")
    fig.tight_layout()
    plt.savefig(name)
    plt.show()


def process_patch(u, image_energy, rho, image):
    result = image_energy.E_dE_du_d2E_du2_dE_drho_d2E_drho2(u.ravel(), rho, image)
    saved_data = image_energy.saved_data
    return result, saved_data

def compute_image_energy_operators(
    image_energies: list[VirtualImageCorrelationEnergyElem], 
    mesh: Mesh, 
    image: np.ndarray, 
    u_field: np.ndarray[np.floating], 
    rho: float, 
    verbose: bool=True, 
    disable_parallel: bool=False
    ) -> tuple[float, np.ndarray[np.floating], sps.spmatrix, float, float]:
    
    separated_u_field = mesh.unique_to_separated(u_field)
    args_list = [(separated_u_field[patch], image_energies[patch], rho) for patch in range(mesh.connectivity.nb_patchs)]
    res = parallel_blocks(process_patch, args_list, verbose=verbose, pbar_title="Compute image energy", shared_mem_last_arg=image, disable_parallel=disable_parallel)

    # Merge results
    E, grads, Hs, dE_drho, d2E_drho2 = 0, [], [], 0, 0
    for patch, ((E_, g, H, dE, d2E), saved_data) in enumerate(res):
        image_energies[patch].saved_data = saved_data
        E += E_
        grads.append(g)
        Hs.append(H)
        dE_drho += dE
        d2E_drho2 += d2E

    grad = mesh.assemble_grads(grads, 3)
    H = mesh.assemble_hessians(Hs, 3)
    
    return E, grad, H, dE_drho, d2E_drho2

# %%

def compute_distance_field_patch(image_energy, f: np.ndarray[np.floating], rho: float, max_iter: int=20) -> np.ndarray[np.floating]:
        """
        Calculate the distance between the target profile and the real profiles 
        in the image using Gauss-Newton optimization method.

        Parameters
        ----------
        f : np.ndarray[np.floating]
            Real image profiles to compare with the target profile.
        rho : float
            Parameter of the virtual image.
        eps : float, optional
            Tolerance for convergence, by default 1e-3.
        max_iter : int, optional
            Maximum number of iterations, by default 20.

        Returns
        -------
        d : np.ndarray[np.floating]
            Displacement field representing the difference between real and target profiles.
        """
        real_profiles = f.reshape((image_energy.xi.size, image_energy.eta.size, image_energy.gamma.size))
        d = np.zeros(real_profiles.shape[:-1], dtype='float')
        real_profiles_p = np.gradient(real_profiles, image_energy.gamma, axis=2)
        Epp = (real_profiles_p*real_profiles_p).sum(axis=2)
        for i in range(max_iter):
            gamma_d = image_energy.gamma[None, None, :] - d[:, :, None]
            target_profiles = image_energy.virtual_image(np.zeros(1), np.zeros(1), gamma_d.ravel(), rho)[0].reshape(gamma_d.shape)
            residu = real_profiles - target_profiles
            Ep = (residu*real_profiles_p).sum(axis=2)
            d += -Ep/Epp
        d[d>image_energy.h] = image_energy.h
        d[d<-image_energy.h] = -image_energy.h
        return d

def compute_distance_field(
    mesh: Mesh, 
    image_energies: list[VirtualImageCorrelationEnergyElem], 
    verbose: bool=True, 
    disable_parallel: bool=False
    ) -> list[np.ndarray[np.floating]]:
    args_list = [(image_energies[patch], image_energies[patch].saved_data['last_saved_f'], image_energies[patch].saved_data['last_saved_rho']) for patch in range(mesh.connectivity.nb_patchs)]
    d = parallel_blocks(compute_distance_field_patch, args_list, verbose=verbose, pbar_title="Compute distance field", disable_parallel=disable_parallel)
    d = [d_i[None] for d_i in d]
    return d