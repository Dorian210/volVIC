import numpy as np
from numpy.typing import NDArray


def g_slide(
    xi: NDArray[np.floating],
    eta: NDArray[np.floating],
    gamma: NDArray[np.floating],
    rho: float,
    bg: float = 0.0,
    fg: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate the virtual image function and its derivative with respect to `rho`.

    Parameters
    ----------
    xi : NDArray[np.floating]
        Array representing the xi coordinate.
    eta : NDArray[np.floating]
        Array representing the eta coordinate.
    gamma : NDArray[np.floating]
        Array representing the gamma coordinate.
    rho : float
        Half width of transition from `bg` to `fg` graylevel.
    bg : float, optional
        Background graylevel value, default is 0.
    fg : float, optional
        Foreground graylevel value, default is 1.

    Returns
    -------
    g : NDArray[np.floating]
        Virtual image function evaluation.
    g_prime : NDArray[np.floating]
        Derivative of the virtual image function with respect to `rho`.
    """
    g = np.zeros((xi.size, eta.size, gamma.size), dtype=gamma.dtype)
    g[:, :, gamma >= rho] = fg
    g[:, :, gamma <= -rho] = bg
    inside = np.abs(gamma) < rho
    gamma_rho_inside = gamma[None, None, inside] / rho
    g[:, :, inside] = 0.5 * (fg - bg) * (
        -0.5 * gamma_rho_inside**3 + 1.5 * gamma_rho_inside
    ) + 0.5 * (fg + bg)
    g_prime = np.zeros((xi.size, eta.size, gamma.size), dtype=gamma.dtype)
    g_prime[:, :, inside] = (0.5 * (fg - bg) * (-1.5 * gamma_rho_inside**2 + 1.5)) * (
        -gamma[None, None, inside] / (rho**2)
    )
    return g, g_prime
