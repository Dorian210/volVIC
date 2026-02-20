from typing import Literal, Union
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from scipy.interpolate import Akima1DInterpolator as interp
from scipy.signal import find_peaks


@nb.njit
def hist(img: np.ndarray[np.uint16]):
    """
    Compute the gray-level histogram of an unsigned integer image.

    This function counts the number of occurrences of each gray level
    in the input image. The histogram size is determined from the full
    range of the image data type (e.g. 0–65535 for `uint16`).

    Parameters
    ----------
    img : np.ndarray[np.uint16]
        Input grayscale image with unsigned integer values.

    Returns
    -------
    histogram : np.ndarray[np.uint64]
        Histogram array where `histogram[g]` is the number of pixels
        with gray level `g`. Histogram array of size
        `np.iinfo(img.dtype).max + 1 = 65_536`.

    Notes
    -----
    - Implemented with explicit loops for compatibility with Numba `njit`.
    - The histogram covers the full dynamic range of the input dtype,
      even if some gray levels are not present in the image.
    """
    hist = np.zeros(np.iinfo(img.dtype).max + 1, dtype="uint64")
    for i in img.flat:
        hist[i] += 1
    return hist


@nb.njit
def otsu_threshold(histogram: np.ndarray[np.uint64]) -> tuple[float, float]:
    """
    Estimate foreground and background gray levels using Otsu's method.

    This function applies Otsu's criterion to a gray-level histogram to
    find the threshold that maximizes the inter-class variance. It then
    returns representative mean gray levels for the foreground and
    background classes.

    Parameters
    ----------
    histogram : np.ndarray[np.uint64]
        Gray-level histogram of an image, where each entry represents
        the number of pixels at a given gray level.

    Returns
    -------
    fg_value : float
        Mean gray level of the foreground class.
    bg_value : float
        Mean gray level of the background class.

    Notes
    -----
    - The method assumes a bimodal histogram.
    - Returned values are floating-point means, not necessarily integer
      gray levels.
    - The foreground is defined as the class with the higher mean gray level.
    """
    total_pixels = histogram.sum()
    sum_total = np.sum(histogram * np.arange(histogram.size))
    sumB = 0
    weightB = 0
    current_max = 0
    mu1, mu2 = 0, 0
    for t in range(histogram.size):
        weightB += histogram[t]
        if weightB == 0:
            continue
        weightF = total_pixels - weightB
        if weightF == 0:
            break
        sumB += t * histogram[t]
        meanB = sumB / weightB
        meanF = (sum_total - sumB) / weightF
        var_between = weightB * weightF * (meanB - meanF) ** 2
        if var_between > current_max:
            current_max = var_between
            mu1 = meanB
            mu2 = meanF
    fg, bg = max(mu1, mu2), min(mu1, mu2)
    return fg, bg


def interp_fg_bg(histogram: np.ndarray[np.uint64]) -> tuple[float, float]:
    """
    Estimate foreground and background gray levels by histogram interpolation.

    This method analyzes the gray-level histogram of an image and estimates
    representative background and foreground gray levels by:
      1. Cropping the histogram to its significant support.
      2. Applying a low-pass filtering in the log-histogram domain by progressively
         truncating the FFT spectrum.
      3. Identifying major modes (peaks) and separating them using valleys.
      4. Refining the peak locations by interpolation to obtain sub-bin
         estimates of foreground and background gray levels.

    The approach is designed to be robust to noise and small secondary peaks,
    and is intended for bimodal or quasi-bimodal histograms commonly encountered
    in volumetric image correlation problems.

    Parameters
    ----------
    histogram : np.ndarray[np.uint64]
        Gray-level histogram of an image, where each entry represents the number
        of pixels at a given gray level.

    Returns
    -------
    fg_value : float
        Estimated foreground gray level.
    bg_value : float
        Estimated background gray level.

    Notes
    -----
    - The histogram is first cropped to exclude bins with negligible population
      (below 1% of the average bin count).
    - A low-frequency approximation of the log-histogram is constructed using
      a limited number of Fourier coefficients, controlled by a maximum number
      of sign changes in its derivative.
    - Foreground and background modes are identified as the dominant peaks in
      two regions separated by a valley in the smoothed histogram.
    - Sub-bin peak locations are obtained by differentiating and rooting an
      interpolated histogram representation.
    - This method is heuristic in nature but often more stable than Otsu's
      thresholding for unbalanced class distributions.
    """
    (wh,) = np.where(histogram >= 0.01 * histogram.sum() / histogram.size)
    a, b = wh[0], wh[-1]
    count = histogram[a : (b + 1)]
    x = np.arange(a, (b + 1))
    fft_log = np.fft.rfft(np.log(count + 1))
    fft_log_low_freq = np.zeros_like(fft_log)  # start with all the frequencies at 0
    n_max = 4  # max number of sign change in the derivative
    i = (
        n_max // 2
    )  # start the algorithm when there are enough sinusoids to display n_max sign changes in the derivative
    fft_log_low_freq[:i] = fft_log[:i]
    nb_der_zero = 0
    while (
        i < count.size and nb_der_zero <= n_max
    ):  # loops until the filtered function displays more than n_max sign changes in the derivative
        fft_log_low_freq[i] = fft_log[i]
        count_log_low_freq = np.fft.irfft(fft_log_low_freq, count.size)
        der = np.diff(
            count_log_low_freq, prepend=count_log_low_freq[-1]
        )  # compute the derivative
        increasing = der > 0  # mask the negative derivatives
        nb_der_zero = np.count_nonzero(
            np.diff(increasing, prepend=increasing[-1])
        )  # count the number of sign changes
        i += 1
    i -= 2
    fft_log_low_freq[i + 1] = 0
    count_log_low_freq = np.fft.irfft(fft_log_low_freq, count.size)
    count_low_freq = np.exp(count_log_low_freq)
    peaks, _ = find_peaks(count_low_freq)
    valleys, _ = find_peaks(-count_low_freq)
    # _, _, inf_borders, sup_borders = peak_widths(count_low_freq, peaks)
    if valleys[0] > peaks[0]:
        inf_bg, sup_bg = 0, int(valleys[0])
        inf_fg, sup_fg = int(valleys[0]), int(valleys[1])
    else:
        inf_bg, sup_bg = int(valleys[0]), int(valleys[1])
        inf_fg, sup_fg = int(valleys[1]), (count.size - 1)
    # inf_bg, sup_bg = int(inf_borders[1]), int(sup_borders[1] + 1)
    interpolator_bg = interp(x[inf_bg : (sup_bg + 1)], count[inf_bg : (sup_bg + 1)])
    loc_max_bg = interpolator_bg.derivative().roots()
    loc_max_bg = loc_max_bg[~np.isnan(loc_max_bg)]
    bg_counts = interpolator_bg(loc_max_bg)
    bg_value = loc_max_bg[np.argmax(bg_counts)]
    ind_bg = np.argmax(bg_counts)
    bg_value = loc_max_bg[ind_bg]
    # inf_fg, sup_fg = int(inf_borders[0]), int(sup_borders[0] + 1)
    interpolator_fg = interp(x[inf_fg : (sup_fg + 1)], count[inf_fg : (sup_fg + 1)])
    loc_max_fg = interpolator_fg.derivative().roots()
    loc_max_fg = loc_max_fg[~np.isnan(loc_max_fg)]
    fg_counts = interpolator_fg(loc_max_fg)
    ind_fg = np.argmax(fg_counts)
    fg_value = loc_max_fg[ind_fg]

    return fg_value, bg_value


def find_fg_bg(
    img: np.ndarray[np.uint16],
    method: Literal["otsu", "interp"] = "otsu",
    save_file: Union[str, None] = None,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Estimate background and foreground gray levels from a grayscale image histogram.

    This function analyzes the gray-level histogram of an unsigned integer image
    (typically `uint16`) and estimates representative background (`bg`) and
    foreground (`fg`) gray levels using either Otsu's method or a histogram
    interpolation strategy.

    Optionally, a histogram visualization can be displayed and saved, showing
    the estimated background and foreground levels.

    Parameters
    ----------
    img : np.ndarray[np.uint16]
        Input grayscale image with unsigned integer values (typically `uint16`).
    method : {"otsu", "interp"}, optional
        Method used to separate foreground and background gray levels:
        - `"otsu"`: Otsu thresholding applied to the histogram.
        - `"interp"`: Histogram-based interpolation heuristic.
        Default is `"otsu"`.
    save_file : str or None, optional
        Filename used to save the histogram plot (matplotlib compatible formats accepted).
        If `None`, the plot is not saved. Default is `None`.
    verbose : bool, optional
        If `True`, prints a short summary of the estimation and displays
        the histogram plot. Default is `True`.

    Returns
    -------
    fg_value : float
        Estimated foreground gray level.
    bg_value : float
        Estimated background gray level.

    Notes
    -----
    - The histogram is internally cropped to remove near-empty bins before
      visualization (the full histogram is used for estimation).
    - The returned values are floats, even though the input image is integer-valued.
    """
    histogram = hist(img)

    if method == "otsu":
        fg_value, bg_value = otsu_threshold(histogram)
    elif method == "interp":
        fg_value, bg_value = interp_fg_bg(histogram)
    else:
        raise ValueError(
            f"Unrecognised method '{method}'. method can only be set to 'otsu' or 'interp'"
        )

    if verbose:
        (wh,) = np.where(histogram >= 0.01 * histogram.sum() / histogram.size)
        a, b = wh[0], wh[-1]
        cropped_hist = histogram[a : (b + 1)]
        x = np.arange(a, (b + 1))
        bg_count, fg_count = np.interp([bg_value, fg_value], x, cropped_hist)

        fig, ax = (
            plt.subplots()
        )  # palette : #1B9E77 #D95F02 #7570B3 #E7298A #66A61E #E6AB02 #A6761D #666666
        lines1 = ax.fill_between(x, cropped_hist, color="#E6AB02", alpha=0.25)  # type: ignore
        ax.scatter([bg_value, fg_value], [bg_count, fg_count], c="#D95F02", zorder=10)
        text_bg = "background graylevel $bg=" + f"{bg_value:.1f}" + "$"
        text_fg = "foreground graylevel $fg=" + f"{fg_value:.1f}" + "$"
        ax.text(
            bg_value,
            bg_count * 1.1,
            text_bg,
            bbox=dict(
                facecolor="#7570B3",
                alpha=0.5,
                edgecolor="k",
                boxstyle="round",
                pad=0.25,
            ),
        )
        ax.text(
            fg_value,
            fg_count * 1.1,
            text_fg,
            bbox=dict(
                facecolor="#7570B3",
                alpha=0.5,
                edgecolor="k",
                boxstyle="round",
                pad=0.25,
            ),
        )
        ax.set_xlabel("graylevel")
        ax.set_ylabel("frequency")
        ax.set_yscale("log")
        ax.set_title("Histogram of the image")
        if save_file is not None:
            if verbose:
                print(f"Saving histogram to '{save_file}'")
            fig.savefig(save_file)
        plt.show()

    return fg_value, bg_value


def find_sigma_hat(image: np.ndarray[np.uint16], fg: float, bg: float) -> float:
    """
    Estimate the standard deviation of noise in a CT image by analyzing voxels
    confidently assigned to a single phase.

    This function computes the unbiased second moment estimate of the noise standard
    deviation, σ, by considering only voxels whose intensities are either below the
    background gray level (`bg`, corresponding to air) or above the foreground gray
    level (`fg`, corresponding to solid material). Voxels with intensities in the
    ambiguous range [`bg`, `fg`] are excluded to avoid partial volume effects. For each
    selected voxel, the deviation from its nominal phase value is calculated, and σ is
    estimated as the square root of the mean squared deviation.

    Parameters
    ----------
    image : np.ndarray[np.uint16]
        The input CT image as a `np.ndarray` of `np.uint16`, where voxel intensities
        represent X-ray absorption.
    fg : float
        The foreground gray level (nominal value for solid material).
    bg : float
        The background gray level (nominal value for air).

    Returns
    -------
    sigma_hat : float
        The estimated standard deviation of noise, computed from the clean phase voxels.

    Notes
    -----
    - Only voxels with intensities `< bg` (air) or `> fg` (material) are used for the estimation.
    - Voxels in the range [`bg`, `fg`] are excluded to avoid bias from partial volume effects.
    - The estimate is based on the second moment of deviations from the nominal phase values,
    following the folded normal distribution model.
    """
    gl_bg = image[image < bg]
    gl_fg = image[image > fg]
    sigma_hat = np.sqrt(
        (
            gl_bg.size * (gl_bg.std(ddof=1) ** 2 + (gl_bg.mean() - bg) ** 2)
            + gl_fg.size * (gl_fg.std(ddof=1) ** 2 + (gl_fg.mean() - fg) ** 2)
        )
        / (gl_bg.size + gl_fg.size)
    )
    return sigma_hat
