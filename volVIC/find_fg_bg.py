# %%
from typing import Union
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from scipy.interpolate import Akima1DInterpolator as interp
from scipy.signal import find_peaks

@nb.njit
def hist(img: np.ndarray[np.uint16]):
    hist = np.zeros(np.iinfo(img.dtype).max + 1, dtype='uint64')
    for i in img.flat:
        hist[i] += 1
    return hist

@nb.njit
def otsu_threshold(histogram: np.ndarray[np.uint64]) -> tuple[int, float, float]:
    total_pixels = histogram.sum()
    sum_total = np.sum(histogram*np.arange(histogram.size))
    sumB = 0
    weightB = 0
    current_max = 0
    threshold = 0
    mu1, mu2 = 0, 0
    for t in range(histogram.size):
        weightB += histogram[t]
        if weightB==0:
            continue
        weightF = total_pixels - weightB
        if weightF==0:
            break
        sumB += t*histogram[t]
        meanB = sumB/weightB
        meanF = (sum_total - sumB)/weightF
        var_between = weightB*weightF*(meanB - meanF)**2
        if var_between>current_max:
            current_max = var_between
            threshold = t
            mu1 = meanB
            mu2 = meanF
    return threshold, mu1, mu2

def find_fg_bg(img: np.ndarray[np.uint16], save_file_base: Union[str, None]="./fg_bg") -> tuple[float, float]:
    """
    Calculate the background and foreground gray levels of a unsigned integer filled image.

    Parameters
    ----------
    image : np.ndarray[np.uint16]
        The input unsigned integer (uint16) filled image.
    save_file_base : Union[str, None], optional
        The base filename for saving the histogram visualization, default is "./fg_bg".
        If `None`, don't make the plot.

    Returns
    -------
    fg_value : float
        The foreground gray level.
    bg_value : float
        The background gray level.
    """
    init_count = hist(img)

    wh, = np.where(init_count>=0.01*img.size/init_count.size)
    a, b = wh[0], wh[-1]
    count = init_count[a:(b + 1)]
    x = np.arange(a, (b + 1))
    fft_log = np.fft.rfft(np.log(count + 1))
    fft_log_low_freq = np.zeros_like(fft_log) # start with all the frequencies at 0
    n_max = 4 # max number of sign change in the derivative
    i = n_max//2 # start the algorithm when there are enough sinusoids to display n_max sign changes in the derivative
    fft_log_low_freq[:i] = fft_log[:i]
    nb_der_zero = 0
    while i<count.size and nb_der_zero<=n_max: # loops until the filtered function displays more than n_max sign changes in the derivative
        fft_log_low_freq[i] = fft_log[i]
        count_log_low_freq = np.fft.irfft(fft_log_low_freq, count.size)
        der = np.diff(count_log_low_freq, prepend=count_log_low_freq[-1]) # compute the derivative
        increasing = der>0 # mask the negative derivatives
        nb_der_zero = np.count_nonzero(np.diff(increasing, prepend=increasing[-1])) # count the number of sign changes
        i += 1
    i -= 2
    fft_log_low_freq[i + 1] = 0
    count_log_low_freq = np.fft.irfft(fft_log_low_freq, count.size)
    count_low_freq = np.exp(count_log_low_freq)
    peaks, _ = find_peaks(count_low_freq)
    valleys, _ = find_peaks(-count_low_freq)
    # _, _, inf_borders, sup_borders = peak_widths(count_low_freq, peaks)
    if valleys[0]>peaks[0]:
        inf_bg, sup_bg = 0, int(valleys[0])
        inf_fg, sup_fg = int(valleys[0]), int(valleys[1])
    else:
        inf_bg, sup_bg = int(valleys[0]), int(valleys[1])
        inf_fg, sup_fg = int(valleys[1]), (count.size - 1)
    # inf_bg, sup_bg = int(inf_borders[1]), int(sup_borders[1] + 1)
    interpolator_bg = interp(x[inf_bg:(sup_bg + 1)], count[inf_bg:(sup_bg + 1)])
    loc_max_bg = interpolator_bg.derivative().roots()
    loc_max_bg = loc_max_bg[~np.isnan(loc_max_bg)]
    bg_counts = interpolator_bg(loc_max_bg)
    bg_value = loc_max_bg[np.argmax(bg_counts)]
    ind_bg = np.argmax(bg_counts)
    bg_value = loc_max_bg[ind_bg]
    bg_count = bg_counts[ind_bg]
    # inf_fg, sup_fg = int(inf_borders[0]), int(sup_borders[0] + 1)
    interpolator_fg = interp(x[inf_fg:(sup_fg + 1)], count[inf_fg:(sup_fg + 1)])
    loc_max_fg = interpolator_fg.derivative().roots()
    loc_max_fg = loc_max_fg[~np.isnan(loc_max_fg)]
    fg_counts = interpolator_fg(loc_max_fg)
    ind_fg = np.argmax(fg_counts)
    fg_value = loc_max_fg[ind_fg]
    fg_count = fg_counts[ind_fg]

    if save_file_base is not None:
        fig, ax = plt.subplots() # palette : #1B9E77 #D95F02 #7570B3 #E7298A #66A61E #E6AB02 #A6761D #666666
        lines1 = ax.fill_between(x, count, color='#E6AB02', alpha=0.25) # type: ignore
        ax.scatter([bg_value, fg_value], [bg_count, fg_count], c='#D95F02', zorder=10)
        text_bg = r"background graylevel \$\bg=" + f"{bg_value:.1f}" + r"\$"
        text_fg = r"foreground graylevel \$\fg=" + f"{fg_value:.1f}" + r"\$"
        try:
            from textalloc import allocate_text
            allocate_text(fig, ax, 
                        [bg_value, fg_value], [bg_count, fg_count], [text_bg, text_fg], 
                        x_scatter=[bg_value, fg_value], y_scatter=[bg_count, fg_count], 
                        x_lines=[x, x], y_lines=[count, count_low_freq], 
                        textsize=10, #min_distance=0.05, max_distance=1, 
                        draw_lines=False, 
                        bbox=dict(facecolor='#7570B3', alpha=0.5, edgecolor='k', boxstyle='round', pad=0.25))
        except:
            ax.text(bg_value, bg_count*1.1, text_bg, bbox=dict(facecolor='#7570B3', alpha=0.5, edgecolor='k', boxstyle='round', pad=0.25))
            ax.text(fg_value, fg_count*1.1, text_fg, bbox=dict(facecolor='#7570B3', alpha=0.5, edgecolor='k', boxstyle='round', pad=0.25))
        ax.set_xlabel("graylevel")
        ax.set_ylabel("frequency")
        ax.set_yscale('log')
        ax.set_title("Histogram of the image")
        svg_file = save_file_base + ".svg"
        fig.savefig(svg_file)
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
    gl_bg = image[image<bg]
    gl_fg = image[image>fg]
    sigma_hat = np.sqrt((
        gl_bg.size*(gl_bg.std()**2 + (gl_bg.mean() - bg)**2)
      + gl_fg.size*(gl_fg.std()**2 + (gl_fg.mean() - fg)**2)
    ) / (gl_bg.size + gl_fg.size))
    return sigma_hat


if __name__=='__main__':
    from skimage.io import imread
    image = imread("/home-local/dbichet/Documents/These/code/VolVIC/exemples/metalic_BCC_traction/SlicesY-Lattice_BCC_traction_binned.tiff")
    fg, bg = find_fg_bg(image)
    print(fg, bg)
    y = hist(image)
    wh, = np.where(y>=0.01*image.size/y.size)
    a, b = wh[0], wh[-1]
    y = y[a:(b + 1)]
    x = np.arange(a, b + 1)
    
    from scipy.stats import norm
    def gaussian_mixture(x, h1, h12, h2, sigma, bg=bg, fg=fg, n=10):
        mus = np.linspace(bg, fg, n + 2)
        g = np.zeros_like(x, dtype='float')
        A = np.sqrt(2*np.pi)*sigma
        g += h1*A*norm.pdf(x, mus[0], sigma)
        for mu in mus[1:-1]:
            g += h12*A*norm.pdf(x, mu, sigma)
        g += h2*A*norm.pdf(x, mus[-1], sigma)
        return g
    
    from scipy.optimize import curve_fit
    popt, _ = curve_fit(gaussian_mixture, x, y, [0.5, 0., 0.5, 1])
    
    fig, ax = plt.subplots()
    ax.fill_between(x, y, color='#E6AB02', alpha=0.25) # type: ignore
    ax.set_yscale("log")
    ylim = ax.get_ylim()
    ax.plot(x, gaussian_mixture(x, 70381.71649086, 1111.69351907, 2416.09899673, 120.849257))
    ax.set_ylim(ylim)
    ax.set_xlim([a, b])


# %%