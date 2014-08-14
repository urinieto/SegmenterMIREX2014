"""
Foote method for segmentation, published here:

Foote, J. (2000). Automatic Audio Segmentation Using a Measure Of Audio
Novelty. In Proc. of the IEEE International Conference of Multimedia and Expo
(pp. 452-455). New York City, NY, USA.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

# Local stuff
import utils


def segmentation(F, M, Mg, L, plot=False):
    """Computes the Foote segmentator.

    Parameters
    ----------
    F : np.array((N,M))
        Features matrix of N beats x M features.
    M : int
        Median filter size for the audio features (in beats).
    Mg : int
        Gaussian kernel size (in beats).
    L : int
        Median filter size for the adaptive threshold

    Return
    ------
    bound_idx : np.array
        Array containing the indices of the boundaries.
    """
    # Filter
    F = utils.median_filter(F, M=M)

    # Self Similarity Matrix
    S = utils.compute_ssm(F)

    # Compute gaussian kernel
    G = utils.compute_gaussian_krnl(Mg)

    # Compute the novelty curve
    nc = utils.compute_nc(S, G)

    # Find peaks in the novelty curve
    return utils.pick_peaks(nc, L=L, plot=plot)
