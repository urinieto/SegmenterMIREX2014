"""
2D-FMC for segmentation, published here:

Nieto, O., & Bello, J. P. (2014). Music Segment Similarity Using 2D-Fourier
Magnitude Coefficients. In Proc. of the 39th IEEE International Conference on
Acoustics Speech and Signal Processing (pp. 664-668). Florence, Italy.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import numpy as np
import scipy.cluster.vq as vq
import logging

# Local stuff
import utils_2dfmc as utils2d
from xmeans import XMeans


MIN_LEN = 4     # Minimum lenght for the segments

def get_pcp_segments(PCP, bound_idxs):
    """Returns a set of segments defined by the bound_idxs."""
    pcp_segments = []
    for i in xrange(len(bound_idxs)-1):
        pcp_segments.append(PCP[bound_idxs[i]:bound_idxs[i+1], :])
    return pcp_segments


def pcp_segments_to_2dfmc_fixed(pcp_segments, N=75):
    """From a list of PCP segments, return a list of 2D-Fourier Magnitude
        Coefs using a fixed segment size (N) and aggregating."""

    fmcs = []
    for pcp_segment in pcp_segments:
        X = []

        # Append so that we never lose a segment
        skip = False
        while pcp_segment.shape[0] < MIN_LEN:
            try:
                pcp_segment = np.vstack((pcp_segment,
                                         pcp_segment[-1][np.newaxis, :]))
            except:
                logging.warning("Error: Can't stack PCP arrays, "
                                "skipping segment")
                skip = True
                break

        if skip:
            continue

        curr_len = pcp_segment.shape[0]

        if curr_len > N:
            # Sub segment if greater than minimum
            for i in xrange(curr_len - N + 1):
                sub_segment = pcp_segment[i:i + N]
                X.append(utils2d.compute_ffmc2d(sub_segment))

            # Aggregate
            X = np.max(np.asarray(X), axis=0)

            fmcs.append(X)

        elif curr_len <= N:
            # Zero-pad
            X = np.zeros((N, pcp_segment.shape[1]))
            X[:curr_len, :] = pcp_segment

            # 2D-FMC
            fmcs.append(utils2d.compute_ffmc2d(X))

    return np.asarray(fmcs)


def compute_labels_kmeans(fmcs, k=6):
    # Removing the higher frequencies seem to yield better results
    fmcs = fmcs[:, fmcs.shape[1]/2:]

    fmcs = np.log1p(fmcs)
    wfmcs = vq.whiten(fmcs)

    dic, dist = vq.kmeans(wfmcs, k, iter=100)
    labels, dist = vq.vq(wfmcs, dic)

    return labels


def compute_similarity(PCP, bound_idxs, xmeans=False, k=5, N=32, seed=None):
    """Main function to compute the segment similarity of file file_struct."""

    # Get PCP segments
    pcp_segments = get_pcp_segments(PCP, bound_idxs)

    # Get the 2d-FMCs segments
    fmcs = pcp_segments_to_2dfmc_fixed(pcp_segments, N=N)
    if fmcs == [] or fmcs is None:
        return np.arange(len(bound_idxs) - 1)

    # Compute the labels using kmeans
    if xmeans:
        xm = XMeans(fmcs, plot=False, seed=seed)
        k = xm.estimate_K_knee(th=0.01, maxK=8)
    est_labels = compute_labels_kmeans(fmcs, k=k)

    # Plot results
    #plot_pcp_wgt(PCP, bound_idxs)

    return est_labels
