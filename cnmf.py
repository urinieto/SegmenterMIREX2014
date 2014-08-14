"""
C-NMF method for segmentation, modified from here:

Nieto, O., Jehan, T., Convex Non-negative Matrix Factorization For Automatic
Music Structure Identification. Proc. of the 38th IEEE International Conference
on Acoustics, Speech, and Signal Processing (ICASSP). Vancouver, Canada, 2013.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import numpy as np
import pymf

# Local stuff
import utils


def cnmf(S, rank, niter=500):
    """(Convex) Non-Negative Matrix Factorization.

    Parameters
    ----------
    S: np.array(p, N)
        Features matrix. p row features and N column observations.
    rank: int
        Rank of decomposition
    niter: int
        Number of iterations to be used

    Returns
    -------
    F: np.array
        Cluster matrix (decomposed matrix)
    G: np.array
        Activation matrix (decomposed matrix)
        (s.t. S ~= F * G)
    """
    nmf_mdl = pymf.CNMF(S, num_bases=rank)
    nmf_mdl.factorize(niter=niter)
    F = np.asarray(nmf_mdl.W)
    G = np.asarray(nmf_mdl.H)
    return F, G


def most_frequent(x):
    """Returns the most frequent value in x."""
    return np.argmax(np.bincount(x))


def compute_labels(X, rank, R, bound_idxs, niter=300):
    """Computes the labels using the bounds."""

    X = X.T
    try:
        F, G = cnmf(X, rank, niter=niter)
    except:
        return [1]

    label_frames = filter_activation_matrix(G.T, R)
    label_frames = np.asarray(label_frames, dtype=int)

    # Get labels from the label frames
    labels = []
    bound_inters = zip(bound_idxs[:-1], bound_idxs[1:])
    for bound_inter in bound_inters:
        if bound_inter[1] - bound_inter[0] <= 0:
            labels.append(np.max(label_frames) + 1)
        else:
            labels.append(most_frequent(
                label_frames[bound_inter[0]:bound_inter[1]]))

    return labels


def filter_activation_matrix(G, R):
    """Filters the activation matrix G, and returns a flattened copy."""
    idx = np.argmax(G, axis=1)
    max_idx = np.arange(G.shape[0])
    max_idx = (max_idx, idx.flatten())
    G[:, :] = 0
    G[max_idx] = idx + 1
    G = np.sum(G, axis=1)
    G = utils.median_filter(G[:, np.newaxis], R)
    return G.flatten()


def segmentation(X, rank, R, h, niter=300):
    """
    Gets the segmentation (boundaries and labels) from the factorization
    matrices.

    Parameters
    ----------
    X: np.array()
        Features matrix (e.g. chromagram)
    rank: int
        Rank of decomposition
    R: int
        Size of the median filter for activation matrix
    niter: int
        Number of iterations for k-means
    bound_idxs : list
        Use previously found boundaries (None to detect them)

    Returns
    -------
    bounds_idx: np.array
        Bound indeces found
    labels: np.array
        Indeces of the labels representing the similarity between segments.
    """

    # Filter
    X = utils.median_filter(X, M=h)
    X = X.T

    # Find non filtered boundaries
    bound_idxs = None
    while True:
        if bound_idxs is None:
            try:
                F, G = cnmf(X, rank, niter=niter)
            except:
                return np.empty(0), [1]

            # Filter G
            G = filter_activation_matrix(G.T, R)
            if bound_idxs is None:
                bound_idxs = np.where(np.diff(G) != 0)[0] + 1

        if len(np.unique(bound_idxs)) <= 2:
            rank += 1
            bound_idxs = None
        else:
            break

    return bound_idxs
