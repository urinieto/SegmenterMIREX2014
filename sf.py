#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using the Structural
Features method:

Serrà, J., Müller, M., Grosche, P., & Arcos, J. L. (2012). Unsupervised
Detection of Music Boundaries by Time Series Structure Features.
In Proc. of the 26th AAAI Conference on Artificial Intelligence
(pp. 1613–1619).

Toronto, Canada.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"


import numpy as np
from scipy.spatial import distance
from scipy import signal
from scipy.ndimage import filters
import pylab as plt

# Local stuff
import utils


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in xrange(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def gaussian_filter(X, M=8, axis=0):
    """Gaussian filter along the first axis of the feature matrix X."""
    for i in xrange(X.shape[axis]):
        if axis == 1:
            X[:, i] = filters.gaussian_filter(X[:, i], sigma=M / 2.)
        elif axis == 0:
            X[i, :] = filters.gaussian_filter(X[i, :], sigma=M / 2.)
    return X


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Serra's paper."""
    g = signal.gaussian(M, M / 3., sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M / 2:, :M / 2] = -G[M / 2:, :M / 2]
    G[:M / 2, M / 1:] = -G[:M / 2, M / 1:]
    return G


def compute_nc(X):
    """Computes the novelty curve from the structural features."""
    N = X.shape[0]
    # nc = np.sum(np.diff(X, axis=0), axis=1) # Difference between SF's

    nc = np.zeros(N)
    for i in xrange(N - 1):
        nc[i] = distance.euclidean(X[i, :], X[i + 1, :])

    # Normalize
    nc += np.abs(nc.min())
    nc /= nc.max()
    return nc


def pick_peaks(nc, L=16, offset_denom=0.1):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() * float(offset_denom)
    th = filters.median_filter(nc, size=L) + offset
    peaks = []
    for i in xrange(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    #plt.plot(nc)
    #plt.plot(th)
    #for peak in peaks:
        #plt.axvline(peak, color="m")
    #plt.show()
    return peaks


def circular_shift(X):
    """Shifts circularly the X squre matrix in order to get a
        time-lag matrix."""
    N = X.shape[0]
    L = np.zeros(X.shape)
    for i in xrange(N):
        L[i, :] = np.asarray([X[(i + j) % N, j] for j in xrange(N)])
    return L


def embedded_space(X, m, tau=1):
    """Time-delay embedding with m dimensions and tau delays."""
    N = X.shape[0] - int(np.ceil(m))
    Y = np.zeros((N, int(np.ceil(X.shape[1] * m))))
    for i in xrange(N):
        rem = int((m % 1) * X.shape[1])  # Reminder for float m
        Y[i, :] = np.concatenate((X[i:i + int(m), :].flatten(),
                                 X[i + int(m), :rem]))
    return Y


def segmentation(F):
    """Main process."""

    # Structural Features params
    Mp = 32         # Size of the adaptive threshold for peak picking
    od = 0.1        # Offset coefficient for adaptive thresholding

    M = 16          # Size of gaussian kernel in beats
    m = 3           # Number of embedded dimensions
    k = 0.06        # k*N-nearest neighbors for the recurrence plot

    # Emedding the feature space (i.e. shingle)
    E = embedded_space(F, m)

    # Recurrence matrix
    R = utils.recurrence_matrix(E.T, k=k * int(F.shape[0]),
                                width=0,  # zeros from the diagonal
                                metric="seuclidean",
                                sym=True).astype(np.float32)

    # Check size in case the track is too short
    if R.shape[0] > 0:
        # Circular shift
        L = circular_shift(R)

        # Obtain structural features by filtering the lag matrix
        SF = gaussian_filter(L.T, M=M, axis=1)
        SF = gaussian_filter(L.T, M=1, axis=0)
        # plt.imshow(SF.T, interpolation="nearest", aspect="auto"); plt.show()

        # Compute the novelty curve
        nc = compute_nc(SF)

        # Find peaks in the novelty curve
        est_bounds = pick_peaks(nc, L=Mp, offset_denom=od)

        # Re-align embedded space
        est_bound_idxs = np.asarray(est_bounds) + int(np.ceil(m / 2.))
    else:
        est_bound_idxs = []

    if len(est_bound_idxs) == 0:
        est_bound_idxs = np.asarray([0])  # Return first one

    return est_bound_idxs
