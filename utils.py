"""
Useful functions that are quite common for music segmentation
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import copy
import numpy as np
import os
import scipy
from scipy.spatial import distance
from scipy.ndimage import filters
from scipy import signal
import pylab as plt


def lognormalize_chroma(C):
    """Log-normalizes chroma such that each vector is between -80 to 0."""
    C += np.abs(C.min()) + 0.1
    C = C/C.max(axis=0)
    C = 80 * np.log10(C)  # Normalize from -80 to 0
    return C


def normalize_matrix(X):
    """Nomalizes a matrix such that it's maximum value is 1 and minimum is 0."""
    X += np.abs(X.min())
    X /= X.max()
    return X


def ensure_dir(directory):
    """Makes sure that the given directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in xrange(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Foote's paper."""
    g = signal.gaussian(M, M / 3., sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M / 2:, :M / 2] = -G[M / 2:, :M / 2]
    G[:M / 2, M / 2:] = -G[:M / 2, M / 2:]
    return G


def compute_ssm(X, metric="seuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    D /= D.max()
    return 1 - D


def compute_nc(X, G):
    """Computes the novelty curve from the self-similarity matrix X and
        the gaussian kernel G."""
    N = X.shape[0]
    M = G.shape[0]
    nc = np.zeros(N)

    for i in xrange(M / 2, N - M / 2 + 1):
        nc[i] = np.sum(X[i - M / 2:i + M / 2, i - M / 2:i + M / 2] * G)

    # Normalize
    nc += nc.min()
    nc /= nc.max()
    return nc


def resample_mx(X, incolpos, outcolpos):
    """
    Method from Librosa
    Y = resample_mx(X, incolpos, outcolpos)
    X is taken as a set of columns, each starting at 'time'
    colpos, and continuing until the start of the next column.
    Y is a similar matrix, with time boundaries defined by
    outcolpos.  Each column of Y is a duration-weighted average of
    the overlapping columns of X.
    2010-04-14 Dan Ellis dpwe@ee.columbia.edu  based on samplemx/beatavg
    -> python: TBM, 2011-11-05, TESTED
    """
    noutcols = len(outcolpos)
    Y = np.zeros((X.shape[0], noutcols))
    # assign 'end times' to final columns
    if outcolpos.max() > incolpos.max():
        incolpos = np.concatenate([incolpos,[outcolpos.max()]])
        X = np.concatenate([X, X[:,-1].reshape(X.shape[0],1)], axis=1)
    outcolpos = np.concatenate([outcolpos, [outcolpos[-1]]])
    # durations (default weights) of input columns)
    incoldurs = np.concatenate([np.diff(incolpos), [1]])

    for c in range(noutcols):
        firstincol = np.where(incolpos <= outcolpos[c])[0][-1]
        firstincolnext = np.where(incolpos < outcolpos[c+1])[0][-1]
        lastincol = max(firstincol,firstincolnext)
        # default weights
        wts = copy.deepcopy(incoldurs[firstincol:lastincol+1])
        # now fix up by partial overlap at ends
        if len(wts) > 1:
            wts[0] = wts[0] - (outcolpos[c] - incolpos[firstincol])
            wts[-1] = wts[-1] - (incolpos[lastincol+1] - outcolpos[c+1])
        wts = wts * 1. /sum(wts)
        Y[:,c] = np.dot(X[:,firstincol:lastincol+1], wts)
    # done
    return Y


def chroma_to_tonnetz(C):
    """Transforms chromagram to Tonnetz (Harte, Sandler, 2006)."""
    N = C.shape[0]
    T = np.zeros((N, 6))

    r1 = 1      # Fifths
    r2 = 1      # Minor
    r3 = 0.5    # Major

    # Generate Transformation matrix
    phi = np.zeros((6, 12))
    for i in range(6):
        for j in range(12):
            if i % 2 == 0:
                fun = np.sin
            else:
                fun = np.cos

            if i < 2:
                phi[i, j] = r1 * fun(j * 7 * np.pi / 6.)
            elif i >= 2 and i < 4:
                phi[i, j] = r2 * fun(j * 3 * np.pi / 2.)
            else:
                phi[i, j] = r3 * fun(j * 2 * np.pi / 3.)

    # Do the transform to tonnetz
    for i in range(N):
        for d in range(6):
            denom = float(C[i, :].sum())
            if denom == 0:
                T[i, d] = 0
            else:
                T[i, d] = 1 / denom * (phi[d, :] * C[i, :]).sum()

    return T


def most_frequent(x):
    """Returns the most frequent value in x."""
    return np.argmax(np.bincount(x))


def pick_peaks(nc, L=16, plot=False):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() / 3
    th = filters.median_filter(nc, size=L) + offset
    peaks = []
    for i in xrange(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    if plot:
        plt.plot(nc)
        plt.plot(th)
        for peak in peaks:
            plt.axvline(peak, color="m")
        plt.show()
    return peaks


def recurrence_matrix(data, k=None, width=1, metric='sqeuclidean', sym=False):
    '''
    Note: Copied from librosa

    Compute the binary recurrence matrix from a time-series.

    ``rec[i,j] == True`` <=> (``data[:,i]``, ``data[:,j]``) are
    k-nearest-neighbors and ``|i-j| >= width``

    :usage:
        >>> mfcc    = librosa.feature.mfcc(y=y, sr=sr)
        >>> R       = librosa.segment.recurrence_matrix(mfcc)

        >>> # Or fix the number of nearest neighbors to 5
        >>> R       = librosa.segment.recurrence_matrix(mfcc, k=5)

        >>> # Suppress neighbors within +- 7 samples
        >>> R       = librosa.segment.recurrence_matrix(mfcc, width=7)

        >>> # Use cosine similarity instead of Euclidean distance
        >>> R       = librosa.segment.recurrence_matrix(mfcc, metric='cosine')

        >>> # Require mutual nearest neighbors
        >>> R       = librosa.segment.recurrence_matrix(mfcc, sym=True)

    :parameters:
      - data : np.ndarray
          feature matrix (d-by-t)

      - k : int > 0 or None
          the number of nearest-neighbors for each sample

          Default: ``k = 2 * ceil(sqrt(t - 2 * width + 1))``,
          or ``k = 2`` if ``t <= 2 * width + 1``

      - width : int > 0
          only link neighbors ``(data[:, i], data[:, j])``
          if ``|i-j| >= width``

      - metric : str
          Distance metric to use for nearest-neighbor calculation.

          See ``scipy.spatial.distance.cdist()`` for details.

      - sym : bool
          set ``sym=True`` to only link mutual nearest-neighbors

    :returns:
      - rec : np.ndarray, shape=(t,t), dtype=bool
          Binary recurrence matrix
    '''

    t = data.shape[1]

    if k is None:
        if t > 2 * width + 1:
            k = 2 * np.ceil(np.sqrt(t - 2 * width + 1))
        else:
            k = 2

    k = int(k)

    def _band_infinite():
        '''Suppress the diagonal+- of a distance matrix'''

        band = np.empty((t, t))
        band.fill(np.inf)
        band[np.triu_indices_from(band, width)] = 0
        band[np.tril_indices_from(band, -width)] = 0

        return band

    # Build the distance matrix
    D = scipy.spatial.distance.cdist(data.T, data.T, metric=metric)

    # Max out the diagonal band
    D = D + _band_infinite()

    # build the recurrence plot
    rec = np.zeros((t, t), dtype=bool)

    # get the k nearest neighbors for each point
    for i in range(t):
        for j in np.argsort(D[i])[:k]:
            rec[i, j] = True

    # symmetrize
    if sym:
        rec = rec * rec.T

    return rec
