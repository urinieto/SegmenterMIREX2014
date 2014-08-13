#!/usr/bin/env python

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import logging
import numpy as np
import time
import features
import pymf
import scipy.cluster.vq as vq

import utils

# Local stuff
import utils_2dfmc as utils2d
from xmeans import XMeans


#### Algorithm Parameters ####
# C-NMF
h = 8           # Size of median filter for features in C-NMF
R = 15          # Size of the median filter for the activation matrix C-NMF
# Foote
M = 2           # Median filter for the audio features (in beats)
Mg = 32         # Gaussian kernel size
L = 16          # Size of the median filter for the adaptive threshold
# 2D-FMC
MIN_LEN = 4     # Minimum lenght for the segments (for 2D-FMC)
N = 16          # Size of the fixed length segments (for 2D-FMC)


def write_results(out_path, bound_times, labels):
    """Writes the results into the output file."""
    # Sanity check
    assert len(bound_times) - 1 == len(labels)

    logging.info("Writing results in %s..." % out_path)

    bound_inters = zip(bound_times[:-1], bound_times[1:])
    out_str = ""
    for (start, end), label in zip(bound_inters, labels):
        out_str += str(start) + "\t" + str(end) + "\t" + str(label) + "\n"
    with open(out_path, "w") as f:
        f.write(out_str)


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


def compute_similarity(PCP, bound_idxs, xmeans=False, k=5):
    """Main function to compute the segment similarity of file file_struct."""

    # Get PCP segments
    pcp_segments = get_pcp_segments(PCP, bound_idxs)

    # Get the 2d-FMCs segments
    fmcs = pcp_segments_to_2dfmc_fixed(pcp_segments, N=32)
    if fmcs == [] or fmcs is None:
        return np.arange(len(bound_idxs) - 1)

    # Compute the labels using kmeans
    if xmeans:
        xm = XMeans(fmcs, plot=False)
        k = xm.estimate_K_knee(th=0.01, maxK=8)
    est_labels = compute_labels_kmeans(fmcs, k=k)

    # Plot results
    #plot_pcp_wgt(PCP, bound_idxs)

    return est_labels


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


def cnmf_segmentation(X, rank, R, niter=300):
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


def foote_segmentation(F, plot=False):
    """Computes the Foote segmentator."""
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


#def read_ref_bounds(audio_path, beats):
    #"""Reads the boundaries based on the audio path. Warning: this is a hack"""
    #ref_file = os.path.join(
        #"beatlesISO", os.path.basename(audio_path).replace(".wav", ".lab"))
    #ref_inters, labels = mir_eval.io.load_labeled_intervals(
        #ref_file, delimiter="\t")
    #ref_times = np.concatenate((ref_inters[:, 0], [ref_inters[-1][-1]]))

    #ref_idxs = []
    #for ref_time in ref_times:
        #k = 0
        #for beat in beats:
            #if ref_time <= beat:
                #break
            #k += 1
        #ref_idxs.append(k)

    #return ref_idxs


def process(audio_path, out_path, foote=False, plot=False):
    """Main process to segment the audio file and save the results in the
        specified output."""

    # Get features and stack them
    feats = features.compute_all_features(audio_path)
    F = np.hstack((feats["hpcp"], feats["mfcc"]))

    # Estimate bounds_idx
    logging.info("Estimating Boundaries...")
    if foote:
        est_bound_idxs = foote_segmentation(F)
    else:
        est_bound_idxs = cnmf_segmentation(F, rank=4, R=R)

    # Compute the labels from all the boundaries
    logging.info("Estimating Segment Similarity (Labeling)...")
    all_est_bound_idxs = np.unique(np.concatenate(([0], est_bound_idxs,
                                                   [len(F)])))
    est_labels = compute_similarity(feats["hpcp"], all_est_bound_idxs,
                                    xmeans=True)

    # Get boundary times while adding first and last boundary
    est_bound_times = np.concatenate(([feats["beats"][0]],
                                      feats["beats"][est_bound_idxs],
                                      [feats["beats"][-1]]))
    est_bound_times = np.unique(est_bound_times)

    # Write results
    write_results(out_path, est_bound_times, est_labels)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Segments the given audio file sampled at 44100, 16 bits, mono.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_path",
                        action="store",
                        help="Path to the input audio file")
    parser.add_argument("-f",
                        action="store_true",
                        dest="foote",
                        help="Use the Foote method for segmentation",
                        default=False)
    parser.add_argument("-o",
                        action="store",
                        dest="out_path",
                        help="Path to the output results file",
                        default="output.lab")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.audio_path, args.out_path, foote=args.foote)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
