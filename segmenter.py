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
import os
import pymf
import pylab as plt

import utils
import mir_eval


# Algorithm Parameters
h = 8      # Size of median filter for features in C-NMF
R = 15      # Size of the median filter for the activation matrix C-NMF


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


def read_ref_bounds(audio_path, beats):
    """Reads the boundaries based on the audio path. Warning: this is a hack"""
    ref_file = os.path.join(
        "beatlesISO", os.path.basename(audio_path).replace(".wav", ".lab"))
    ref_inters, labels = mir_eval.io.load_labeled_intervals(
        ref_file, delimiter="\t")
    ref_times = np.concatenate((ref_inters[:, 0], [ref_inters[-1][-1]]))

    ref_idxs = []
    for ref_time in ref_times:
        k = 0
        for beat in beats:
            if ref_time <= beat:
                break
            k += 1
        ref_idxs.append(k)

    return ref_idxs


def process(audio_path, out_path, plot=False):
    """Main process to segment the audio file and save the results in the
        specified output."""

    # Get features and stack them
    feats = features.compute_all_features(audio_path)
    F = np.hstack((feats["hpcp"], feats["mfcc"], feats["tonnetz"]))
    F = np.hstack((feats["tonnetz"], feats["mfcc"]))
    F = np.hstack((feats["hpcp"], feats["mfcc"]))

    #ref_bound_idxs = read_ref_bounds(audio_path, feats["beats"])
    #S = utils.compute_ssm(F)
    #plt.imshow(S, interpolation="nearest", aspect="auto")
    #for bound in ref_bound_idxs:
        #plt.axhline(bound, color="g", linewidth=2)
        #plt.axvline(bound, color="g", linewidth=2)
    #plt.show()

    ## Estimate bounds using C-NMF method
    est_bound_idxs = cnmf_segmentation(F, rank=4, R=R)

    #for bound in est_bound_idxs:
        #plt.axvline(bound, color="m", ymin=0.5)
    #for bound in ref_bound_idxs:
        #plt.axvline(bound, color="g", alpha=0.6, linewidth=2)
    #plt.show()

    # Get boundary times while adding first and last boundary
    est_bound_times = np.concatenate(([feats["beats"][0]],
                                      feats["beats"][est_bound_idxs],
                                      [feats["beats"][-1]]))
    est_bound_times = np.unique(est_bound_times)

    # Write results
    print est_bound_times
    est_labels = np.ones(len(est_bound_times) - 1, dtype=int)
    write_results(out_path, est_bound_times, est_labels)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Segments the given audio file sampled at 44100, 16 bits, mono.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_path",
                        action="store",
                        help="Path to the input audio file")
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
    process(args.audio_path, args.out_path)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
