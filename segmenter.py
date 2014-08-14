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

# Local stuff
import cnmf as cnmf_S
import foote as foote_S
import fmc2d as fmc2d_S


#### Algorithm Parameters ####
# C-NMF
h = 8           # Size of median filter for features in C-NMF
R = 15          # Size of the median filter for the activation matrix C-NMF
rank = 4        # Rank of decomposition for the boundaries
# Foote
M = 2           # Median filter for the audio features (in beats)
Mg = 32         # Gaussian kernel size
L = 16          # Size of the median filter for the adaptive threshold
# 2D-FMC
N = 8          # Size of the fixed length segments (for 2D-FMC)


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
        est_bound_idxs = foote_S.foote_segmentation(F, M, Mg, L)
    else:
        est_bound_idxs = cnmf_S.cnmf_segmentation(F, rank=rank, R=R, h=h)

    # Compute the labels from all the boundaries
    logging.info("Estimating Segment Similarity (Labeling)...")
    all_est_bound_idxs = np.unique(np.concatenate(([0], est_bound_idxs,
                                                   [len(F)])))
    est_labels = fmc2d_S.compute_similarity(
        feats["hpcp"], all_est_bound_idxs, xmeans=True, N=N)

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
