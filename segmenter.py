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
import pylab as plt
import utils
from scipy.ndimage import filters

# Algorithm Parameters
M = 4       # Median filter for the audio features (in beats)
Mg = 32     # Gaussian kernel size
L = 16      # Size of the median filter for the adaptive threshold


def pick_peaks(nc, L=16, plot=False):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() / 3.
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


def process(audio_path, out_path, plot=False):
    """Main process to segment the audio file and save the results in the
        specified output."""

    # Get features and stack them
    feats = features.compute_all_features(audio_path)
    F = np.hstack((feats["hpcp"], feats["mfcc"], feats["tonnetz"]))
    F = utils.median_filter(F, M=M)

    # Self Similarity Matrix
    S = utils.compute_ssm(F)

    #plt.imshow(feats["mfcc"].T, interpolation="nearest", aspect="auto")
    #plt.imshow(feats["hpcp"].T, interpolation="nearest", aspect="auto")
    #plt.show()
    #plt.imshow(S, interpolation="nearest")
    #plt.show()

    # Compute gaussian kernel
    G = utils.compute_gaussian_krnl(Mg)

    # Compute the novelty curve
    nc = utils.compute_nc(S, G)

    # Find peaks in the novelty curve
    est_bound_idxs = pick_peaks(nc, L=L, plot=plot)

    # Get boundary times while adding first and last boundary
    est_bound_times = np.concatenate(([feats["beats"][0]],
                                      feats["beats"][est_bound_idxs],
                                      [feats["beats"][-1]]))

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
