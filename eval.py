#!/usr/bin/env python
"""Evaluates the segmenter using mir_eval
"""

import argparse
import glob
import logging
import os
import sys
import time
import pandas as pd
from joblib import Parallel, delayed

import mir_eval


def eval_track(ref_file, est_file, i=-1, N=-1):
    """Evaluates a single file."""
    # Progress bar
    sys.stdout.write("\r%.1f %%" % (100 * i / float(N)))
    sys.stdout.flush()

    assert os.path.basename(ref_file) == os.path.basename(est_file)

    # Read intervals
    delimiter = "\t"
    ref_inters, ref_labels = mir_eval.io.load_labeled_intervals(
        ref_file, delimiter=delimiter)
    est_inters, est_labels = mir_eval.io.load_labeled_intervals(
        est_file, delimiter=delimiter)

    # Remove the first and last stupid boundary from the ISO Beatles if needed
    if ref_inters[0][0] >= ref_inters[0][1]:
        ref_inters = ref_inters[1:]
    if ref_inters[-1][0] >= ref_inters[-1][1]:
        ref_inters = ref_inters[:-1]

    # Compute evaluations and store them in a dictionary
    evals = {}

    # Boundary Eval
    evals["Hit05_P"], evals["Hit05_R"], evals["Hit05_F"] = \
        mir_eval.boundary.detection(ref_inters, est_inters, window=0.5)
    evals["Hit3_P"], evals["Hit3_R"], evals["Hit3_F"] = \
        mir_eval.boundary.detection(ref_inters, est_inters, window=3)
    evals["Hit05t_P"], evals["Hit05t_R"], evals["Hit05t_F"] = \
        mir_eval.boundary.detection(ref_inters, est_inters, window=0.5,
                                    trim=True)
    evals["Hit3t_P"], evals["Hit3t_R"], evals["Hit3t_F"] = \
        mir_eval.boundary.detection(ref_inters, est_inters, window=3,
                                    trim=True)

    # Similarity Eval
    ref_inters, ref_labels = mir_eval.util.adjust_intervals(
        ref_inters, ref_labels, t_min=0)
    est_inters, est_labels = mir_eval.util.adjust_intervals(
        est_inters, est_labels, t_min=0, t_max=ref_inters.max())
    evals["Pwf_P"], evals["Pwf_R"], evals["Pwf_F"] = \
        mir_eval.structure.pairwise(ref_inters, ref_labels,
                                    est_inters, est_labels)
    evals["NCE_o"], evals["NCE_u"], evals["NCE_F"] = \
        mir_eval.structure.nce(ref_inters, ref_labels,
                               est_inters, est_labels)
    return evals


def process(ref_path, est_path, n_jobs=4):
    """Evaluates the segmentator using mir_eval."""
    ref_files = glob.glob(os.path.join(ref_path, "*.lab"))
    est_files = glob.glob(os.path.join(est_path, "*.lab"))
    logging.info("Evaluating %d files..." % len(ref_files))

    # Data frame to store all the results
    results = pd.DataFrame()

    # Compute evaluations in parallel
    evals = Parallel(n_jobs=n_jobs)(delayed(eval_track)(
        ref_file, est_file, i, len(ref_files))
        for i, (ref_file, est_file) in enumerate(zip(ref_files, est_files)))

    # Collect results
    for e in evals:
        if e != []:
            results = results.append(e, ignore_index=True)

    # Print out
    sys.stdout.write("\r")
    sys.stdout.flush()
    logging.info("Results:")
    print results.mean()


def main():
    """Main routine to start the process."""
    parser = argparse.ArgumentParser(description=
        "Evaluates the segmentator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ref_path",
                        action="store",
                        help="Path to the reference annotations folder")
    parser.add_argument("est_path",
                        action="store",
                        help="Path to the estimate results folder")
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        help="Number of parallel jobs to launch",
                        type=int,
                        default=4)
    args = parser.parse_args()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Start the process and time it
    start_time = time.time()
    process(args.ref_path, args.est_path, n_jobs=args.n_jobs)
    print "Done! Took %.1f seconds" % (time.time() - start_time)

if __name__ == "__main__":
    main()
