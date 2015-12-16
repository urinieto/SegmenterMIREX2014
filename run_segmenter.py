#!/usr/bin/env python
'''Runs the segmenter across an entire folder containing wav files.
'''

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import glob
import os
import argparse
import time
import logging
from joblib import Parallel, delayed

import segmenter as S
import utils


def process_track(audio_file, out_path, bounds_type, labels_type, seed):
    """Processes one track, for paralelization purposes."""
    logging.info("Segmenting %s" % audio_file)

    out_file = os.path.join(
        out_path, os.path.basename(audio_file).replace(".wav", ".lab"))
    S.process(audio_file, out_file, bounds_type=bounds_type,
              labels_type=labels_type, seed=seed)


def process(in_path, out_path, bounds_type="cnmf", labels_type="cnmf",
            n_jobs=4, seed=None):
    """Main process."""

    # Make sure output folder exists
    utils.ensure_dir(out_path)

    # Get relevant files
    audio_files = glob.glob(os.path.join(in_path, "*.wav"))

    # Call in parallel
    Parallel(n_jobs=n_jobs)(delayed(process_track)(
        audio_file, out_path, bounds_type, labels_type, seed)
        for audio_file in audio_files)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the segmenter in all the wav files in the specified folder "
        "and saves the results in the output folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Folder to Wav Files")
    parser.add_argument("-r",
                        dest="random_seed",
                        action="store",
                        type=int,
                        default=None,
                        help="Random Seed")
    parser.add_argument("-o",
                        action="store",
                        dest="out_path",
                        default="estimations",
                        help="Output folder to store the lab files")
    parser.add_argument("-b",
                        action="store",
                        dest="bounds_type",
                        help="Which algortihm to use to extract the "
                        "boundaries",
                        default="cnmf",
                        choices=["cnmf", "foote", "sf"])
    parser.add_argument("-s",
                        action="store",
                        dest="labels_type",
                        help="Which algortihm to use to extract the "
                        "segment similarity (labeling)",
                        default="cnmf",
                        choices=["cnmf", "2dfmc"])
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        default=4,
                        type=int,
                        help="The number of threads to use")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.out_path, bounds_type=args.bounds_type,
            labels_type=args.labels_type, n_jobs=args.n_jobs, seed=args.random_seed)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
