#!/usr/bin/env python


__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import essentia
import essentia.standard as ES
import json
import logging
import os
import numpy as np
import utils

# Setup main params
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 1024
#WINDOW_TYPE = "blackmanharris74"
WINDOW_TYPE = "hann"
OUTPUT_FEATURES = "features"


class STFTFeature:
    """Class to easily compute the features that require a frame based
        spectrum process (or STFT)."""
    def __init__(self, frame_size, hop_size, window_type, feature,
            beats, sample_rate):
        """STFTFeature constructor."""
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window_type = window_type
        self.w = ES.Windowing(type=window_type)
        self.spectrum = ES.Spectrum()
        self.feature = feature  # Essentia feature object
        self.beats = beats
        self.sample_rate = sample_rate

    def compute_features(self, audio):
        """Computes the specified Essentia features from the audio array."""
        features = []

        for frame in ES.FrameGenerator(audio,
                frameSize=self.frame_size, hopSize=self.hop_size):
            if self.feature.name() == "MFCC":
                bands, coeffs = self.feature(self.spectrum(self.w(frame)))
            elif self.feature.name() == "HPCP":
                spectral_peaks = ES.SpectralPeaks()
                freqs, mags = spectral_peaks(self.spectrum(self.w(frame)))
                coeffs = self.feature(freqs, mags)
            features.append(coeffs)

        # Convert to Essentia Numpy array
        features = essentia.array(features)

        if self.beats != []:
            framerate = self.sample_rate / float(self.hop_size)
            tframes = np.arange(features.shape[0]) / float(framerate)
            features = utils.resample_mx(features.T, tframes, self.beats).T

        return features


def compute_beats(audio):
    """Computes the beats using Essentia."""
    logging.info("Computing Beats...")
    ticks, conf = ES.BeatTrackerMultiFeature()(audio)
    return ticks, conf


def compute_beatsync_features(ticks, audio):
    """Computes the HPCP and MFCC beat-synchronous features given a set
        of beats (ticks)."""
    MFCC = STFTFeature(FRAME_SIZE, HOP_SIZE, WINDOW_TYPE,
                       ES.MFCC(numberCoefficients=14), ticks, SAMPLE_RATE)
    HPCP = STFTFeature(FRAME_SIZE, HOP_SIZE, WINDOW_TYPE, ES.HPCP(),
                       ticks, SAMPLE_RATE)
    logging.info("Computing Beat-synchronous MFCCs...")
    mfcc = MFCC.compute_features(audio)
    logging.info("Computing Beat-synchronous HPCPs...")
    hpcp = HPCP.compute_features(audio)
    #plt.imshow(hpcp.T, interpolation="nearest", aspect="auto"); plt.show()
    tonnetz = utils.chroma_to_tonnetz(hpcp)

    return mfcc.tolist(), hpcp.tolist(), tonnetz.tolist()


def list_to_array(features):
    """Convets the features to numpy arrays."""
    for key in features.keys():
        features[key] = np.asarray(features[key])
    return features


def compute_all_features(audio_file, audio_beats=False):
    """Computes all the features for a specific audio file and its respective
        human annotations.

    Returns
    -------
    features : dict
        Dictionary with the following features:
            mfcc : np.array
                Mel Frequency Cepstral Coefficients representation
            hpcp : np.array
                Harmonic Pitch Class Profiles
            tonnets : np.array
                Tonal Centroids (or Tonnetz)
    """

    # Makes sure the output features folder exists
    utils.ensure_dir(OUTPUT_FEATURES)
    features_file = os.path.join(OUTPUT_FEATURES,
                                 os.path.basename(audio_file) + ".json")

    # If already precomputed, read and return
    if os.path.exists(features_file):
        with open(features_file, "r") as f:
            features = json.load(f)
        return list_to_array(features)

    # Load Audio
    logging.info("Loading audio file %s" % os.path.basename(audio_file))
    audio = ES.MonoLoader(filename=audio_file, sampleRate=SAMPLE_RATE)()

    # Estimate Beats
    ticks, conf = compute_beats(audio)
    ticks = np.concatenate(([0], ticks))  # Add first time
    ticks = essentia.array(np.unique(ticks))

    ticks, conf = ES.BeatTrackerMultiFeature()(audio)

    # Compute Beat-sync features
    features = {}
    features["mfcc"], features["hpcp"], features["tonnetz"] = \
        compute_beatsync_features(ticks, audio)

    # Save output as audio file
    if audio_beats:
        logging.info("Saving Beats as an audio file")
        marker = ES.AudioOnsetsMarker(onsets=ticks, type='beep',
                                      sampleRate=SAMPLE_RATE)
        marked_audio = marker(audio)
        ES.MonoWriter(filename='beats.wav',
                      sampleRate=SAMPLE_RATE)(marked_audio)

    # Save features
    with open(features_file, "w") as f:
        json.dump(features, f)

    return list_to_array(features)
