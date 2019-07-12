#!/usr/bin/env python
"""
This script discovers polyphonic musical patterns from a mono 16-bit wav file
sampled at 44.1kHz. It also needs the BPMs of the audio track and the csv file
from which to read the MIDI pitches.

It is based on:

Nieto, O., Farbood, M., Identifying Polyphonic Musical Patterns From Audio
Recordings Using Music Segmentation Techniques. Proc. of the 15th International
Society for Music Information Retrieval Conference (ISMIR).
Taipei, Taiwan, 2014.

#############

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2013, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import logging
import numpy as np
import os
import time
from . import utils
from . import ismir
import matplotlib.pyplot as plt


def get_bpm(wav_file):
    """Gets the correct bpm based on the wav_file name. If the wav_file is not
    contained in the JKU dataset, raises error.

    Parameters
    ----------
    wav_file : str
        Path to the wav file to obtain its bpm.

    Returns
    -------
    bpm : int
        BPM of the piece, as described in the JKU dataset.
    """
    bpm_dict = {"wtc2f20-poly" : 84,
                "sonata01-3-poly" : 192,
                "mazurka24-4-poly" : 138,
                "silverswan-poly" : 54,
                "sonata04-2-poly" : 120
                }
    wav_file = os.path.basename(wav_file).split(".")[0]
    if wav_file not in bpm_dict.keys():
        raise Exception("%s not in the JKU dataset, you need to input a BPM" %
                        wav_file)
    return bpm_dict[wav_file]


def print_patterns(patterns, h):
    """Prints the patterns and the occurrences included in pattterns.

    Parameters
    ----------
    patterns : list
        Patterns list with all of its occurrences.
    h : float
        Hop size.
    """
    logging.info("Printing Extracted Motives (all times are in seconds):")
    for i, p in enumerate(patterns):
        print("Pattern %d:" % (i + 1))
        for j, occ in enumerate(p):
            # Get start and end times
            start = occ[2]
            end = occ[3]
            print("\tOccurrence %d: (%.2f, %.2f)" % (j + 1, start, end))


def occurrence_to_csv(start, end, midi_score):
    """Given an occurrence, return the csv formatted one into a
        list (onset,midi).

    Parameters
    ----------
    start : float
        Start index of the occurrence.
    end : float
        End index of the occurrence.
    midi_score : list
        The score of the piece (read form CSV format).

    Returns
    -------
    occ : list
        Occurrence in the csv format list(onset, midi).
    """
    occ = []
    start = round(start)
    end = round(end)
    h = 0.125  # Resolution of csv files
    for i in np.arange(start, end, h):
        idxs = np.argwhere(midi_score[:, utils.CSV_ONTIME] == i)
        # Get all available staves
        if len(idxs) > 0:
            for idx in idxs:
                onset = midi_score[idx, utils.CSV_ONTIME][0]
                midi = midi_score[idx, utils.CSV_MIDI][0]
                occ.append([onset, midi, idx])
    return occ


def patterns_to_csv(patterns, midi_score, h, offset, raw = False):
    """Formats the patterns into the csv format.

    Parameters
    ----------
    pattersn : list
        List of patterns with its occurrences.
    midi_score : list
        The score of the piece (read from CSV).
    h : float
        Hop size of the ssm.

    Returns
    -------
    csv_patterns : list
        List of the patterns in the csv format to be analyzed by MIREX.
    """
    csv_patterns = []
    for p in patterns:
        new_p = []
        for occ in p:
            start = occ[2] * h + offset
            end = occ[3] * h + offset
            csv_occ = occurrence_to_csv(start, end, midi_score)
            if raw:
                csv_occ = onsets_to_raw(csv_occ, h, offset, 0)
            if csv_occ != []:
                new_p.append(csv_occ)
        if new_p != [] and len(new_p) >= 2:
            csv_patterns.append(new_p)

    return csv_patterns


def onsets_to_raw(csv_occ, h, offset, correct):
    occ = []
    for onset, midi, idx in csv_occ:
        onraw = int(1. * (onset + 1 - offset) / h) - correct
        occ.append([onraw, midi, idx])
    
    return occ


def obtain_patterns(segments, max_diff):
    """Given a set of segments, find its occurrences and thus obtain the
    possible patterns.

    Parameters
    ----------
    segments : list
        List of the repetitions found in the self-similarity matrix.
    max_diff : float
        Maximum difference to decide whether we found a segment or not.

    Returns
    -------
    patters : list
        List of patterns found.
    """
    patterns = []
    N = len(segments)

    # Initially, all patterns must be checked
    checked_patterns = np.zeros(N)

    for i in range(N):
        if checked_patterns[i]:
            continue

        # Store new pattern
        new_p = []
        s = segments[i]
        # Add diagonal occurrence
        new_p.append([s[0], s[1], s[0], s[1]])
        # Add repetition
        new_p.append(s)

        checked_patterns[i] = 1

        # Find occurrences
        for j in range(N):
            if checked_patterns[j]:
                continue
            ss = segments[j]
            if (s[0] + max_diff >= ss[0] and s[0] - max_diff <= ss[0]) and \
                    (s[1] + max_diff >= ss[1] and s[1] - max_diff <= ss[1]):
                new_p.append(ss)
                checked_patterns[j] = 1
        patterns.append(new_p)

    return patterns


def compute_ssm(wav_file, h, ssm_read_pk, is_ismir=False, tonnetz=False):
    """Computes the self similarity matrix from an audio file.

    Parameters
    ----------
    wav_file: str
        Path to the wav file to be read.
    h : float
        Hop size.
    ssm_read_pk : bool
        Whether to read the ssm from a pickle file or not (note: this function
        utomatically saves the ssm in a pickle file).
    is_ismir : bool
        Produce the plots that appear on the ISMIR paper.
    tonnetz : bool
        Compute tonnetz instead of Chroma features.

    Returns
    -------
    X : np.array((N, N))
        Self-similarity matrix
    """
    if not ssm_read_pk:
        # Read WAV file
        logging.info("Reading the WAV file...")
        C = utils.compute_audio_chromagram(wav_file, h)
        C = utils.median_filter(C, L=9)

        if is_ismir:
            ismir.plot_chroma(C)

        # Compute Tonnetz if needed
        F = C
        if tonnetz:
            F = utils.chroma_to_tonnetz(C)

        # Compute the self similarity matrix
        logging.info("Computing key-invariant self-similarity matrix...")
        X = utils.compute_key_inv_ssm(F, h)

        #plt.imshow(X, interpolation="nearest", aspect="auto")
        #plt.show()

        utils.write_cPickle(wav_file + "-audio-ssm.pk", X)
    else:
        X = utils.read_cPickle(wav_file + "-audio-ssm.pk")

    if is_ismir:
        #X = X**2.5
        ismir.plot_ssm(X)
        ismir.plot_score_examples(X)

    return X


def save_hist(values, title, filename):
    '''
    Save a histogram of values titled with title in file filename.
    '''
    plt.clf()
    plt.hist(values.flatten(), bins=50)
    plt.xlabel('Value')
    plt.ylabel('Amount')
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_hist(values, title, filename):
    return save_hist(values, title, filename)


def prepro(X):
    X = X - np.median(X)
    return X


def process(wav_file, outfile, domain, csv_file=None, bpm=None, tol=0.1,
            ssm_read_pk=False, read_pk=False, rho=2, is_ismir=False,
            tonnetz=False, sonify=False):
    """Main process to find the patterns in a polyphonic audio file.

    Parameters
    ----------
    wav_file : str
        Path to the wav file to be analyzed.
    csv_file : str
        Path to the csv containing the midi_score of the input audio file
        (needed to produce a result that can be read for JKU dataset).
    outfile : str
        Path to file to save the results.
    bpm : int
        Beats per minute of the piece. If None, bpms are read from the JKU.
    tol : float
        Tolerance to find the segments in the SSM.
    ssm_read_pk : bool
        Whether to read the SSM from a pickle file.
    read_pk : bool
        Whether to read the segments from a pickle file.
    rho : int
        Positive integer to compute the score of the segments.
    is_ismir : bool
        Produce the plots that appear on the ISMIR paper.
    tonnetz : bool
        Whether to use Tonnetz or Chromas.
    sonify : bool
        Whether to sonify the patterns or not.
    """

    # Algorithm parameters
    min_notes = 16
    max_diff_notes = 5

    # to process
    if wav_file.endswith("wav"):
        # Get the correct bpm if needed
        if bpm is None:
            bpm = get_bpm(wav_file)
        
        h = bpm / 60. / 8.  # Hop size /8 works better than /4, but it takes longer
        # Obtain the Self Similarity Matrix
        X = compute_ssm(wav_file, h, ssm_read_pk, is_ismir, tonnetz)
    elif wav_file.endswith("npy"):
        X = np.load(wav_file)
        X = prepro(X)
        if domain == "symbolic":
            h = .25 # 2. # for symbolic (16th notes)
        else:
            if bpm is None:
                bpm = get_bpm(wav_file)
            h = 0.0886 * bpm / 60

    offset = 0
    # Read CSV file
    if csv_file is not None:
        logging.info("Reading the CSV file for MIDI pitches...")
        midi_score = utils.read_csv(csv_file)
        # Don't use offset, as these were already encoded in the MIDI files
        if domain == "audio":
            offset = utils.get_offset(midi_score)
        else:
            # Compensate for two inserted empty beats
            offset = -2. + utils.get_offset(midi_score)

    patterns = []
    csv_patterns = []
    while patterns == [] or csv_patterns == []:
        # Find the segments inside the self similarity matrix
        logging.info("Finding segments in the self-similarity matrix...")
        max_diff = int(max_diff_notes / float(h))
        min_dur = int(np.ceil(min_notes / float(h)))
        #print min_dur, min_notes, h, max_diff
        if not read_pk:
            segments = []
            while segments == []:
                logging.info(("{0}: \ttrying tolerance %.2f" % tol).format(wav_file))
                segments = np.asarray(utils.find_segments(X, min_dur, th=tol, rho=rho))
                tol -= 0.001
            #utils.write_cPickle(wav_file + "-audio.pk", segments)
        else:
            segments = utils.read_cPickle(wav_file + "-audio.pk")

        # Obtain the patterns from the segments and split them if needed
        logging.info("Obtaining the patterns from the segments...")
        patterns = obtain_patterns(segments, max_diff)

        # Get the csv patterns if they exist
        if csv_file is not None:
            try:
                csv_patterns = patterns_to_csv(patterns, midi_score, h, offset)
                raw_patterns = patterns_to_csv(patterns, midi_score, h, offset,
                                               raw = True)
            except Exception as e:
                print(e)
        else:
            csv_patterns = [0]
            raw_patterns = [0]

    # Sonify patterns if needed
    if sonify:
        logging.info("Sonifying Patterns...")

        utils.sonify_patterns(wav_file, patterns, h)

    # Formatting csv patterns and save results
    if csv_file is not None:
        try:
            logging.info("Writting results to %s" % outfile)
            utils.save_results(csv_patterns, outfile=outfile)
            utils.save_results(raw_patterns, outfile=outfile + "_")
        except Exception as e:
            print(e)

    utils.save_results_raw(patterns, outfile=outfile + "raw")

    if is_ismir:
        ismir.plot_segments(X, segments)

    # Alright, we're done :D
    logging.info("Algorithm finished.")


def main():
    """Main function to run the audio polyphonic version of patterns
        finding."""
    parser = argparse.ArgumentParser(
        description="Discovers the audio polyphonic motives given a WAV file"
        " and a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_file", action="store", help="Input WAV file")
    parser.add_argument("-c", dest="csv_file", action="store", default=None,
                        help="Input CSV file (to read MIDI notes for output)")
    parser.add_argument("-b", dest="bpm", action="store", type=float,
                        default=None, help="Beats Per Minute of the wave file")
    parser.add_argument("-o", action="store", default="results.txt",
                        dest="output", help="Output results")
    parser.add_argument("-pk", action="store_true", default=False,
                        dest="read_pk", help="Read Pickle File")
    parser.add_argument("-spk", action="store_true", default=False,
                        dest="ssm_read_pk", help="Read SSM Pickle File")
    parser.add_argument("-th", action="store", default=0.01, type=float,
                        dest="tol", help="Tolerance level, from 0 to 1")
    parser.add_argument("-r", action="store", default=2, type=int,
                        dest="rho", help="Positive integer number for "
                        "calculating the score")
    parser.add_argument("-ismir", action="store_true", default=False,
                        dest="is_ismir", help="Produce the plots that appear "
                        "on the ISMIR paper.")
    parser.add_argument("-t", action="store_true", default=False,
                        dest="tonnetz", help="Whether to use Tonnetz or not.")
    parser.add_argument("-s", action="store_true", default=False,
                        dest="sonify", help="Sonify the patterns")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.wav_file, args.output, csv_file=args.csv_file, bpm=args.bpm,
            tol=args.tol, read_pk=args.read_pk, ssm_read_pk=args.ssm_read_pk,
            rho=args.rho, is_ismir=args.is_ismir, tonnetz=args.tonnetz,
            sonify=args.sonify)

    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == "__main__":
    main()
