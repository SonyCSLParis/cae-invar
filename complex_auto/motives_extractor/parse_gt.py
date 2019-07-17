#!/usr/bin/env python
"""
This script parses the ground truth annotations (csv) into the MIREX format.

Adapted on July 05, 2019
by Stefan Lattner

Sony CSL Paris, France
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import csv
import glob
import logging
import os
import time
import utils


def get_out_file(patterns, out_dir):
    """Given a set of patterns corresponding to a single musical piece and the
    output directory, get the output file path.

    Parameters
    ----------
    patterns: list of list of strings (files)
        Set of all the patterns with the occurrences of a given piece.
    out_dir: string (path)
        Path to the output directory.

    Returns
    -------
    out_file: string (path)
        Path to the output file.
    """
    assert len(patterns) > 0
    name_split = patterns[0][0].split("/")
    idx_offset = name_split.index("groundTruth") - 1
    return os.path.join(out_dir, name_split[idx_offset + 2] + "-" +
                        name_split[idx_offset + 3] + ".txt")


def parse_patterns(patterns, out_file):
    """Parses the set of patterns and saves the results into the output file.

    Parameters
    ----------
    patterns: list of list of strings (files)
        Set of all the patterns with the occurrences of a given piece.
    out_file: string (path)
        Path to the output file to save the set of patterns in the MIREX
        format.
    """
    out_str = ""
    pattern_n = 1
    for pattern in patterns:
        out_str += "pattern" + str(pattern_n) + "\n"
        occ_n = 1
        for occ_file in pattern:
            out_str += "occurrence" + str(occ_n) + "\n"
            with open(occ_file, "r") as f:
                file_reader = csv.reader(f)
                for fields in file_reader:
                    out_str += fields[0] + ", " + fields[1] + "\n"
            occ_n += 1
        pattern_n += 1

    # Save file
    with open(out_file, "w") as f:
        f.write(out_str)


def get_gt_patterns(annotators):
    """Obtains the set of files containing the patterns and its occcurrences
    given the annotator directories.

    Parameters
    ----------
    annotators: list of strings (files)
        List containing a set of paths to all the annotators of a given piece.

    Returns
    -------
    P: list of list of strings (files)
       Paths to all the patterns with all the occurrences for the current
       annotator. e.g. P = [[pat1_occ1, pat1_occ2],[pat2_occ1, ...],...]
    """
    P = []
    for annotator in annotators:
        # Get all the patterns from this annotator
        patterns = glob.glob(os.path.join(annotator, "*"))
        for pattern in patterns:
            if os.path.isdir(pattern):
                # Get all the occurrences for the current pattern
                occurrences = glob.glob(os.path.join(pattern, "occurrences",
                                                     "csv", "*.csv"))
                O = []
                [O.append(occurrence) for occurrence in occurrences]
                P.append(O)
    return P


def process(jku_dir, out_dir):
    """Main process to parse the ground truth csv files.

    Parameters
    ----------
    jku_dir: string
        Directory where the JKU Dataset is located.
    out_dir: string
        Directory in which to put the parsed files.
    """
    # Make sure the output dir exists
    utils.ensure_dir(out_dir)

    # Get all the music pieces in the ground truth
    pieces = glob.glob(os.path.join(jku_dir, "groundTruth", "*"))

    # Two types of patterns for each piece
    types = ["monophonic", "polyphonic"]

    # Main loop to retrieve all the patterns from the GT
    all_patterns = []
    for piece in pieces:
        logging.info("Parsing piece %s" % piece)
        for type in types:
            # Get all the annotators for the current piece
            annotators = glob.glob(os.path.join(piece, type,
                                                "repeatedPatterns", "*"))

            # Based on the readme.txt of JKU, these are the valid annotators
            # (thanks Colin! :-)
            if type == "polyphonic":
                valid_annotators = ['barlowAndMorgensternRevised',
                                    'bruhn',
                                    'schoenberg',
                                    'sectionalRepetitions',
                                    'tomCollins']
                for annotator in annotators:
                    if os.path.split(annotator)[1] not in valid_annotators:
                        annotators.remove(annotator)

            all_patterns.append(get_gt_patterns(annotators))

    # For the patterns of one given file, parse them into a single file
    for patterns in all_patterns:
        out_file = get_out_file(patterns, out_dir)
        parse_patterns(patterns, out_file)


def main():
    """Main function to parse the ground truth."""
    parser = argparse.ArgumentParser(description=
        "Parses the ground truth in csv format into the MIREX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("jku_dir",
                        action="store",
                        help="Input JKU dataset dir")
    parser.add_argument("out_dir",
                        action="store",
                        help="Output dir")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.jku_dir, args.out_dir)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
