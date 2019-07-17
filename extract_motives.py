#!/usr/bin/env python
"""
Script to run the extractor on an entire folder.

To run the script:
./run_extractor.py jku_input outputdir

This should procude the output reported in the ISMIR paper.

For more options:
./run_extractor.py -h

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

Adapted by Stefan Lattner

On July 05, 2019

Sony CSL Paris, France

"""
from complex_auto.motives_extractor import utils, extractor
from complex_auto.util import read_file

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2013, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import logging
import os
import time
from joblib import Parallel, delayed

#import extractor as EX
#import utils


def process_piece(fn_ss_matrix, outdir, tol, domain, ssm_read_pk, read_pk,
                  tonnetz, rho, csv_file=None):
    
    assert domain in ("audio", "symbolic")
    f_base = os.path.basename(fn_ss_matrix)
    base_name = os.path.join(outdir, f_base.split(".")[0] + ".seg")

    logging.info("Running algorithm for %s" % f_base)
    #out = os.path.join(outdir, out) + ".txt"
    #print "./extractor.py %s -c %s -o %s -th %f" % (wav, csv, out, tol)
    print("Processing file {0}...".format(fn_ss_matrix))
    extractor.process(fn_ss_matrix, base_name, domain, csv_file=csv_file,
                      tol=tol, ssm_read_pk=ssm_read_pk,
                      read_pk=read_pk, tonnetz=tonnetz, rho=rho)


def process_audio_poly(files, outdir, domain, tol, ssm_read_pk, read_pk, rho,
                       n_jobs=5, csv_files=None,
                       tonnetz=False):
    utils.ensure_dir(outdir)
    if csv_files is None:
        csv_files = [None] * len(files)

    Parallel(n_jobs=n_jobs)(delayed(process_piece)(
        wav, outdir, tol, domain, ssm_read_pk, read_pk, tonnetz, rho, csv)
        for wav, csv in zip(files, csv_files))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description=
        "Runs the algorithm of pattern discovery on the polyphonic csv files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_keyword', type=str, default="experiment1",
                        help='keyword used for input path')
    #parser.add_argument("outdir", action="store", help="Output Folder")
    parser.add_argument("-pk", action="store_true", default=False,
                        dest="read_pk", help="Read Pickle File")
    parser.add_argument("-th", action="store", default=0.0105, type=float,
                        dest="tol", help="threshold to consider a repetition")
    parser.add_argument("-r", action="store", default=1, type=int, dest="rho",
                        help="Positive integer number for calculating the "
                        "score")
    parser.add_argument("-dom", action="store", default="audio", type=str,
                        dest="dom", help="Domain (symbolic / audio)")
    parser.add_argument("-spk", action="store_true", default=False,
                        dest="ssm_read_pk", help="Read SSM Pickle File")
    parser.add_argument("-j", action="store", default=10, type=int,
                        dest="n_jobs",
                        help="Number of processors to use to divide the task.")
    parser.add_argument("-t", action="store_true", default=False,
                        dest="tonnetz", help="Whether to use Tonnetz or not.")
    parser.add_argument("-csv", action="store", default=None, type=str,
                        dest="csv_files", help="filelist with csv files to "
                                               "determine offsets.")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    out_dir = os.path.join("output", args.run_keyword)
    input_filelist = os.path.join(out_dir, "ss_matrices_filelist.txt")
    assert os.path.exists(input_filelist), \
        "A file named 'ss_matrices_filelist.txt' listing *.npy files with " \
        "stored self-similarity matrices has to exist in folder " \
        f"{out_dir}. Run 'to_self_sim_matrix.py' before " \
        "'extract_motives.py', or check if run_keyword " \
        f"'{args.run_keyword}' points to the intended folder."

    inputs = read_file(input_filelist)

    if args.csv_files is not None:
        csv_files = read_file(args.csv_files)
    else:
        csv_files = None

    # Run the algorithm
    process_audio_poly(inputs, out_dir, tol=args.tol, rho=args.rho,
                       domain=args.dom, csv_files=csv_files,
                       ssm_read_pk=args.ssm_read_pk, read_pk=args.read_pk,
                       n_jobs=args.n_jobs, tonnetz=args.tonnetz)

    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == "__main__":
    main()
