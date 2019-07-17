#!/usr/bin/env python
"""
Script to run the evaluations.

To run the script:
    ./eval.py annotations/ estimations/

Both directories must have the same file names.

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

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2013, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import glob
import logging
import mir_eval
import os
import time
import pandas as pd


def process(refdir, estdir):
    references = glob.glob(os.path.join(refdir, "*.txt"))
    estimations = glob.glob(os.path.join(estdir, "*.seg"))
    references = sorted(references)
    estimations = sorted(estimations)
    results = pd.DataFrame()
    for ref, est in zip(references, estimations):
        assert os.path.splitext(os.path.basename(ref))[0] == \
               os.path.splitext(os.path.basename(est))[0]
        logging.info("Evaluating file: %s", est)
        ref_pat = mir_eval.io.load_patterns(ref)
        est_pat = mir_eval.io.load_patterns(est)

        res = {}
        res["Est_F"], res["Est_P"], res["Est_R"] = \
            mir_eval.pattern.establishment_FPR(ref_pat, est_pat)
        res["Occ.75_F"], res["Occ.75_P"], res["Occ.75_R"] = \
            mir_eval.pattern.occurrence_FPR(ref_pat, est_pat, thres=.75)
        res["ThreeLayer_F"], res["ThreeLayer_P"], res["ThreeLayer_R"] = \
            mir_eval.pattern.three_layer_FPR(ref_pat, est_pat)
        res["Occ.5_F"], res["Occ.5_P"], res["Occ.5_R"] = \
            mir_eval.pattern.occurrence_FPR(ref_pat, est_pat, thres=.5)
        res["Std_F"], res["Std_P"], res["Std_R"] = \
            mir_eval.pattern.standard_FPR(ref_pat, est_pat)
        results = results.append(res, ignore_index=True)

    logging.info("Results per piece:")
    print(results)
    logging.info("Average Results:")
    print(results.mean())
    
    with open(os.path.join(estdir, "results.txt"), "a") as f:
        f.write(str(results))
        f.write(str(results.mean()))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description=
        "Evals the algorithm using mir_eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-refdir", action="store",
                        default="complex_auto/motives_extractor/groundtruth",
                        help="Directory with the annotations")
    parser.add_argument("run_keyword", action="store",
                        help="Directory with the estimations")

    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    est_dir = os.path.join("output", args.run_keyword)

    # Run the algorithm
    process(args.refdir, est_dir)

    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == "__main__":
    main()
