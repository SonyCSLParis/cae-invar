MotivesExtractor
================

This script extracts the most repeated harmonic patterns from an audio file
sampled at 11025Hz. It is based on the following paper:

Nieto, O., Farbood, M., Identifying Polyphonic Patterns From Audio Recordings 
Using Music Segmentation Techniques. Proc. of the 15th International 
wSociety for Music Information Retrieval Conference (ISMIR). Taipei, Taiwan, 2014. 

Examples
--------

To run the extractor on a single file without CSV annotations (the results
will be printed on the screen):
    
    ./extractor.py wav_file

where `wav_file` is the path to a wav file sampled at 11025Hz with 16 bits.
You can find the wav files from the [JKU dataset](https://dl.dropbox.com/u/11997856/JKU/JKUPDD-Aug2013.zip)
in the folder `jku_input`.

To run the extractor on a single file with CSV annotations:
    
    ./extractor.py wav_file -c csv_file [-o output.txt]

where `csv_file` is the path to the corresponsing CSV file using the JKU format.
Examples of CSV files are included in the `jku_input` folder. The output is
saved using the MIREX format. If the output file is not provided, 
the results will be saved in `results.txt`.

To run the extractor on multiple files:

    ./run_extractor.py input_folder output_folder [-j 8]

It will analyze all the wav files and their corresponding CSV files from the
`input_folder` and write the results into the `output_folder`. The parameter 
`-j` indicates how many processors you want to run in parallel (default is 4).

To evaluate an entire folder:

    ./eval.py references_folder estimations_folder

where these folders contain the patterns using the MIREX format, and both
folders contains the exact same file names.

To obtain the ISMIR numbers:
    
    ./run_extractor.py jku_input/ results/ -th .33 -r 2
    ./eval.py parsed_jku/ results/

To plot the ISMIR paper plots:

    ./extractor.py jku_input/mazurka24-4-poly.wav -th .33 -r 2 -ismir

For more options, please type

    ./run_extractor.py -h


Requirements
------------

* [Python >=2.7](https://www.python.org/download/releases/2.7/)
* [audiolab](https://pypi.python.org/pypi/scikits.audiolab/)
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [joblib](https://pythonhosted.org/joblib/)
* [pylab](http://wiki.scipy.org/PyLab) (For plotting only)
* [pandas](http://pandas.pydata.org/) (For evaluating only)
* [mir_eval](https://github.com/craffel/mir_eval) (For evaluating only)

Author
------

[Oriol Nieto](https://files.nyu.edu/onc202/public/)

