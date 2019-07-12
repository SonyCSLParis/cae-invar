#!/bin/sh

python train.py $1 filelist_audio.txt config_cqt.ini --refresh-cache
python convert.py $1 jku_input_audio.txt config_cqt.ini --self-sim-matrix
python extract_motives.py $1 -csv jku_csv_files.txt -r 2
python extract_motives_eval.py $1

