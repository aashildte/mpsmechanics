#!/bin/sh

# run from root folder

f=test_input.csv

# main scripts

echo 'analysis'
python3 IA/analysis.py $f "0 1 2 3 4 5 6 7" -p "0 1 2 3 4 5 6 7"
echo 'plot_mean_stds'
python3 IA/plot_mean_stds_drug_studies.py Output/Analysis/test_input.csv
echo 'plot_disp_str'
python3 IA/plot_disp_str.py $f 9 150
echo 'find_range'
python3 IA/find_range.py $f

