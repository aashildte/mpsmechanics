#!/bin/sh

f=test_input.csv

echo 'analysis'
python3 IA/analysis.py $f
echo 'fibre_direction'
python3 IA/fibre_direction.py $f 118 25 118 25
echo 'heart_beat'
python3 IA/heart_beat.py $f
echo 'io_funs'
python3 IA/io_funs.py $f
echo 'mechanical_properties'
python3 IA/mechanical_properties.py $f
echo 'preprocessing'
python3 IA/preprocessing.py $f
echo 'plot_disp_str'
python3 IA/plot_disp_str.py $f 9 150
echo 'find_range'
python3 IA/find_range.py $f
echo 'least_sq_solver'
python3 IA/least_sq_solver.py $f
echo 'metrics'
python3 IA/metrics.py $f
echo 'plot_vector_field'
python3 IA/plot_vector_field.py $f
