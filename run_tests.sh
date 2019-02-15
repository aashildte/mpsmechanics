#!/bin/sh

f=test_input.csv

echo 'init'
python3 __init__.py
echo 'analysis'
python3 analysis.py $f test_output.csv
echo 'fibre_direction'
python3 fibre_direction.py $f 118 25 118 25
echo 'heart_beat'
python3 heart_beat.py $f
echo 'io_funs'
python3 io_funs.py $f
echo 'mechanical_properties'
python3 mechanical_properties.py $f
echo 'preprocessing'
python3 preprocessing.py $f
echo 'plot_disp_str'
python3 plot_disp_str.py $f 9 150
echo 'find_range'
python3 find_range.py $f
echo 'least_sq_solver'
python3 least_sq_solver.py $f
echo 'metrics'
python3 metrics.py $f
echo 'plot_vector_field'
python3 plot_vector_field.py $f
