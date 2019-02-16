# Motivation

Project for image processing -> value in itself, input other models

# Fibre orientation

Calculate vector field representing movement / direction script as

    python3 fibre_direction.py [input file] [M] [N] [X] [Y]

where input file is a csv file, M x N the desired size of the function space, X x Y the dimensions of the image corresponding to the input file, e.g. as
    
    python3 fibre_direction.py H12.csv 255 144 40 340

# Analysis

Perform an analysis by running the script as

    python3 analysis.py [list of csv files] [output file]

eg as

    python3 analysis.py L2.csv - output.csv

or, to read in all csv files in a given folder,

    python3 analysis.py [path]/* -o output_file       (for Linux and Mac)
    python3 analysis.py [path]\* -o output_file       (for Windows - maybe?)

where the list of csv files contains T x X x Y lines with 2 values on each line (with the first line giving the dimensions) – these are assumed to be displacement data.

Note that the output file is always assumed to be the last file name given, and that *this file will be overwritten without any warning* if it already exists.

The output will be saved as plots (in a folder called Plots) for visual checks as well as key values in an output file called values.csv

# Displacement and strain

Run first

    python3 get_range.py

for all files of interest; then

    python3 calc_maxima [list of output files]

where list of output files are as given in get\_range.py; typically something like
    "Output values/get_range/*"

(at least on mac/linux), which will print two values, max displacement and max principal strain; and then finally

    python3 plot_disp_strain.py [input file] [max displacement] [max principal strain] 

The 3-step split is necessesary because the first can run in parallel (on different cores, no communication needed); the second step is a syncronization step; the third step can again be done independently. An option would obviously be to implent the whole thing as a program using e.g. MPI; however at the current state of this project this is probably sufficient.
