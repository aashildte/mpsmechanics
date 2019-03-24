# Motivation

Project for image processing -> value in itself, input other models

# Fibre orientation

Calculate vector field representing movement / direction script as

    python3 fibre_direction.py [input file] [M] [N] [X] [Y]

where input file is a csv file, M x N the desired size of the function space, X x Y the dimensions of the image corresponding to the input file, e.g. as
    
    python3 fibre_direction.py H12.csv 255 144 40 340

# Analysis

Perform an analysis by running the script as

    python3 analysis.py [input file] indices -p indices

where the input file is a csv file as specified (OR a nd2 file), and indices a string of integers (e.g. "1 3 6 7", quotation marks included) specifying what properties we want to compute. The integers correspond to one of the following numbers / properties:
    - 0: average beat rate
    - 1: average displacement
    - 2: average x motion
    - 3: average y motion
    - 4: average prevalence
    - 5: average principal strain
    - 6: average principal strain in x direction (x strain)
    - 7: average principal strain in y direction (y strain)

and the list after "-p" indicates which of these that are to be plotted similtanously.

Example:

    python3 analysis.py test.csv "0 1 2 3 7" -p "2 3"

The output will be saved in Output -> Analysis, plots will be saved in Figures -> Analysis.

# Displacement and strain

Run first

    python3 get_range.py

for all files of interest; then

    python3 calc_maxima [list of output files]

where list of output files are as given in get\_range.py; typically something like
    "Output/get_range/*"

(at least on mac/linux), which will print two values, max displacement and max principal strain; and then finally

    python3 plot_disp_strain.py [input file] [max displacement] [max principal strain] 

The 3-step split is necessesary because the first can run in parallel (on different cores, no communication needed); the second step is a syncronization step; the third step can again be done independently. An option would obviously be to implent the whole thing as a program using e.g. MPI; however at the current state of this project this is probably sufficient.


# Tracking points

Run

    python3 track_points.py [file1] [file2]

where file1 give displacement, file2 points of interest. See test\_input.csv, test\_points.csv for examples.
