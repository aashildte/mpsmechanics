# project for image processing -> value in itself, input other models

Do analysis running

    python(3) analysis.py [list of csv files]

eg as

    python3 analysis.py L2.csv

where the list of csv files contains T x X x Y lines with 2 values on each line
(with the first line giving the dimensions) – these are assumed to be displacement data.

The output will be saved as plots (in a folder called Plots) for visual checks as well as key values in an output file called values.csv
