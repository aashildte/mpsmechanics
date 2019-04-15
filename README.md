# Motivation

Project for image processing/analysis to analyze mechanical properties -> value in itself, input other models

# Setup

## Dependencies

Depends on numpy, scipy and matplotlib – install these manually for now; we can do it properly later maybe – as well as Henrik's MPS script. Eventually it will also depend on David's script for finding the pillars (?).

## Initializing git

To download the code for the relevant scripts you can use git's 'pull' command. In order to have access to the code you need to tell git who you are, and tell bitbucket that you are allowed to pull the code from a given computer.

Set up git by entering

    git config --global user.name "Your Name"
    git config --global user.email your@email.com
    git init

Locate your ssh key, or generate it (check e.g. https://help.github.com/en/articles/connecting-to-github-with-ssh for how) and add it to the list of known ssh keys in the bitbucket repository – either for all your projects (Account settings -> Security -> SSH keys) or for this one only (Repository settings -> General -> Access keys).

## Installing the scripts

To install the module and the relevant scripts run

    python setup.py install

which can be done either globally (you might want to be in sudo mode) or in a local environment. You can specify where to install it using the "--prefix" option and you might want to update your PYTHONPATH variable to point to the given location.

You should now be able to access the code in mpsmechanics as a module (import it in your python script) or run the scripts located in the "scripts" folder anywhere.

# Running the script

At the moment the two scripts developed are *analyze_mechanics* and *track_pillars*. We might add others eventually, feel free to suggest new ones if you have any ideas.

## Input files

Both scripts handle displacement data sets; *track_pillars* also takes a csv file for inital position of the pillars as input.

The displacement files are either nd2 files (better description needed) or csv files. For the csv files, the first three numbers give T, X and Y; then we have T x X x Y lines of two components - x and y displacement for each time step and each spacial coordinate. The displacement data are relative to number of points we're given; each number determines how many blocks in x/y direction a given macroblock has moved.

The initial position of the pillars are given as csv files, having numbers for x position, y position and radius at each line.

## Analysis

Perform an analysis by running the script as

    analyze_mechanics [input file] indices -p indices

where the input file is a csv file or a nd2 file as specified, and indices a string of integers (e.g. "1 3 6 7", quotation marks included) specifying what properties we want to compute. The integers correspond to one of the following numbers / properties:
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

    analyze_mechanics test.csv "0 1 2 3 7" -p "2 3"

The output will be saved in Output -> Analysis, plots will be saved in Figures -> Analysis.

## Tracking pillars

Run

    track_pillars.py [file1] [file2]

where file1 give displacement, file2 points of interest. See test\_input.csv, test\_points.csv for examples.
