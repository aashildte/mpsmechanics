# Motivation

Project for image processing/analysis to analyze mechanical properties -> value in itself, input other models

# Setup

## Dependencies

Depends on numpy, scipy and matplotlib – install these manually for now; we can do it properly later maybe – as well as Henrik's MPS script.

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

# Scripts

To calculate all values for general purposes, run

$ *analyze\_mechanics* [BF files/folders] [options]

for which the output saved in [folder name]/mpsmechanics/analyze\_mechanics.npy, where [folder name] is a folder named the same as the input file (minus the .nd2 suffix). This can be run by itself. For any other script in which the values depends on the mechanical analysis, each will call this script for you if it's not already calculated. 

Other scripts:
* Metrics script: *calculate_metrics*
* Visualize each of the metrics, as movies or plots (at peak); decomposed (6 subplots): *visualize_mechanics*
* Animation/plot of mesh over a (small) part of the chip: *visualize_mesh_over_movie*
* Distribution plots, for each of the metrics: *plot_distributions*
* Animation/plot of calcium over time, original and corrected (2 subplots): *visualize_calcium*
* Vector field plots over the (whole) chip, i.e. displacement, velocity and principal strain: *visualize_vectorfield*


For *all* of the scripts above the positional arguments are assumed to be nd2 files and/or folders containing such files. For most of the scripts (the expection being visualize\_calcium, which works with Cyan files) only the BF files will be included (everything else will just be ignored).

For all of the files we also have two flags that apply:
* Overwrite information at current layer: *-o* or *--overwrite*
* Overwrite information at all layers: *-oa* or *--overwrite\_all*
* Debug mode - stops the script if anything is wrong + write out a more useful error message: *-d* or *--debug\_mode*

For all scripts which can do animation, the default is just to plot at peak. To actually do the animation, add *-a* or *--animate*. To change the framerate, add *-s* or *--scaling\_factor*. If the scaling factor is set to 1 you get frame rate corresponding to real time; if you set it to be less than 1 (you probably would not want the movie to go faster), it will be scaled accordingly (e.g. -s 0.5 for half speed).

For *mesh\_over\_movie* there are additional options for specifying focus area, in terms of coordinates, size and number of mesh points.
* -w / --width give width (in pixels) for area included in the plot/movie; default value 100
* -x / --xcoord and -y / --ycoord gives coordinates; coordinates given will be in the middle of the area included. xcoord is here taken to be along the chamber, ycoord across.
* -st / --step determines how many mesh point to include in the animation. -s 1 (also default) will include all points tracked. -st 2 will give every second point (in each direction) etc. Higher st makes sense for wider pictures/movies.
