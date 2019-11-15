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
