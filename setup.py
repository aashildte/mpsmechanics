
# -*- coding: utf-8 -*-

"""

Åshild Telle, Henrik Finsberg / Simula Research Laboratory / 2019

"""

from __future__ import print_function

import os
import sys
import platform
import glob

from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)

VERSION = "0.1"
NAME = "mpsmechanics"
AUTHORS = "Åshlid Telle"
SCRIPTS = glob.glob("scripts/*")

def adapt_to_windows():
    """
    In the Windows command prompt we can't execute Python scripts
    without a .py extension. A solution is to create batch files
    that runs the different scripts.
    """
    batch_files = []
    for script in SCRIPTS:
        if os.path.splitext(script)[-1] == ".bat":
            continue
        batch_filename = script + ".bat"
        batch_file = open(batch_filename, "w")
        script_file = os.path.split(script)[1]
        batch_file.write('python "%~dp0{}" %*'.format(script_file))
        batch_file.close()
        batch_files.append(batch_filename)
    SCRIPTS.extend(batch_files)

def run_install():
    """
    Run installation
    """

    if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
        adapt_to_windows()

    # Call distutils to perform installation
    setup(
        name=NAME,
        description="...",
        version=VERSION,
        author=AUTHORS,
        license="...",
        author_email="aashild@simula.no",
        platforms=["Windows", "Linux", "Mac OS-X"],
        packages=find_packages("."),
        package_dir={"mpsmechanics": "mpsmechanics"},
        install_requires=[],
        # Additional build targets
        scripts=SCRIPTS,
        zip_safe=False,
    )


if __name__ == "__main__":
    run_install()
