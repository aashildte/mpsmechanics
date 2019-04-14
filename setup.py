

# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import platform
import glob

from setuptools import setup, find_packages, Command


if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)

VERSION = "0.1"
NAME = "IA"

scripts = glob.glob("scripts/*.py")

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        if os.path.splitext(script)[-1] == ".bat":
            continue
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%~dp0{}" %*'.format(os.path.split(script)[1]))
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)


AUTHORS = """Åshlid Telle"""

def run_install():
    "Run installation"

    # Call distutils to perform installation
    setup(
        name=NAME,
        description="...",
        version=VERSION,
        author=AUTHORS,
        license="...",
        author_email="aashild@simula.no",
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        packages=find_packages("."),
        package_dir={"IA": "IA"},
        install_requires=[],
        # Additional build targets
        scripts=scripts,
        zip_safe=False,
    )


if __name__ == "__main__":
    run_install()

