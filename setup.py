#!/usr/bin/env python
from codecs import open  # pylint: disable=redefined-builtin
from os.path import realpath, dirname, join
import re
from setuptools import setup, find_packages

DISTNAME = 'pymc4'
# pylint: disable=line-too-long
DESCRIPTION = "Pre-release development of high-level probabilistic programming interface for TensorFlow"
AUTHOR = 'PyMC Developers'
AUTHOR_EMAIL = 'pymc.devs@gmail.com'
URL = "http://github.com/pymc-devs/pymc4"
LICENSE = "Apache License, Version 2.0"

CLASSIFIERS = ['Development Status :: 2 - Pre-Alpha',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

PROJECT_ROOT = dirname(realpath(__file__))

# Get the long description from the README file
with open(join(PROJECT_ROOT, 'README.md'), encoding='utf-8') as buff:
    LONG_DESCRIPTION = buff.read()

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    INSTALL_REQS = f.read().splitlines()

TEST_REQS = ['pytest', 'pytest-cov']


def get_version():
    versionfile = join('pymc4', '__init__.py')
    lines = open(versionfile, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        match = re.search(version_regex, line, re.M)
        if match:
            return match.group(1)
    raise RuntimeError('Unable to find version in %s.' % (versionfile,))

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=get_version(),
          maintainer=AUTHOR,
          maintainer_email=AUTHOR_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          packages=find_packages(),
          package_data={},
          include_package_data=True,
          classifiers=CLASSIFIERS,
          install_requires=INSTALL_REQS,
          tests_require=TEST_REQS,)
