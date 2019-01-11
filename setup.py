import codecs
import os
import re

from setuptools import setup, find_packages


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')
README_FILE = os.path.join(PROJECT_ROOT, 'README.md')
VERSION_FILE = os.path.join(PROJECT_ROOT, 'pymc4', '__init__.py')


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_long_description():
    with codecs.open(README_FILE, 'rt') as buff:
        return buff.read()


def get_version():
    lines = open(VERSION_FILE, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version in %s.' % (VERSION_FILE,))

setup(
    name='PyMC4',
    version=get_version(),
    description='Four times the Bayes',
    author='PyMC Developers',
    url="https://github.com/pymc-devs/pymc4",
    packages=find_packages(),
    install_requires=get_requirements(),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    include_package_data=True,
)
