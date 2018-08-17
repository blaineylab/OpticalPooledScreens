"""Adapted from https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='ops-lasagna',
    version='0.1',
    description='Analysis code for pooled optical screening.',  # Required
    # long_description=long_description,
    url='http://github.com/blaineylab/opticalpooledscreens',
    author='blaineylab/feldman',  # Optional
    author_email='feldman@broadinstitute.org',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    packages=['ops'],
)
