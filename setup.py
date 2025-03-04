#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

import re
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

with open('README.rst', encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding="utf-8") as history_file:
    history = re.sub(r':py:class:`([^`]+)`', r'\1', 
        history_file.read())

requirements = ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'tqdm', 'ultranest']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Johannes Buchner",
    author_email='johannes.buchner.acad@gmx.com',
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*, !=3.6.*, !=3.7.*, !=3.8.*',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
    ],
    description="Optimized Nested Sampling",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    keywords='optns',
    name='optns',
    packages=['optns'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JohannesBuchner/OptNS',
    version='1.0.0',
)
