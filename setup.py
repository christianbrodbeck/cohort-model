#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='cohort',
    version='0.1dev1',
    description='Cohort model implementation for calculating information theoretic variables',
    author='Christian Brodbeck',
    author_email='christianmbrodbeck@gmail.com',
    install_requires=[
            'attrs',
            'eelbrain',
            'trftools @ https://github.com/christianbrodbeck/TRF-Tools/archive/refs/heads/main.zip',
    ],
    packages=find_packages(),
)
