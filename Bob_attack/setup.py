#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import find_packages
from bob.extension.utils import load_requirements

install_requires = load_requirements()


setup(

    name='bob.bio.facexzoo',
    version=open("version.txt").read().rstrip(),
    description='New package',

    url='https://gitlab.idiap.ch/bob/bob.bio.facexzoo',
    license='MIT',

    # there may be multiple authors (separate entries by comma)
    author='Hatef OTROSHI',
    author_email='hatef.otroshi@idiap.ch',

    # there may be a maintainer apart from the author - you decide
    maintainer='Hatef OTROSHI',
    maintainer_email='hatef.otroshi@idiap.ch',

    # you may add more keywords separating those by commas (a, b, c, ...)
    keywords = "bob",

    long_description=open('README.md').read(),

    # leave this here, it is pretty standard
    packages=find_packages(),
    include_package_data=True,
    zip_safe = False,

    install_requires=install_requires,

    entry_points={
      # add entry points (scripts, bob resources here, if any)
      },

    # check classifiers, add and remove as you see fit
    # full list here: https://pypi.org/classifiers/
    # don't remove the Bob framework unless it's not a bob package
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

)