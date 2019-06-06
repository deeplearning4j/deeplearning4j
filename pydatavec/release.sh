#!/bin/bash

# Note: this needs manual upgrading of version in setup.py to work (can't override old versions)

# remove old wheels
sudo rm -rf dist/*

# Build Python 2 & 3 wheels for current version
sudo python2 setup.py sdist bdist_wheel
sudo python3 setup.py sdist bdist_wheel

# Upload to PyPI with twine. Needs full "skymind" credentials in ~/.pypirc
twine upload dist/*