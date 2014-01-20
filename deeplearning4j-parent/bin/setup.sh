#!/bin/bash
yum -y update
sudo yum install -y libpng-devel python-setuptools blas java-1.7.0-openjdk.x86_64 zip unzip freetype-devel
sudo pip install numpy matplotlib

