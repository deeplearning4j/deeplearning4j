#!/bin/bash
wget https://s3.amazonaws.com/dl4j-distribution/deeplearning4j-dist-bin.tar.gz
tar xvf deeplearning4j-dist-bin.tar.gz
mkdir lib
mv *.jar lib
sudo yum -y install blas
