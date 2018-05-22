#!/bin/bash
CXX_PATH=`/usr/bin/g++ --print-search-dirs | awk '/install/{print $2;}'`
export CXX_PATH
CXX=/usr/bin/g++
export CXX

make -j4 && layers_tests/minifier -l -o nd4j_minilib.h 

