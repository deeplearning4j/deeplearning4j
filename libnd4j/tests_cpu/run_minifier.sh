#!/bin/bash
CXX=`gcc --print-search-dirs | awk '/install/{print $2;}'`
export CXX
make -j4 && layers_tests/minifier -l -o nd4j_minilib.h 

