#!/bin/bash
CXX=/usr/bin/g++
CXX_PATH=`$CXX --print-search-dirs | awk '/install/{print $2;}'`
export CXX_PATH
export CXX

make -j4 && layers_tests/runtests --gtest_output="xml:cpu_test_results.xml"
