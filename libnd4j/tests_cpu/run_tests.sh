#!/bin/bash

make -j4 && layers_tests/runtests --gtest_output="xml:cpu_test_results.xml"
