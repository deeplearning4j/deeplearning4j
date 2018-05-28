#!/bin/bash

cmake -G "Unix Makefiles" && make -j4 && layers_tests/runtests --gtest_output="xml:../target/surefire-reports/cpu_test_results.xml"
