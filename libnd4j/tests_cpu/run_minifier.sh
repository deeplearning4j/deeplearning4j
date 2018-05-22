#!/bin/bash

make -j4 && layers_tests/minifier -l -o nd4j_minilib.h

