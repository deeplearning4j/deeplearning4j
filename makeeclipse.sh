#!/usr/bin/env bash
cmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE .
python ./nsight-err-parse-patch.py ./project

