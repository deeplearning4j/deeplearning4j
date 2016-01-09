#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Please specify an argument"
else
    command="$1"
    echo "Running $1"
    if [ "$1" == "clean" ]; then
       rm -f CMakeCache.txt
       rm -rf build
       echo "Deleted build"
    elif [ "$1" ==  "eclipse" ]; then
            cmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE .
            python ./nsight-err-parse-patch.py ./project

    else
           rm -rf build
           echo "Running CMAKE"
           mkdir -p build
           cd build && cmake ..
fi

fi
