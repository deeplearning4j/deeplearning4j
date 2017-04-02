#!/usr/bin/env bash
set -eu

# cd to the directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

install() {
    echo "RUNNING INSTALL"
    if hash nvcc 2>/dev/null; then
    echo "BUILDING CUDA"
        ./buildnativeoperations.sh blas cuda release
        ./buildnativeoperations.sh blas cpu release
    else
        echo "BUILDING CPU"
        ./buildnativeoperations.sh blas cpu release
    fi
}

install
