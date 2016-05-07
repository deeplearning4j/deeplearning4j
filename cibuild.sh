#!/usr/bin/env bash

install() {
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