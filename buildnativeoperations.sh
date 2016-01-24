#!/usr/bin/env bash

export CPPUTEST_HOME=/home/agibsonccc/code/cpputest

if [ "$#" -ne 1 ]; then
    echo "Please specify an argument"
else
    command="$1"
    echo "Running $1"
    if [ "$1" == "clean" ]; then
       rm -rf cmake_install.cmake
       rm -rf cubinbuild
       rm -rf ptxbuild
       rm -rf CMakeFiles 
       rm -f CMakeCache.txt
       rm -rf testbuild
       rm -rf eclipse/CMakeFiles

       echo "Deleted build"
    elif [ "$1" ==  "eclipse" ]; then
            cd eclipse
            cmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ..
            python ./nsight-err-parse-patch.py ./project
            mv eclipse/.cproject .
            mv eclipse/.project .
     elif [ "$1" ==  "test" ]; then
           rm -rf testbuild
          mkdir testbuild
           cd testbuild
           cmake -DRUN_TEST=TRUE ..
            make && cd ..
            mv testbuild/test/libnd4jtests .
            ./libnd4jtests
           echo "FINISHING BUILD"
     elif [ "$1" == "cubin" ]; then
            rm -rf cubinbuild
           mkdir cubinbuild
           cd cubinbuild
           cmake -DCUBIN=TRUE ..
           make && cd ..
           echo "FINISHING BUILD"
           mv cubinbuild/cubin/cuda_compile_cubin_generated_all.cu.cubin all.cubin
      elif [ "$1" == "ptx" ]; then
           rm -rf ptxbuild
           mkdir ptxbuild
           cd ptxbuild
           cmake -DPTX=TRUE ..
           make && cd ..
           echo "FINISHING BUILD"
           mv ptxbuild/ptx/cuda_compile_ptx_generated_all.cu.ptx all.ptx
fi

fi
