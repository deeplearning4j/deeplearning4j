#!/usr/bin/env bash

export OMP_NUM_THREADS=1


if [ "$(uname)" == "Darwin" ]; then
    echo "RUNNING OSX CLANG"
    # Do something under Mac OS X platform
    export CC=clang-omp++
    export CXX=clang-omp++
    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    # Do something under Windows NT platform
    echo "Running windows"
fi

if [ "$#" -lt 1 ]; then
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
     elif [ "$1" ==  "lib" ]; then
         rm -rf library  build
         mkdir librarybuild
         cd librarybuild
         cmake -DLIBRARY=TRUE ..
         make && cd ..
     elif [ "$1" ==  "test" ]; then
           if [ "$#" -gt "1" ]; then
                rm -rf testbuild
                mkdir testbuild
                cd testbuild
                cmake -DRUN_TEST=TRUE ..
                make && cd ..
                mv testbuild/test/libnd4jtests .
               ./libnd4jtests -n "$2"
           else
               rm -rf testbuild
               mkdir testbuild
               cd testbuild
               cmake -DRUN_TEST=TRUE ..
               make && cd ..
               mv testbuild/test/libnd4jtests .
               ./libnd4jtests
           fi

           echo "FINISHING BUILD"
     elif [ "$1" == "cubin" ]; then
            rm -rf cubinbuild
           mkdir cubinbuild
           cd cubinbuild
           cmake -DCUBIN=TRUE ..
           make && cd ..
           echo "FINISHING BUILD"
           mv cubinbuild/cubin/cuda_compile_cubin_generated_all.cu.cubin all.cubin
      elif [ "$1" == "buffer" ]; then
            rm -rf bufferbuild
           mkdir bufferbuild
           cd bufferbuild
           cmake -DBUFFER=TRUE ..
           make && cd ..
           echo "FINISHING BUILD"
     elif [ "$1" == "blas" ]; then
            rm -rf blasbuild
           mkdir blasbuild
           cd blasbuild
           if [ "$#" -gt "1" ]; then
              if [ "$2" == "cuda" ]; then
                   cmake -DCUDA_BLAS=true -DBLAS=TRUE ..
                   make && cd ..
                  echo "FINISHING BUILD"
              elif [ "$2" == "cpu" ]; then
                   cmake -DCPU_BLAS=true -DBLAS=TRUE ..
                   make && cd ..
                   echo "FINISHING BUILD"
              else
                   echo "Please specify cpu or gpu"

              fi

            else

                  cmake  -DCPU_BLAS=true -DBLAS=TRUE ..
                  make && cd ..
                  echo "FINISHING BUILD"
           fi


      elif [ "$1" == "ptx" ]; then
           rm -rf ptxbuild
           mkdir ptxbuild
           cd ptxbuild
           cmake -DPTX=TRUE ..
           make && cd ..
           echo "FINISHING BUILD"
           mv ptxbuild/ptx/cuda_compile_ptx_generated_all.cu.ptx all.ptx
     elif [ "$1" == "fatbin" ]; then
           rm -rf fatbuild
           mkdir fatbuild
           cd fatbuild
           cmake -DFATBIN=TRUE ..
           make && cd ..
           echo "FINISHING BUILD"
           mv fatbuild/fatbin/cuda_compile_fatbin_generated_all.cu.fatbin all.fatbin
fi

fi
