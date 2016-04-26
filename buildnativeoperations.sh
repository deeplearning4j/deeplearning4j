#!/usr/bin/env bash

#export OMP_NUM_THREADS=1

export CMAKE_COMMAND="cmake"
export MAKE_COMMAND="make"
echo eval $CMAKE_COMMAND
if [ "$(uname)" == "Darwin" ]; then
    echo "RUNNING OSX CLANG"
    # Do something under Mac OS X platform
    #export CC=clang-omp++
    export CXX=clang-omp++
elif [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ] || [ "$(expr substr $(uname -s) 1 4)" == "MSYS" ]; then
    # Do something under Windows NT platform
    if [ "$2" == "cuda" ]; then
        export CMAKE_COMMAND="cmake -G \"NMake Makefiles\" -DCMAKE_BUILD_TYPE=Release"
        export MAKE_COMMAND="nmake"
    else
        export CMAKE_COMMAND="cmake -G \"MSYS Makefiles\""
        export MAKE_COMMAND="make"
    fi
    # Try some defaults for Visual Studio 2013 if user has not run vcvarsall.bat or something
    if [ -z "$VCINSTALLDIR" ]; then
        export VisualStudioVersion=12.0
        export VSINSTALLDIR="C:\\Program Files (x86)\\Microsoft Visual Studio $VisualStudioVersion"
        export VCINSTALLDIR="$VSINSTALLDIR\\VC"
        export WindowsSdkDir="C:\\Program Files (x86)\\Windows Kits\\8.1"
        export Platform=X64
        export INCLUDE="$VCINSTALLDIR\\INCLUDE;$WindowsSdkDir\\include\\shared;$WindowsSdkDir\\include\\um"
        export LIB="$VCINSTALLDIR\\LIB\\amd64;$WindowsSdkDir\\lib\\winv6.3\\um\\x64"
        export LIBPATH="$VCINSTALLDIR\\LIB\\amd64;$WindowsSdkDir\\References\\CommonConfiguration\\Neutral"
        export PATH="$PATH:$VCINSTALLDIR\\BIN\\amd64:$WindowsSdkDir\\bin\\x64:$WindowsSdkDir\\bin\\x86"
    fi
    # Make sure we are using 64-bit MinGW-w64
    export PATH=/mingw64/bin/:$PATH
   CC=/mingw64/bin/gcc
    CXX=/mingw64/bin/g++
    echo "Running windows"
   # export GENERATOR="MSYS Makefiles"

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
            export GENERATOR="Eclipse CDT4 - Unix Makefiles"
            eval $CMAKE_COMMAND -DDEV=FALSE -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ..
            python ./nsight-err-parse-patch.py ./project
            mv eclipse/.cproject .
            mv eclipse/.project .
     elif [ "$1" ==  "lib" ]; then
         rm -rf library  build
         mkdir librarybuild
         cd librarybuild
          eval $CMAKE_COMMAND -DDEV=FALSE -DLIBRARY=TRUE ..
         eval $MAKE_COMMAND && cd ..
     elif [ "$1" ==  "test" ]; then
           if [ "$#" -gt "1" ]; then
                rm -rf testbuild
                mkdir testbuild
                cd testbuild
                 eval $CMAKE_COMMAND  -DDEV=FALSE -DRUN_TEST=TRUE ..
                eval $MAKE_COMMAND && cd ..
                mv testbuild/test/libnd4jtests .
               ./libnd4jtests -n "$2"
           else
               rm -rf testbuild
               mkdir testbuild
               cd testbuild
               eval $CMAKE_COMMAND -DDEV=FALSE -DRUN_TEST=TRUE ..
               eval $MAKE_COMMAND && cd ..
               mv testbuild/test/libnd4jtests .
               ./libnd4jtests
           fi

     elif [ "$1" == "cubin" ]; then
            rm -rf cubinbuild
           mkdir cubinbuild
           cd cubinbuild
            eval $CMAKE_COMMAND -DDEV=FALSE -DCUBIN=TRUE ..
           eval $MAKE_COMMAND && cd ..
           mv cubinbuild/cubin/cuda_compile_cubin_generated_all.cu.cubin all.cubin
     elif [ "$1" == "blas" ]; then
           if [ "$#" -gt 1 ]; then
              if [ "$2" == "cuda" ]; then
              if [ "$#" -gt 2 ]; then
                   echo "CUDA BUILD $3"
                   rm -rf blasbuild/cuda
                   mkdir -p blasbuild/cuda
                   cd blasbuild/cuda
                   eval $CMAKE_COMMAND -DCUDA_BLAS=true -DDEV=FALSE -DBLAS=TRUE -DCMAKE_BUILD_TYPE=$3 ../..
                   eval $MAKE_COMMAND && cd ../..
              else
                   echo "CUDA BUILD RELEASE"
                   rm -rf blasbuild/cuda
                   mkdir -p blasbuild/cuda
                   cd blasbuild/cuda
                   eval $CMAKE_COMMAND -DCUDA_BLAS=true -DBLAS=TRUE -DDEV=FALSE -DCMAKE_BUILD_TYPE=Release../..
                   eval $MAKE_COMMAND && cd ../..
              fi
              elif [ "$2" == "cpu" ]; then
                    if [ "$#" -gt 2 ]; then
                     echo "CPU BUILD DEBUG"
                     echo "RUNNING COMMAND $CMAKE_COMMAND"
                     rm -rf blasbuild/cpu
                     mkdir -p blasbuild/cpu
                   cd blasbuild/cpu
                   eval $CMAKE_COMMAND -DCPU_BLAS=true -DBLAS=TRUE -DDEV=FALSE -DCMAKE_BUILD_TYPE=$3  ../..
                   eval $MAKE_COMMAND && cd ../..

              else
                    echo "RUNNING COMMAND $CMAKE_COMMAND"
                        rm -rf blasbuild/cpu

                        mkdir -p blasbuild/cpu
                    cd blasbuild/cpu
                    eval $CMAKE_COMMAND -DCPU_BLAS=true -DBLAS=TRUE -DDEV=FALSE -DCMAKE_BUILD_TYPE=Release ../..
                   eval $MAKE_COMMAND && cd ../..
              fi


              else
                   echo "Please specify cpu or cuda"

              fi

            else

                   eval $CMAKE_COMMAND  -DCPU_BLAS=true -DDEV=FALSE -DBLAS=TRUE ..
                  eval $MAKE_COMMAND && cd ..
           fi


      elif [ "$1" == "ptx" ]; then
           rm -rf ptxbuild
           mkdir ptxbuild
           cd ptxbuild
            eval $CMAKE_COMMAND -DDEV=FALSE -DPTX=TRUE ..
           eval $MAKE_COMMAND && cd ..
           echo "FINISHING BUILD"
           mv ptxbuild/ptx/cuda_compile_ptx_generated_all.cu.ptx all.ptx
     elif [ "$1" == "fatbin" ]; then
           rm -rf fatbuild
           mkdir fatbuild
           cd fatbuild
            eval $CMAKE_COMMAND -DDEV=FALSE -DFATBIN=TRUE ..
           eval $MAKE_COMMAND && cd ..
           echo "FINISHING BUILD"
           mv fatbuild/fatbin/cuda_compile_fatbin_generated_all.cu.fatbin all.fatbin
fi

fi
