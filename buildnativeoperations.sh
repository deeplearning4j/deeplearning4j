#!/usr/bin/env bash


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
# Use > 1 to consume two arguments per pass in the loop (e.g. each
# argument has a corresponding value to go with it).
# Use > 0 to consume one or more arguments per pass in the loop (e.g.
# some arguments don't have a corresponding value to go with it such
# as in the --default example).
# note: if this is set to > 0 the /etc/hosts part is not recognized ( may be a bug )
while [[ $# > 1 ]]
do
key="$1"
#Build type (release/debug), packaging type, chip: cpu,gpu,lib type (static/dynamic)
case $key in
    -b|--build-type)
    BUILD="$2"
    shift # past argument
    ;;
    -p|--packaging)
    PACKAGING="$2"
    shift # past argument
    ;;
    -c|--chip)
    CHIP="$2"
    shift # past argument
    ;;
     -l|--libtype)
    LIBTYPE="$2"
    shift # past argument
    ;;
    --default)
    DEFAULT=YES
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

if [ -z "$BUILD" ]; then
 BUILD="release"

fi

if [ -z "$CHIP" ]; then
 CHIP="cpu"
fi

if [ -z "$LIBTYPE" ]; then
 LIBTYPE="dynamic"
fi

if [ -z "$PACKAGING" ]; then
 PACKAGING="none"
fi

if [ "$CHIP" == "cpu" ]; then
  BLAS_ARG="-DCPU_BLAS=true -DBLAS=TRUE"
  else
       BLAS_ARG="-DCUDA_BLAS=true -DBLAS=TRUE"
fi

if [ "$LIBTYPE" == "dynamic" ]; then
     SHARED_LIBS_ARG="-DBUILD_SHARED_LIBS=OFF"
     else
         SHARED_LIBS_ARG="-DBUILD_SHARED_LIBS=ON"
fi

if [ "$BUILD" == "release" ]; then
    BUILD_TYPE="-DCMAKE_BUILD_TYPE=Release"
    else
        BUILD_TYPE="-DCMAKE_BUILD_TYPE=Debug"

fi

if [ "$PACKAGING" == "none" ]; then
    PACKAGING_ARG="-DPACKAGING=none"
fi

if [ "$PACKAGING" == "rpm" ]; then
    PACKAGING_ARG="-DPACKAGING=rpm"
fi

if [ "$PACKAGING" == "deb" ]; then
    PACKAGING_ARG="-DPACKAGING=deb"
fi

if [ "$PACKAGING" == "msi" ]; then
    PACKAGING_ARG="-DPACKAGING=msi"
fi


mkbuilddir() {
if [ "$CHIP" == "cpu" ]; then
     rm -rf blasbuild/cpu
     mkdir -p blasbuild/cpu
     cd blasbuild/cpu
  else
        rm -rf blasbuild/cuda
        mkdir -p blasbuild/cuda
        cd blasbuild/cuda
fi

}


echo PACKAGING  = "${PACKAGING}"
echo BUILD  = "${BUILD}"
echo CHIP     = "${CHIP}"
echo LIBRARY TYPE    = "${LIBTYPE}"
mkbuilddir
pwd
eval $CMAKE_COMMAND  "$BLAS_ARG" "$SHARED_LIBS_ARG"  "$BUILD_TYPE" "$PACKAGING_ARG" -DDEV=FALSE ../..
eval $MAKE_COMMAND && cd ../../..


