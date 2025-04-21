#!/usr/bin/env bash
#
# /* ******************************************************************************
#  *
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  *  See the NOTICE file distributed with this work for additional
#  *  information regarding copyright ownership.
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#
set -eu

# cd to the directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

setwindows_msys() {
  if [[ $KERNEL == *"windows"* ]]; then
    export CMAKE_COMMAND="$CMAKE_COMMAND -G \"MSYS Makefiles\""
  fi
}

setandroid_defaults() {
  if [[ -z ${ANDROID_NDK:-} ]]; then
    export ANDROID_NDK=$HOME/Android/android-ndk/
    echo "No ANDROID_NDK variable set. Setting to default of $ANDROID_NDK"
  else
    echo "USING ANDROID NDK $ANDROID_NDK"
  fi

  if [[ -z ${ANDROID_VERSION:-} ]]; then
    export ANDROID_VERSION=21
    echo "No ANDROID_VERSION variable set. Setting to default of $ANDROID_VERSION"
  else
    echo "USING ANDROID VERSION $ANDROID_VERSION"
    # android needs static linking
  fi
}

export CMAKE_COMMAND="cmake"
if which cmake3 &> /dev/null; then
    export CMAKE_COMMAND="cmake3"
fi
export MAKE_COMMAND="make"
export MAKE_ARGUMENTS=
echo eval $CMAKE_COMMAND

[[ -z ${MAKEJ:-} ]] && MAKEJ=4

PARALLEL="true"
OS=
CHIP=
BUILD=
COMPUTE=
ARCH=
LIBTYPE=
PACKAGING=
CHIP_EXTENSION=
CHIP_VERSION=
EXPERIMENTAL=
# OPERATIONS AND DATATYPES argument are lists with semicolon ; separator
OPERATIONS=
DATATYPES=
CLEAN="false"
MINIFIER="false"
TESTS="false"
PRINT_INDICES="OFF"
VERBOSE="true"
VERBOSE_ARG="VERBOSE=1"
HELPER=
CHECK_VECTORIZATION="OFF"
NAME=
OP_OUTPUT_FILE="include/generated/include_ops.h"
USE_LTO=
SANITIZE="OFF"
OPTIMIZATION_LEVEL=
# NOTE WHEN SETTING THIS VALUE. THREAD AND ADDRESS CAN NOT BE USED TOGETHER. THAT IS WHY THIS OPTION EXISTS.
# FOR THREADS USE: thread,undefined,float-divide-by-zero,float-cast-overflow
# FOR ADDRESS USE: address,undefined,float-divide-by-zero,float-cast-overflow
SANITIZERS="address,undefined,float-divide-by-zero,float-cast-overflow"
FUNC_TRACE="OFF"
LOG_OUTPUT="none"
PRINT_MATH="OFF"
KEEP_NVCC="OFF"
PREPROCESS="ON"  # Initialize PREPROCESS variable
CMAKE_ARGUMENTS=""
PTXAS_INFO="OFF"
while [[ $# -gt 0 ]]
do
    key="$1"
    value="${2:-}"
    # Build type (release/debug), packaging type, chip: cpu,cuda, lib type (static/dynamic)
    case $key in
      --generate-flatc)
                export GENERATE_FLATC="ON"
                CMAKE_ARGUMENTS="$CMAKE_ARGUMENTS -DGENERATE_FLATC=ON"
                shift # past argument
                ;;
    -ptxas|--ptxas-info)
            PTXAS_INFO="$value"
            shift # past argument
            ;;
        -ol|--optimization-level)
            OPTIMIZATION_LEVEL="$value"
            shift # past argument
            ;;
        -pi|--print-indices)
            PRINT_INDICES="$value"
            shift # past argument
            ;;
        -pm|--print-math)
            PRINT_MATH="$value"
            shift # past argument
            ;;
        -h|--helper)
            HELPER="$value"
            shift # past argument
            ;;
        -o|-platform|--platform)
            OS="$value"
            shift # past argument
            ;;
        -ft|-functrace|--functrace)
            FUNC_TRACE="$value"
            shift # past argument
            ;;
        -b|--build-type)
            BUILD="$value"
            shift # past argument
            ;;
        -p|--packaging)
            PACKAGING="$value"
            shift # past argument
            ;;
        -kno|--keep-nvcc-output)
            KEEP_NVCC="$value"
            shift # past argument
            ;;
        -c|--chip)
            CHIP="$value"
            shift # past argument
            ;;
        -cc|--compute)
            COMPUTE="$value"
            echo COMPUTE="$value"
            shift # past argument
            ;;
        -a|--arch)
            ARCH="$value"
            shift # past argument
            ;;
        -l|--libtype)
            LIBTYPE="$value"
            shift # past argument
            ;;
        -e|--chip-extension)
            CHIP_EXTENSION="$value"
            shift # past argument
            ;;
        -v|--chip-version)
            CHIP_VERSION="$value"
            shift # past argument
            ;;
        -op|--operations)
            OPERATIONS="$value"
            shift # past argument
            ;;
        -dt|--datatypes)
            DATATYPES="$value"
            shift # past argument
            ;;
        --use_lto)
            USE_LTO="-DSD_USE_LTO=$value"
            shift # past argument
            ;;
        --name)
            NAME="$value"
            shift # past argument
            ;;
        --check-vectorization)
            CHECK_VECTORIZATION="$value"
            shift # past argument
            ;;
        -j)
            MAKEJ="$value"
            shift # past argument
            ;;
        clean)
            CLEAN="true"
            shift # past argument
            ;;
        -m|--minifier)
            MINIFIER="true"
            shift # past argument
            ;;
        -t|--tests)
            TESTS="true"
            shift # past argument
            ;;
        -V|--verbose)
            VERBOSE="true"
            shift # past argument
            ;;
        -l|--log-output)
            LOG_OUTPUT="$value"
            shift # past argument
            ;;
        -sa|--sanitize)
            SANITIZE="$value"
            shift # past argument
            ;;
        -sar|--sanitizers)
            SANITIZERS="$value"
            shift # past argument
            ;;
        -of|--op-output-file)
            OP_OUTPUT_FILE="$value"
            shift # past argument
            ;;
        --preprocess)
            PREPROCESS="$value"
            shift # past argument
            ;;
        --default)
            DEFAULT=YES
            shift # past argument
            ;;
        *)
            # unknown option
            shift # past argument
            ;;
    esac
done

HOST=$(uname -s | tr [A-Z] [a-z])
KERNEL=$HOST-$(uname -m | tr [A-Z] [a-z])

if [ "$(uname)" == "Darwin" ]; then
    HOST="macosx"
    KERNEL="darwin-x86_64"
    echo "RUNNING OSX CLANG"
elif [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ] || [ "$(expr substr $(uname -s) 1 4)" == "MSYS" ]; then
    HOST="windows"
    KERNEL="windows-x86_64"
    echo "Running windows"
elif [ "$(uname -m)" == "ppc64le" ]; then
    if [ -z "$ARCH" ]; then
        ARCH="power8"
    fi
    KERNEL="linux-ppc64le"
fi

if [ -z "$OS" ]; then
    OS="$HOST"
fi

if [[ -z ${ANDROID_NDK:-} ]]; then
    export ANDROID_NDK=$HOME/Android/android-ndk/
fi

case "$OS" in
    linux-armhf)
        if [ -z "$ARCH" ]; then
            ARCH="armv7-a"
        fi
        if [ ! -z ${RPI_BIN+set} ]; then
            export CMAKE_COMMAND="$CMAKE_COMMAND -D CMAKE_TOOLCHAIN_FILE=cmake/rpi.cmake"
        fi
        export CMAKE_COMMAND="$CMAKE_COMMAND -DSD_ARM_BUILD=true -DSD_SANITIZE=OFF "
        ;;
    linux-arm64)
        if [ -z "$ARCH" ]; then
            ARCH="armv8-a"
        fi
        if [ ! -z ${RPI_BIN+set} ]; then
            export CMAKE_COMMAND="$CMAKE_COMMAND -D CMAKE_TOOLCHAIN_FILE=cmake/rpi.cmake"
        fi
        export CMAKE_COMMAND="$CMAKE_COMMAND -DSD_ARM_BUILD=true"
        ;;
    android-arm)
        if [ -z "$ARCH" ]; then
            ARCH="armv7-a"
        fi

        setandroid_defaults

        # Note here for android 32 bit prefix on the binutils is different
        # See https://developer.android.com/ndk/guides/other_build_systems
        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi/prebuilt/$KERNEL/"
        export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
        export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL/bin/clang"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-$ANDROID_VERSION/arch-arm/"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/android-arm.cmake -DSD_ANDROID_BUILD=true"
        setwindows_msys
        ;;
    android-arm64)
        if [ -z "$ARCH" ]; then
            ARCH="armv8-a"
        fi

        setandroid_defaults

        echo "BUILDING ANDROID ARM with KERNEL $KERNEL"
        export ANDROID_BIN="$ANDROID_NDK/toolchains/aarch64-linux-android-4.9/prebuilt/$KERNEL/"
        export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
        export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL/bin/clang"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-$ANDROID_VERSION/arch-arm64/"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/android-arm64.cmake -DSD_ANDROID_BUILD=true"
        setwindows_msys
        ;;
    android-x86)
        if [ -z "$ARCH" ]; then
            ARCH="i686"
        fi

        setandroid_defaults
        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/$KERNEL/"
        export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
        export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL/bin/clang"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-$ANDROID_VERSION/arch-x86/"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/android-x86.cmake -DSD_ANDROID_BUILD=true"
        setwindows_msys
        ;;
    android-x86_64)

        if [ -z "$ARCH" ]; then
            ARCH="x86-64"
        fi
        echo "BUILDING ANDROID x86_64"

        setandroid_defaults


        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/$KERNEL/"
        export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
        export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL/bin/clang"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-$ANDROID_VERSION/arch-x86_64/"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/android-x86_64.cmake -DSD_ANDROID_BUILD=true"
        setwindows_msys
        ;;

    ios-x86_64)
        LIBTYPE="static"
        ARCH="x86-64"
        if xcrun --sdk iphoneos --show-sdk-version &> /dev/null; then
            export IOS_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
        else
            export IOS_VERSION="10.3"
        fi
        XCODE_PATH="$(xcode-select --print-path)"
        export IOS_SDK="$XCODE_PATH/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator$IOS_VERSION.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-x86_64.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    ios-x86)
        LIBTYPE="static"
        ARCH="i386"
        if xcrun --sdk iphoneos --show-sdk-version &> /dev/null; then
            export IOS_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
        else
            export IOS_VERSION="10.3"
        fi
        XCODE_PATH="$(xcode-select --print-path)"
        export IOS_SDK="$XCODE_PATH/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator$IOS_VERSION.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-x86.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    ios-arm64)
        LIBTYPE="static"
        ARCH="arm64"
        if xcrun --sdk iphoneos --show-sdk-version &> /dev/null; then
            export IOS_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
        else
            export IOS_VERSION="10.3"
        fi
        XCODE_PATH="$(xcode-select --print-path)"
        export IOS_SDK="$XCODE_PATH/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS$IOS_VERSION.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-arm64.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    ios-arm)
        LIBTYPE="static"
        ARCH="armv7"
        if xcrun --sdk iphoneos --show-sdk-version &> /dev/null; then
            export IOS_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
        else
            export IOS_VERSION="10.3"
        fi
        XCODE_PATH="$(xcode-select --print-path)"
        export IOS_SDK="$XCODE_PATH/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS$IOS_VERSION.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-arm.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    ios-armv7)
        # change those 2 parameters and make sure the IOS_SDK exists
        export iPhoneOS="iPhoneOS"
        export IOS_VERSION="10.3"
        LIBTYPE="static"
        ARCH="armv7"
        export IOS_SDK="/Applications/Xcode.app/Contents/Developer/Platforms/${iPhoneOS}.platform/Developer/SDKs/${iPhoneOS}${IOS_VERSION}.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-armv7.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    linux*)
        ;;

    macosx*)
        export CC=clang
        export CXX=clang++
        PARALLEL="true"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_MACOSX_RPATH=ON -DSD_APPLE_BUILD=true"
        ;;

    windows*)
        # Do something under Windows NT platform
        if [ "$CHIP" == "cuda" ]; then
            export CMAKE_COMMAND="cmake -G \"Ninja\""
            export MAKE_COMMAND="ninja"
            export CC="cl.exe"
            export CXX="cl.exe"
            PARALLEL="true"
            VERBOSE_ARG="-v"
        else
            # NOTE when running windows, do NOT use the gcc compiler in usr/ always use the one in mingw/bin
            # Another note on this particular gcc: ensure the path is setup in msys2 with /c/msys64/mingw64/bin
            # prefixed. If the one in USR is used, errors like error: 'RTLD_LAZY' was not declared in this scope
            # may show up
            export CMAKE_COMMAND="cmake -G \"MSYS Makefiles\""
            export MAKE_COMMAND="make"
            export CC=gcc
            export CXX=g++
            PARALLEL="true"
        fi

        # Try some defaults for Visual Studio 2013 if user has not run vcvarsall.bat or something
        if [ -z "${VCINSTALLDIR:-}" ]; then
            echo "NEED TO SET DEFAULTS FOR VISUAL STUDIO, NO VCINSTALLDIR environment variable found"
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
        export PATH=/mingw64/bin/:/mingw64/lib:$PATH
        # export GENERATOR="MSYS Makefiles"
        ;;
esac

if [ ! -d "include/generated" ]; then
    mkdir -p "include/generated"
fi

if [ -f "$OP_OUTPUT_FILE" ]; then
    rm -f "${OP_OUTPUT_FILE}"
fi

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

export CMAKE_COMMAND="$CMAKE_COMMAND -DSD_SANITIZE=$SANITIZE -DSD_SANITIZERS=$SANITIZERS"

if [ "$CHIP_EXTENSION" == "avx512" ] || [ "$ARCH" == "avx512" ]; then
    CHIP_EXTENSION="avx512"
    ARCH="skylake-avx512"
elif [ "$CHIP_EXTENSION" == "avx2" ] || [ "$ARCH" == "avx2" ]; then
    CHIP_EXTENSION="avx2"
    ARCH="x86-64"
elif [ "$CHIP_EXTENSION" == "x86_64" ] || [ "$ARCH" == "x86_64" ]; then
    CHIP_EXTENSION="x86_64"
    ARCH="x86-64"
fi

if [ -z "$ARCH" ]; then
    ARCH="x86-64"
fi

if [ -z "$COMPUTE" ]; then
    COMPUTE="all"
fi

# Enable call stacking
if [ "$FUNC_TRACE" == "ON" ]; then
    export CMAKE_COMMAND="$CMAKE_COMMAND -DSD_GCC_FUNCTRACE=ON"
fi

OPERATIONS_ARG=

if [ -z "$OPERATIONS" ]; then
    OPERATIONS_ARG="-DSD_ALL_OPS=true"
else
    OPERATIONS_ARG="-DSD_OPS_LIST=\"$OPERATIONS\" -DSD_ALL_OPS=false"
fi

DATATYPES_ARG=

if [ -n "$DATATYPES" ]; then
    DATATYPES_ARG="-DSD_TYPES_LIST=\"$DATATYPES\""
fi

if [ -z "$EXPERIMENTAL" ]; then
    EXPERIMENTAL="no"
fi

if [ "$CHIP" == "cpu" ]; then
    BLAS_ARG="-DSD_CPU=true -DBLAS=TRUE"
elif [ "$CHIP" == "aurora" ]; then
    BLAS_ARG="-DSD_AURORA=true -DBLAS=TRUE"
elif [ "$CHIP" == "cuda" ]; then
    BLAS_ARG="-DSD_CUDA=true -DBLAS=TRUE"
fi

if [ -z "$NAME" ]; then
    if [ "$CHIP" == "cpu" ]; then
        NAME="nd4jcpu"
    elif [ "$CHIP" == "cuda" ]; then
        NAME="nd4jcuda"
    elif [ "$CHIP" == "aurora" ]; then
        NAME="nd4jaurora"
    fi
fi

if [ "$LIBTYPE" == "dynamic" ]; then
    SHARED_LIBS_ARG="-DSD_SHARED_LIB=ON -DSD_STATIC_LIB=OFF"
else
    SHARED_LIBS_ARG="-DSD_SHARED_LIB=OFF -DSD_STATIC_LIB=ON"
fi

# Set build type
if [ "$FUNC_TRACE" == "ON" ]; then
    BUILD_TYPE="-DCMAKE_BUILD_TYPE=none"
elif [ "$BUILD" == "release" ]; then
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

# Use parent of output file to mean source include directory
OP_OUTPUT_FILE_ARG="-DOP_OUTPUT_FILE=../${OP_OUTPUT_FILE}"

EXPERIMENTAL_ARG=""
MINIFIER_ARG="-DSD_BUILD_MINIFIER=false"
TESTS_ARG="-DSD_BUILD_TESTS=OFF"
NAME_ARG="-DSD_LIBRARY_NAME=$NAME"

if [ "$EXPERIMENTAL" == "yes" ]; then
    EXPERIMENTAL_ARG="-DSD_EXPERIMENTAL=yes"
fi

if [ "$MINIFIER" == "true" ]; then
    MINIFIER_ARG="-DSD_BUILD_MINIFIER=true"
fi

if [ "$TESTS" == "true" ]; then
    MINIFIER_ARG="-DSD_BUILD_MINIFIER=true"
    TESTS_ARG="-DSD_BUILD_TESTS=ON"
fi

ARCH_ARG="-DSD_ARCH=$ARCH -DSD_EXTENSION=$CHIP_EXTENSION"

CUDA_COMPUTE="-DCOMPUTE=\"$COMPUTE\""

if [ "$CHIP" == "cuda" ] && [ -n "$CHIP_VERSION" ]; then
    case $OS in
        linux*)
            if [ "${CUDA_PATH-}" == "" ]; then
                export CUDA_PATH="/usr/local/cuda-$CHIP_VERSION/"
            fi
            ;;
        macosx*)
            export CUDA_PATH="/Developer/NVIDIA/CUDA-$CHIP_VERSION/"
            ;;
        windows*)
            export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v$CHIP_VERSION/"
            ;;
    esac
fi

[[ -z ${OPENBLAS_PATH:-} ]] && OPENBLAS_PATH=""
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

if [[ ! -f "$OPENBLAS_PATH/include/openblas_config.h" ]]; then
    echo "Could not find OpenBLAS, please make sure to run the build with Maven or set the OPENBLAS_PATH variable"
    OPENBLAS_PATH=""
fi

# Replace any backslash with a slash
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"

mkbuilddir() {
    if [ "$CLEAN" == "true" ]; then
        echo "Removing blasbuild"
        rm -Rf blasbuild
    fi
    mkdir -p "blasbuild/$CHIP"
    cd "blasbuild/$CHIP"
}

HELPERS=""
if [ "$HELPER" == "" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                 WARNING!                                                  !!"
    echo "!!                                      No helper packages configured!                                       !!"
    echo "!!                          You can specify helper by using -h key. I.e. <-h onednn>                         !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
else
    # If helpers were defined, we'll propagate them to CMake
    IFS=','
    read -ra HLP <<< "$HELPER"
    for i in "${HLP[@]}"; do
        HELPERS="${HELPERS} -DHELPERS_$i=true"
    done
    IFS=' '
fi

echo PACKAGING           = "${PACKAGING}"
echo BUILD               = "${BUILD}"
echo CHIP                = "${CHIP}"
echo ARCH                = "${ARCH}"
echo CHIP_EXTENSION      = "${CHIP_EXTENSION}"
echo CHIP_VERSION        = "${CHIP_VERSION}"
echo GPU_COMPUTE_CAPABILITY = "${COMPUTE}"
echo EXPERIMENTAL        = "${EXPERIMENTAL}"
echo LIBRARY TYPE        = "${LIBTYPE}"
echo OPERATIONS          = "${OPERATIONS_ARG}"
echo DATATYPES           = "${DATATYPES_ARG}"
echo MINIFIER            = "${MINIFIER_ARG}"
echo TESTS               = "${TESTS_ARG}"
echo NAME                = "${NAME_ARG}"
echo OPENBLAS_PATH       = "$OPENBLAS_PATH"
echo CHECK_VECTORIZATION = "$CHECK_VECTORIZATION"
echo HELPERS             = "$HELPERS"
echo OP_OUTPUT_FILE      = "$OP_OUTPUT_FILE"
echo USE_LTO             = "$USE_LTO"
echo SANITIZE            = "$SANITIZE"
echo FUNC_TRACE          = "$FUNC_TRACE"
echo LOG_OUTPUT          = "$LOG_OUTPUT"
echo KEEP_NVCC           = "$KEEP_NVCC"
echo PRINT_INDICES       = "$PRINT_INDICES"
echo PRINT_MATH          = "$PRINT_MATH"
echo PREPROCESS          = "$PREPROCESS"

mkbuilddir
pwd

# ----------------------- CMake Configuration -----------------------

# Configure CMake
echo "$CMAKE_COMMAND - -DSD_KEEP_NVCC_OUTPUT=$KEEP_NVCC -DSD_GCC_FUNCTRACE=$FUNC_TRACE $BLAS_ARG $ARCH_ARG $NAME_ARG $OP_OUTPUT_FILE_ARG -DSD_SANITIZERS=${SANITIZERS} -DSD_SANITIZE=${SANITIZE} -DSD_CHECK_VECTORIZATION=${CHECK_VECTORIZATION} $USE_LTO $HELPERS $SHARED_LIBS_ARG $MINIFIER_ARG $OPERATIONS_ARG $DATATYPES_ARG $BUILD_TYPE $PACKAGING_ARG $EXPERIMENTAL_ARG $TESTS_ARG $CUDA_COMPUTE -DOPENBLAS_PATH=$OPENBLAS_PATH -DDEV=FALSE -DCMAKE_NEED_RESPONSE=YES -DMKL_MULTI_THREADED=TRUE ../.."

if [ "$LOG_OUTPUT" == "none" ]; then
    eval "$CMAKE_COMMAND" \
        -DPRINT_MATH="$PRINT_MATH" \
        -DPRINT_INDICES="$PRINT_INDICES" \
        -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
        -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
        -DSD_PTXAS="$PTXAS_INFO" \
        "$BLAS_ARG" \
        "$ARCH_ARG" \
        "$NAME_ARG" \
        "$OP_OUTPUT_FILE_ARG" \
        -DSD_SANITIZE="${SANITIZE}" \
        -DSD_CHECK_VECTORIZATION="${CHECK_VECTORIZATION}" \
        "$USE_LTO" \
        "$HELPERS" \
        "$SHARED_LIBS_ARG" \
        "$MINIFIER_ARG" \
        "$OPERATIONS_ARG" \
        "$DATATYPES_ARG" \
        "$BUILD_TYPE" \
        "$PACKAGING_ARG" \
        "$TESTS_ARG" \
        "$CUDA_COMPUTE" \
        -DOPENBLAS_PATH="$OPENBLAS_PATH" \
        -DDEV=FALSE \
        -DCMAKE_NEED_RESPONSE=YES \
        -DMKL_MULTI_THREADED=TRUE \
        ../..
else
    eval "$CMAKE_COMMAND" \
        -DPRINT_MATH="$PRINT_MATH" \
        -DPRINT_INDICES="$PRINT_INDICES" \
        -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
        -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
        -DSD_PTXAS="$PTXAS_INFO" \
        "$BLAS_ARG" \
        "$ARCH_ARG" \
        "$NAME_ARG" \
        "$OP_OUTPUT_FILE_ARG" \
        -DSD_SANITIZE="${SANITIZE}" \
        -DSD_CHECK_VECTORIZATION="${CHECK_VECTORIZATION}" \
        "$USE_LTO" \
        "$HELPERS" \
        "$SHARED_LIBS_ARG" \
        "$MINIFIER_ARG" \
        "$OPERATIONS_ARG" \
        "$DATATYPES_ARG" \
        "$BUILD_TYPE" \
        "$PACKAGING_ARG" \
        "$TESTS_ARG" \
        "$CUDA_COMPUTE" \
        -DOPENBLAS_PATH="$OPENBLAS_PATH" \
        -DDEV=FALSE \
        -DCMAKE_NEED_RESPONSE=YES \
        -DMKL_MULTI_THREADED=TRUE \
        ../.. >> "$LOG_OUTPUT" 2>&1
fi

# ----------------------- Preprocessing Step -----------------------

# Handle the PREPROCESS flag
if [ "$PREPROCESS" == "ON" ]; then
   if [ "$LOG_OUTPUT" == "none" ]; then
       eval "$CMAKE_COMMAND" \
           -DPRINT_MATH="$PRINT_MATH" \
           -DPRINT_INDICES="$PRINT_INDICES" \
           -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
           -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
            -DSD_PREPROCESS="$PREPROCESS" \
           -DSD_PTXAS="$PTXAS_INFO" \
           "$BLAS_ARG" \
           "$ARCH_ARG" \
           "$NAME_ARG" \
           "$OP_OUTPUT_FILE_ARG" \
           -DSD_SANITIZE="${SANITIZE}" \
           "$USE_LTO" \
           "$HELPERS" \
           "$SHARED_LIBS_ARG" \
           "$MINIFIER_ARG" \
           "$OPERATIONS_ARG" \
           "$DATATYPES_ARG" \
           "$BUILD_TYPE" \
           "$PACKAGING_ARG" \
           "$TESTS_ARG" \
           "$CUDA_COMPUTE" \
           -DOPENBLAS_PATH="$OPENBLAS_PATH" \
           -DDEV=FALSE \
           -DCMAKE_NEED_RESPONSE=YES \
           -DMKL_MULTI_THREADED=TRUE \
           ../..
   else
       eval "$CMAKE_COMMAND" \
           -DPRINT_MATH="$PRINT_MATH" \
           -DPRINT_INDICES="$PRINT_INDICES" \
           -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
           -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
           -DSD_PREPROCESS="$PREPROCESS" \
           "$BLAS_ARG" \
           "$ARCH_ARG" \
           "$NAME_ARG" \
           "$OP_OUTPUT_FILE_ARG" \
           -DSD_PTXAS="$PTXAS_INFO" \
           -DSD_SANITIZE="${SANITIZE}" \
           -DSD_CHECK_VECTORIZATION="${CHECK_VECTORIZATION}" \
           "$USE_LTO" \
           "$HELPERS" \
           "$SHARED_LIBS_ARG" \
           "$MINIFIER_ARG" \
           "$OPERATIONS_ARG" \
           "$DATATYPES_ARG" \
           "$BUILD_TYPE" \
           "$PACKAGING_ARG" \
           "$TESTS_ARG" \
           "$CUDA_COMPUTE" \
           -DOPENBLAS_PATH="$OPENBLAS_PATH" \
           -DDEV=FALSE \
           -DCMAKE_NEED_RESPONSE=YES \
           -DMKL_MULTI_THREADED=TRUE \
           ../.. >> "$LOG_OUTPUT" 2>&1
   fi
    echo "Running preprocessing step..."
    exit 0
fi

# --------------------- End of Preprocessing Step ----------------------

# Set Make arguments based on user options
if [ "$PARALLEL" == "true" ]; then
    MAKE_ARGUMENTS="$MAKE_ARGUMENTS -j $MAKEJ"
fi
if [ "$VERBOSE" == "true" ]; then
    MAKE_ARGUMENTS="$MAKE_ARGUMENTS $VERBOSE_ARG"
fi

# Build the project


if [ "$LOG_OUTPUT" == "none" ]; then
    eval "$MAKE_COMMAND" "$MAKE_ARGUMENTS" && cd ../../..
else
    eval "$MAKE_COMMAND" "$MAKE_ARGUMENTS" >> "$LOG_OUTPUT" 2>&1 && cd ../../..
fi

if [ "$GENERATE_FLATC" == "ON" ]; then
    echo "Copying flatc generated for java"
    # ensure proper flatc sources are in place
    bash copy-flatc-java.sh
fi

echo "Build process completed successfully."
