#!/bin/bash
#
# /* ******************************************************************************
#  * Copyright (c) 2021 Deeplearning4j Contributors
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#

TARGET=armv7-a
BLAS_TARGET_NAME=ARMV7
ARMCOMPUTE_TARGET=armv7a
#BASE_DIR=${HOME}/pi
#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SOURCE="${BASH_SOURCE[0]}"
ARMCOMPUTE_DEBUG=1
LIBND4J_BUILD_MODE=Release
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
BASE_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
CMAKE=cmake #/snap/bin/cmake

mkdir -p ${BASE_DIR}/helper_bin/

CROSS_COMPILER_URL=https://sourceforge.net/projects/raspberry-pi-cross-compilers/files/Raspberry%20Pi%20GCC%20Cross-Compiler%20Toolchains/Buster/GCC%208.3.0/Raspberry%20Pi%203A%2B%2C%203B%2B%2C%204/cross-gcc-8.3.0-pi_3%2B.tar.gz/download
CROSS_COMPILER_DIR=${BASE_DIR}/helper_bin/cross_compiler

SCONS_LOCAL_URL=http://prdownloads.sourceforge.net/scons/scons-local-3.1.1.tar.gz
SCONS_LOCAL_DIR=${BASE_DIR}/helper_bin/scons_local

THIRD_PARTY=${BASE_DIR}/third_party_libs

ARMCOMPUTE_GIT_URL=https://github.com/ARM-software/ComputeLibrary.git
ARMCOMPUTE_TAG=v20.05
ARMCOMPUTE_DIR=${THIRD_PARTY}/arm_compute_dir

OPENBLAS_GIT_URL="https://github.com/xianyi/OpenBLAS.git"
OPENBLAS_DIR=${THIRD_PARTY}/OpenBLAS


LIBND4J_SRC_DIR=${BASE_DIR}

LIBND4J_BUILD_DIR=${BASE_DIR}/build_pi

#for some downloads
XRTACT_STRIP="--strip-components=1"

HAS_ARMCOMPUTE=1
mkdir -p ${BASE_DIR}
mkdir -p ${THIRD_PARTY}

#change directory to base
cd $BASE_DIR

function message {
	echo "BUILDER:::: ${@}"
}


function check_requirements {
	for i in "${@}"
	do
      if [ ! -e "$i" ]; then
         message "missing: ${i}"
		 exit -2
	  fi
	done
}

function download_extract {
	#$1 is url #2 is dir $3 is extract argument
	if [ ! -f ${2}_file ]; then
		message "download"
		wget --quiet --show-progress -O ${2}_file ${1}
	fi
 
	message "extract"
    #extract
	mkdir -p ${2}
	command="tar -xzf ${2}_file --directory=${2} ${3} "
	message $command
	$command

	check_requirements "${2}"
}

function git_check {
	#$1 is url #$2 is dir #$3 is tag or branch if optional
	command="git clone --quiet ${1} ${2}"
	message "$command"
	$command 
	if [ -n "$3" ]; then
		cd ${2}
		command="git checkout ${3}"
		message "$command"
		$command 
		cd ${BASE_DIR}
	fi
	check_requirements "${2}"
}


if [ ! -d ${CROSS_COMPILER_DIR} ]; then
	#out file
	message "download CROSS_COMPILER"
	download_extract ${CROSS_COMPILER_URL} ${CROSS_COMPILER_DIR} ${XRTACT_STRIP}
fi

#useful exports
export PI_FOLDER=${CROSS_COMPILER_DIR}
export RPI_BIN=${PI_FOLDER}/bin/arm-linux-gnueabihf
export PI_SYS_ROOT=${PI_FOLDER}/arm-linux-gnueabihf/libc
export LD_LIBRARY_PATH=${PI_FOLDER}/lib:$LD_LIBRARY_PATH
export CC=${RPI_BIN}-gcc
export FC=${RPI_BIN}-gfortran
export CXX=${RPI_BIN}-g++
export CPP=${RPI_BIN}-cpp
export RANLIB=${RPI_BIN}-gcc-ranlib
export LD="${RPI_BIN}-ld"
export AR="${RPI_BIN}-ar"


#lets build OpenBlas 
if [ ! -d "${OPENBLAS_DIR}" ]; then 
	message "download OpenBLAS"
	git_check "${OPENBLAS_GIT_URL}" "${OPENBLAS_DIR}"
fi

if [ ! -f "${THIRD_PARTY}/lib/libopenblas.so" ]; then
	message "build and install OpenBLAS" 
	cd ${OPENBLAS_DIR}

	command="make TARGET=${BLAS_TARGET_NAME} HOSTCC=gcc CC=${CC} USE_THREAD=0 NOFORTRAN=1 CFLAGS=--sysroot=${PI_SYS_ROOT} LDFLAGS=\"-L${PI_SYS_ROOT}/../lib/ -lm\"  &>/dev/null"
	message $command
	eval $command 
    message "install it"
	command="make PREFIX=${THIRD_PARTY} install"
	message $command
	$command
	cd $BASE_DIR

fi
check_requirements ${THIRD_PARTY}/lib/libopenblas.so



if [ ! -d ${SCONS_LOCAL_DIR} ]; then
	#out file
	message "download Scons local"
	download_extract ${SCONS_LOCAL_URL} ${SCONS_LOCAL_DIR}
fi
check_requirements ${SCONS_LOCAL_DIR}/scons.py


if [ ! -d "${ARMCOMPUTE_DIR}" ]; then 
	message "download ArmCompute Source" 
	git_check ${ARMCOMPUTE_GIT_URL} "${ARMCOMPUTE_DIR}" "tags/${ARMCOMPUTE_TAG}" 
fi

#build armcompute
if [ ! -f "${ARMCOMPUTE_DIR}/build/libarm_compute-static.a" ]; then
message "build arm compute"
cd ${ARMCOMPUTE_DIR}
command="CC=gcc CXX=g++ python3 ${SCONS_LOCAL_DIR}/scons.py Werror=1 -j$(nproc) toolchain_prefix=${RPI_BIN}- debug=${ARMCOMPUTE_DEBUG}  neon=1 opencl=0 extra_cxx_flags=-fPIC os=linux build=cross_compile arch=${ARMCOMPUTE_TARGET} &>/dev/null"
message $command
eval $command
cd ${BASE_DIR} 
fi
check_requirements "${ARMCOMPUTE_DIR}/build/libarm_compute-static.a" "${ARMCOMPUTE_DIR}/build/libarm_compute_core-static.a"



message "build cmake for LIBND4J. output: ${LIBND4J_BUILD_DIR}"

TOOLCHAIN=${LIBND4J_SRC_DIR}/cmake/rpi.cmake
cmake_cmd="${CMAKE}  -G \"Unix Makefiles\"  -B${LIBND4J_BUILD_DIR} -S${LIBND4J_SRC_DIR}  -DCMAKE_BUILD_TYPE=${LIBND4J_BUILD_MODE} -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN} -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DSD_ALL_OPS=true  -DSD_CPU=true -DSD_LIBRARY_NAME=nd4jcpu -DSD_BUILD_TESTS=ON -DSD_ARM_BUILD=true -DOPENBLAS_PATH=${THIRD_PARTY} -DSD_ARCH=${TARGET} -DARMCOMPUTE_ROOT=${ARMCOMPUTE_DIR} -DHELPERS_armcompute=${HAS_ARMCOMPUTE}"
message $cmake_cmd
eval $cmake_cmd

#build
message "lets build"

cd ${LIBND4J_BUILD_DIR}
make -j $(nproc)






