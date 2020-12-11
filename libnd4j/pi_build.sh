#!/usr/bin/env bash

BUILD_USING_MAVEN=
if [ "$1" = "maven" ]  ||  [ "$1" = "mvn" ]; then
	BUILD_USING_MAVEN="maven"
fi

TARGET=armv7-a
BLAS_TARGET_NAME=ARMV7
ARMCOMPUTE_TARGET=armv7a
#BASE_DIR=${HOME}/pi
#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SOURCE="${BASH_SOURCE[0]}"
ARMCOMPUTE_DEBUG=1
LIBND4J_BUILD_MODE=Debug
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
BASE_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
CMAKE=cmake #/snap/bin/cmake

mkdir -p ${BASE_DIR}/helper_bin/


CROSS_COMPILER_URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz?revision=e09a1c45-0ed3-4a8e-b06b-db3978fd8d56&la=en&hash=93ED4444B8B3A812B893373B490B90BBB28FD2E3"
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

PREFIX=arm-linux-gnueabihf
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



function download_extract_base {
	#$1 is url #2 is dir $3 is extract argument
	if [ ! -f ${3}_file ]; then
		message "download"
		wget --quiet --show-progress -O ${3}_file ${2}
	fi
 
	message "extract $@"
    #extract
	mkdir -p ${3} 
	command="tar ${1}  ${3}_file --directory=${3} ${4} "
	message $command
	$command

	check_requirements "${3}"
}

function download_extract {
	download_extract_base -xzf $@ 
}

function download_extract_xz {
	download_extract_base -xf $@ 
}


function git_check {
	#$1 is url #$2 is dir #$3 is tag or branch if optional
	command=
	if [ -n "$3" ]; then
	    command="git clone --quiet --depth 1 --branch ${3} ${1} ${2}"	
	else 
	command="git clone --quiet ${1} ${2}"
	fi
	message "$command"
	$command 
	check_requirements "${2}"
}


if [ ! -d ${CROSS_COMPILER_DIR} ]; then
	#out file
	message "download CROSS_COMPILER"
	download_extract_xz ${CROSS_COMPILER_URL} ${CROSS_COMPILER_DIR} ${XRTACT_STRIP} 
fi


export PI_FOLDER=${CROSS_COMPILER_DIR}
export PI_SYS_ROOT=${PI_FOLDER}/${PREFIX}/libc
export BINUTILS_BIN=${PI_FOLDER}/${PREFIX}/bin
export RPI_BIN_PREFIX=${PI_FOLDER}/bin/arm-linux-gnueabihf
export PI_SYS_ROOT=${PI_FOLDER}/arm-linux-gnueabihf/libc
export LD_LIBRARY_PATH=${PI_FOLDER}/lib:$LD_LIBRARY_PATH
export CC=${RPI_BIN_PREFIX}-gcc
export FC=${RPI_BIN_PREFIX}-gfortran
export CXX=${RPI_BIN_PREFIX}-g++
export CPP=${RPI_BIN_PREFIX}-cpp
export RANLIB="${BINUTILS_BIN}/ranlib"
export LD="${BINUTILS_BIN}/ld"
export AR="${BINUTILS_BIN}/ar"

check_requirements ${RPI_BIN_PREFIX}-gcc

if [ -z "${BUILD_USING_MAVEN}" ] ;then
#lets build OpenBlas 
if [ ! -d "${OPENBLAS_DIR}" ]; then 
	message "download OpenBLAS"
	git_check "${OPENBLAS_GIT_URL}" "${OPENBLAS_DIR}" "v0.3.10"
fi

if [ ! -f "${THIRD_PARTY}/lib/libopenblas.so" ]; then
	message "build and install OpenBLAS" 
	cd ${OPENBLAS_DIR}

	command="make TARGET=${BLAS_TARGET_NAME} HOSTCC=gcc CC=${CC} USE_THREAD=0 NOFORTRAN=1 CFLAGS=--sysroot=${PI_SYS_ROOT} LDFLAGS=\"-L${PI_SYS_ROOT}/../lib/ -lm\" "
	message $command
	eval $command 
    message "install it"
	command="make  TARGET=${BLAS_TARGET_NAME} PREFIX=${THIRD_PARTY} install"
	message $command
	$command
	cd $BASE_DIR

fi
check_requirements ${THIRD_PARTY}/lib/libopenblas.so
export OPENBLAS_PATH=${THIRD_PARTY}

fi # end if [ -z "${BUILD_USING_MAVEN}"];then

if [ ! -d ${SCONS_LOCAL_DIR} ]; then
	#out file
	message "download Scons local"
	download_extract ${SCONS_LOCAL_URL} ${SCONS_LOCAL_DIR}
fi
check_requirements ${SCONS_LOCAL_DIR}/scons.py


if [ ! -d "${ARMCOMPUTE_DIR}" ]; then 
	message "download ArmCompute Source" 
	git_check ${ARMCOMPUTE_GIT_URL} "${ARMCOMPUTE_DIR}" "${ARMCOMPUTE_TAG}" 
fi

#build armcompute
if [ ! -f "${ARMCOMPUTE_DIR}/build/libarm_compute-static.a" ]; then
message "build arm compute"
cd ${ARMCOMPUTE_DIR}
command="CC=gcc CXX=g++ python3 ${SCONS_LOCAL_DIR}/scons.py Werror=1 -j$(nproc) toolchain_prefix=${RPI_BIN_PREFIX}- debug=${ARMCOMPUTE_DEBUG}  neon=1 opencl=0 extra_cxx_flags=-fPIC os=linux build=cross_compile arch=${ARMCOMPUTE_TARGET} &>/dev/null"
message $command
eval $command
cd ${BASE_DIR} 
fi
check_requirements "${ARMCOMPUTE_DIR}/build/libarm_compute-static.a" "${ARMCOMPUTE_DIR}/build/libarm_compute_core-static.a"

export ARMCOMPUTE_ROOT="${ARMCOMPUTE_DIR}"
export OS_PLATFORM=linux-armhf 

if [ -z "${BUILD_USING_MAVEN}" ] ;then
message "lets build just library"
bash ./buildnativeoperations.sh -o ${OS_PLATFORM} -t -h armcompute -V -j $(nproc)
else
cd ..  
export JAVA_LIBRARY_PATH=${PI_FOLDER}/${PREFIX}/lib 
message "lets build jars"
mvn  install  -Dlibnd4j.platform=${OS_PLATFORM} -Dlibnd4j.helper=armcompute  -Djavacpp.platform=${OS_PLATFORM} -DprotocCommand=protoc -Djavacpp.platform.compiler=${CXX} -Djava.library.path=${JAVA_LIBRARY_PATH} -Dmaven.test.skip=true -Dmaven.javadoc.skip=true
fi
 