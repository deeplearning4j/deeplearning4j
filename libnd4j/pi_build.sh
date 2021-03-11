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

function message {
	echo "BUILDER:::: ${@}"
}
if [ -z "${BUILD_USING_MAVEN}" ]; then export BUILD_USING_MAVEN=; fi
if [ -z "${CURRENT_TARGET}" ]; then export  CURRENT_TARGET=arm32; fi
if [ -z "${ARMCOMPUTE_DEBUG}" ]; then export  ARMCOMPUTE_DEBUG=1; fi
if [ -z "${HAS_ARMCOMPUTE}" ]; then export  HAS_ARMCOMPUTE=1; fi
if [ -z "${ARMCOMPUTE_TAG}" ]; then export  ARMCOMPUTE_TAG=v20.05; fi
if [ -z "${LIBND4J_BUILD_MODE}" ]; then export  LIBND4J_BUILD_MODE=Release; fi
if [ -z "${ANDROID_VERSION}" ]; then export  ANDROID_VERSION=21; fi
if [ -z "${CUDA_VER}" ]; then export  CUDA_VER=10.2; fi

OTHER_ARGS=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -a|--arch)
    CURRENT_TARGET="$2"
    shift
    shift
    ;;
    -m|--mvn)
    BUILD_USING_MAVEN="mvn"
    shift
    ;;
    *)
    OTHER_ARGS+=("$1")
    shift
    ;;
esac
done

CC_URL32="https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz?revision=e09a1c45-0ed3-4a8e-b06b-db3978fd8d56&la=en&hash=93ED4444B8B3A812B893373B490B90BBB28FD2E3"
CC_URL64="https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz?revision=2e88a73f-d233-4f96-b1f4-d8b36e9bb0b9&la=en&hash=167687FADA00B73D20EED2A67D0939A197504ACD"
CC_ANDROID="https://dl.google.com/android/repository/android-ndk-r21d-linux-x86_64.zip"
CC_L4T64="https://developer.nvidia.com/embedded/dlc/l4t-gcc-7-3-1-toolchain-64-bit"
TARGET_ARRS=( arm32 arm64 android-arm android-arm64 android-x86 android-x86_64 jetson_arm64)
COMPILER_ARRS=( "${CC_URL32}" "${CC_URL64}" "${CC_ANDROID}" "${CC_ANDROID}" "${CC_ANDROID}" "${CC_ANDROID}" "${CC_L4T64}" )
COMPILER_DOWNLOAD_CMD_LIST=( download_extract_xz download_extract_xz download_extract_unzip download_extract_unzip download_extract_unzip download_extract_unzip download_extract_xz)
COMPILER_DESTDIR=( "arm32" "arm64" "android" "android" "android" "android" "l4t" )

OPENBLAS_TARGETS=( ARMV7 ARMV8 ARMV7 ARMV8 ATOM ATOM ARMV8)
ARMCOMPUTE_TARGETS=( armv7a arm64-v8a armv7a arm64-v8a None None None)
OS_LIST=( linux linux android android android android linux)
LIBND4J_PLATFORM_EXT_LIST=( armhf arm64 arm arm64 x86 x86_64 arm64)
PREFIXES=( arm-linux-gnueabihf aarch64-linux-gnu arm-linux-androideabi aarch64-linux-android  i686-linux-android x86_64-linux-android aarch64-linux-gnu)
TARGET_INDEX=-1

for i in "${!TARGET_ARRS[@]}"; do
   if [[ "${TARGET_ARRS[$i]}" = "${CURRENT_TARGET}" ]]; then
       TARGET_INDEX=${i}
   fi
done

if [ ${TARGET_INDEX} -eq -1 ];then
	message "could not find  ${CURRENT_TARGET} in ${TARGET_ARRS[@]}"
	exit -1
fi

#BASE_DIR=${HOME}/pi
#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
BASE_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

export CROSS_COMPILER_URL=${COMPILER_ARRS[$TARGET_INDEX]}
export CROSS_COMPILER_DIR=${BASE_DIR}/compile_tools/cross_compiler_${COMPILER_DESTDIR[$TARGET_INDEX]}
export COMPILER_DOWNLOAD_CMD=${COMPILER_DOWNLOAD_CMD_LIST[$TARGET_INDEX]}
export DETECT=${DETECT_LIST[$TARGET_INDEX]}
export LIBND4J_PLATFORM_EXT=${LIBND4J_PLATFORM_EXT_LIST[$TARGET_INDEX]}
export BLAS_TARGET_NAME=${OPENBLAS_TARGETS[$TARGET_INDEX]}
export ARMCOMPUTE_TARGET=${ARMCOMPUTE_TARGETS[$TARGET_INDEX]}
export TARGET_OS=${OS_LIST[$TARGET_INDEX]}
export LIBND4J_PLATFORM=${TARGET_OS}-${LIBND4J_PLATFORM_EXT}
export PREFIX=${PREFIXES[$TARGET_INDEX]}

export CMAKE=cmake #/snap/bin/cmake
mkdir -p ${BASE_DIR}/compile_tools/

SCONS_LOCAL_URL=http://prdownloads.sourceforge.net/scons/scons-local-3.1.1.tar.gz
SCONS_LOCAL_DIR=${BASE_DIR}/compile_tools/scons_local

THIRD_PARTY=${BASE_DIR}/third_party_libs${TARGET_INDEX}

ARMCOMPUTE_GIT_URL=https://github.com/ARM-software/ComputeLibrary.git
ARMCOMPUTE_DIR=${THIRD_PARTY}/arm_compute_dir

OPENBLAS_GIT_URL="https://github.com/xianyi/OpenBLAS.git"
OPENBLAS_DIR=${THIRD_PARTY}/OpenBLAS


mkdir -p ${BASE_DIR}
mkdir -p ${THIRD_PARTY}

#change directory to base
cd $BASE_DIR

function check_requirements {
	for i in "${@}"
	do
      if [ ! -e "$i" ]; then
         message "missing: ${i}"
		 exit -2
	  fi
	done
}

function rename_top_folder {
	for dir in ${1}/*
	do
		if [ -d "$dir" ]
		then
		    mv "${dir}" "${1}/folder/"
			message "${dir} => ${1}/folder/"
			break
		fi
	done
}

function download_extract_base {
	#$1 is extract arg, $2 is url, $3 is dir
	xtract_arg=${1}
	down_url=${2}
	down_dir=$(dirname "${3}/__")
	down_file="${down_dir}_file"
	if [ ! -f ${down_file} ]; then
		message "download ${down_url}"
		wget --quiet --show-progress -O ${down_file} ${down_url}
	fi
 
	message "extract $@"
    #extract
	mkdir -p ${down_dir} 
	if [ ${xtract_arg} = "-unzip" ]; then
		command="unzip -qq ${down_file} -d ${down_dir} "
	else
		command="tar ${xtract_arg} ${down_file} --directory=${down_dir} "
	fi
	message $command
	$command
	check_requirements "${down_dir}"
}

function download_extract {
	download_extract_base -xzf $@
}

function download_extract_xz {
	download_extract_base -xf $@
}

function download_extract_unzip {
	download_extract_base -unzip $@
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

#fix py debug linkage manually and also makes it use gold
function fix_pi_linker {
  #$1 BINUTILS folder
  if [ ! -f ${1}/ld.original ]; then
    mv ${1}/ld ${1}/ld.original
  fi
  rm -f ${1}/ld
  printf '#!/usr/bin/env bash\n'"${1}/ld.gold --long-plt \$*">${1}/ld
  chmod +x ${1}/ld 
}

function cuda_cross_setup {
		# $1 is local  cuda toolkit version
		# $2 the folder where cross cuda toolkit will be
		loc_VER=${1-10.2}
		loc_DIR=${2-/tmp}
		CUDA_DIR=${CUDA_DIR-/usr/local/cuda-${loc_VER}}
		CUDA_TARGET_STUBS=${loc_DIR}/target_stub${loc_VER}
		CUDA_TARGET_STUB_URL=https://github.com/quickwritereader/temp_jars/releases/download/1.2/aarch64_linux_cuda_${loc_VER}_cudnn8.tar.gz
		download_extract ${CUDA_TARGET_STUB_URL} ${CUDA_TARGET_STUBS}
		message "lets setup cuda toolkit by combining local cuda-${loc_VER} and target ${CUDA_TARGET_STUBS}"
		message "cuda cross folder: ${loc_DIR}"
		check_requirements ${CUDA_DIR} ${CUDA_TARGET_STUBS}/aarch64-linux/
		if [ ! -d ${loc_DIR}/cuda/bin ];then
			mkdir -p ${loc_DIR}/cuda/bin
			cd ${CUDA_DIR}/bin/
			# we are obliged to symlink inner files and folders to avoid errors happening 
			# when  relative folders are used from symlinks 
			for i in $(find . -maxdepth 1)
			do 
				ln -s ${CUDA_DIR}/bin/${i} ${loc_DIR}/cuda/bin/${i}
			done
			cd -
			ln -s ${CUDA_DIR}/nvvm ${loc_DIR}/cuda/nvvm
			ln -s ${CUDA_TARGET_STUBS} ${loc_DIR}/cuda/targets
		fi
		export CUDACXX=${loc_DIR}/cuda/bin/nvcc
		export CUDNN_ROOT_DIR=${CUDA_TARGET_STUBS}/aarch64-linux
		export CUDA_TOOLKIT_ROOT=${loc_DIR}/cuda
}

if [ ! -d ${CROSS_COMPILER_DIR}/folder ]; then
	#out file
	message "download CROSS_COMPILER"
	${COMPILER_DOWNLOAD_CMD} ${CROSS_COMPILER_URL} ${CROSS_COMPILER_DIR}
	message "rename top folder  (instead of --strip-components=1)"
	rename_top_folder ${CROSS_COMPILER_DIR}
fi

export CROSS_COMPILER_DIR=${CROSS_COMPILER_DIR}/folder

if [ "${TARGET_OS}" = "android" ];then
	export ANDROID_TOOLCHAIN=${CROSS_COMPILER_DIR}/toolchains/llvm/prebuilt/linux-x86_64
	export COMPILER_PREFIX="${ANDROID_TOOLCHAIN}/bin/${PREFIX}${ANDROID_VERSION}"
	export TOOLCHAIN_PREFIX="${ANDROID_TOOLCHAIN}/bin/${PREFIX}"
	if [ "$BLAS_TARGET_NAME" = "ARMV7" ];then
	    BLAS_XTRA="ARM_SOFTFP_ABI=1 "
		COMPILER_PREFIX="${ANDROID_TOOLCHAIN}/bin/armv7a-linux-androideabi${ANDROID_VERSION}"
	fi
	export CC_EXE="clang"
	export CXX_EXE="clang++"
	export AR="${TOOLCHAIN_PREFIX}-ar"
	export RANLIB="${TOOLCHAIN_PREFIX}-ranlib"
	export BLAS_XTRA="CC=${COMPILER_PREFIX}-${CC_EXE} AR=${AR} RANLIB=${RANLIB} ${BLAS_XTRA}"
else
	export BINUTILS_BIN=${CROSS_COMPILER_DIR}/${PREFIX}/bin
	export COMPILER_PREFIX=${CROSS_COMPILER_DIR}/bin/${PREFIX}
	export TOOLCHAIN_PREFIX=${COMPILER_PREFIX}
	export SYS_ROOT=${CROSS_COMPILER_DIR}/${PREFIX}/libc
	#LD_LIBRARY_PATH=${CROSS_COMPILER_DIR}/lib:$LD_LIBRARY_PATH
	export CC_EXE="gcc"
	export CXX_EXE="g++"
	export RANLIB="${BINUTILS_BIN}/ranlib"
	export LD="${BINUTILS_BIN}/ld"
	export AR="${BINUTILS_BIN}/ar"
	export BLAS_XTRA="CC=${COMPILER_PREFIX}-${CC_EXE} AR=${AR} RANLIB=${RANLIB}  CFLAGS=--sysroot=${SYS_ROOT} LDFLAGS=\"-L${SYS_ROOT}/../lib/ -lm\""
fi

check_requirements ${CC}

if [ -z "${BUILD_USING_MAVEN}" ] ;then
#lets build OpenBlas 
if [ ! -d "${OPENBLAS_DIR}" ]; then
	message "download OpenBLAS"
	git_check "${OPENBLAS_GIT_URL}" "${OPENBLAS_DIR}" "v0.3.10"
fi

if [ ! -f "${THIRD_PARTY}/lib/libopenblas.so" ]; then
	message "build and install OpenBLAS" 
	cd ${OPENBLAS_DIR}

	command="make TARGET=${BLAS_TARGET_NAME} HOSTCC=gcc  NOFORTRAN=1 ${BLAS_XTRA} "
	message $command
	eval $command  &>/dev/null
    message "install it"
	command="make TARGET=${BLAS_TARGET_NAME} PREFIX=${THIRD_PARTY} install &>/dev/null"
	message $command
	$command
	cd $BASE_DIR

fi
check_requirements ${THIRD_PARTY}/lib/libopenblas.so

export OPENBLAS_PATH=${THIRD_PARTY}

fi # end if [ -z "${BUILD_USING_MAVEN}"];then


XTRA_ARGS=""
XTRA_MVN_ARGS=""

if [ "${ARMCOMPUTE_TARGET}" != "None" ];then
message "~~~ARMCOMPUTE~~~"
XTRA_ARGS=" -h armcompute "
XTRA_MVN_ARGS=" -Dlibnd4j.helper=armcompute "

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
command="CC=${CC_EXE} CXX=${CXX_EXE} python3 ${SCONS_LOCAL_DIR}/scons.py Werror=1 -j$(nproc) toolchain_prefix=${TOOLCHAIN_PREFIX}-  compiler_prefix=${COMPILER_PREFIX}- debug=${ARMCOMPUTE_DEBUG}  neon=1 opencl=0 extra_cxx_flags=-fPIC os=${TARGET_OS} build=cross_compile arch=${ARMCOMPUTE_TARGET} "
message $command
eval $command &>/dev/null
cd ${BASE_DIR} 
fi
check_requirements "${ARMCOMPUTE_DIR}/build/libarm_compute-static.a" "${ARMCOMPUTE_DIR}/build/libarm_compute_core-static.a"

export ARMCOMPUTE_ROOT="${ARMCOMPUTE_DIR}"

fi #if armcompute


if [ "${TARGET_OS}" = "android" ];then
	export ANDROID_NDK=${CROSS_COMPILER_DIR}
	XTRA_MVN_ARGS="${XTRA_MVN_ARGS} -pl \":libnd4j,:nd4j-native\" "
else
	 if [ "${CURRENT_TARGET}" == "jetson_arm64" ];then
	 	message  "jetson cuda build "
		cuda_cross_setup ${CUDA_VER}
		XTRA_ARGS="${XTRA_ARGS} -c cuda  -h cudnn  "
		XTRA_MVN_ARGS="${XTRA_MVN_ARGS} -Djavacpp.version=1.5.3 -Dcuda.version=${CUDA_VER} -Dlibnd4j.cuda=${CUDA_VER} -Dlibnd4j.chip=cuda -Dlibnd4j.compute=5.3 "
		XTRA_MVN_ARGS="${XTRA_MVN_ARGS}  -Dlibnd4j.helper=cudnn  "
		export SYSROOT=${CROSS_COMPILER_DIR}/${PREFIX}/libc
	else
		XTRA_MVN_ARGS="${XTRA_MVN_ARGS} -pl \":libnd4j,:nd4j-native\" "
	fi
	export RPI_BIN=${CROSS_COMPILER_DIR}/bin/${PREFIX}
	export JAVA_LIBRARY_PATH=${CROSS_COMPILER_DIR}/${PREFIX}/lib
	fix_pi_linker ${BINUTILS_BIN}
fi


#because of the toolchain passive detection we have to delete build folder manually
detect=$(cat ${BASE_DIR}/blasbuild/cpu/CMakeCache.txt | grep -o ${PREFIX})
if [ -z "${detect}" ] ;then
message "remove blasbuild folder "
rm -rf $BASE_DIR/blasbuild/
else
message "keep blasbuild folder"
fi

if [ -z "${BUILD_USING_MAVEN}" ] ;then
message "lets build just library"
bash ./buildnativeoperations.sh -o ${LIBND4J_PLATFORM} -t -j $(nproc) ${XTRA_ARGS} 
else
message "cd $BASE_DIR/.. "
cd $BASE_DIR/..
message "lets build jars"
if [ "${DEPLOY-}" != "" ]; then
  message "Deploying to maven"
  mvn  -P"${PUBLISH_TO}" deploy  --batch-mode  -Dlibnd4j.platform=${LIBND4J_PLATFORM} -Djavacpp.platform=${LIBND4J_PLATFORM} ${XTRA_MVN_ARGS} -DprotocCommand=protoc -Djavacpp.platform.compiler=${COMPILER_PREFIX}-${CC_EXE} -Djava.library.path=${JAVA_LIBRARY_PATH}  --also-make -DskipTests -Dmaven.test.skip=true -Dmaven.javadoc.skip=true
else
  message "Installing to local repo"
  mvn  install  -Dlibnd4j.platform=${LIBND4J_PLATFORM} -Djavacpp.platform=${LIBND4J_PLATFORM}  ${XTRA_MVN_ARGS}  -DprotocCommand=protoc -Djavacpp.platform.compiler=${COMPILER_PREFIX}-${CC_EXE} -Djava.library.path=${JAVA_LIBRARY_PATH}  --also-make -DskipTests -Dmaven.test.skip=true -Dmaven.javadoc.skip=true
fi

fi
