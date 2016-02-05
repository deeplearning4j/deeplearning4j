##
## Author: Hank Anderson <hank@statease.com>
## Description: Ported from OpenBLAS/Makefile.system
##

set(NETLIB_LAPACK_DIR "${CMAKE_SOURCE_DIR}/lapack-netlib")

# TODO: Makefile.system detects Darwin (mac) and switches to clang here -hpa
# http://stackoverflow.com/questions/714100/os-detecting-makefile

# TODO: Makefile.system sets HOSTCC = $(CC) here if not already set -hpa

# TARGET_CORE will override TARGET which is used in DYNAMIC_ARCH=1.
if (DEFINED TARGET_CORE)
  set(TARGET ${TARGET_CORE})
endif ()

# Force fallbacks for 32bit
if (DEFINED BINARY AND DEFINED TARGET AND BINARY EQUAL 32)
  message(STATUS "Compiling a ${BINARY}-bit binary.")
  set(NO_AVX 1)
  if (${TARGET} STREQUAL "HASWELL" OR ${TARGET} STREQUAL "SANDYBRIDGE")
    set(TARGET "NEHALEM")
  endif ()
  if (${TARGET} STREQUAL "BULLDOZER" OR ${TARGET} STREQUAL "PILEDRIVER")
    set(TARGET "BARCELONA")
  endif ()
endif ()

if (DEFINED TARGET)
  message(STATUS "Targetting the ${TARGET} architecture.")
  set(GETARCH_FLAGS "-DFORCE_${TARGET}")
endif ()

if (INTERFACE64)
  message(STATUS "Using 64-bit integers.")
  set(GETARCH_FLAGS	"${GETARCH_FLAGS} -DUSE64BITINT")
endif ()

if (NOT DEFINED GEMM_MULTITHREAD_THRESHOLD)
  set(GEMM_MULTITHREAD_THRESHOLD 4)
endif ()
message(STATUS "GEMM multithread threshold set to ${GEMM_MULTITHREAD_THRESHOLD}.")
set(GETARCH_FLAGS	"${GETARCH_FLAGS} -DGEMM_MULTITHREAD_THRESHOLD=${GEMM_MULTITHREAD_THRESHOLD}")

if (NO_AVX)
  message(STATUS "Disabling Advanced Vector Extensions (AVX).")
  set(GETARCH_FLAGS "${GETARCH_FLAGS} -DNO_AVX")
endif ()

if (NO_AVX2)
  message(STATUS "Disabling Advanced Vector Extensions 2 (AVX2).")
  set(GETARCH_FLAGS "${GETARCH_FLAGS} -DNO_AVX2")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
  set(GETARCH_FLAGS "${GETARCH_FLAGS} -g")
endif ()

# TODO: let CMake handle this? -hpa
#if (${QUIET_MAKE})
#  set(MAKE "${MAKE} -s")
#endif()

if (NOT DEFINED NO_PARALLEL_MAKE)
  set(NO_PARALLEL_MAKE 0)
endif ()
set(GETARCH_FLAGS	"${GETARCH_FLAGS} -DNO_PARALLEL_MAKE=${NO_PARALLEL_MAKE}")

if (CMAKE_CXX_COMPILER STREQUAL loongcc)
  set(GETARCH_FLAGS	"${GETARCH_FLAGS} -static")
endif ()

#if don't use Fortran, it will only compile CBLAS.
if (ONLY_CBLAS)
  set(NO_LAPACK 1)
else ()
  set(ONLY_CBLAS 0)
endif ()

include("${CMAKE_SOURCE_DIR}/cmake/prebuild.cmake")

if (NOT DEFINED NUM_THREADS)
  set(NUM_THREADS ${NUM_CORES})
endif ()

if (${NUM_THREADS} EQUAL 1)
  set(USE_THREAD 0)
endif ()

if (DEFINED USE_THREAD)
  if (NOT ${USE_THREAD})
    unset(SMP)
  else ()
    set(SMP 1)
  endif ()
else ()
  # N.B. this is NUM_THREAD in Makefile.system which is probably a bug -hpa
  if (${NUM_THREADS} EQUAL 1)
    unset(SMP)
  else ()
    set(SMP 1)
  endif ()
endif ()

if (${SMP})
  message(STATUS "SMP enabled.")
endif ()

if (NOT DEFINED NEED_PIC)
  set(NEED_PIC 1)
endif ()

# TODO: I think CMake should be handling all this stuff -hpa
unset(ARFLAGS)
set(CPP "${COMPILER} -E")
set(AR "${CROSS_SUFFIX}ar")
set(AS "${CROSS_SUFFIX}as")
set(LD "${CROSS_SUFFIX}ld")
set(RANLIB "${CROSS_SUFFIX}ranlib")
set(NM "${CROSS_SUFFIX}nm")
set(DLLWRAP "${CROSS_SUFFIX}dllwrap")
set(OBJCOPY "${CROSS_SUFFIX}objcopy")
set(OBJCONV "${CROSS_SUFFIX}objconv")

# OS dependent settings
include("${CMAKE_SOURCE_DIR}/cmake/os.cmake")

# Architecture dependent settings
include("${CMAKE_SOURCE_DIR}/cmake/arch.cmake")

# C Compiler dependent settings
include("${CMAKE_SOURCE_DIR}/cmake/cc.cmake")

if (NOT NOFORTRAN)
  # Fortran Compiler dependent settings
  include("${CMAKE_SOURCE_DIR}/cmake/fc.cmake")
endif ()

if (BINARY64)
  if (INTERFACE64)
    # CCOMMON_OPT += -DUSE64BITINT
  endif ()
endif ()

if (NEED_PIC)
  if (${CMAKE_C_COMPILER} STREQUAL "IBM")
    set(CCOMMON_OPT "${CCOMMON_OPT} -qpic=large")
  else ()
    set(CCOMMON_OPT "${CCOMMON_OPT} -fPIC")
  endif ()

  if (${F_COMPILER} STREQUAL "SUN")
    set(FCOMMON_OPT "${FCOMMON_OPT} -pic")
  else ()
    set(FCOMMON_OPT "${FCOMMON_OPT} -fPIC")
  endif ()
endif ()

if (DYNAMIC_ARCH)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DDYNAMIC_ARCH")
endif ()

if (NO_LAPACK)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DNO_LAPACK")
  #Disable LAPACK C interface
  set(NO_LAPACKE 1)
endif ()

if (NO_LAPACKE)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DNO_LAPACKE")
endif ()

if (NO_AVX)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DNO_AVX")
endif ()

if (${ARCH} STREQUAL "x86")
  set(CCOMMON_OPT "${CCOMMON_OPT} -DNO_AVX")
endif ()

if (NO_AVX2)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DNO_AVX2")
endif ()

if (SMP)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DSMP_SERVER")

  if (${ARCH} STREQUAL "mips64")
    if (NOT ${CORE} STREQUAL "LOONGSON3B")
      set(USE_SIMPLE_THREADED_LEVEL3 1)
    endif ()
  endif ()

  if (USE_OPENMP)
    # USE_SIMPLE_THREADED_LEVEL3 = 1
    # NO_AFFINITY = 1
    set(CCOMMON_OPT "${CCOMMON_OPT} -DUSE_OPENMP")
  endif ()

  if (BIGNUMA)
    set(CCOMMON_OPT "${CCOMMON_OPT} -DBIGNUMA")
  endif ()

endif ()

if (NO_WARMUP)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DNO_WARMUP")
endif ()

if (CONSISTENT_FPCSR)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DCONSISTENT_FPCSR")
endif ()

# Only for development
# set(CCOMMON_OPT "${CCOMMON_OPT} -DPARAMTEST")
# set(CCOMMON_OPT "${CCOMMON_OPT} -DPREFETCHTEST")
# set(CCOMMON_OPT "${CCOMMON_OPT} -DNO_SWITCHING")
# set(USE_PAPI 1)

if (USE_PAPI)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DUSE_PAPI")
  set(EXTRALIB "${EXTRALIB} -lpapi -lperfctr")
endif ()

if (DYNAMIC_THREADS)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DDYNAMIC_THREADS")
endif ()

set(CCOMMON_OPT "${CCOMMON_OPT} -DMAX_CPU_NUMBER=${NUM_THREADS}")

if (USE_SIMPLE_THREADED_LEVEL3)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DUSE_SIMPLE_THREADED_LEVEL3")
endif ()

if (DEFINED LIBNAMESUFFIX)
  set(LIBPREFIX "libopenblas_${LIBNAMESUFFIX}")
else ()
  set(LIBPREFIX "libopenblas")
endif ()

if (NOT DEFINED SYMBOLPREFIX)
  set(SYMBOLPREFIX "")
endif ()

if (NOT DEFINED SYMBOLSUFFIX)
  set(SYMBOLSUFFIX "")
endif ()

set(KERNELDIR	"${CMAKE_SOURCE_DIR}/kernel/${ARCH}")

# TODO: nead to convert these Makefiles
# include ${CMAKE_SOURCE_DIR}/cmake/${ARCH}.cmake

if (${CORE} STREQUAL "PPC440")
  set(CCOMMON_OPT "${CCOMMON_OPT} -DALLOC_QALLOC")
endif ()

if (${CORE} STREQUAL "PPC440FP2")
  set(STATIC_ALLOCATION 1)
endif ()

if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(NO_AFFINITY 1)
endif ()

if (NOT ${ARCH} STREQUAL "x86_64" AND NOT ${ARCH} STREQUAL "x86" AND NOT ${CORE} STREQUAL "LOONGSON3B")
  set(NO_AFFINITY 1)
endif ()

if (NO_AFFINITY)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DNO_AFFINITY")
endif ()

if (FUNCTION_PROFILE)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DFUNCTION_PROFILE")
endif ()

if (HUGETLB_ALLOCATION)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DALLOC_HUGETLB")
endif ()

if (DEFINED HUGETLBFILE_ALLOCATION)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DALLOC_HUGETLBFILE -DHUGETLB_FILE_NAME=${HUGETLBFILE_ALLOCATION})")
endif ()

if (STATIC_ALLOCATION)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DALLOC_STATIC")
endif ()

if (DEVICEDRIVER_ALLOCATION)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DALLOC_DEVICEDRIVER -DDEVICEDRIVER_NAME=\"/dev/mapper\"")
endif ()

if (MIXED_MEMORY_ALLOCATION)
  set(CCOMMON_OPT "${CCOMMON_OPT} -DMIXED_MEMORY_ALLOCATION")
endif ()

if (${CMAKE_SYSTEM_NAME} STREQUAL "SunOS")
  set(TAR	gtar)
  set(PATCH	gpatch)
  set(GREP ggrep)
else ()
  set(TAR tar)
  set(PATCH patch)
  set(GREP grep)
endif ()

if (NOT DEFINED MD5SUM)
  set(MD5SUM md5sum)
endif ()

set(AWK awk)

set(REVISION "-r${OpenBLAS_VERSION}")
set(MAJOR_VERSION ${OpenBLAS_MAJOR_VERSION})

if (DEBUG)
  set(COMMON_OPT "${COMMON_OPT} -g")
endif ()

if (NOT DEFINED COMMON_OPT)
  set(COMMON_OPT "-O2")
endif ()

#For x86 32-bit
if (DEFINED BINARY AND BINARY EQUAL 32)
if (NOT MSVC)
  set(COMMON_OPT "${COMMON_OPT} -m32")
endif()
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_OPT} ${CCOMMON_OPT}")
if(NOT MSVC)
set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${COMMON_OPT} ${CCOMMON_OPT}")
endif()
# TODO: not sure what PFLAGS is -hpa
set(PFLAGS "${PFLAGS} ${COMMON_OPT} ${CCOMMON_OPT} -I${TOPDIR} -DPROFILE ${COMMON_PROF}")

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${COMMON_OPT} ${FCOMMON_OPT}")
# TODO: not sure what FPFLAGS is -hpa
set(FPFLAGS "${FPFLAGS} ${COMMON_OPT} ${FCOMMON_OPT} ${COMMON_PROF}")

#For LAPACK Fortran codes.
set(LAPACK_FFLAGS "${LAPACK_FFLAGS} ${CMAKE_Fortran_FLAGS}")
set(LAPACK_FPFLAGS "${LAPACK_FPFLAGS} ${FPFLAGS}")

#Disable -fopenmp for LAPACK Fortran codes on Windows.
if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(FILTER_FLAGS "-fopenmp;-mp;-openmp;-xopenmp=parralel")
  foreach (FILTER_FLAG ${FILTER_FLAGS})
    string(REPLACE ${FILTER_FLAG} "" LAPACK_FFLAGS ${LAPACK_FFLAGS})
    string(REPLACE ${FILTER_FLAG} "" LAPACK_FPFLAGS ${LAPACK_FPFLAGS})
  endforeach ()
endif ()

if ("${F_COMPILER}" STREQUAL "GFORTRAN")
  # lapack-netlib is rife with uninitialized warnings -hpa
  set(LAPACK_FFLAGS "${LAPACK_FFLAGS} -Wno-maybe-uninitialized")
endif ()

set(LAPACK_CFLAGS "${CMAKE_C_CFLAGS} -DHAVE_LAPACK_CONFIG_H")
if (INTERFACE64)
  set(LAPACK_CFLAGS "${LAPACK_CFLAGS} -DLAPACK_ILP64")
endif ()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(LAPACK_CFLAGS "${LAPACK_CFLAGS} -DOPENBLAS_OS_WINDOWS")
endif ()

if (${CMAKE_C_COMPILER} STREQUAL "LSB" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(LAPACK_CFLAGS "${LAPACK_CFLAGS} -DLAPACK_COMPLEX_STRUCTURE")
endif ()

if (NOT DEFINED SUFFIX)
  set(SUFFIX o)
endif ()

if (NOT DEFINED PSUFFIX)
  set(PSUFFIX po)
endif ()

if (NOT DEFINED LIBSUFFIX)
  set(LIBSUFFIX a)
endif ()

if (DYNAMIC_ARCH)
  if (DEFINED SMP)
    set(LIBNAME "${LIBPREFIX}p${REVISION}.${LIBSUFFIX}")
    set(LIBNAME_P	"${LIBPREFIX}p${REVISION}_p.${LIBSUFFIX}")
  else ()
    set(LIBNAME "${LIBPREFIX}${REVISION}.${LIBSUFFIX}")
    set(LIBNAME_P	"${LIBPREFIX}${REVISION}_p.${LIBSUFFIX}")
  endif ()
else ()
  if (DEFINED SMP)
    set(LIBNAME "${LIBPREFIX}_${LIBCORE}p${REVISION}.${LIBSUFFIX}")
    set(LIBNAME_P	"${LIBPREFIX}_${LIBCORE}p${REVISION}_p.${LIBSUFFIX}")
  else ()
    set(LIBNAME	"${LIBPREFIX}_${LIBCORE}${REVISION}.${LIBSUFFIX}")
    set(LIBNAME_P	"${LIBPREFIX}_${LIBCORE}${REVISION}_p.${LIBSUFFIX}")
  endif ()
endif ()


set(LIBDLLNAME "${LIBPREFIX}.dll")
set(LIBSONAME "${LIBNAME}.${LIBSUFFIX}.so")
set(LIBDYNNAME "${LIBNAME}.${LIBSUFFIX}.dylib")
set(LIBDEFNAME "${LIBNAME}.${LIBSUFFIX}.def")
set(LIBEXPNAME "${LIBNAME}.${LIBSUFFIX}.exp")
set(LIBZIPNAME "${LIBNAME}.${LIBSUFFIX}.zip")

set(LIBS "${CMAKE_SOURCE_DIR}/${LIBNAME}")
set(LIBS_P "${CMAKE_SOURCE_DIR}/${LIBNAME_P}")


set(LIB_COMPONENTS BLAS)
if (NOT NO_CBLAS)
  set(LIB_COMPONENTS "${LIB_COMPONENTS} CBLAS")
endif ()

if (NOT NO_LAPACK)
  set(LIB_COMPONENTS "${LIB_COMPONENTS} LAPACK")
  if (NOT NO_LAPACKE)
    set(LIB_COMPONENTS "${LIB_COMPONENTS} LAPACKE")
  endif ()
endif ()

if (ONLY_CBLAS)
  set(LIB_COMPONENTS CBLAS)
endif ()


# For GEMM3M
set(USE_GEMM3M 0)

if (DEFINED ARCH)
  if (${ARCH} STREQUAL "x86" OR ${ARCH} STREQUAL "x86_64" OR ${ARCH} STREQUAL "ia64" OR ${ARCH} STREQUAL "MIPS")
    set(USE_GEMM3M 1)
  endif ()

  if (${CORE} STREQUAL "generic")
    set(USE_GEMM3M 0)
  endif ()
endif ()


#export OSNAME
#export ARCH
#export CORE
#export LIBCORE
#export PGCPATH
#export CONFIG
#export CC
#export FC
#export BU
#export FU
#export NEED2UNDERSCORES
#export USE_THREAD
#export NUM_THREADS
#export NUM_CORES
#export SMP
#export MAKEFILE_RULE
#export NEED_PIC
#export BINARY
#export BINARY32
#export BINARY64
#export F_COMPILER
#export C_COMPILER
#export USE_OPENMP
#export CROSS
#export CROSS_SUFFIX
#export NOFORTRAN
#export NO_FBLAS
#export EXTRALIB
#export CEXTRALIB
#export FEXTRALIB
#export HAVE_SSE
#export HAVE_SSE2
#export HAVE_SSE3
#export HAVE_SSSE3
#export HAVE_SSE4_1
#export HAVE_SSE4_2
#export HAVE_SSE4A
#export HAVE_SSE5
#export HAVE_AVX
#export HAVE_VFP
#export HAVE_VFPV3
#export HAVE_VFPV4
#export HAVE_NEON
#export KERNELDIR
#export FUNCTION_PROFILE
#export TARGET_CORE
#
#export SGEMM_UNROLL_M
#export SGEMM_UNROLL_N
#export DGEMM_UNROLL_M
#export DGEMM_UNROLL_N
#export QGEMM_UNROLL_M
#export QGEMM_UNROLL_N
#export CGEMM_UNROLL_M
#export CGEMM_UNROLL_N
#export ZGEMM_UNROLL_M
#export ZGEMM_UNROLL_N
#export XGEMM_UNROLL_M
#export XGEMM_UNROLL_N
#export CGEMM3M_UNROLL_M
#export CGEMM3M_UNROLL_N
#export ZGEMM3M_UNROLL_M
#export ZGEMM3M_UNROLL_N
#export XGEMM3M_UNROLL_M
#export XGEMM3M_UNROLL_N


#if (USE_CUDA)
#  export CUDADIR
#  export CUCC
#  export CUFLAGS
#  export CULIB
#endif

#.SUFFIXES: .$(PSUFFIX) .$(SUFFIX) .f
#
#.f.$(SUFFIX):
#	$(FC) $(FFLAGS) -c $<  -o $(@F)
#
#.f.$(PSUFFIX):
#	$(FC) $(FPFLAGS) -pg -c $<  -o $(@F)

# these are not cross-platform
#ifdef BINARY64
#PATHSCALEPATH	= /opt/pathscale/lib/3.1
#PGIPATH		= /opt/pgi/linux86-64/7.1-5/lib
#else
#PATHSCALEPATH	= /opt/pathscale/lib/3.1/32
#PGIPATH		= /opt/pgi/linux86/7.1-5/lib
#endif

#ACMLPATH	= /opt/acml/4.3.0
#ifneq ($(OSNAME), Darwin)
#MKLPATH         = /opt/intel/mkl/10.2.2.025/lib
#else
#MKLPATH         = /Library/Frameworks/Intel_MKL.framework/Versions/10.0.1.014/lib
#endif
#ATLASPATH	= /opt/atlas/3.9.17/opteron
#FLAMEPATH	= $(HOME)/flame/lib
#ifneq ($(OSNAME), SunOS)
#SUNPATH		= /opt/sunstudio12.1
#else
#SUNPATH		= /opt/SUNWspro
#endif

