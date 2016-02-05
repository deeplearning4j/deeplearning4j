##
## Author: Hank Anderson <hank@statease.com>
## Description: Ported from the OpenBLAS/c_check perl script.
##              This is triggered by prebuild.cmake and runs before any of the code is built.
##              Creates config.h and Makefile.conf.

# CMake vars set by this file:
# OSNAME (use CMAKE_SYSTEM_NAME)
# ARCH
# C_COMPILER (use CMAKE_C_COMPILER)
# BINARY32
# BINARY64
# FU
# CROSS_SUFFIX
# CROSS
# CEXTRALIB

# Defines set by this file:
# OS_
# ARCH_
# C_
# __32BIT__
# __64BIT__
# FUNDERSCORE
# PTHREAD_CREATE_FUNC

# N.B. c_check (and ctest.c) is not cross-platform, so instead try to use CMake variables.
set(FU "")
if(APPLE)
set(FU "_")
elseif(MSVC)
set(FU "_")
elseif(UNIX)
set(FU "")
endif()

# Convert CMake vars into the format that OpenBLAS expects
string(TOUPPER ${CMAKE_SYSTEM_NAME} HOST_OS)
if (${HOST_OS} STREQUAL "WINDOWS")
  set(HOST_OS WINNT)
endif ()

# added by hpa - check size of void ptr to detect 64-bit compile
if (NOT DEFINED BINARY)
  set(BINARY 32)
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BINARY 64)
  endif ()
endif ()

if (BINARY EQUAL 64)
  set(BINARY64 1)
else ()
  set(BINARY32 1)
endif ()

# CMake docs define these:
# CMAKE_SYSTEM_PROCESSOR - The name of the CPU CMake is building for.
# CMAKE_HOST_SYSTEM_PROCESSOR - The name of the CPU CMake is running on.
#
# TODO: CMAKE_SYSTEM_PROCESSOR doesn't seem to be correct - instead get it from the compiler a la c_check
set(ARCH ${CMAKE_SYSTEM_PROCESSOR})
if (${ARCH} STREQUAL "AMD64")
  set(ARCH "x86_64")
endif ()

# If you are using a 32-bit compiler on a 64-bit system CMAKE_SYSTEM_PROCESSOR will be wrong
if (${ARCH} STREQUAL "x86_64" AND BINARY EQUAL 32)
  set(ARCH x86)
endif ()

if (${ARCH} STREQUAL "X86")
  set(ARCH x86)
endif ()

set(COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
if (${COMPILER_ID} STREQUAL "GNU")
  set(COMPILER_ID "GCC")
endif ()

string(TOUPPER ${ARCH} UC_ARCH)

file(WRITE ${TARGET_CONF}
  "#define OS_${HOST_OS}\t1\n"
  "#define ARCH_${UC_ARCH}\t1\n"
  "#define C_${COMPILER_ID}\t1\n"
  "#define __${BINARY}BIT__\t1\n"
  "#define FUNDERSCORE\t${FU}\n")

