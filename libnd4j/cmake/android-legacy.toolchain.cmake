# Copyright (C) 2016 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configurable variables.
# Modeled after the ndk-build system.
# For any variables defined in:
#         https://developer.android.com/ndk/guides/android_mk.html
#         https://developer.android.com/ndk/guides/application_mk.html
# if it makes sense for CMake, then replace LOCAL, APP, or NDK with ANDROID, and
# we have that variable below.
#
# ANDROID_TOOLCHAIN
# ANDROID_ABI
# ANDROID_PLATFORM
# ANDROID_STL
# ANDROID_PIE
# ANDROID_CPP_FEATURES
# ANDROID_ALLOW_UNDEFINED_SYMBOLS
# ANDROID_ARM_MODE
# ANDROID_DISABLE_FORMAT_STRING_CHECKS
# ANDROID_CCACHE
# ANDROID_SANITIZE


# Inhibit all of CMake's own NDK handling code.
set(CMAKE_SYSTEM_VERSION 1)

# Android NDK
get_filename_component(ANDROID_NDK_EXPECTED_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if(NOT ANDROID_NDK)
  set(ANDROID_NDK "${ANDROID_NDK_EXPECTED_PATH}")
else()
  # Allow the user to specify their own NDK path, but emit a warning. This is an
  # uncommon use case, but helpful if users want to use a bleeding edge
  # toolchain file with a stable NDK.
  # https://github.com/android-ndk/ndk/issues/473
  get_filename_component(ANDROID_NDK "${ANDROID_NDK}" ABSOLUTE)
  if(NOT "${ANDROID_NDK}" STREQUAL "${ANDROID_NDK_EXPECTED_PATH}")
    message(WARNING "Using custom NDK path (ANDROID_NDK is set): ${ANDROID_NDK}")
  endif()
endif()
unset(ANDROID_NDK_EXPECTED_PATH)
file(TO_CMAKE_PATH "${ANDROID_NDK}" ANDROID_NDK)

# Android NDK revision
# Possible formats:
# * r16, build 1234: 16.0.1234
# * r16b, build 1234: 16.1.1234
# * r16 beta 1, build 1234: 16.0.1234-beta1
#
# Canary builds are not specially marked.
file(READ "${ANDROID_NDK}/source.properties" ANDROID_NDK_SOURCE_PROPERTIES)

set(ANDROID_NDK_REVISION_REGEX
  "^Pkg\\.Desc = Android NDK\nPkg\\.Revision = ([0-9]+)\\.([0-9]+)\\.([0-9]+)(-beta([0-9]+))?")
if(NOT ANDROID_NDK_SOURCE_PROPERTIES MATCHES "${ANDROID_NDK_REVISION_REGEX}")
  message(SEND_ERROR "Failed to parse Android NDK revision: ${ANDROID_NDK}/source.properties.\n${ANDROID_NDK_SOURCE_PROPERTIES}")
endif()

set(ANDROID_NDK_MAJOR "${CMAKE_MATCH_1}")
set(ANDROID_NDK_MINOR "${CMAKE_MATCH_2}")
set(ANDROID_NDK_BUILD "${CMAKE_MATCH_3}")
set(ANDROID_NDK_BETA "${CMAKE_MATCH_5}")
if(ANDROID_NDK_BETA STREQUAL "")
  set(ANDROID_NDK_BETA "0")
endif()
set(ANDROID_NDK_REVISION
  "${ANDROID_NDK_MAJOR}.${ANDROID_NDK_MINOR}.${ANDROID_NDK_BUILD}${CMAKE_MATCH_4}")

# Touch toolchain variable to suppress "unused variable" warning.
# This happens if CMake is invoked with the same command line the second time.
if(CMAKE_TOOLCHAIN_FILE)
endif()

# Compatibility for configurable variables.
# Compatible with configurable variables from the other toolchain file:
#         https://github.com/taka-no-me/android-cmake
# TODO: We should consider dropping compatibility to simplify things once most
# of our users have migrated to our standard set of configurable variables.
if(ANDROID_TOOLCHAIN_NAME AND NOT ANDROID_TOOLCHAIN)
  if(ANDROID_TOOLCHAIN_NAME MATCHES "-clang([0-9].[0-9])?$")
    set(ANDROID_TOOLCHAIN clang)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "-[0-9].[0-9]$")
    set(ANDROID_TOOLCHAIN gcc)
  endif()
endif()
if(ANDROID_ABI STREQUAL "armeabi-v7a with NEON")
  set(ANDROID_ABI armeabi-v7a)
elseif(ANDROID_TOOLCHAIN_NAME AND NOT ANDROID_ABI)
  if(ANDROID_TOOLCHAIN_NAME MATCHES "^arm-linux-androideabi-")
    set(ANDROID_ABI armeabi-v7a)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^aarch64-linux-android-")
    set(ANDROID_ABI arm64-v8a)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^x86-")
    set(ANDROID_ABI x86)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^x86_64-")
    set(ANDROID_ABI x86_64)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^mipsel-linux-android-")
    set(ANDROID_ABI mips)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^mips64el-linux-android-")
    set(ANDROID_ABI mips64)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^riscv64-")
    set(ANDROID_ABI riscv64)
  endif()
endif()
if(ANDROID_NATIVE_API_LEVEL AND NOT ANDROID_PLATFORM)
  if(ANDROID_NATIVE_API_LEVEL MATCHES "^android-[0-9]+$")
    set(ANDROID_PLATFORM ${ANDROID_NATIVE_API_LEVEL})
  elseif(ANDROID_NATIVE_API_LEVEL MATCHES "^[0-9]+$")
    set(ANDROID_PLATFORM android-${ANDROID_NATIVE_API_LEVEL})
  endif()
endif()
if(DEFINED ANDROID_APP_PIE AND NOT DEFINED ANDROID_PIE)
  set(ANDROID_PIE "${ANDROID_APP_PIE}")
endif()
if(ANDROID_STL_FORCE_FEATURES AND NOT DEFINED ANDROID_CPP_FEATURES)
  set(ANDROID_CPP_FEATURES "rtti exceptions")
endif()
if(DEFINED ANDROID_NO_UNDEFINED AND NOT DEFINED ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  if(ANDROID_NO_UNDEFINED)
    set(ANDROID_ALLOW_UNDEFINED_SYMBOLS FALSE)
  else()
    set(ANDROID_ALLOW_UNDEFINED_SYMBOLS TRUE)
  endif()
endif()
if(DEFINED ANDROID_SO_UNDEFINED AND NOT DEFINED ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  set(ANDROID_ALLOW_UNDEFINED_SYMBOLS "${ANDROID_SO_UNDEFINED}")
endif()
if(DEFINED ANDROID_FORCE_ARM_BUILD AND NOT ANDROID_ARM_MODE)
  if(ANDROID_FORCE_ARM_BUILD)
    set(ANDROID_ARM_MODE arm)
  else()
    set(ANDROID_ARM_MODE thumb)
  endif()
endif()
if(NDK_CCACHE AND NOT ANDROID_CCACHE)
  set(ANDROID_CCACHE "${NDK_CCACHE}")
endif()

# Default values for configurable variables.
if(NOT ANDROID_TOOLCHAIN)
  set(ANDROID_TOOLCHAIN clang)
elseif(ANDROID_TOOLCHAIN STREQUAL gcc)
  message(FATAL_ERROR "GCC is no longer supported. See "
  "https://android.googlesource.com/platform/ndk/+/master/docs/ClangMigration.md.")
endif()
if(NOT ANDROID_ABI)
  set(ANDROID_ABI armeabi-v7a)
endif()

if(ANDROID_ABI STREQUAL armeabi)
  message(FATAL_ERROR "armeabi is no longer supported. Use armeabi-v7a.")
elseif(ANDROID_ABI MATCHES "^(mips|mips64)$")
  message(FATAL_ERROR "MIPS and MIPS64 are no longer supported.")
endif()

if(DEFINED ANDROID_ARM_NEON AND NOT ANDROID_ARM_NEON)
  message(FATAL_ERROR "Disabling Neon is no longer supported")
endif()

if(ANDROID_ABI STREQUAL armeabi-v7a)
  set(ANDROID_ARM_NEON TRUE)
endif()

include(${ANDROID_NDK}/build/cmake/abis.cmake)
include(${ANDROID_NDK}/build/cmake/platforms.cmake)

# If no platform version was chosen by the user, default to the minimum version
# supported by this NDK.
if(NOT ANDROID_PLATFORM)
  message(STATUS "\
ANDROID_PLATFORM not set. Defaulting to minimum supported version
${NDK_MIN_PLATFORM_LEVEL}.")

  set(ANDROID_PLATFORM "android-${NDK_MIN_PLATFORM_LEVEL}")
endif()

if(ANDROID_PLATFORM STREQUAL "latest")
  message(STATUS
    "Using latest available ANDROID_PLATFORM: ${NDK_MAX_PLATFORM_LEVEL}.")
  set(ANDROID_PLATFORM "android-${NDK_MAX_PLATFORM_LEVEL}")
  string(REPLACE "android-" "" ANDROID_PLATFORM_LEVEL ${ANDROID_PLATFORM})
endif()

string(REPLACE "android-" "" ANDROID_PLATFORM_LEVEL ${ANDROID_PLATFORM})

# Aliases defined by meta/platforms.json include codename aliases for platform
# API levels as well as cover any gaps in platforms that may not have had NDK
# APIs.
if(NOT "${NDK_PLATFORM_ALIAS_${ANDROID_PLATFORM_LEVEL}}" STREQUAL "")
  message(STATUS "\
${ANDROID_PLATFORM} is an alias for \
${NDK_PLATFORM_ALIAS_${ANDROID_PLATFORM_LEVEL}}. Adjusting ANDROID_PLATFORM to \
match.")
  set(ANDROID_PLATFORM "${NDK_PLATFORM_ALIAS_${ANDROID_PLATFORM_LEVEL}}")
  string(REPLACE "android-" "" ANDROID_PLATFORM_LEVEL ${ANDROID_PLATFORM})
endif()

# Pull up to the minimum supported version if an old API level was requested.
if(ANDROID_PLATFORM_LEVEL LESS NDK_MIN_PLATFORM_LEVEL)
  message(STATUS "\
${ANDROID_PLATFORM} is unsupported. Using minimum supported version \
${NDK_MIN_PLATFORM_LEVEL}.")
  set(ANDROID_PLATFORM "android-${NDK_MIN_PLATFORM_LEVEL}")
  string(REPLACE "android-" "" ANDROID_PLATFORM_LEVEL ${ANDROID_PLATFORM})
endif()

# Pull up any ABI-specific minimum API levels.
set(min_for_abi ${NDK_ABI_${ANDROID_ABI}_MIN_OS_VERSION})

if(ANDROID_PLATFORM_LEVEL LESS min_for_abi)
  message(STATUS
    "${ANDROID_PLATFORM} is not supported for ${ANDROID_ABI}. Using minimum "
    "supported ${ANDROID_ABI} version ${min_for_abi}.")
  set(ANDROID_PLATFORM android-${min_for_abi})
  set(ANDROID_PLATFORM_LEVEL ${min_for_abi})
endif()

# ANDROID_PLATFORM beyond the maximum is an error. The correct way to specify
# the latest version is ANDROID_PLATFORM=latest.
if(ANDROID_PLATFORM_LEVEL GREATER NDK_MAX_PLATFORM_LEVEL)
  message(SEND_ERROR "\
${ANDROID_PLATFORM} is above the maximum supported version \
${NDK_MAX_PLATFORM_LEVEL}. Choose a supported API level or set \
ANDROID_PLATFORM to \"latest\".")
endif()

if(NOT ANDROID_STL)
  set(ANDROID_STL c++_static)
endif()

if("${ANDROID_STL}" STREQUAL "gnustl_shared" OR
    "${ANDROID_STL}" STREQUAL "gnustl_static" OR
    "${ANDROID_STL}" STREQUAL "stlport_shared" OR
    "${ANDROID_STL}" STREQUAL "stlport_static")
  message(FATAL_ERROR "\
${ANDROID_STL} is no longer supported. Please switch to either c++_shared or \
c++_static. See https://developer.android.com/ndk/guides/cpp-support.html \
for more information.")
endif()

if("hwaddress" IN_LIST ANDROID_SANITIZE AND "${CMAKE_ANDROID_STL_TYPE}" STREQUAL "c++_static")
  message(FATAL_ERROR "\
  hwaddress does not support c++_static. Use system or c++_shared.")
endif()

set(ANDROID_PIE TRUE)
if(NOT ANDROID_ARM_MODE)
  set(ANDROID_ARM_MODE thumb)
endif()

# Export configurable variables for the try_compile() command.
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  ANDROID_ABI
  ANDROID_ALLOW_UNDEFINED_SYMBOLS
  ANDROID_ARM_MODE
  ANDROID_ARM_NEON
  ANDROID_CCACHE
  ANDROID_CPP_FEATURES
  ANDROID_DISABLE_FORMAT_STRING_CHECKS
  ANDROID_PIE
  ANDROID_PLATFORM
  ANDROID_STL
  ANDROID_TOOLCHAIN
  ANDROID_USE_LEGACY_TOOLCHAIN_FILE
)

# Standard cross-compiling stuff.
set(ANDROID TRUE)
set(CMAKE_SYSTEM_NAME Android)

# https://github.com/android-ndk/ndk/issues/890
#
# ONLY doesn't do anything when CMAKE_FIND_ROOT_PATH is empty. Without this,
# CMake will wrongly search host sysroots for headers/libraries. The actual path
# used here is fairly meaningless since CMake doesn't handle the NDK sysroot
# layout (per-arch and per-verion subdirectories for libraries), so find_library
# is handled separately by CMAKE_SYSTEM_LIBRARY_PATH.
list(APPEND CMAKE_FIND_ROOT_PATH "${ANDROID_NDK}")

# Allow users to override these values in case they want more strict behaviors.
# For example, they may want to prevent the NDK's libz from being picked up so
# they can use their own.
# https://github.com/android-ndk/ndk/issues/517
if(NOT CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()

# ABI.
set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
if(ANDROID_ABI STREQUAL armeabi-v7a)
  set(ANDROID_SYSROOT_ABI arm)
  set(ANDROID_TOOLCHAIN_NAME arm-linux-androideabi)
  set(CMAKE_SYSTEM_PROCESSOR armv7-a)
  set(ANDROID_LLVM_TRIPLE armv7-none-linux-androideabi)
elseif(ANDROID_ABI STREQUAL arm64-v8a)
  set(ANDROID_SYSROOT_ABI arm64)
  set(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(ANDROID_TOOLCHAIN_NAME aarch64-linux-android)
  set(ANDROID_LLVM_TRIPLE aarch64-none-linux-android)
elseif(ANDROID_ABI STREQUAL x86)
  set(ANDROID_SYSROOT_ABI x86)
  set(CMAKE_SYSTEM_PROCESSOR i686)
  set(ANDROID_TOOLCHAIN_NAME i686-linux-android)
  set(ANDROID_LLVM_TRIPLE i686-none-linux-android)
elseif(ANDROID_ABI STREQUAL x86_64)
  set(ANDROID_SYSROOT_ABI x86_64)
  set(CMAKE_SYSTEM_PROCESSOR x86_64)
  set(ANDROID_TOOLCHAIN_NAME x86_64-linux-android)
  set(ANDROID_LLVM_TRIPLE x86_64-none-linux-android)
elseif(ANDROID_ABI STREQUAL riscv64)
  set(ANDROID_SYSROOT_ABI riscv64)
  set(CMAKE_SYSTEM_PROCESSOR riscv64)
  set(ANDROID_TOOLCHAIN_NAME riscv64-linux-android)
  set(ANDROID_LLVM_TRIPLE riscv64-none-linux-android)
else()
  message(FATAL_ERROR "Invalid Android ABI: ${ANDROID_ABI}.")
endif()

set(ANDROID_LLVM_TRIPLE "${ANDROID_LLVM_TRIPLE}${ANDROID_PLATFORM_LEVEL}")

set(ANDROID_COMPILER_FLAGS)
set(ANDROID_COMPILER_FLAGS_CXX)
set(ANDROID_COMPILER_FLAGS_DEBUG)
set(ANDROID_COMPILER_FLAGS_RELEASE)
set(ANDROID_LINKER_FLAGS)
set(ANDROID_LINKER_FLAGS_EXE)
set(ANDROID_LINKER_FLAGS_RELEASE)
set(ANDROID_LINKER_FLAGS_RELWITHDEBINFO)
set(ANDROID_LINKER_FLAGS_MINSIZEREL)

# STL.
set(ANDROID_CXX_STANDARD_LIBRARIES)
if(ANDROID_STL STREQUAL system)
  list(APPEND ANDROID_COMPILER_FLAGS_CXX "-stdlib=libstdc++")
  if(NOT "x${ANDROID_CPP_FEATURES}" STREQUAL "x")
    list(APPEND ANDROID_CXX_STANDARD_LIBRARIES "-lc++abi")
  endif()
elseif(ANDROID_STL STREQUAL c++_static)
  list(APPEND ANDROID_LINKER_FLAGS "-static-libstdc++")
elseif(ANDROID_STL STREQUAL c++_shared)
elseif(ANDROID_STL STREQUAL none)
  list(APPEND ANDROID_COMPILER_FLAGS_CXX "-nostdinc++")
  list(APPEND ANDROID_LINKER_FLAGS "-nostdlib++")
else()
  message(FATAL_ERROR "Invalid STL: ${ANDROID_STL}.")
endif()

if(CMAKE_HOST_SYSTEM_NAME MATCHES Linux|Android)
  set(ANDROID_HOST_TAG linux-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL Darwin)
  set(ANDROID_HOST_TAG darwin-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
  set(ANDROID_HOST_TAG windows-x86_64)
endif()

if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
  set(ANDROID_TOOLCHAIN_SUFFIX .exe)
endif()

# Toolchain.
set(ANDROID_TOOLCHAIN_ROOT
  "${ANDROID_NDK}/toolchains/llvm/prebuilt/${ANDROID_HOST_TAG}")

list(APPEND CMAKE_PREFIX_PATH "${ANDROID_TOOLCHAIN_ROOT}")

# NB: This variable causes CMake to automatically pass --sysroot to the
# toolchain. Studio currently relies on this to recognize Android builds. If
# this variable is removed, ensure that flag is still passed.
# TODO: Teach Studio to recognize Android builds based on --target.
set(CMAKE_SYSROOT "${ANDROID_TOOLCHAIN_ROOT}/sysroot")

# Allows CMake to find headers in the architecture-specific include directories.
set(CMAKE_LIBRARY_ARCHITECTURE "${ANDROID_TOOLCHAIN_NAME}")

# In addition to <root>/<prefix>/lib/<arch>, cmake also searches <root>/<prefix>.
# Adding the API specific path to the beginning of CMAKE_SYSTEM_PREFIX_PATH, to
# make sure it is searched first.
set(CMAKE_SYSTEM_PREFIX_PATH
  "/usr/lib/${ANDROID_TOOLCHAIN_NAME}/${ANDROID_PLATFORM_LEVEL}"
  "${CMAKE_SYSTEM_PREFIX_PATH}"
  )

set(ANDROID_HOST_PREBUILTS "${ANDROID_NDK}/prebuilt/${ANDROID_HOST_TAG}")

set(ANDROID_C_COMPILER
  "${ANDROID_TOOLCHAIN_ROOT}/bin/clang${ANDROID_TOOLCHAIN_SUFFIX}")
set(ANDROID_CXX_COMPILER
  "${ANDROID_TOOLCHAIN_ROOT}/bin/clang++${ANDROID_TOOLCHAIN_SUFFIX}")
set(ANDROID_ASM_COMPILER
  "${ANDROID_TOOLCHAIN_ROOT}/bin/clang${ANDROID_TOOLCHAIN_SUFFIX}")
set(CMAKE_C_COMPILER_TARGET   ${ANDROID_LLVM_TRIPLE})
set(CMAKE_CXX_COMPILER_TARGET ${ANDROID_LLVM_TRIPLE})
set(CMAKE_ASM_COMPILER_TARGET ${ANDROID_LLVM_TRIPLE})
set(ANDROID_AR
  "${ANDROID_TOOLCHAIN_ROOT}/bin/llvm-ar${ANDROID_TOOLCHAIN_SUFFIX}")
set(ANDROID_RANLIB
  "${ANDROID_TOOLCHAIN_ROOT}/bin/llvm-ranlib${ANDROID_TOOLCHAIN_SUFFIX}")
set(ANDROID_STRIP
  "${ANDROID_TOOLCHAIN_ROOT}/bin/llvm-strip${ANDROID_TOOLCHAIN_SUFFIX}")

if(${CMAKE_VERSION} VERSION_LESS "3.19")
    # Older CMake won't pass -target when running the compiler identification
    # test, which causes the test to fail on flags like -mthumb.
    # https://github.com/android/ndk/issues/1427
    message(WARNING "An old version of CMake is being used that cannot "
      "automatically detect compiler attributes. Compiler identification is "
      "being bypassed. Some values may be wrong or missing. Update to CMake "
      "3.19 or newer to use CMake's built-in compiler identification.")
    set(CMAKE_C_COMPILER_ID_RUN TRUE)
    set(CMAKE_CXX_COMPILER_ID_RUN TRUE)
    set(CMAKE_C_COMPILER_ID Clang)
    set(CMAKE_CXX_COMPILER_ID Clang)
    # No need to auto-detect the computed standard defaults because CMake 3.6
    # doesn't know about anything past C11 or C++14 (neither does 3.10, so no
    # need to worry about 3.7-3.9), and any higher standards that Clang might
    # use are clamped to those values.
    set(CMAKE_C_STANDARD_COMPUTED_DEFAULT 11)
    set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT 14)
    set(CMAKE_C_COMPILER_FRONTEND_VARIANT "GNU")
    set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "GNU")
    include(${ANDROID_NDK}/build/cmake/compiler_id.cmake)
endif()

# Generic flags.
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
  -fdata-sections
  -ffunction-sections
  -funwind-tables
  -fstack-protector-strong
  -no-canonical-prefixes)

if(ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES)
  list(APPEND ANDROID_COMPILER_FLAGS -D__BIONIC_NO_PAGE_SIZE_MACRO)
  if(ANDROID_ABI STREQUAL arm64-v8a OR ANDROID_ABI STREQUAL x86_64)
    list(APPEND ANDROID_LINKER_FLAGS -Wl,-z,max-page-size=16384)
  endif()
endif()

if(ANDROID_WEAK_API_DEFS)
  list(APPEND ANDROID_COMPILER_FLAGS
      -D__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__
      -Werror=unguarded-availability)
endif()

if("hwaddress" IN_LIST ANDROID_SANITIZE)
  list(APPEND ANDROID_COMPILER_FLAGS -fsanitize=hwaddress -fno-omit-frame-pointer)
  list(APPEND ANDROID_LINKER_FLAGS -fsanitize=hwaddress)
endif()

if("memtag" IN_LIST ANDROID_SANITIZE)
  list(APPEND ANDROID_COMPILER_FLAGS -fsanitize=memtag-stack -fno-omit-frame-pointer)
  list(APPEND ANDROID_LINKER_FLAGS -fsanitize=memtag-stack,memtag-heap -fsanitize-memtag-mode=sync)
  if(ANDROID_ABI STREQUAL arm64-v8a)
    list(APPEND ANDROID_COMPILER_FLAGS -march=armv8-a+memtag)
    list(APPEND ANDROID_LINKER_FLAGS -march=armv8-a+memtag)
  endif()
endif()

# https://github.com/android/ndk/issues/885
# If we're using LLD we need to use a slower build-id algorithm to work around
# the old version of LLDB in Android Studio, which doesn't understand LLD's
# default hash ("fast").
list(APPEND ANDROID_LINKER_FLAGS -Wl,--build-id=sha1)
if(ANDROID_PLATFORM_LEVEL LESS 30)
  # https://github.com/android/ndk/issues/1196
  # https://github.com/android/ndk/issues/1589
  list(APPEND ANDROID_LINKER_FLAGS -Wl,--no-rosegment)
endif()

if (NOT ANDROID_ALLOW_UNDEFINED_VERSION_SCRIPT_SYMBOLS)
  list(APPEND ANDROID_LINKER_FLAGS -Wl,--no-undefined-version)
endif()

list(APPEND ANDROID_LINKER_FLAGS -Wl,--fatal-warnings)

# --gc-sections should not be present for debug builds because that can strip
# functions that the user may want to evaluate while debugging.
list(APPEND ANDROID_LINKER_FLAGS_RELEASE -Wl,--gc-sections)
list(APPEND ANDROID_LINKER_FLAGS_RELWITHDEBINFO -Wl,--gc-sections)
list(APPEND ANDROID_LINKER_FLAGS_MINSIZEREL -Wl,--gc-sections)

# Debug and release flags.
list(APPEND ANDROID_COMPILER_FLAGS_RELEASE -O3)
list(APPEND ANDROID_COMPILER_FLAGS_RELEASE -DNDEBUG)
if(ANDROID_TOOLCHAIN STREQUAL clang)
  list(APPEND ANDROID_COMPILER_FLAGS_DEBUG -fno-limit-debug-info)
endif()

# Toolchain and ABI specific flags.
if(ANDROID_ABI STREQUAL x86 AND ANDROID_PLATFORM_LEVEL LESS 24)
  # http://b.android.com/222239
  # http://b.android.com/220159 (internal http://b/31809417)
  # x86 devices have stack alignment issues.
  list(APPEND ANDROID_COMPILER_FLAGS -mstackrealign)
endif()

list(APPEND ANDROID_COMPILER_FLAGS -D_FORTIFY_SOURCE=2)

set(CMAKE_C_STANDARD_LIBRARIES_INIT "-latomic -lm")
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_C_STANDARD_LIBRARIES_INIT}")
if(ANDROID_CXX_STANDARD_LIBRARIES)
  string(REPLACE ";" "\" \"" ANDROID_CXX_STANDARD_LIBRARIES "\"${ANDROID_CXX_STANDARD_LIBRARIES}\"")
  set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_CXX_STANDARD_LIBRARIES}")
endif()

# Configuration specific flags.

# PIE is supported on all currently supported Android releases, but it is not
# supported with static executables, so we still provide ANDROID_PIE as an
# escape hatch for those.
if(ANDROID_PIE)
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()

if(ANDROID_CPP_FEATURES)
  separate_arguments(ANDROID_CPP_FEATURES)
  foreach(feature ${ANDROID_CPP_FEATURES})
    if(NOT ${feature} MATCHES "^(rtti|exceptions|no-rtti|no-exceptions)$")
      message(FATAL_ERROR "Invalid Android C++ feature: ${feature}.")
    endif()
    list(APPEND ANDROID_COMPILER_FLAGS_CXX
      -f${feature})
  endforeach()
  string(REPLACE ";" " " ANDROID_CPP_FEATURES "${ANDROID_CPP_FEATURES}")
endif()
if(NOT ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  list(APPEND ANDROID_LINKER_FLAGS
    -Wl,--no-undefined)
endif()
if(ANDROID_ABI MATCHES "armeabi")
  # Clang does not set this up properly when using -fno-integrated-as.
  # https://github.com/android-ndk/ndk/issues/906
  list(APPEND ANDROID_COMPILER_FLAGS "-march=armv7-a")
  if(ANDROID_ARM_MODE STREQUAL thumb)
    list(APPEND ANDROID_COMPILER_FLAGS -mthumb)
  elseif(ANDROID_ARM_MODE STREQUAL arm)
    # Default behavior.
  else()
    message(FATAL_ERROR "Invalid Android ARM mode: ${ANDROID_ARM_MODE}.")
  endif()
endif()

# CMake automatically forwards all compiler flags to the linker, and clang
# doesn't like having -Wa flags being used for linking. To prevent CMake from
# doing this would require meddling with the CMAKE_<LANG>_COMPILE_OBJECT rules,
# which would get quite messy.
list(APPEND ANDROID_LINKER_FLAGS -Qunused-arguments)

if(ANDROID_DISABLE_FORMAT_STRING_CHECKS)
  list(APPEND ANDROID_COMPILER_FLAGS
    -Wno-error=format-security)
else()
  list(APPEND ANDROID_COMPILER_FLAGS
    -Wformat -Werror=format-security)
endif()

# Convert these lists into strings.
string(REPLACE ";" " " ANDROID_COMPILER_FLAGS         "${ANDROID_COMPILER_FLAGS}")
string(REPLACE ";" " " ANDROID_COMPILER_FLAGS_CXX     "${ANDROID_COMPILER_FLAGS_CXX}")
string(REPLACE ";" " " ANDROID_COMPILER_FLAGS_DEBUG   "${ANDROID_COMPILER_FLAGS_DEBUG}")
string(REPLACE ";" " " ANDROID_COMPILER_FLAGS_RELEASE "${ANDROID_COMPILER_FLAGS_RELEASE}")
string(REPLACE ";" " " ANDROID_LINKER_FLAGS           "${ANDROID_LINKER_FLAGS}")
string(REPLACE ";" " " ANDROID_LINKER_FLAGS_EXE       "${ANDROID_LINKER_FLAGS_EXE}")
string(REPLACE ";" " " ANDROID_LINKER_FLAGS_RELEASE   "${ANDROID_LINKER_FLAGS_RELEASE}")
string(REPLACE ";" " " ANDROID_LINKER_FLAGS_RELWITHDEBINFO "${ANDROID_LINKER_FLAGS_RELWITHDEBINFO}")
string(REPLACE ";" " " ANDROID_LINKER_FLAGS_MINSIZEREL "${ANDROID_LINKER_FLAGS_MINSIZEREL}")

if(ANDROID_CCACHE)
  set(CMAKE_C_COMPILER_LAUNCHER   "${ANDROID_CCACHE}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${ANDROID_CCACHE}")
endif()
set(CMAKE_C_COMPILER "${ANDROID_C_COMPILER}")
set(CMAKE_CXX_COMPILER "${ANDROID_CXX_COMPILER}")
set(CMAKE_AR "${ANDROID_AR}" CACHE FILEPATH "Archiver")
set(CMAKE_RANLIB "${ANDROID_RANLIB}" CACHE FILEPATH "Ranlib")
set(CMAKE_STRIP "${ANDROID_STRIP}" CACHE FILEPATH "Strip")

if(ANDROID_ABI STREQUAL "x86" OR ANDROID_ABI STREQUAL "x86_64")
  set(CMAKE_ASM_NASM_COMPILER
    "${ANDROID_TOOLCHAIN_ROOT}/bin/yasm${ANDROID_TOOLCHAIN_SUFFIX}")
  set(CMAKE_ASM_NASM_COMPILER_ARG1 "-DELF")
endif()

# Set or retrieve the cached flags.
# This is necessary in case the user sets/changes flags in subsequent
# configures. If we included the Android flags in here, they would get
# overwritten.
set(CMAKE_C_FLAGS ""
  CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_CXX_FLAGS ""
  CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_ASM_FLAGS ""
  CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_C_FLAGS_DEBUG ""
  CACHE STRING "Flags used by the compiler during debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG ""
  CACHE STRING "Flags used by the compiler during debug builds.")
set(CMAKE_ASM_FLAGS_DEBUG ""
  CACHE STRING "Flags used by the compiler during debug builds.")
set(CMAKE_C_FLAGS_RELEASE ""
  CACHE STRING "Flags used by the compiler during release builds.")
set(CMAKE_CXX_FLAGS_RELEASE ""
  CACHE STRING "Flags used by the compiler during release builds.")
set(CMAKE_ASM_FLAGS_RELEASE ""
  CACHE STRING "Flags used by the compiler during release builds.")
set(CMAKE_MODULE_LINKER_FLAGS ""
  CACHE STRING "Flags used by the linker during the creation of modules.")
set(CMAKE_SHARED_LINKER_FLAGS ""
  CACHE STRING "Flags used by the linker during the creation of dll's.")
set(CMAKE_EXE_LINKER_FLAGS ""
  CACHE STRING "Flags used by the linker.")

set(CMAKE_C_FLAGS             "${ANDROID_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS           "${ANDROID_COMPILER_FLAGS} ${ANDROID_COMPILER_FLAGS_CXX} ${CMAKE_CXX_FLAGS}")
set(CMAKE_ASM_FLAGS           "${ANDROID_COMPILER_FLAGS} ${CMAKE_ASM_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG       "${ANDROID_COMPILER_FLAGS_DEBUG} ${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_DEBUG     "${ANDROID_COMPILER_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_ASM_FLAGS_DEBUG     "${ANDROID_COMPILER_FLAGS_DEBUG} ${CMAKE_ASM_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE     "${ANDROID_COMPILER_FLAGS_RELEASE} ${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE   "${ANDROID_COMPILER_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_ASM_FLAGS_RELEASE   "${ANDROID_COMPILER_FLAGS_RELEASE} ${CMAKE_ASM_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS "${ANDROID_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${ANDROID_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS    "${ANDROID_LINKER_FLAGS} ${ANDROID_LINKER_FLAGS_EXE} ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${ANDROID_LINKER_FLAGS_RELEASE} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${ANDROID_LINKER_FLAGS_RELEASE} ${CMAKE_MODULE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE    "${ANDROID_LINKER_FLAGS_RELEASE} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${ANDROID_LINKER_FLAGS_RELWITHDEBINFO} ${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}")
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "${ANDROID_LINKER_FLAGS_RELWITHDEBINFO} ${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO}")
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO    "${ANDROID_LINKER_FLAGS_RELWITHDEBINFO} ${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")
set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${ANDROID_LINKER_FLAGS_MINSIZEREL} ${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL}")
set(CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL "${ANDROID_LINKER_FLAGS_MINSIZEREL} ${CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL}")
set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL    "${ANDROID_LINKER_FLAGS_MINSIZEREL} ${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL}")

# Compatibility for read-only variables.
# Read-only variables for compatibility with the other toolchain file.
# We'll keep these around for the existing projects that still use them.
# TODO: All of the variables here have equivalents in our standard set of
# configurable variables, so we can remove these once most of our users migrate
# to those variables.
set(ANDROID_NATIVE_API_LEVEL ${ANDROID_PLATFORM_LEVEL})
if(ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  set(ANDROID_SO_UNDEFINED TRUE)
else()
  set(ANDROID_NO_UNDEFINED TRUE)
endif()
set(ANDROID_FUNCTION_LEVEL_LINKING TRUE)
set(ANDROID_GOLD_LINKER TRUE)
set(ANDROID_NOEXECSTACK TRUE)
set(ANDROID_RELRO TRUE)
if(ANDROID_ARM_MODE STREQUAL arm)
  set(ANDROID_FORCE_ARM_BUILD TRUE)
endif()
if(ANDROID_CPP_FEATURES MATCHES "rtti"
    AND ANDROID_CPP_FEATURES MATCHES "exceptions")
  set(ANDROID_STL_FORCE_FEATURES TRUE)
endif()
if(ANDROID_CCACHE)
  set(NDK_CCACHE "${ANDROID_CCACHE}")
endif()
if(ANDROID_TOOLCHAIN STREQUAL clang)
  set(ANDROID_TOOLCHAIN_NAME ${ANDROID_TOOLCHAIN_NAME}-clang)
else()
  set(ANDROID_TOOLCHAIN_NAME ${ANDROID_TOOLCHAIN_NAME}-4.9)
endif()
set(ANDROID_NDK_HOST_X64 TRUE)
set(ANDROID_NDK_LAYOUT RELEASE)
if(ANDROID_ABI STREQUAL armeabi-v7a)
  set(ARMEABI_V7A TRUE)
  if(ANDROID_ARM_NEON)
    set(NEON TRUE)
  endif()
elseif(ANDROID_ABI STREQUAL arm64-v8a)
  set(ARM64_V8A TRUE)
elseif(ANDROID_ABI STREQUAL x86)
  set(X86 TRUE)
elseif(ANDROID_ABI STREQUAL x86_64)
  set(X86_64 TRUE)
elseif(ANDROID_ABI STREQUAL riscv64)
  set(RISCV64 TRUE)
endif()
set(ANDROID_NDK_HOST_SYSTEM_NAME ${ANDROID_HOST_TAG})
set(ANDROID_NDK_ABI_NAME ${ANDROID_ABI})
set(ANDROID_NDK_RELEASE r${ANDROID_NDK_REVISION})
set(ANDROID_ARCH_NAME ${ANDROID_SYSROOT_ABI})
set(TOOL_OS_SUFFIX ${ANDROID_TOOLCHAIN_SUFFIX})
if(ANDROID_TOOLCHAIN STREQUAL clang)
  set(ANDROID_COMPILER_IS_CLANG TRUE)
endif()

# CMake 3.7+ compatibility.
if (CMAKE_VERSION VERSION_GREATER 3.7.0)
  set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
  set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)

  set(CMAKE_ANDROID_STL_TYPE ${ANDROID_STL})

  if(ANDROID_ABI MATCHES "^armeabi(-v7a)?$")
    set(CMAKE_ANDROID_ARM_NEON ${ANDROID_ARM_NEON})
    set(CMAKE_ANDROID_ARM_MODE ${ANDROID_ARM_MODE})
  endif()

  # https://github.com/android/ndk/issues/861
  if(ANDROID_ABI STREQUAL armeabi-v7a)
    set(CMAKE_ANDROID_ARCH arm)
  elseif(ANDROID_ABI STREQUAL arm64-v8a)
    set(CMAKE_ANDROID_ARCH arm64)
  elseif(ANDROID_ABI STREQUAL x86)
    set(CMAKE_ANDROID_ARCH x86)
  elseif(ANDROID_ABI STREQUAL x86_64)
    set(CMAKE_ANDROID_ARCH x86_64)
  elseif(ANDROID_ABI STREQUAL riscv64)
    set(CMAKE_ANDROID_ARCH riscv64)
  endif()

  # https://github.com/android/ndk/issues/1012
  set(CMAKE_ASM_ANDROID_TOOLCHAIN_MACHINE "${ANDROID_TOOLCHAIN_NAME}")
  set(CMAKE_C_ANDROID_TOOLCHAIN_MACHINE "${ANDROID_TOOLCHAIN_NAME}")
  set(CMAKE_CXX_ANDROID_TOOLCHAIN_MACHINE "${ANDROID_TOOLCHAIN_NAME}")

  set(CMAKE_ASM_ANDROID_TOOLCHAIN_SUFFIX "${ANDROID_TOOLCHAIN_SUFFIX}")
  set(CMAKE_C_ANDROID_TOOLCHAIN_SUFFIX "${ANDROID_TOOLCHAIN_SUFFIX}")
  set(CMAKE_CXX_ANDROID_TOOLCHAIN_SUFFIX "${ANDROID_TOOLCHAIN_SUFFIX}")
endif()
