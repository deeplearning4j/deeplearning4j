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


# CMake invokes the toolchain file twice during the first build, but only once
# during subsequent rebuilds. This was causing the various flags to be added
# twice on the first build, and on a rebuild ninja would see only one set of the
# flags and rebuild the world.
# https://github.com/android-ndk/ndk/issues/323
if(ANDROID_NDK_TOOLCHAIN_INCLUDED)
   return()
endif(ANDROID_NDK_TOOLCHAIN_INCLUDED)
set(ANDROID_NDK_TOOLCHAIN_INCLUDED true)

if(DEFINED ANDROID_USE_LEGACY_TOOLCHAIN_FILE)
   set(_USE_LEGACY_TOOLCHAIN_FILE ${ANDROID_USE_LEGACY_TOOLCHAIN_FILE})
else()
   # Default to the legacy toolchain file to avoid changing the behavior of
   # CMAKE_CXX_FLAGS. See https://github.com/android/ndk/issues/1693.
   set(_USE_LEGACY_TOOLCHAIN_FILE true)
endif()
if(_USE_LEGACY_TOOLCHAIN_FILE)
   include("${CMAKE_CURRENT_LIST_DIR}/android-legacy.toolchain.cmake")
   return()
endif()

# Android NDK path
get_filename_component(ANDROID_NDK_EXPECTED_PATH
        "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if(NOT ANDROID_NDK)
   set(CMAKE_ANDROID_NDK "${ANDROID_NDK_EXPECTED_PATH}")
else()
   # Allow the user to specify their own NDK path, but emit a warning. This is an
   # uncommon use case, but helpful if users want to use a bleeding edge
   # toolchain file with a stable NDK.
   # https://github.com/android-ndk/ndk/issues/473
   get_filename_component(ANDROID_NDK "${ANDROID_NDK}" ABSOLUTE)
   if(NOT "${ANDROID_NDK}" STREQUAL "${ANDROID_NDK_EXPECTED_PATH}")
      message(WARNING "Using custom NDK path (ANDROID_NDK is set): ${ANDROID_NDK}")
   endif()
   set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
endif()
unset(ANDROID_NDK_EXPECTED_PATH)
file(TO_CMAKE_PATH "${CMAKE_ANDROID_NDK}" CMAKE_ANDROID_NDK)

# Android NDK revision
# Possible formats:
# * r16, build 1234: 16.0.1234
# * r16b, build 1234: 16.1.1234
# * r16 beta 1, build 1234: 16.0.1234-beta1
#
# Canary builds are not specially marked.
file(READ "${CMAKE_ANDROID_NDK}/source.properties" ANDROID_NDK_SOURCE_PROPERTIES)

set(ANDROID_NDK_REVISION_REGEX
        "^Pkg\\.Desc = Android NDK\nPkg\\.Revision = ([0-9]+)\\.([0-9]+)\\.([0-9]+)(-beta([0-9]+))?")
if(NOT ANDROID_NDK_SOURCE_PROPERTIES MATCHES "${ANDROID_NDK_REVISION_REGEX}")
   message(SEND_ERROR "Failed to parse Android NDK revision: ${CMAKE_ANDROID_NDK}/source.properties.\n${ANDROID_NDK_SOURCE_PROPERTIES}")
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

# Determine the ABI.
if(NOT CMAKE_ANDROID_ARCH_ABI)
   if(ANDROID_ABI STREQUAL "armeabi-v7a with NEON")
      set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
   elseif(ANDROID_ABI)
      set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
   elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^arm-linux-androideabi-")
      set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
   elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^aarch64-linux-android-")
      set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
   elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^x86-")
      set(CMAKE_ANDROID_ARCH_ABI x86)
   elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^x86_64-")
      set(CMAKE_ANDROID_ARCH_ABI x86_64)
   elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^riscv64-")
      set(CMAKE_ANDROID_ARCH_ABI riscv64)
   else()
      set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
   endif()
endif()

if(DEFINED ANDROID_ARM_NEON AND NOT ANDROID_ARM_NEON)
   message(FATAL_ERROR "Disabling Neon is no longer supported")
endif()

if(CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
   set(CMAKE_ANDROID_ARM_NEON TRUE)

   if(NOT DEFINED CMAKE_ANDROID_ARM_MODE)
      if(DEFINED ANDROID_FORCE_ARM_BUILD)
         set(CMAKE_ANDROID_ARM_MODE ${ANDROID_FORCE_ARM_BUILD})
      elseif(DEFINED ANDROID_ARM_MODE)
         if(ANDROID_ARM_MODE STREQUAL "arm")
            set(CMAKE_ANDROID_ARM_MODE TRUE)
         elseif(ANDROID_ARM_MODE STREQUAL "thumb")
            set(CMAKE_ANDROID_ARM_MODE FALSE)
         else()
            message(FATAL_ERROR "Invalid Android ARM mode: ${ANDROID_ARM_MODE}.")
         endif()
      endif()
   endif()
endif()

# PIE is supported on all currently supported Android releases, but it is not
# supported with static executables, so we still provide ANDROID_PIE as an
# escape hatch for those.
if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
   if(DEFINED ANDROID_PIE)
      set(CMAKE_POSITION_INDEPENDENT_CODE ${ANDROID_PIE})
   elseif(DEFINED ANDROID_APP_PIE)
      set(CMAKE_POSITION_INDEPENDENT_CODE ${ANDROID_APP_PIE})
   else()
      set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
   endif()
endif()

# Default values for configurable variables.
if(NOT ANDROID_TOOLCHAIN)
   set(ANDROID_TOOLCHAIN clang)
elseif(ANDROID_TOOLCHAIN STREQUAL gcc)
   message(FATAL_ERROR "GCC is no longer supported. See "
           "https://android.googlesource.com/platform/ndk/+/master/docs/ClangMigration.md.")
endif()

if(ANDROID_NATIVE_API_LEVEL AND NOT ANDROID_PLATFORM)
   if(ANDROID_NATIVE_API_LEVEL MATCHES "^android-[0-9]+$")
      set(ANDROID_PLATFORM ${ANDROID_NATIVE_API_LEVEL})
   elseif(ANDROID_NATIVE_API_LEVEL MATCHES "^[0-9]+$")
      set(ANDROID_PLATFORM android-${ANDROID_NATIVE_API_LEVEL})
   endif()
endif()
include(${CMAKE_ANDROID_NDK}/build/cmake/adjust_api_level.cmake)
adjust_api_level("${ANDROID_PLATFORM}" CMAKE_SYSTEM_VERSION)

if(NOT DEFINED CMAKE_ANDROID_STL_TYPE AND DEFINED ANDROID_STL)
   set(CMAKE_ANDROID_STL_TYPE ${ANDROID_STL})
endif()

if("hwaddress" IN_LIST ANDROID_SANITIZE AND "${CMAKE_ANDROID_STL_TYPE}" STREQUAL "c++_static")
   message(FATAL_ERROR "\
  hwaddress does not support c++_static. Use system or c++_shared.")
endif()

if("${CMAKE_ANDROID_STL_TYPE}" STREQUAL "gnustl_shared" OR
        "${CMAKE_ANDROID_STL_TYPE}" STREQUAL "gnustl_static" OR
        "${CMAKE_ANDROID_STL_TYPE}" STREQUAL "stlport_shared" OR
        "${CMAKE_ANDROID_STL_TYPE}" STREQUAL "stlport_static")
   message(FATAL_ERROR "\
${CMAKE_ANDROID_STL_TYPE} is no longer supported. Please switch to either c++_shared \
or c++_static. See https://developer.android.com/ndk/guides/cpp-support.html \
for more information.")
endif()

# Standard cross-compiling stuff.
set(CMAKE_SYSTEM_NAME Android)

# STL.
if(ANDROID_STL)
   set(CMAKE_ANDROID_STL_TYPE ${ANDROID_STL})
endif()

if(NDK_CCACHE AND NOT ANDROID_CCACHE)
   set(ANDROID_CCACHE "${NDK_CCACHE}")
endif()
if(ANDROID_CCACHE)
   set(CMAKE_C_COMPILER_LAUNCHER   "${ANDROID_CCACHE}")
   set(CMAKE_CXX_COMPILER_LAUNCHER "${ANDROID_CCACHE}")
endif()

# Configuration specific flags.
if(ANDROID_STL_FORCE_FEATURES AND NOT DEFINED ANDROID_CPP_FEATURES)
   set(ANDROID_CPP_FEATURES "rtti exceptions")
endif()

if(ANDROID_CPP_FEATURES)
   separate_arguments(ANDROID_CPP_FEATURES)
   foreach(feature ${ANDROID_CPP_FEATURES})
      if(NOT ${feature} MATCHES "^(rtti|exceptions|no-rtti|no-exceptions)$")
         message(FATAL_ERROR "Invalid Android C++ feature: ${feature}.")
      endif()
      if(${feature} STREQUAL "rtti")
         set(CMAKE_ANDROID_RTTI TRUE)
      endif()
      if(${feature} STREQUAL "no-rtti")
         set(CMAKE_ANDROID_RTTI FALSE)
      endif()
      if(${feature} STREQUAL "exceptions")
         set(CMAKE_ANDROID_EXCEPTIONS TRUE)
      endif()
      if(${feature} STREQUAL "no-exceptions")
         set(CMAKE_ANDROID_EXCEPTIONS FALSE)
      endif()
   endforeach()
   string(REPLACE ";" " " ANDROID_CPP_FEATURES "${ANDROID_CPP_FEATURES}")
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
        ANDROID_SANITIZE
)

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

# Exports compatible variables defined in exports.cmake.
set(_ANDROID_EXPORT_COMPATIBILITY_VARIABLES TRUE)

if(CMAKE_HOST_SYSTEM_NAME MATCHES Linux|Android)
   set(ANDROID_HOST_TAG linux-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL Darwin)
   set(ANDROID_HOST_TAG darwin-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
   set(ANDROID_HOST_TAG windows-x86_64)
endif()

# Toolchain.
set(ANDROID_TOOLCHAIN_ROOT
        "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/${ANDROID_HOST_TAG}")

# NB: This variable causes CMake to automatically pass --sysroot to the
# toolchain. Studio currently relies on this to recognize Android builds. If
# this variable is removed, ensure that flag is still passed.
# TODO: Teach Studio to recognize Android builds based on --target.
set(CMAKE_SYSROOT "${ANDROID_TOOLCHAIN_ROOT}/sysroot")
