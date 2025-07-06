# Copyright (C) 2020 The Android Open Source Project
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

# This is a hook file that will be included by cmake at the beginning of
# Modules/Platform/Android/Determine-Compiler.cmake.

# Skip hook for the legacy toolchain workflow.
if(CMAKE_SYSTEM_VERSION EQUAL 1)
  return()
endif()

if(${CMAKE_VERSION} VERSION_LESS "3.22.0")
  # If we don't explicitly set the target CMake will ID the compiler using the
  # default target, causing MINGW to be defined when a Windows host is used.
  # https://github.com/android/ndk/issues/1581
  # https://gitlab.kitware.com/cmake/cmake/-/issues/22647
  if(CMAKE_ANDROID_ARCH_ABI STREQUAL armeabi-v7a)
    set(ANDROID_LLVM_TRIPLE armv7-none-linux-androideabi)
  elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL arm64-v8a)
    set(ANDROID_LLVM_TRIPLE aarch64-none-linux-android)
  elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL x86)
    set(ANDROID_LLVM_TRIPLE i686-none-linux-android)
  elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL x86_64)
    set(ANDROID_LLVM_TRIPLE x86_64-none-linux-android)
  else()
    message(FATAL_ERROR "Invalid Android ABI: ${ANDROID_ABI}.")
  endif()
  set(CMAKE_ASM_COMPILER_TARGET "${ANDROID_LLVM_TRIPLE}${CMAKE_SYSTEM_VERSION}")
  set(CMAKE_C_COMPILER_TARGET "${ANDROID_LLVM_TRIPLE}${CMAKE_SYSTEM_VERSION}")
  set(CMAKE_CXX_COMPILER_TARGET "${ANDROID_LLVM_TRIPLE}${CMAKE_SYSTEM_VERSION}")
endif()
