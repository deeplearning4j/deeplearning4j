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

# Read-only variables for compatibility with the other toolchain file.
# We'll keep these around for the existing projects that still use them.
# TODO: All of the variables here have equivalents in the standard set of
# cmake configurable variables, so we can remove these once most of our
# users migrate to those variables.

# From legacy toolchain file.
set(ANDROID_NDK "${CMAKE_ANDROID_NDK}")
set(ANDROID_ABI "${CMAKE_ANDROID_ARCH_ABI}")
set(ANDROID_COMPILER_IS_CLANG TRUE)
set(ANDROID_PLATFORM "android-${CMAKE_SYSTEM_VERSION}")
set(ANDROID_PLATFORM_LEVEL "${CMAKE_SYSTEM_VERSION}")
set(ANDROID_ARM_NEON TRUE)
if(CMAKE_ANDROID_ARM_MODE)
  set(ANDROID_ARM_MODE "arm")
  set(ANDROID_FORCE_ARM_BUILD TRUE)
else()
  set(ANDROID_ARM_MODE "thumb")
endif()
set(ANDROID_ARCH_NAME "${CMAKE_ANDROID_ARCH}")
set(ANDROID_LLVM_TRIPLE "${CMAKE_ANDROID_ARCH_LLVM_TRIPLE}${CMAKE_SYSTEM_VERSION}")
set(ANDROID_TOOLCHAIN_ROOT "${CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED}")
set(ANDROID_HOST_TAG "${CMAKE_ANDROID_NDK_TOOLCHAIN_HOST_TAG}")
set(ANDROID_HOST_PREBUILTS "${CMAKE_ANDROID_NDK}/prebuilt/${CMAKE_ANDROID_NDK_TOOLCHAIN_HOST_TAG}")
set(ANDROID_AR "${CMAKE_AR}")
set(ANDROID_RANLIB "${CMAKE_RANLIB}")
set(ANDROID_STRIP "${CMAKE_STRIP}")
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
  set(ANDROID_TOOLCHAIN_SUFFIX ".exe")
endif()

# From other toolchain file.
set(ANDROID_NATIVE_API_LEVEL "${ANDROID_PLATFORM_LEVEL}")
if(ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  set(ANDROID_SO_UNDEFINED TRUE)
else()
  set(ANDROID_NO_UNDEFINED TRUE)
endif()
set(ANDROID_FUNCTION_LEVEL_LINKING TRUE)
set(ANDROID_GOLD_LINKER TRUE)
set(ANDROID_NOEXECSTACK TRUE)
set(ANDROID_RELRO TRUE)
if(ANDROID_CPP_FEATURES MATCHES "rtti"
    AND ANDROID_CPP_FEATURES MATCHES "exceptions")
  set(ANDROID_STL_FORCE_FEATURES TRUE)
endif()
if(ANDROID_CCACHE)
  set(NDK_CCACHE "${ANDROID_CCACHE}")
endif()
set(ANDROID_NDK_HOST_X64 TRUE)
set(ANDROID_NDK_LAYOUT RELEASE)
if(CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
  set(ARMEABI_V7A TRUE)
  set(NEON TRUE)
elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
  set(ARM64_V8A TRUE)
elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
  set(X86 TRUE)
elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
  set(X86_64 TRUE)
elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "riscv64")
  set(RISCV64 TRUE)
endif()
set(ANDROID_NDK_HOST_SYSTEM_NAME "${ANDROID_HOST_TAG}")
set(ANDROID_NDK_ABI_NAME "${CMAKE_ANDROID_ARCH_ABI}")
set(ANDROID_NDK_RELEASE "r${ANDROID_NDK_REVISION}")
set(TOOL_OS_SUFFIX "${ANDROID_TOOLCHAIN_SUFFIX}")
