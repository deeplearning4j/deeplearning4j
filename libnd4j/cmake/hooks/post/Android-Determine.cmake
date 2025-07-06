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

# This is a hook file that will be included by cmake at the end of
# Modules/Platform/Android-Determine.cmake.

# android.toolchain.cmake may set this to export old variables.
if(_ANDROID_EXPORT_COMPATIBILITY_VARIABLES)
  file(READ "${CMAKE_ANDROID_NDK}/build/cmake/exports.cmake" _EXPORTS)
  string(APPEND CMAKE_SYSTEM_CUSTOM_CODE "\n${_EXPORTS}\n")
endif()
