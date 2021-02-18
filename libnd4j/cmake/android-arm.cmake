################################################################################
#
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# See the NOTICE file distributed with this work for additional
# information regarding copyright ownership.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################
# CMake toolchain to build for Android 5.0 or newer. Sample usage:
#
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK}")
set(CMAKE_ANDROID_STL_TYPE c++_static)
set(CMAKE_SYSTEM_VERSION  "$ENV{ANDROID_VERSION}")
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)

set(ANDROID TRUE)
if (WIN32)
   set(CMAKE_C_COMPILER   "$ENV{ANDROID_CC}.exe")
   set(CMAKE_CXX_COMPILER "$ENV{ANDROID_CC}++.exe")
   else()
   set(CMAKE_C_COMPILER   "$ENV{ANDROID_CC}")
   set(CMAKE_CXX_COMPILER "$ENV{ANDROID_CC}++")
endif (WIN32)



add_definitions(-D__ANDROID_API__=$ENV{ANDROID_VERSION} -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -target  armv7a-linux-androideabi)

