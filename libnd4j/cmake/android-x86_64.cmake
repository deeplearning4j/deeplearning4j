################################################################################
#
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# See the NOTICE file distributed with this work for additional
# information regarding copyright ownership.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

# CMake toolchain for modern Android x86_64 builds

# 1. Activate CMake's native Android support. This is the main switch.
set(CMAKE_SYSTEM_NAME Android)

# 2. Specify the target architecture.
set(CMAKE_ANDROID_ARCH_ABI x86_64)

# 3. Specify the target Android API level.
#    This is the modern variable for what was CMAKE_SYSTEM_VERSION.
#    An explicit check for the environment variable is good practice.
if(DEFINED ENV{ANDROID_VERSION})
   set(CMAKE_ANDROID_API "$ENV{ANDROID_VERSION}")
else()
   # Default to a common API level if the environment variable is not set.
   set(CMAKE_ANDROID_API 21)
endif()

# 4. Point CMake to your NDK installation.
#    CMake will automatically find the necessary toolchain files from here.
set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK}")

# 5. Select the C++ standard library.
set(CMAKE_ANDROID_STL_TYPE c++_shared)

