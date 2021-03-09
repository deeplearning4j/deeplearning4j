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

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)

set(CMAKE_C_COMPILER   "$ENV{RPI_BIN}-gcc" )
set(CMAKE_CXX_COMPILER "$ENV{RPI_BIN}-g++" )

if (SD_CUDA)
    if(${SD_ARCH} MATCHES "armv8")
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
    else()
        set(CMAKE_SYSTEM_PROCESSOR ${SD_ARCH})
    endif()
    set(CUDA_TARGET_CPU_ARCH ${CMAKE_SYSTEM_PROCESSOR})
    set(CUDA_TARGET_OS_VARIANT "linux")
    if (SD_CUDA)
        set(ENV{CUDAHOSTCXX} "${CMAKE_CXX_COMPILER}")
    endif()
endif()

set(CMAKE_FIND_ROOT_PATH "")

if(DEFINED ENV{SYSROOT})
    set(CMAKE_SYSROOT "$ENV{SYSROOT}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --sysroot=$ENV{SYSROOT}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=$ENV{SYSROOT}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=$ENV{SYSROOT}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} --sysroot=$ENV{SYSROOT}")
    list(APPEND CMAKE_FIND_ROOT_PATH "$ENV{SYSROOT}")
endif()

if(DEFINED ENV{CUDNN_ROOT_DIR})
    list(APPEND CMAKE_FIND_ROOT_PATH "$ENV{CUDNN_ROOT_DIR}")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
#search only in target path
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)