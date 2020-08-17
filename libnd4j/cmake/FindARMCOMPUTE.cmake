################################################################################
# Copyright (c) 2020 Konduit K.K.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################



### Find ARM COMPUTE LIBRARY STATIC libraries

SET (COMPUTE_INCLUDE_DIRS 
    /usr/include
    ${ARMCOMPUTE_ROOT}
    ${ARMCOMPUTE_ROOT}/include
    ${ARMCOMPUTE_ROOT}/applications 
    ${ARMCOMPUTE_ROOT}/applications/arm_compute    
)


SET (COMPUTE_LIB_DIRS  
     /lib
     /usr/lib
    ${ARMCOMPUTE_ROOT}
    ${ARMCOMPUTE_ROOT}/lib 
    ${ARMCOMPUTE_ROOT}/build
)

find_path(ARMCOMPUTE_INCLUDE arm_compute/core/CL/ICLKernel.h
            PATHS ${COMPUTE_INCLUDE_DIRS}
            NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

find_path(ARMCOMPUTE_INCLUDE arm_compute/core/CL/ICLKernel.h)

find_path(HALF_INCLUDE half/half.hpp)
find_path(HALF_INCLUDE half/half.hpp
              PATHS ${ARMCOMPUTE_ROOT}/include
              NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
include_directories(SYSTEM ${HALF_INCLUDE})

# Find the Arm Compute libraries if not already specified 
if (NOT DEFINED ARMCOMPUTE_LIBRARIES)
 
    find_library(ARMCOMPUTE_LIBRARY NAMES arm_compute-static
                    PATHS ${COMPUTE_LIB_DIRS}
                    PATH_SUFFIXES "Release"
                    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    find_library(ARMCOMPUTE_CORE_LIBRARY NAMES arm_compute_core-static
                    PATHS ${COMPUTE_LIB_DIRS}
                    PATH_SUFFIXES "Release"
                    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    # In case it wasn't there, try a default search (will work in cases where
    # the library has been installed into a standard location) 
    find_library(ARMCOMPUTE_LIBRARY NAMES arm_compute-static) 
    find_library(ARMCOMPUTE_CORE_LIBRARY NAMES arm_compute_core-static)
    
    set(ARMCOMPUTE_LIBRARIES  ${ARMCOMPUTE_LIBRARY} ${ARMCOMPUTE_CORE_LIBRARY} )
endif()
 
 
INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(ARMCOMPUTE REQUIRED_VARS ARMCOMPUTE_INCLUDE ARMCOMPUTE_LIBRARIES)

