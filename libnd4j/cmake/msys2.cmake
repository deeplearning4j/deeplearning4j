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
# Use this command to build the Windows port of Allegro
# with a mingw cross compiler:
#
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-mingw.cmake .
#
# or for out of source:
#
#   cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchain-mingw.cmake ..
#
# You will need at least CMake 2.6.0.
#
# Adjust the following paths to suit your environment.
#
# This file was based on http://www.cmake.org/Wiki/CmakeMingw

# the name of the target operating system
set(CMAKE_SYSTEM_NAME Windows)
message("Using msys2")
# Assume the target architecture.
# XXX for some reason the value set here gets cleared before we reach the
# main CMakeLists.txt; see that file for a workaround.
# set(CMAKE_SYSTEM_PROCESSOR i686)

# Which compilers to use for C and C++, and location of target
# environment.
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    # First look in standard location as used by Debian/Ubuntu/etc.
    set(CMAKE_C_COMPILER /mingw64/bin/gcc)
    set(CMAKE_CXX_COMPILER /mingw64/bin/g++)
    set(CMAKE_RC_COMPILER /mingw64/bin/windres)
    set(CMAKE_FIND_ROOT_PATH /mingw64)
else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    # First look in standard location as used by Debian/Ubuntu/etc.
    set(CMAKE_C_COMPILER  /mingw32/bin/gcc)
    set(CMAKE_CXX_COMPILER /mingw32/bin/g++)
    set(CMAKE_RC_COMPILER /mingw64/bin/windres)
    set(CMAKE_FIND_ROOT_PATH /mingw32)
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )


# Adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Tell pkg-config not to look at the target environment's .pc files.
# Setting PKG_CONFIG_LIBDIR sets the default search directory, but we have to
# set PKG_CONFIG_PATH as well to prevent pkg-config falling back to the host's
# path.
set(ENV{PKG_CONFIG_LIBDIR} ${CMAKE_FIND_ROOT_PATH}/lib/pkgconfig)
set(ENV{PKG_CONFIG_PATH} ${CMAKE_FIND_ROOT_PATH}/lib/pkgconfig)

set(ENV{MINGDIR} ${CMAKE_FIND_ROOT_PATH})