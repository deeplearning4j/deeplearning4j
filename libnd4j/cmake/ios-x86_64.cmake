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
# CMake toolchain to build libnd4j for 64-bit iOS Simulator. Sample usage:
#
# cmake -DCMAKE_TOOLCHAIN_FILE=ios-x86_64.cmake -DCMAKE_INSTALL_PREFIX=..
#

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR "x86_64")
set(IOS TRUE)
set(CFLAGS, "-miphoneos-version-min=6.0 -arch x86_64")
set(CMAKE_C_COMPILER   "clang")
set(CMAKE_CXX_COMPILER "clang")
set(CMAKE_C_LINK_EXECUTABLE    "libtool -static <FLAGS> <CMAKE_C_LINK_FLAGS>   <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -L$ENV{IOS_SDK}/usr/lib/ -syslibroot $ENV{IOS_SDK}")
set(CMAKE_CXX_LINK_EXECUTABLE    "libtool -static <FLAGS> <CMAKE_CXX_LINK_FLAGS>   <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -L$ENV{IOS_SDK}/usr/lib/ -syslibroot $ENV{IOS_SDK}")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY  "libtool -o <TARGET> <OBJECTS> <LINK_LIBRARIES> -syslibroot $ENV{IOS_SDK} -L$ENV{IOS_SDK}/usr/lib/")
add_definitions("-DIOS -stdlib=libc++ -miphoneos-version-min=6.0 -arch x86_64 -isysroot $ENV{IOS_SDK} -I/usr/local/opt/llvm/4.0.0/include/c++/v1 -I/usr/local/opt/llvm/4.0.0/lib/clang/4.0.0/include -fPIC -ffunction-sections -funwind-tables -fstack-protector -fomit-frame-pointer -fstrict-aliasing")
