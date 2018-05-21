cmake_minimum_required(VERSION 3.6)
project(libnd4j)
set( CMAKE_VERBOSE_MAKEFILE on)
option(NATIVE "Optimize for build machine (might not work on others)" OFF)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#ensure we create lib files
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

set(CMAKE_CXX_FLAGS_RELEASE  " -O3 -std=c++11 -fassociative-math -funsafe-math-optimizations")
set(CMAKE_CXX_FLAGS_DEBUG " -g -O0 -std=c++11")


if(NATIVE)
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    ENDIF()
endif()


if (NOT DEFINED ENV{CLION_IDE})
    message("NOT CLION")
    include_directories(blas/ include/ include/helpers include/loops include/graph include/ops include/types include/array include/cnpy)
    add_subdirectory(blas)
endif ()

if ($ENV{CLION_IDE})
    add_subdirectory(tests_cpu)
endif ()

set (CMAKE_INSTALL_PREFIX $ENV{ND4J_HOME}/nd4j-native-parent/nd4j-native/src/main/resources)

# Set package information
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations for nd4j.")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "François Garillot <francois@skymind.io>")
set(CPACK_PACKAGE_VENDOR "Skymind")
set(CPACK_SETDESTDIR "false")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/lib")
set(CPACK_PACKAGE_NAME "libnd4j")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "8")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "libnd4j")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Determine distribution and release — may require redhat-lsb-core installed on CentOS / RH
execute_process(COMMAND lsb_release -si OUTPUT_VARIABLE DISTRIBUTION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND lsb_release -sc OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND uname -i OUTPUT_VARIABLE ARCHITECTURE)

# Set package name and type (deb vs rpm)
if(DISTRIBUTION STREQUAL "Ubuntu")

  # Set Ubuntu-specific information (see http://www.cmake.org/Wiki/CMake:CPackPackageGenerators)
  if(ARCHITECTURE MATCHES ".*x86_64.*")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
  else()
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
  endif()
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "François Garillot")
  set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
  set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "cuda")
  # For Ubuntu <= 12, libatlas3gf-base, liblapack3gf
  # Build deps: libatlas3-base liblapack3 libopenblas-dev libatlas-dev liblapack-dev gcc-5 g++-5
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
  set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/deeplearning4j/libnd4j")
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${RELEASE}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})
  set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst;${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm;" )

elseif(DISTRIBUTION STREQUAL "CentOS")

  # Set Fedora-specific information (see http://www.cmake.org/Wiki/CMake:CPackPackageGenerators)
  execute_process(COMMAND lsb_release -sr OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(ARCHITECTURE MATCHES ".*x86_64.*")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
  else()
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "i686")
  endif()
  set(CPACK_PACKAGE_CONTACT "François Garillot")
  set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
  set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
  set(CPACK_RPM_PACKAGE_SUGGESTS "cuda")
  # Build deps: atlas blas lapack cmake3 devtoolset-4-gcc devtoolset-4-gcc-c++
  set(CPACK_RPM_PACKAGE_REQUIRES "")
  set(CPACK_RPM_PACKAGE_URL "https://github.com/deeplearning4j/libnd4j")
  set(CPACK_GENERATOR "RPM")
  set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.fc${RELEASE}.${CPACK_RPM_PACKAGE_ARCHITECTURE})
  set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst")
  set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm")
  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/lib")

endif()

include(CPack)
