project(libnd4j)
set(CMAKE_VERBOSE_MAKEFILE OFF)
option(NATIVE "Optimize for build machine (might not work on others)" OFF)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#ensure we create lib files
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
option(CHECK_VECTORIZATION "checks for vectorization" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" OFF)
set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Hack to disable flatc build" FORCE)

set(CMAKE_CXX_STANDARD 11)
if (CUDA_BLAS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)

    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
endif()

# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

set(X86_BUILD false)

if (NOT IOS_BUILD AND NOT ANDROID_BUILD AND NOT ${ARCH} MATCHES "power*" AND NOT ${ARCH} MATCHES "arm*")
    set(X86_BUILD true)
endif()

# -fsanitize=address
# -fsanitize=leak
if (ANDROID_BUILD)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif (APPLE)
    set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O0 -g -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
elseif(WIN32)
    set(X86_BUILD true)
    if (CUDA_BLAS)
        set(CMAKE_CXX_FLAGS_RELEASE  "-D_RELEASE=true")
        set(CMAKE_CXX_FLAGS_DEBUG  "  /FS /EHsc")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -fPIC -fmax-errors=2 -D_RELEASE=true")
        set(CMAKE_CXX_FLAGS_DEBUG  " -g -O2 -fPIC -fmax-errors=2")
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -fPIC -fmax-errors=2 -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O0 -fPIC -fmax-errors=2")

    if (CPU_BLAS)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    endif()
endif()

if(NATIVE)
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
        set(X86_BUILD false)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    ENDIF()
endif()


if(NOT CUDA_BLAS)
    # we need this definition to avoid global memory use within mkldnn
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true)

    # there's a chance, we have no BLAS provided externally
    if ("${OPENBLAS_PATH}" STREQUAL "")
        #we don't want static OpenBLAS on Apple
        set(BLA_STATIC ON)
        if (NOT APPLE)
            set(BLA_VENDOR "OpenBLAS")
        endif()

        # look around for system blas instead
        find_package(BLAS REQUIRED)
        if (BLAS_FOUND)
            message("Original library: ${BLAS_LIBRARIES}")
            # workaround for for cmake being unable to find static blas library
            SET(_TMP_B "")
            if (APPLE)
                string(REGEX REPLACE "\\.dylib$" ".lib" _TMP_B "${BLAS_LIBRARIES}")
            elseif (WIN32)
                string(REGEX REPLACE "\\.dll" ".lib" _TMP_B "${BLAS_LIBRARIES}")
            else()
                string(REGEX REPLACE "\\.so$" ".a" _TMP_B "${BLAS_LIBRARIES}")
            endif()
            set(BLAS_LIBRARIES "${_TMP_B}")

            message("Found external BLAS implementation: ${BLAS_LIBRARIES} ")
            add_definitions(-D__EXTERNAL_BLAS__=true)
        elseif(WIN32)
            message("BLAS not found, using downloaded OpenBLAS instead")
            add_definitions(-D__EXTERNAL_BLAS__=true)
        endif()
    else()
        # if we have externally provided OPENBLAS_PATH - let's use it
        set(HAVE_OPENBLAS 1)
        include_directories(${OPENBLAS_PATH}/include/)
        link_directories(${OPENBLAS_PATH} ${OPENBLAS_PATH}/lib/)
        set(OPENBLAS_LIBRARIES openblas)
    endif()

    # building cpu_features
    if (X86_BUILD)
        add_definitions(-DCPU_FEATURES=true)
        set(BUILD_PIC "ON" CACHE STRING "Hack to enforce fPIC mode" FORCE)
        configure_file(./CMakeLists.txt.cpu_features.in cpu_features-download/CMakeLists.txt)
        message("CMAKE_COMMAND: ${CMAKE_COMMAND}")
        execute_process(COMMAND ${CMAKE_COMMAND} -DBUILD_PIC=ON -G "${CMAKE_GENERATOR}" .
                RESULT_VARIABLE result
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpu_features-download )

        if(result)
            message(FATAL_ERROR "CMake step for cpu_features failed: ${result}")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} --build .
                RESULT_VARIABLE result
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpu_features-download )
        if(result)
            message(FATAL_ERROR "Build step for cpu_features failed: ${result}")
        endif()

        add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/cpu_features-src
                ${CMAKE_CURRENT_BINARY_DIR}/cpu_features-build
                EXCLUDE_FROM_ALL)
        set(CPUF_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu_features-src)
        include_directories(${CPUF_SOURCE_DIR}/include)
        set(CPU_FEATURES cpu_features)
    endif()
endif()


# new mkl-dnn entry
if (${HELPERS_mkldnn})
    message("Going to pull & build mkldnn")
    set(HAVE_MKLDNN 1)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Hack to enforce static mode" FORCE)

    configure_file(./CMakeLists.txt.mkldnn.in mkldnn-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mkldnn-download )
    if(result)
        message(FATAL_ERROR "CMake step for mkldnn failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mkldnn-download )
    if(result)
        message(FATAL_ERROR "Build step for mkldnn failed: ${result}")
    endif()

    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/mkldnn-src
            ${CMAKE_CURRENT_BINARY_DIR}/mkldnn-build
            EXCLUDE_FROM_ALL)

    set(mkldnn_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/mkldnn-build)
    set(mkldnn_EXT_DIR ${CMAKE_CURRENT_BINARY_DIR}/mkldnn-src)
    set(MKLDNN_PATH "${mkldnn_SOURCE_DIR}")
    include_directories(${mkldnn_SOURCE_DIR}/include ${mkldnn_EXT_DIR}/include ${mkldnn_SOURCE_DIR})
    set(MKLDNN dnnl)
endif()


if (${HELPERS_cudnn})
    if (NOT CUDA_BLAS)
        message(FATAL_ERROR "Can't build cuDNN on non-CUDA platform")
    endif()

    set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")

    # FIXME: we don't want static library in master
    SET(CUDNN_LIBNAME "cudnn")
    SET(CULIBOS_LIBNAME "culibos")

    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES cuda/include include)

    find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

    find_library(CULIBOS_LIBRARY ${CULIBOS_LIBNAME}
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)


    if (CUDNN_LIBRARY)
        set(HAVE_CUDNN true)
        set(CUDNN ${CUDNN_LIBRARY} ${CULIBOS_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find cuDNN")
    endif()
endif()

# Download and unpack flatbuffers at configure time
configure_file(CMakeLists.txt.in flatbuffers-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download )
if(result)
  message(FATAL_ERROR "CMake step for flatbuffers failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download )
if(result)
  message(FATAL_ERROR "Build step for flatbuffers failed: ${result}")
endif()

# Add flatbuffers directly to our build.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src
                 ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
                 EXCLUDE_FROM_ALL)

set(HAVE_FLATBUFFERS 1)
set(FLATBUFFERS_PATH ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src)
include_directories(${FLATBUFFERS_PATH}/include)



configure_file(include/config.h.in include/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

if (NOT DEFINED ENV{CLION_IDE})
    message("NOT CLION")
    include_directories(blas/ include/ include/helpers include/loops include/graph include/execution include/ops include/types include/array include/cnpy include/exceptions)
    add_subdirectory(blas)
    if(BUILD_TESTS)
        # tests are always compiled with all ops included
        set(LIBND4J_ALL_OPS true)
        set(LIBND4J_BUILD_MINIFIER true)
        add_subdirectory(tests_cpu)
    endif()
endif ()

if ($ENV{CLION_IDE})
    add_subdirectory(tests_cpu)
endif ()

if (MSVC_DEV)
    set(LIBND4J_BUILD_MINIFIER false)
endif ()

set (CMAKE_INSTALL_PREFIX $ENV{ND4J_HOME}/nd4j-native-parent/nd4j-native/src/main/resources)

# Set package information
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations for nd4j.")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "raver119 <raver119@skymind.io>")
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
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "raver119")
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
  set(CPACK_PACKAGE_CONTACT "raver119")
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
