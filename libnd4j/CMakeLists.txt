cmake_minimum_required(VERSION 3.15)
project(libnd4j)

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")

#ensure we create lib files
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_WINDOWS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_WINDOWS_BUILD=true")
endif()

option(SD_NATIVE "Optimize for build machine (might not work on others)" OFF)
option(SD_CHECK_VECTORIZATION "checks for vectorization" OFF)
option(SD_BUILD_TESTS "Build tests" OFF)
option(SD_STATIC_LIB "Build static library" OFF)
option(SD_SHARED_LIB "Build shared library" ON)
option(SD_SANITIZE "Enable Address Sanitizer" ON)
option(SD_USE_LTO "Use link time optimization" OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" OFF)
set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Hack to disable flatc build" FORCE)

if(SD_AURORA)
    set(SD_CUDA false)
    set(CMAKE_CXX_STANDARD 14)
else()

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
        SET(INFORMATIVE_FLAGS  "-fmax-errors=3")
    endif()

endif()

if (SD_CUDA AND NOT SD_AURORA)
    message("Enabling cuda")
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    # Add a platform agnostic way of showing we are compiling for cuda
    # The normal way of using __NVCC__ and __CUDACC__ does not show up
    # for cpp files which makes it difficult to add guards for specific functions.
    add_compile_definitions(SD_CUDA=true)
    if("${SD_ARCH}" MATCHES "armv8-a")
        message("Adding jetson nano specific settings")
        # Need to manually specify stubbed links in order
        # for cublas and cusolver to be resolved
        if(NOT DEFINED CUDA_cublas_LIBRARY OR "${CMAKE_cublas_LIBRARY}" MATCHES "NOTFOUND")
            set(CUDA_cublas_LIBRARY $ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so)


        if(NOT DEFINED CUDA_cusolver_LIBRARY OR "${CUDA_cusolver_LIBRARY}" MATCHES "NOTFOUND")
            set(CUDA_cusolver_LIBRARY $ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so)
        endif()
    endif()

    if(CMAKE_CXX_COMPILER MATCHES "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.29" )
        message("Working around windows build failure with visual studio. Visual studio 16.10 introduced a compiler bug with compilng CUDA code with C++14. Set the cuda standard to 17 as a workaround.")
        set(CMAKE_CXX_STANDARD 11)
        set(CMAKE_CUDA_STANDARD 11)
    else()
        set(CMAKE_CXX_STANDARD 11)
        message("Set CUDA C++ Standard to ${CMAKE_CUDA_STANDARD}")

    endif()

    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")


    set(CMAKE_CXX_STANDARD 11)

endif()


#///////////////////////////////////////////////////////////////////////////////
# genCompilation: Generates cpp, cu files
# INPUT:
# $FILE_ITEM template-configuration that utilizes libnd4j type, macros helpers
# defined inside { include/types/types.h, include/system/type_boilerplate.h}
# OUTPUT:
# $CUSTOMOPS_GENERIC_SOURCES  generated files will be added into this List
#////////////////////////////////////////////////////////////////////////////////
#  A simple template-configuration file example:
# // hints and defines what types will be generated
# #cmakedefine SD_COMMON_TYPES_GEN
# #cmakedefine SD_FLOAT_TYPES_GEN
# // below if defines blocks are needed for correctly handling multiple types
# #if defined(SD_COMMON_TYPES_GEN) && defined(SD_COMMON_TYPES_@FL_TYPE_INDEX@)
#  BUILD_DOUBLE_TEMPLATE(template void someFunc, (arg_list,..),
#                          SD_COMMON_TYPES_@FL_TYPE_INDEX@, SD_INDEXING_TYPES);
# #endif
# #if defined(SD_FLOAT_TYPES_GEN) && defined(SD_FLOAT_TYPES_@FL_TYPE_INDEX@)
#  BUILD_SINGLE_TEMPLATE(template class SomeClass,, SD_FLOAT_TYPES_@FL_TYPE_INDEX@);
# #endif
#////////////////////////////////////////////////////////////////////////////////

set_property(GLOBAL PROPERTY JOB_POOLS one_jobs=1 two_jobs=2)


function(genCompilation FILE_ITEM)
    get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)

    set(EXTENSION "cpp")

    if(FL_ITEM MATCHES "cu.in$")
        set(EXTENSION "cu")
    endif()

    file(READ ${FL_ITEM} CONTENT_FL)
    #check content for types

    #set all to false
    set (SD_FLOAT_TYPES_GEN     0)
    set (SD_INTEGER_TYPES_GEN       0)
    set (SD_COMMON_TYPES_GEN   0)
    set (SD_PAIRWISE_TYPES_GEN  0)
    set (RANGE_STOP         -1)

    string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL})

    #STOP=COUNT(SD_COMMON_TYPES)-1
    set(SD_INTEGER_TYPES_END 7)
    set(SD_COMMON_TYPES_END 12)
    set(SD_FLOAT_TYPES_END 3)
    set(SD_PAIRWISE_TYPES_END 12)

    foreach(TYPEX ${TYPE_MATCHES})
        set(STOP -1)
        if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
            set (SD_INTEGER_TYPES_GEN  1)
            set(STOP ${SD_INTEGER_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
            set (SD_COMMON_TYPES_GEN 1)
            set(STOP ${SD_COMMON_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
            set (SD_FLOAT_TYPES_GEN 1)
            set(STOP ${SD_FLOAT_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
            set (SD_PAIRWISE_TYPES_GEN  1)
            set(STOP ${SD_PAIRWISE_TYPES_END})
        endif()
        if(STOP GREATER RANGE_STOP)
            set(RANGE_STOP ${STOP})
        endif()

    endforeach()

    if(RANGE_STOP GREATER -1)
        foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP})
            # set OFF if the index is above
            if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                set (SD_FLOAT_TYPES_GEN     0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                set (SD_INTEGER_TYPES_GEN     0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                set (SD_COMMON_TYPES_GEN   0)
            endif()
            set(GENERATED_SOURCE  "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
            configure_file(  "${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)
            LIST(APPEND CUSTOMOPS_GENERIC_SOURCES ${GENERATED_SOURCE} )
        endforeach()
    endif()

    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()


# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

set(SD_X86_BUILD false)

if (NOT SD_IOS_BUILD AND NOT SD_ANDROID_BUILD AND NOT ${SD_ARCH} MATCHES "power*" AND NOT ${SD_ARCH} MATCHES "arm*" AND NOT SD_AURORA)
    set(SD_X86_BUILD true)
endif()

# -fsanitize=address
# -fsanitize=leak
if (SD_AURORA)
    message("Aurora build in process")
    set(SD_X86_BUILD false)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -w -O3 -fPIC -finline-functions -finline-max-times=64 -finline-max-depth=64 -fno-inline-copy-arguments -fdiag-inline=2 -fdiag-parallel=2 -fdiag-vector=2 -DSD_AURORA=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -w -g -O0 -fPIC -DSD_AURORA=true")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,muldefs,-rpath,$ENV{NLC_ROOT}/lib/")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,muldefs,-rpath,$ENV{NLC_ROOT}/lib/")
elseif (SD_ANDROID_BUILD)
    set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif (APPLE)
    set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O0 -g -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
elseif(WIN32)
    set(SD_X86_BUILD true)
    if (SD_CUDA)
        set(CMAKE_CXX_FLAGS_RELEASE  "-D_RELEASE=true")
        set(CMAKE_CXX_FLAGS_DEBUG  "  /FS /EHsc")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -fPIC -D_RELEASE=true")
        set(CMAKE_CXX_FLAGS_DEBUG  " -g -O2 -fPIC")
    endif()
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Aurora")
    set(CMAKE_CXX_FLAGS_RELEASE  "-w -O3 -fPIC -finline-functions -finline-max-depth=10 -fopenmp -fassociative-math -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "-w -g -O0 -fPIC -fno-openmp -fassociative-math")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,muldefs,-rpath,$ENV{NLC_ROOT}/lib/")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,muldefs,-rpath,$ENV{NLC_ROOT}/lib/")

    set(HAVE_OPENBLAS 1)
    include_directories($ENV{NLC_ROOT}/include/)
    link_directories($ENV{NLC_ROOT}/lib/)
    string(TOUPPER ${CMAKE_BUILD_TYPE} UPPERCASE_CMAKE_BUILD_TYPE)
    if(UPPERCASE_CMAKE_BUILD_TYPE MATCHES RELEASE)
        set(OPENBLAS_LIBRARIES cblas blas_openmp)
    else()
        set(OPENBLAS_LIBRARIES cblas blas_sequential)
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -fPIC -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O0 -fPIC")

    if (SD_CPU AND SD_SANITIZE)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    endif()
endif()

if(SD_NATIVE)
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
        set(SD_X86_BUILD false)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    ENDIF()
endif()


if(NOT SD_CUDA)
    # we need this definition to avoid global memory use within onednn
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true)
    if (SD_AURORA)
        message("In aurora using NLC ROOT $ENV{NLC_ROOT}")
        include_directories("$ENV{NLC_ROOT}/include")
        link_directories("$ENV{NLC_ROOT}/lib/")

        # FIXME: we don't want sequential version in master
        SET(BLAS_LIBNAME "blas_openmp")
        SET(CBLAS_LIBNAME "cblas")

        find_library(BLAS_LIBRARY ${BLAS_LIBNAME}
                HINTS $ENV{NLC_ROOT}
                PATH_SUFFIXES lib)

        find_library(CBLAS_LIBRARY ${CBLAS_LIBNAME}
                HINTS $ENV{NLC_ROOT}
                PATH_SUFFIXES lib)

        if (BLAS_LIBRARY)
            set(HAVE_OPENBLAS 1)
            set(BLAS_LIBRARIES ${CBLAS_LIBRARY} ${BLAS_LIBRARY})
            add_definitions(-D__EXTERNAL_BLAS__=true)
        else()
            message(FATAL_ERROR "Unable to find BLAS")
        endif()
        # there's a chance, we have no BLAS provided externally
    elseif ("${OPENBLAS_PATH}" STREQUAL "")
        #we don't want OpenBLAS on Apple
        if (NOT APPLE)
            # note: this is not a typo
            set(BLA_VENDOR "OpenBLAS")
        endif()

        # look around for system blas instead, see: https://cmake.org/cmake/help/latest/module/FindBLAS.html
        find_package(BLAS REQUIRED)
        if (BLAS_FOUND)
            message("Found external BLAS implementation: ${BLAS_LIBRARIES} ")
            add_definitions(-D__EXTERNAL_BLAS__=true)
        endif()
    else()
        # if we have externally provided OPENBLAS_PATH - let's use it
        set(HAVE_OPENBLAS 1)
        message("Setting openblas")
        include_directories(${OPENBLAS_PATH}/include/)
        link_directories(${OPENBLAS_PATH} ${OPENBLAS_PATH}/lib/)
        set(OPENBLAS_LIBRARIES openblas)
    endif()

    # building cpu_features
    if (SD_X86_BUILD)
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


#arm-compute entry
if(${HELPERS_armcompute})
    find_package(ARMCOMPUTE REQUIRED)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version ERROR_QUIET OUTPUT_VARIABLE ld_version)
    if ("${ld_version}" MATCHES "GNU gold")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold ")
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_link_options("-Wl,--long-plt")
        endif()
    endif()

    if(ARMCOMPUTE_FOUND)
        message("Found ARMCOMPUTE: ${ARMCOMPUTE_LIBRARIES}")
        set(HAVE_ARMCOMPUTE 1)
        # Add preprocessor definition for ARM Compute NEON
        add_definitions(-DARMCOMPUTENEON_ENABLED)
        include_directories(${ARMCOMPUTE_INCLUDE})
        message("----${ARMCOMPUTE_INCLUDE}---")
    endif()


endif()

# new ve-dnn entry
if (${HELPERS_vednn})
    message("Adds Vednn into dependency projects using ExternalProject_Add")
    set(HAVE_VEDNN 1)
    include(ExternalProject)
    set(VEDNN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/vednn-install")
    set(VEDNN_STATIC_LIB
        "${VEDNN_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}dnnl${CMAKE_STATIC_LIBRARY_SUFFIX}")
  
    if(SD_AURORA)
        set (BASH_CMD_VE  bash build.sh -aq$<$<CONFIG:Debug>:d>)
    else()
        set (BASH_CMD_VE  bash build.sh -vjq$<$<CONFIG:Debug>:dddd>)
    endif()
    set(VEDNN_BUILD_COMMAND NLC_HOME=$ENV{NLC_ROOT} VEDNN_INSTALL_DIR=${VEDNN_PREFIX} ${BASH_CMD_VE})
    set(VEDNN_PATCH_COMMAND "patch" "-Np0" < ${CMAKE_SOURCE_DIR}/vednn.patch "||" true)
    ExternalProject_Add(vednn_project
        GIT_REPOSITORY    https://github.com/necla-ml/gen-dnn
        GIT_TAG           ve-v1.4
        UPDATE_DISCONNECTED 1
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/vednn-src"
        PATCH_COMMAND     ${VEDNN_PATCH_COMMAND}
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE   1
        BUILD_COMMAND     ${VEDNN_BUILD_COMMAND}
        BUILD_BYPRODUCTS  ${VEDNN_STATIC_LIB}
        INSTALL_COMMAND cmake -E echo "Skipping install step as its already inside build."
        )

 
    set(VEDNN_INCLUDE "${VEDNN_PREFIX}/include")
    include_directories(${VEDNN_INCLUDE})
    list(APPEND EXTERNAL_DEPENDENCY_PROJECTS vednn_project)
    list(APPEND EXTERNAL_DEPENDENCY_LIBS ${VEDNN_STATIC_LIB} )
endif()


# new one-dnn entry
if (${HELPERS_onednn})
    message("Going to pull & build onednn")
    set(HAVE_ONEDNN 1)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Hack to enforce static mode" FORCE)

    configure_file(./CMakeLists.txt.onednn.in onednn-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onednn-download )
    if(result)
        message(FATAL_ERROR "CMake step for onednn failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onednn-download )
    if(result)
        message(FATAL_ERROR "Build step for onednn failed: ${result}")
    endif()

    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/onednn-src
            ${CMAKE_CURRENT_BINARY_DIR}/onednn-build
            EXCLUDE_FROM_ALL)

    set(onednn_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/onednn-build)
    set(onednn_EXT_DIR ${CMAKE_CURRENT_BINARY_DIR}/onednn-src)
    set(ONEDNN_PATH "${onednn_SOURCE_DIR}")
    include_directories(${onednn_SOURCE_DIR}/include ${onednn_EXT_DIR}/include ${onednn_SOURCE_DIR})
    set(ONEDNN dnnl)
endif()


if (${HELPERS_cudnn})
    if (NOT SD_CUDA)
        message(FATAL_ERROR "Can't build cuDNN on non-CUDA platform")
    endif()

    #set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")

    SET(CUDNN_LIBNAME "cudnn")

    if(DEFINED ENV{CUDNN_ROOT_DIR})
        message("Using cudnn root directory from environment")
        set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR})
    endif()
    if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
        message("Using cuda root directory from environment")
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
    endif()
    message("Cudnn root dir ${CUDNN_ROOT_DIR} CUDA TOOLKIT ROOT DIR ${CUDA_TOOLKIT_ROOT_DIR}")
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES cuda/include include)


    find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 cuda/lib/x64 lib/x64)


    #find_library(CULIBOS_LIBRARY ${CULIBOS_LIBNAME}
    #        HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    #        PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)


    if (CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message("CUDNN include dir is ${CUDNN_INCLUDE_DIR}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true)
        set(CUDNN ${CUDNN_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find cuDNN, no library found. Please ensure cudnn is installed under the cuda toolkit directory.")
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


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(blas)
if(SD_BUILD_TESTS)
    # tests are always compiled with all ops included
    set(SD_ALL_OPS true)
    set(SD_BUILD_MINIFIER true)
    add_subdirectory(tests_cpu)
endif()


if (MSVC_DEV)
    set(SD_BUILD_MINIFIER false)
endif ()

set (CMAKE_INSTALL_PREFIX $ENV{ND4J_HOME}/nd4j-native-parent/nd4j-native/src/main/resources)

# Set package information
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations for nd4j.")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "agibsonccc <adam@skymind.global>")
set(CPACK_PACKAGE_VENDOR "Eclipse")
set(CPACK_SETDESTDIR "false")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/lib")
set(CPACK_PACKAGE_NAME "libnd4j")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "8")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "libnd4j")
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
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/eclipse/deeplearning4j")
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
    set(CPACK_PACKAGE_CONTACT "agibsonccc")
    set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
    set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
    set(CPACK_RPM_PACKAGE_SUGGESTS "cuda")
    # Build deps: atlas blas lapack cmake3 devtoolset-4-gcc devtoolset-4-gcc-c++
    set(CPACK_RPM_PACKAGE_REQUIRES "")
    set(CPACK_RPM_PACKAGE_URL "https://github.com/eclipse/deeplearning4j/libnd4j")
    set(CPACK_GENERATOR "RPM")
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.fc${RELEASE}.${CPACK_RPM_PACKAGE_ARCHITECTURE})
    set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst")
    set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm")
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/lib")

endif()

include(CPack)
