################################################################################
#
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
#  See the NOTICE file distributed with this work for additional
#   information regarding copyright ownership.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

set(CMAKE_VERBOSE_MAKEFILE OFF)
include(CheckCXXCompilerFlag)
if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()

if(APPLE)
    message("Using apple")
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()

if (SD_APPLE_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_APPLE_BUILD=true -mmacosx-version-min=10.10")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_APPLE_BUILD=true -mmacosx-version-min=10.10")
endif()

if (SD_ARM_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_ARM_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_ARM_BUILD=true")
endif()


if (SD_ANDROID_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_ANDROID_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_ANDROID_BUILD=true")
endif()

if (SD_IOS_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_IOS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_IOS_BUILD=true")
endif()

if(WIN32 AND NOT ANDROID)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()

    # workaround for long command lines
    SET(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
    SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)

    SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
    SET(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")

    SET(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")
endif()

if(SD_USE_LTO)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Use link time optimizations")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

set(DEFINITIONS_CONTENT "")
if ("${SD_ALL_OPS}" OR "${SD_OPS_LIST}" STREQUAL "")
    message("Adding all ops due to empty op list or SD_ALL_OPS definition: SD ALL OPS: ${SD_ALL_OPS} SD_OPS_LIST: ${SD_OPS_LIST} ")
    add_compile_definitions(SD_ALL_OPS=1)
    string(APPEND DEFINITIONS_CONTENT  "#define SD_ALL_OPS 1\n")
else()
    message("_OPS: ${SD_OPS_LIST}")
    foreach(OP ${SD_OPS_LIST})
        add_compile_definitions(OP_${OP}=1)
        message(STATUS "OP: ${OP}")
        string(APPEND DEFINITIONS_CONTENT  "#define OP_${OP} 1\n")
    endforeach()
endif()

LIST(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
if(SD_TYPES_LIST_COUNT GREATER 0)
    add_compile_definitions(SD_SELECTIVE_TYPES)
    string(APPEND DEFINITIONS_CONTENT  "#define SD_SELECTIVE_TYPES\n")
    foreach(SD_TYPE ${SD_TYPES_LIST})
        string(TOUPPER ${SD_TYPE} SD_TYPE_UPPERCASE)
        add_compile_definitions(HAS_${SD_TYPE_UPPERCASE})
        message(STATUS "TYPE: ${SD_TYPE_UPPERCASE}")
        string(APPEND DEFINITIONS_CONTENT  "#define HAS_${SD_TYPE_UPPERCASE}\n")
    endforeach()
endif()
if(OP_OUTPUT_FILE MATCHES ".h$")
    message("definitions will be written to \"${OP_OUTPUT_FILE}\"")
    file(WRITE "${OP_OUTPUT_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")
endif()

IF(${SD_ARCH} MATCHES "armv8")
    set(ARCH_TUNE "-march=${SD_ARCH}")
ELSEIF(${SD_ARCH} MATCHES "armv7")
    set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon ")
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Aurora")
    set_source_files_properties(../include/graph/impl/GraphHolder.cpp PROPERTIES COMPILE_FLAGS -g0)
ELSEIF(${SD_ARCH} MATCHES "power*")
    set(ARCH_TUNE "-mcpu=${SD_ARCH} -mtune=${SD_ARCH} -D__POWER")
ELSEIF(${SD_EXTENSION} MATCHES "avx2")
    message("Building AVX2 binary...")
    set(ARCH_TUNE "-mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mprefetchwt1 -DSD_F16C=true -DF_AVX2=true")
    check_cxx_compiler_flag("-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" NO_AVX256_SPLIT)
    if(NO_AVX256_SPLIT)
        set(ARCH_TUNE "${ARCH_TUNE} -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
    endif(NO_AVX256_SPLIT)
ELSE()
    if ("${SD_ARCH}" STREQUAL "x86-64")
        message("Building x86_64 binary...")
        set(ARCH_TYPE "generic")
        add_compile_definitions(F_X64=true)
    else()
        set(ARCH_TYPE "${SD_ARCH}")
    endif()

    IF(${SD_EXTENSION} MATCHES "avx512")
        message("Building AVX512 binary...")
        # we need to set flag here, that we can use hardware f16 conversion + tell that cpu features should be tracked
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mavx512f -mavx512vl -mavx512bw -mavx512dq  -mavx512cd -mbmi -mbmi2 -mprefetchwt1 -mclflushopt -mxsavec -mxsaves -DSD_F16C=true -DF_AVX512=true")
    endif()

    if (NOT WIN32)
        # we don't want this definition for msvc
        set(ARCH_TUNE "-march=${SD_ARCH} -mtune=${ARCH_TYPE}")
    endif()
ENDIF()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" AND SD_X86_BUILD)
    # apple clang but not ios-arm
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -O3 -fp-model fast")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
    set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"  AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Aurora")
    # using GCC
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} ${INFORMATIVE_FLAGS}  -std=c++11")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,-z,,--verbose")
    # keep this, we need to ensure that all symbols are defined or it may fail on mac based platforms
    if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT(APPLE) AND NOT(WIN32))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic,--verbose")
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic,--verbose")
    endif()

    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        # See: https://github.com/bombela/backward-cpp

        # Introduce variables:
        # * CMAKE_INSTALL_LIBDIR
        # * CMAKE_INSTALL_BINDIR
        # * CMAKE_INSTALL_INCLUDEDIR

        # check if compiler is nvcc or nvcc_wrapper
        set(COMPILER_IS_NVCC false)
        get_filename_component(COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
        if (COMPILER_NAME MATCHES "^nvcc")
            set(COMPILER_IS_NVCC true)
        endif()

        if (DEFINED ENV{OMPI_CXX} OR DEFINED ENV{MPICH_CXX})
            if ( ($ENV{OMPI_CXX} MATCHES "nvcc") OR ($ENV{MPICH_CXX} MATCHES "nvcc") )
                set(COMPILER_IS_NVCC true)
            endif()
        endif()

        # set CXX standard
        set(CMAKE_CXX_STANDARD_REQUIRED True)
        if (${COMPILER_IS_NVCC})
            # GNU CXX extensions are not supported by nvcc
            set(CMAKE_CXX_EXTENSIONS OFF)
        endif()

        # Set C++ compiler and flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MT -pthread  -Bsymbolic   -lbfd -rdynamic  -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls  -rdynamic -finstrument-functions -g -O0")
        add_compile_definitions(SD_GCC_FUNCTRACE)
    endif()
endif()

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include")
    list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
if(!SD_CUDA)
    if(!SD_CPU)
        set(SD_CUDA FALSE)
        set(SD_CPU TRUE)
    endif()
endif()

#if ONEDNN is enabled - we're building onednn-powered helpers
if (HAVE_ONEDNN)
    file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES false ../include/ops/declarable/platform/mkldnn/*.cpp ../include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
endif()

if (HAVE_VEDNN)
    file(GLOB_RECURSE CUSTOMOPS_VEDNN_SOURCES false ../include/ops/declarable/platform/vednn/*.cpp ../include/ops/declarable/platform/vednn/*.h)
    if (HAVE_VEDA)
        file(GLOB_RECURSE VEDA_SOURCES false ../include/ops/declarable/platform/vednn/*.vc ../include/ops/declarable/platform/vednn/*.vcpp ../include/ops/declarable/platform/vednn/*.h)
    endif()
endif()

if(HAVE_ARMCOMPUTE)
    file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES false ../include/ops/declarable/platform/armcompute/*.cpp ../include/ops/declarable/platform/armcompute/*.h)
endif()

if(SD_CUDA)
    message("Build cublas")

    add_definitions(-D__CUDABLAS__=true)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set (CMAKE_CXX_FLAGS "")
    endif()


    if (CUDA_FOUND)
        message("CUDA include directory: ${CUDA_INCLUDE_DIRS}")
        include_directories(${CUDA_INCLUDE_DIRS})
        message("CUDA found!")
        if ("${SD_EXPERIMENTAL}" STREQUAL "yes")
            message("Experimental mode ENABLED")
            set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DSD_EXPERIMENTAL_ENABLED=true")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_EXPERIMENTAL_ENABLED=true -allow-unsupported-compiler")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_EXPERIMENTAL_ENABLED=true -allow-unsupported-compiler")
            set(EXPM " -DSD_EXPERIMENTAL_ENABLED=true")
        endif()


        # the only difference for debug mode here is host/device debug symbols
        set(CMAKE_CUDA_FLAGS_DEBUG " -G -g")
        # we need -fPIC on Linux/GCC
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            message("Enabling fPIC...")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
            #enable gnu extensions
            # functrace works for cuda as well as long as the underlying compiler is gcc
            if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
                # Set C++ compiler and flags
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DSD_GCC_FUNCTRACE=1 -Bsymbolic  -lbfd  -rdynamic  -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls  -rdynamic -finstrument-functions -g -O0")
                # note that cuda can't use -MT and -finstrument-functions
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}   -DSD_GCC_FUNCTRACE=1 -Xcompiler=-DSD_GCC_FUNCTRACE=1 -ldl  -lbfd  -lunwind -ldwv -Xcompiler=-Bsymbolic   -Xcompiler=-rdynamic -Xcompiler=-fno-omit-frame-pointer -Xcompiler=-fno-optimize-sibling-calls  -g -O0")
            else()
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

            endif()
        endif()

        if(WIN32)
            message("In windows, setting cublas library and cusolver library")
            if(NOT DEFINED CUDA_cublas_LIBRARY)
                set(CUDA_cublas_LIBRARY ${CUDA_HOME}/lib/x64/cublas.lib)
            endif()

            if(NOT DEFINED CUDA_cusolver_LIBRARY)
                set(CUDA_cusolver_LIBRARY ${CUDA_HOME}/lib/x64/cusolver.lib)
            endif()
        endif()

        if("${SD_ARCH}" MATCHES "armv8-a" AND UNIX)
            message("Adding jetson nano specific settings")
            # Need to manually specify stubbed links in order
            # for cublas and cusolver to be resolved
            if(NOT DEFINED CUDA_cublas_LIBRARY OR "${CUDA_cublas_LIBRARY}" MATCHES ".*NOTFOUND.*" )
                message("Setting cublas library manually")
                set(CUDA_cublas_LIBRARY  "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so" CACHE STRING "CUDA CUBLAS LIB" FORCE)
                set(CUDA_cublas_LIBRARY  "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so")
                # See: https://stackoverflow.com/questions/47948473/what-this-cmake-error-means-my-variables-are-set-to-notfound
                unset(CUDA_cublas-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND PARENT_SCOPE)


            endif()

            if(NOT DEFINED CUDA_cusolver_LIBRARY OR "${CUDA_cusolver_LIBRARY}" MATCHES ".*NOTFOUND.*")
                message("Setting cusolver library manually")
                set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so" CACHE STRING "CUDA CUSOLVER LIB" FORCE)
                set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so")
                #See: https://stackoverflow.com/questions/47948473/what-this-cmake-error-means-my-variables-are-set-to-notfound
                unset(CUDA_cusolver-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND PARENT_SCOPE)

            endif()

            message("Jetson nano cublas library is ${CUDA_cublas_LIBRARY} and CuSolver library ${CUDA_cusolver_LIBRARY}")
        endif()



        string( TOLOWER "${COMPUTE}" COMPUTE_CMP )
        if ("${COMPUTE_CMP}" STREQUAL "all")
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Common")
        elseif("${COMPUTE_CMP}" STREQUAL "auto")
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Auto")
        elseif(COMPUTE_CMP MATCHES "^[0-9]+$")
            #matches USER COMPUTE old way
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_${COMPUTE},code=sm_${COMPUTE} ")
        else()
            #matches numbers NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
            #NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal
            #NUM:   2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2 et cetera
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${COMPUTE}")
        endif()
        # list to spaces
        string (REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")

        set(CMAKE_CUDA_FLAGS " -allow-unsupported-compiler ${CMAKE_CUDA_FLAGS} -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all ${CUDA_ARCH_FLAGS}")
        file(GLOB_RECURSE PERF_SOURCES false ../include/performance/*.cpp ../include/performance/*.h)
        file(GLOB_RECURSE EXCEPTIONS_SOURCES false ../include/exceptions/*.cpp ../include/exceptions/*.h)
        file(GLOB_RECURSE EXEC_SOURCES false ../include/execution/impl/*.cpp ../include/execution/*.cu ../include/execution/*.h)
        file(GLOB_RECURSE TYPES_SOURCES false ../include/types/*.cpp ../include/types/*.h)
        file(GLOB_RECURSE ARRAY_SOURCES false ../include/array/impl/*.cpp ../include/array/cuda/*.cu ../include/array/*.h)
        file(GLOB_RECURSE MEMORY_SOURCES false ../include/memory/impl/*.cpp ../include/memory/cuda/*.cu ../include/memory/*.h)
        file(GLOB_RECURSE GRAPH_SOURCES false ../include/graph/*.cpp ../include/graph/*.cu ../include/graph/*.h)
        file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../include/ops/declarable/generic/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES false ../include/ops/declarable/helpers/cuda/*.cu ../include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE OPS_SOURCES false ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp  ../include/ops/*.h)
        file(GLOB_RECURSE HELPERS_SOURCES false ../include/build_info.cpp ../include/ConstMessages.cpp ../include/helpers/impl/*.cpp ../include/helpers/*.cu ../include/helpers/*.cupp ../include/helpers/*.h)
        file(GLOB_RECURSE INDEXING_SOURCES false ../include/indexing/*.cpp ../include/indexing/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES false ../include/loops/impl/*.cpp ../include/loops/*.h)
        file(GLOB_RECURSE LEGACY_SOURCES false ../include/legacy/impl/*.cpp  ../include/legacy/*.cu ../include/legacy/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA false ../include/loops/*.cu)
        file(GLOB_RECURSE HELPERS_CPP  false   ../include/helpers/impl/shape.cpp )
        file(GLOB_RECURSE COMPILATION_UNITS false ../include/loops/cuda/compilation_units/*.cu.in
                ../include/ops/impl/compilation_units/*.cpp.in)



        foreach(FL_ITEM ${COMPILATION_UNITS})
            genCompilation(FL_ITEM)
        endforeach()

        if (HAVE_CUDNN)
            message("cuDNN included")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES false ../include/ops/declarable/platform/cudnn/*.cu)
        endif()


        if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
            # TODO: come back for backward
            add_library(samediff_obj OBJECT  ${LOOPS_SOURCES_CUDA} ${LEGACY_SOURCES} ${HELPERS_CPP}
                    ${CUSTOMOPS_HELPERS_SOURCES} ${HELPERS_SOURCES} ${EXEC_SOURCES}
                    ${LOOPS_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
                    ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${EXCEPTIONS_SOURCES} ${OPS_SOURCES} ${PERF_SOURCES} ${CUSTOMOPS_CUDNN_SOURCES} ${CUSTOMOPS_ONEDNN_SOURCES}
                    ${CUSTOMOPS_ARMCOMPUTE_SOURCES} ${CUSTOMOPS_GENERIC_SOURCES}
                    )


        else()
            add_library(samediff_obj OBJECT  ${LOOPS_SOURCES_CUDA} ${LEGACY_SOURCES} ${HELPERS_CPP}
                    ${CUSTOMOPS_HELPERS_SOURCES} ${HELPERS_SOURCES} ${EXEC_SOURCES}
                    ${LOOPS_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
                    ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${EXCEPTIONS_SOURCES} ${OPS_SOURCES} ${PERF_SOURCES} ${CUSTOMOPS_CUDNN_SOURCES} ${CUSTOMOPS_ONEDNN_SOURCES}
                    ${CUSTOMOPS_ARMCOMPUTE_SOURCES} ${CUSTOMOPS_GENERIC_SOURCES}
                    )

        endif()



        target_include_directories(samediff_obj PUBLIC ${EXTERNAL_INCLUDE_DIRS})

        if (WIN32)
            message("MSVC runtime for library: ${MSVC_RT_LIB}")
        endif()

        # build shared library by default or when it's explicitly requested
        if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        endif()

        if (SD_STATIC_LIB AND SD_SHARED_LIB)
            # if both static and shared library are going to be built - static library will have special suffix
            add_library(${SD_LIBRARY_NAME}static STATIC $<TARGET_OBJECTS:samediff_obj>)
            set_property(TARGET ${SD_LIBRARY_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME}static  DESTINATION .)
        elseif(SD_STATIC_LIB)
            # if we only build static library - use this name
            add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
            set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME}  DESTINATION .)
        endif()

        # on windows we want to make sure we use MT or MD, but since we use it in one lib, we must use it everywhere to avoid conflicts
        set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        if(WIN32)
            message("CUDA on Windows: enabling /EHsc")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj /std:c++14")
        endif()

        target_link_libraries(${SD_LIBRARY_NAME}  ${CALLTRACE_LIBS}  ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDNN} ${MKLDNN})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cuda)

        install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)
    endif(CUDA_FOUND)
elseif(SD_CPU OR SD_AURORA)

    if ("${SD_EXPERIMENTAL}" STREQUAL "yes")
        message("Experimental mode ENABLED")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_EXPERIMENTAL_ENABLED=true")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_EXPERIMENTAL_ENABLED=true")
    endif()

    file(GLOB_RECURSE PERF_SOURCES false ../include/performance/*.cpp ../include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES false ../include/exceptions/*.cpp ../include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES false ../include/execution/*.cpp ../include/execution/*.h)
    file(GLOB_RECURSE TYPES_SOURCES false ../include/types/*.cpp ../include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES false ../include/array/*.cpp ../include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES false ../include/memory/*.cpp ../include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES false ../include/graph/*.cpp ../include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_GENERIC_SOURCES false ../include/ops/declarable/helpers/cpu/*.cpp ../include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES false ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp  ../include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES false ../include/indexing/*.cpp ../include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES false ../include/build_info.cpp ../include/ConstMessages.cpp ../include/helpers/*.cpp ../include/helpers/*.h)
    file(GLOB_RECURSE LEGACY_SOURCES false ../include/legacy/impl/*.cpp  ../include/legacy/cpu/*.cpp ../include/legacy/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES false ../include/loops/*.cpp ../include/loops/*.h)


    file(GLOB_RECURSE COMPILATION_UNITS false ../include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in
            ../include/loops/cpu/compilation_units/*.cpp.in ../include/helpers/cpu/loops/*.cpp.in
            ../include/ops/impl/compilation_units/*.cpp.in)

    foreach(FL_ITEM ${COMPILATION_UNITS})
        genCompilation(FL_ITEM)
    endforeach()

    if (SD_X86_BUILD)
        # we disable platform optimizations for certains files for linux/macos
        set_source_files_properties(cpu/NativeOps.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        set_source_files_properties(../include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
    endif()


    if(SD_CHECK_VECTORIZATION)
        set(VECT_FILES cpu/NativeOps.cpp ${OPS_SOURCES} ${HELPERS_SOURCES} ${CUSTOMOPS_GENERIC_SOURCES} ${LOOPS_SOURCES})
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

            if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
                set(CHECK_VECT_FLAGS "-ftree-vectorize -fsave-optimization-record")
                #to process fsave-optimization-record we will need our cython version code
                message("Build Auto vectorization helpers")
                execute_process(COMMAND "python3" "${CMAKE_CURRENT_SOURCE_DIR}/../auto_vectorization/cython_setup.py" "build_ext" "--inplace" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../auto_vectorization/" RESULT_VARIABLE ret)
                message("build='${ret}'")

                #remove fail cases that gcc fails produce sometimes
                file(GLOB_RECURSE FAILURE_CASES false ../include/loops/cpu/compilation_units/reduce3*.cpp)
                #message("*****${FAILURE_CASES}")
                foreach(FL_ITEM ${FAILURE_CASES})
                    message("Removing failure cases ${FL_ITEM}")
                    list(REMOVE_ITEM VECT_FILES ${FL_ITEM})
                endforeach()
            else()
                set(CHECK_VECT_FLAGS "-ftree-vectorize -fopt-info-vec-optimized-missed")
            endif()
            message("CHECK VECTORIZATION ${CHECK_VECT_FLAGS}")
            set_source_files_properties( ${VECT_FILES}  PROPERTIES COMPILE_FLAGS "${CHECK_VECT_FLAGS}" )
        endif()
    endif()

    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)

    if(NOT SD_ALL_OPS)
        message("Not all SD OPS INCLUDED")
        message("Scanning PERF_SOURCES")
        foreach(SRC_FILE ${PERF_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" "${PERF_SOURCES}")
        endforeach()

        message("Scanning EXCEPTIONS_SOURCES")

        foreach(SRC_FILE ${EXCEPTIONS_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${EXCEPTIONS_SOURCES}")
        endforeach()
        message("Scanning EXEC_SOURCES")

        foreach(SRC_FILE ${EXEC_SOURCES})
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${EXEC_SOURCES}")
        endforeach()
        message("Scanning TYPES_SOURCES")

        foreach(SRC_FILE ${TYPES_SOURCES})
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${TYPES_SOURCES}" )
        endforeach()

        message("Scanning ARRAY_SOURCES")

        foreach(SRC_FILE ${ARRAY_SOURCES})
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${ARRAY_SOURCES}")
        endforeach()
        message("Scanning MEMORY_SOURCES")

        foreach(SRC_FILE ${MEMORY_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${MEMORY_SOURCES}")
        endforeach()
        message("Scanning GRAPH_SOURCES")

        foreach(SRC_FILE ${GRAPH_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${GRAPH_SOURCES}")
        endforeach()
        message("Scanning CUSTOMOPS_SOURCES")

        foreach(SRC_FILE ${CUSTOMOPS_SOURCES})
            message("SRC FILE ${SRC_FILE}")

            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${CUSTOMOPS_SOURCES}" )
        endforeach()
        message("Scanning OPS_SOURCES")

        foreach(SRC_FILE ${OPS_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${OPS_SOURCES}")
        endforeach()
        message("Scanning HELPERS_SOURCES")

        foreach(SRC_FILE ${HELPERS_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${HELPERS_SOURCES}" )
        endforeach()
        message("Scanning INDEXING_SOURCES")


        foreach(SRC_FILE ${INDEXING_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}"  LIST_ITEM "${INDEXING_SOURCES}")
        endforeach()
        message("Scanning LOOPS_SOURCES")

        foreach(SRC_FILE ${LOOPS_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${LOOPS_SOURCES}" )
        endforeach()
        message("Scanning LEGACY_SOURCES")

        foreach(SRC_FILE ${LEGACY_SOURCES})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${LEGACY_SOURCES}")
        endforeach()
        message("Scanning LOOPS_SOURCES_CUDA")

        foreach(SRC_FILE ${LOOPS_SOURCES_CUDA})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${LOOPS_SOURCES_CUDA}")
        endforeach()
        message("Scanning HELPERS_CPP")

        foreach(SRC_FILE ${HELPERS_CPP})
            message("SRC FILE ${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${HELPERS_CPP}")
        endforeach()
    endif()

    foreach(FL_ITEM ${COMPILATION_UNITS})
        genCompilation(FL_ITEM)
    endforeach()

    if (HAVE_CUDNN)
        message("cuDNN included")
        file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES false ../include/ops/declarable/platform/cudnn/*.cu)
    endif()

    add_library(samediff_obj OBJECT ${LEGACY_SOURCES} ${CALLSTACK_SRC}
            ${LOOPS_SOURCES} ${CALLSTACK_SRC} ${HELPERS_SOURCES} ${EXEC_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
            ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${EXCEPTIONS_SOURCES} ${INDEXING_SOURCES} ${CUSTOMOPS_ONEDNN_SOURCES}
            ${CUSTOMOPS_VEDNN_SOURCES}
            ${CUSTOMOPS_ARMCOMPUTE_SOURCES} ${CUSTOMOPS_GENERIC_SOURCES} ${OPS_SOURCES} ${PERF_SOURCES})

    target_include_directories(samediff_obj PUBLIC ${EXTERNAL_INCLUDE_DIRS})

    foreach(external_dependency ${EXTERNAL_DEPENDENCY_PROJECTS})
        message("@external ${external_dependency}")
        add_dependencies(samediff_obj ${external_dependency})
    endforeach()
    if(IOS)
        add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
    else()
        # build shared library by default or when it's explicitly requested
        if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
            if(ANDROID)
                # See: https://www.scivision.dev/cmake-ninja-job-pool-limited-memory/
                # See: https://cmake.org/cmake/help/v3.0/command/cmake_host_system_information.html
                # See: https://cmake.org/cmake/help/latest/prop_gbl/JOB_POOLS.html
                cmake_host_system_information(RESULT _logical_cores QUERY NUMBER_OF_LOGICAL_CORES)
                if(_logical_cores LESS 4)
                    set_target_properties(${SD_LIBRARY_NAME} PROPERTIES JOB_POOL_COMPILE one_jobs)
                endif()
            endif()
        endif()

        if (SD_STATIC_LIB AND SD_SHARED_LIB)
            # if both static and shared library are going to be built - static library will have special suffix
            add_library(${SD_LIBRARY_NAME}static STATIC $<TARGET_OBJECTS:samediff_obj>)
            set_property(TARGET ${SD_LIBRARY_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME}static  DESTINATION .)
        elseif(SD_STATIC_LIB)
            # if we only build static library - use this name
            add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
            set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME}  DESTINATION .)
        endif()
    endif()

    # we're including {MKLDNN} here in case of building from sources. in future that'll replace {MKLDNN_LIBRARIES}. same applies to BLAS
    if (NOT BLAS_LIBRARIES)
        set(BLAS_LIBRARIES "")
    endif()
    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()

    #This breaks the build. Normally you want to run tests anyways.
    if(NOT "$ENV{CLION_IDE}")
        target_link_libraries(${SD_LIBRARY_NAME} ${CALLTRACE_LIBS}   ${EXTERNAL_DEPENDENCY_LIBS} ${CALL_STACK} ${ONEDNN}  ${ONEDNN_LIBRARIES} ${ARMCOMPUTE_LIBRARIES} ${OPENBLAS_LIBRARIES} ${BLAS_LIBRARIES} ${CPU_FEATURES})
        target_include_directories (${SD_LIBRARY_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS})
    endif()

    if (HAVE_VEDA)
        message("----${VEDA_SOURCES}---${VEDA_LIBRARY}")
        ADD_LIBRARY (${SD_LIBRARY_NAME}_device SHARED ${VEDA_SOURCES})
        target_link_libraries(${SD_LIBRARY_NAME}_device PRIVATE ${VEDA_DEPENDENCY_LIBS})
        target_include_directories(${SD_LIBRARY_NAME}_device PRIVATE ${VEDA_INCLUDE_DIRS})
        if (CMAKE_BUILD_TYPE STREQUAL "Debug" )
            target_compile_options(${SD_LIBRARY_NAME}_device PRIVATE -O0 -g -traceback )
        else()
            target_compile_options(${SD_LIBRARY_NAME}_device PRIVATE -O4 -fPIC -fno-defer-inline-template-instantiation  -msched-block -finline-functions -finline-max-times=64 -finline-max-depth=64 -fno-inline-copy-arguments -fdiag-inline=2 -fdiag-parallel=2 -fdiag-vector=2)
        endif()
    endif()



    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
        message(FATAL_ERROR "You need at least GCC 4.9")
    endif()

    # OpenMP works well pretty much only with GCC
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()

    install(TARGETS ${SD_LIBRARY_NAME} DESTINATION  .)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cpu)
endif()
