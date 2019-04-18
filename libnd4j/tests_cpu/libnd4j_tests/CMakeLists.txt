cmake_minimum_required(VERSION 3.9)
project(dev_tests)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../cmake" ${CMAKE_MODULE_PATH})

# Download and unpack googletest at configure time
configure_file(../CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# OPTIONAL MKL-DNN
if ("${BUILD_MKLDNN}")
    # Download and unpack mkl-dnn at configure time
    configure_file(./CMakeLists.txt.in mkldnn-download/CMakeLists.txt)
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
    set(mkldnn_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/mkldnn-src)
    set(HAVE_MKLDNN 1)
    add_definitions(-DHAVE_MKLDNN=true)
    include_directories(${mkldnn_SOURCE_DIR}/include ${mkldnn_SOURCE_DIR}/external/mklml_lnx_2019.0.3.20190220/include ${mkldnn_SOURCE_DIR})
endif()

# Download and unpack flatbuffers at configure time
configure_file(../../CMakeLists.txt.in flatbuffers-download/CMakeLists.txt)
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


# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

set(gtest_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest-src)
add_definitions(-D__STANDALONE_BUILD__=true)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(../../include ../../layers ../../include/helpers ../../include/array ../../include/memory ../../include/loops ../../include/graph ../../include/ops ../../include/types ../../include/cnpy ../../blas)
if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    message("Using apple")
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()
if(WIN32)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()
endif()

# -fsanitize=address
# -fsanitize=leak
if (APPLE)
    set(CMAKE_CXX_FLAGS  " -O0 -g -fPIC -std=c++11 -fassociative-math -funsafe-math-optimizations -D__APPLE_OS__=true")
elseif(WIN32)
    set(CMAKE_CXX_FLAGS  " -g -fPIC -std=c++11 -fassociative-math -funsafe-math-optimizations -Wa,-mbig-obj")
else()
    if ("${_RELEASE}" OR CMAKE_BUILD_TYPE STREQUAL "Release")
        message("Release build for tests")
        set(CMAKE_CXX_FLAGS  "-O3 -fPIC -std=c++11 -fassociative-math -funsafe-math-optimizations")
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
            set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -mcpu=native")
        else()
            set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
        endif()
    else()
        set(CMAKE_CXX_FLAGS  " -g -O0 -fPIC -std=c++11 -fassociative-math -funsafe-math-optimizations -fsanitize=address")
    endif()

    if (${F16C})
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mf16c -D__F16C__=true")
    endif()
endif()

if ("${EXPERIMENTAL}" STREQUAL "yes")
    message("Experimental mode ENABLED")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
endif()

# tests are always compiled with all ops included
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DLIBND4J_ALL_OPS=true")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -Wno-logical-op-parentheses -Wno-inconsistent-missing-override -Wno-implicit-conversion-floating-point-to-bool -Wno-delete-non-virtual-dtor")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    message("AppleClang used")
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -Wno-logical-op-parentheses -Wno-inconsistent-missing-override -Wno-implicit-conversion-floating-point-to-bool -Wno-delete-non-virtual-dtor")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -fp-model fast")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fmax-errors=2")
endif()


IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
    message(FATAL_ERROR "You need at least GCC 4.9")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message("OPENMP NOT FOUND")
endif()

if ("${OPENBLAS}" OR CMAKE_BUILD_TYPE STREQUAL "Release")
    find_package(BLAS)
    if (BLAS_FOUND)
        message("Found external BLAS implementation...")
        add_definitions(-D__EXTERNAL_BLAS__=true)
    endif()
endif()

file(GLOB_RECURSE TYPES_SOURCES false ../../include/types/*.cpp ../../include/types/*.h)
file(GLOB_RECURSE ARRAY_SOURCES false ../../include/array/*.cpp ../../include/array/*.h)
file(GLOB_RECURSE MEMORY_SOURCES false ../../include/memory/*.cpp ../../include/memory/*.h)
file(GLOB_RECURSE GRAPH_SOURCES false ../../include/graph/*.cpp ../../include/graph/*.h)
file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../../include/ops/declarable/generic/*.cpp)
file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES false ../../include/ops/declarable/helpers/cpu/*.cpp)
file(GLOB_RECURSE OPS_SOURCES false ../../include/ops/impl/*.cpp ../../include/ops/declarable/impl/*.cpp  ../../include/ops/*.h)
file(GLOB_RECURSE INDEXING_SOURCES false ../../include/indexing/*.cpp ../../include/indexing/*.h)
file(GLOB_RECURSE HELPERS_SOURCES false ../../include/helpers/*.cpp ../../include/helpers/*.h)
file(GLOB_RECURSE LOOPS_SOURCES false ../../include/loops/*.cpp ../../include/loops/*.h)

message("CPU BLAS")
add_definitions(-D__CPUBLAS__=true)


file(GLOB_RECURSE TEST_SOURCES false ../layers_tests/*.cpp ../layers_tests/*.h)
# file(GLOB_RECURSE TEST_SOURCES false ../layers_tests/DeclarableOpsTests6.cpp ../layers_tests/*.h)


# Filter out any source files from */CMakeFiles/* paths. these tend to cause problems such a multiple main definitions.
set (EXCLUDE_DIR "/CMakeFiles/")
foreach (TMP_PATH ${TEST_SOURCES})
    string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
    if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
        list (REMOVE_ITEM TEST_SOURCES ${TMP_PATH})
    endif ()
endforeach(TMP_PATH)


add_executable(runtests ${LOOPS_SOURCES} ../../blas/cpu/NativeOps.cpp ../../blas/cpu/GraphExecutioner.cpp
    ../../blas/cpu/NativeOpExcutioner.cpp ../../blas/cpu/NDArray.cpp ../../blas/cpu/NDArrayFactory.cpp
    ../../include/cnpy/cnpy.cpp  ../../include/nd4jmemset.h ../../include/nd4jmalloc.h
    ../../blas/Environment.cpp ../../blas/Environment.h  ${ARRAY_SOURCES} ${TYPES_SOURCES}
    ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES}  ${CUSTOMOPS_HELPERS_SOURCES}
    ${OPS_SOURCES} ${TEST_SOURCES})

target_link_libraries(runtests gtest mkldnn gtest_main ${BLAS_LIBRARIES})

