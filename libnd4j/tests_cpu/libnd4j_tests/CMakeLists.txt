cmake_minimum_required(VERSION 3.9)
project(dev_tests)
message("Starting up tests build")
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
    configure_file(../../CMakeLists.txt.mkldnn.in mkldnn-download/CMakeLists.txt)
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
    add_definitions("-DHAVE_MKLDNN")
    include_directories(${mkldnn_SOURCE_DIR}/include ${mkldnn_SOURCE_DIR}/external/mklml_lnx_2019.0.3.20190220/include ${mkldnn_SOURCE_DIR})
    set(MKLDNN dnnl)
endif()

if (${HELPERS_armcompute})
 find_package(ARMCOMPUTE REQUIRED)

 if(ARMCOMPUTE_FOUND)
    message("Found ARMCOMPUTE: ${ARMCOMPUTE_LIBRARIES}")
    set(HAVE_ARMCOMPUTE 1)
    # Add preprocessor definition for ARM Compute NEON
    add_definitions(-DARMCOMPUTENEON_ENABLED)
    #build our library with neon support
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    include_directories(${ARMCOMPUTE_INCLUDE})
 endif()

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
include_directories(../../include)
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
    set(CMAKE_CXX_FLAGS  " -O0 -g -fPIC -std=c++11 -D__APPLE_OS__=true -DSD_APPLE_BUILD=true")
elseif(WIN32)
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		set(CMAKE_CXX_FLAGS  " -g -fPIC -std=c++11 -Wa,-mbig-obj")
	endif()
else()
    set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -ffast-math -DFFAST_MATH=true -DLINUX_BUILD=true")

    if ("${_RELEASE}" OR CMAKE_BUILD_TYPE STREQUAL "Release")
        message("Release build for tests")
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -fPIC -std=c++11 -D_RELEASE=true")
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
            set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -mcpu=native")
        else()
            set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
        endif()
    else()
        set(CMAKE_CXX_FLAGS  " -g -O0 -fPIC -std=c++11 ")
        if (NOT SD_CUDA)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
        endif()
    endif()

    if (${F16C})
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mf16c -DSD_F16C=true")
    endif()
endif()

if ("${_RELEASE}" OR CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRELEASE_BUILD=true")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_BUILD=true")
endif()

if ("${SD_EXPERIMENTAL}" STREQUAL "yes")
    message("Experimental mode ENABLED")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
endif()

# tests are always compiled with all ops included
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DSD_ALL_OPS=true -DDEFAULT_ENGINE=samediff::ENGINE_CPU")

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

message("Looking for OpenMP")
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message("OPENMP NOT FOUND")
endif()

if ("${OPENBLAS}" OR CMAKE_BUILD_TYPE STREQUAL "Release" OR "${BUILD_MKLDNN}")
    message("Looking for BLAS")
    find_package(BLAS REQUIRED)
    if (BLAS_FOUND)
        message("Found external BLAS library: ${BLAS_LIBRARIES}")
        add_definitions(-D__EXTERNAL_BLAS__=true)
    endif()
endif()

file(GLOB_RECURSE PERF_SOURCES false ../../include/performance/*.cpp ../../include/performance/*.h)
file(GLOB_RECURSE EXCEPTIONS_SOURCES false ../../include/exceptions/*.cpp ../../include/exceptions/*.h)
file(GLOB_RECURSE EXEC_SOURCES false ../../include/execution/*.cpp ../../include/execution/*.h)
file(GLOB_RECURSE TYPES_SOURCES false ../../include/types/*.cpp ../../include/types/*.h)
file(GLOB_RECURSE ARRAY_SOURCES false ../../include/array/*.cpp ../../include/array/*.h)
file(GLOB_RECURSE MEMORY_SOURCES false ../../include/memory/*.cpp ../../include/memory/*.h)
file(GLOB_RECURSE GRAPH_SOURCES false ../../include/graph/*.cpp ../../include/graph/*.h)
file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../../include/ops/declarable/generic/*.cpp)
file(GLOB_RECURSE CUSTOMOPS_GENERIC_SOURCES false ../../include/ops/declarable/helpers/cpu/*.cpp ../../include/ops/declarable/helpers/impl/*.cpp)
file(GLOB_RECURSE OPS_SOURCES false ../../include/ops/impl/*.cpp ../../include/ops/declarable/impl/*.cpp  ../../include/ops/*.h)
file(GLOB_RECURSE INDEXING_SOURCES false ../../include/indexing/*.cpp ../../include/indexing/*.h)
file(GLOB_RECURSE HELPERS_SOURCES false ../../include/helpers/*.cpp)
file(GLOB_RECURSE LEGACY_SOURCES false ../../include/legacy/impl/*.cpp  ../../include/legacy/cpu/*.cpp ../../include/legacy/*.h)
file(GLOB_RECURSE LOOPS_SOURCES false ../../include/loops/*.cpp ../../include/loops/*.h)

# optionally build mkldnn
if ("${BUILD_MKLDNN}")
    file(GLOB_RECURSE CUSTOMOPS_PLATFORM_SOURCES false ../../include/ops/declarable/platform/mkldnn/*.cpp)
endif()

if(HAVE_ARMCOMPUTE)
    file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES false ../include/ops/declarable/platform/armcompute/*.cpp ../include/ops/declarable/platform/armcompute/armcomputeUtils.h)
endif()

message("CPU backend")
add_definitions(-D__CPUBLAS__=true)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT(MINGW) AND NOT(APPLE))
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic")
endif()

    file(GLOB_RECURSE COMPILATION_UNITS false ../include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in 
    ../include/loops/cpu/compilation_units/*.cpp.in ../include/helpers/cpu/loops/*.cpp.in)

    foreach(FL_ITEM ${COMPILATION_UNITS})   
        genCompilation(FL_ITEM)
    endforeach() 

# this function strips path from file name, basically making up short file name, i.e. file.cpp
function(SHORTNAME LONG_NAME OUTPUT)
    SET(_TMP_STR "")
    string (REGEX REPLACE ".*/" "" _TMP_STR "${LONG_NAME}")
    set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

# now we ned to join two lists
# first of all we'll build truncated list of files in platform sources
# and list of priority implementations from platform helpers
#set(CUSTOMOPS_HELPERS_SOURCES "")
#set(SHORT_NAMES "")
#foreach(LONG_NAME ${CUSTOMOPS_PLATFORM_SOURCES})
#    SHORTNAME("${LONG_NAME}" "SHORT_NAME")
#    set(CUSTOMOPS_HELPERS_SOURCES ${CUSTOMOPS_HELPERS_SOURCES} ${LONG_NAME})
#    set(SHORT_NAMES ${SHORT_NAMES} ${SHORT_NAME})
#endforeach()

# now we're going to filter generic helpers, to exclude platform implementations
#foreach(LONG_NAME ${CUSTOMOPS_GENERIC_SOURCES})
#    SHORTNAME("${LONG_NAME}" "SHORT_NAME")

    # and now we add this op ONLY if it wasn't announced in platform helpers
#    string(FIND "${SHORT_NAMES}" "${SHORT_NAME}" "LOC")
#    if (${LOC} EQUAL -1)
#        set(CUSTOMOPS_HELPERS_SOURCES ${CUSTOMOPS_HELPERS_SOURCES} ${LONG_NAME})
#    endif()
#endforeach()


file(GLOB_RECURSE TEST_SOURCES false ../layers_tests/*.cpp ../layers_tests/*.h)


# Filter out any source files from */CMakeFiles/* paths. these tend to cause problems such a multiple main definitions.
set (EXCLUDE_DIR "/CMakeFiles/")
foreach (TMP_PATH ${TEST_SOURCES})
    string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
    if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
        list (REMOVE_ITEM TEST_SOURCES ${TMP_PATH})
    endif ()
endforeach(TMP_PATH)


add_executable(runtests ${LOOPS_SOURCES} ${LEGACY_SOURCES} ${EXEC_SOURCES} ${HELPERS_SOURCES}  ${ARRAY_SOURCES} ${TYPES_SOURCES}
    ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${EXCEPTIONS_SOURCES} ${INDEXING_SOURCES} ${CUSTOMOPS_PLATFORM_SOURCES} 
    ${CUSTOMOPS_ARMCOMPUTE_SOURCES} ${CUSTOMOPS_GENERIC_SOURCES}
    ${OPS_SOURCES} ${TEST_SOURCES} ${PERF_SOURCES})

target_link_libraries(runtests gtest ${MKLDNN} ${ARMCOMPUTE_LIBRARIES} gtest_main ${BLAS_LIBRARIES})

