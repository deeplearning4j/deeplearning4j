# cmake/BuildCUDA.cmake
# Contains all logic for building the CUDA library.

enable_language(CUDA)

# --- CUDA Compiler Configuration ---
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_COMPILER_LAUNCHER "" CACHE STRING "CUDA compiler launcher")
if(WIN32)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES ON)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES ON)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LINK_OBJECTS ON)
    set(CMAKE_CUDA_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_CUDA_COMPILE_OPTIONS_USE_RESPONSE_FILE ON)
endif()

if(NOT DEFINED COMPUTE)
    set(COMPUTE "auto")
endif()

string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
if(COMPUTE_CMP STREQUAL "all")
    set(CUDA_ARCH_FLAGS "-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89")
    message(STATUS "Building for all CUDA architectures")
elseif(COMPUTE_CMP STREQUAL "auto")
    set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86")
    message(STATUS "Auto-detecting CUDA architectures")
else()
    string(REPLACE "," ";" ARCH_LIST "${COMPUTE}")
    set(CUDA_ARCH_FLAGS "")
    foreach(ARCH ${ARCH_LIST})
        string(REPLACE "." "" ARCH_CLEAN "${ARCH}")
        if(ARCH_CLEAN MATCHES "^[0-9][0-9]$")
            set(CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS} -gencode arch=compute_${ARCH_CLEAN},code=sm_${ARCH_CLEAN}")
        else()
            message(WARNING "Invalid CUDA architecture: ${ARCH} (cleaned: ${ARCH_CLEAN}). Skipping.")
        endif()
    endforeach()
    string(STRIP "${CUDA_ARCH_FLAGS}" CUDA_ARCH_FLAGS)
    message(STATUS "Using user-specified CUDA architectures: ${COMPUTE}")
endif()

if(NOT CUDA_ARCH_FLAGS OR CUDA_ARCH_FLAGS STREQUAL "")
    message(WARNING "No valid CUDA architecture flags generated. Using default compute_86.")
    set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all")
if(CMAKE_CUDA_COMPILER_VERSION)
    string(REGEX MATCH "^([0-9]+)" CUDA_VERSION_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}")
endif()

# --- Source File Collection ---
file(GLOB_RECURSE PERF_SOURCES ./include/performance/*.cpp ./include/performance/*.h)
file(GLOB_RECURSE EXCEPTIONS_SOURCES ./include/exceptions/*.cpp ./include/exceptions/*.h)
file(GLOB_RECURSE EXEC_SOURCES ./include/execution/impl/*.cpp ./include/execution/cuda/*.cu ./include/execution/cuda/*.h ./include/execution/*.cu ./include/execution/*.h)
file(GLOB_RECURSE TYPES_SOURCES ./include/types/*.cpp ./include/types/*.h)
file(GLOB_RECURSE ARRAY_SOURCES ./include/array/cuda/*.cu ./include/array/impl/*.cpp ./include/array/*.h)
file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/impl/*.cpp ./include/memory/cuda/*.cu ./include/memory/*.h)
file(GLOB_RECURSE GRAPH_SOURCES ./include/graph/*.cpp ./include/graph/*.cu ./include/graph/*.h)
file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES ./include/ops/declarable/helpers/cuda/*.cu ./include/ops/declarable/helpers/impl/*.cpp)
file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp ./include/ops/*.h)
file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp ./include/helpers/cuda/*.cu ./include/helpers/*.h)
file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp ./include/indexing/*.h)
file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/impl/*.cpp ./include/loops/*.h)
file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/*.cu ./include/legacy/*.h)
file(GLOB_RECURSE LOOPS_SOURCES_CUDA ./include/loops/*.cu ./include/loops/cuda/**/*.cu)

set(ALL_SOURCES ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${EXEC_SOURCES} ${TYPES_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${CUSTOMOPS_HELPERS_SOURCES} ${OPS_SOURCES} ${HELPERS_SOURCES} ${INDEXING_SOURCES} ${LOOPS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES_CUDA})

if(HAVE_CUDNN)
    file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ./include/ops/declarable/platform/cudnn/*.cu)
    list(APPEND ALL_SOURCES ${CUSTOMOPS_CUDNN_SOURCES})
endif()
list(REMOVE_DUPLICATES ALL_SOURCES)

# --- Generate CUDA Template Instantiations ---
file(GLOB_RECURSE COMPILATION_UNITS ./include/loops/cuda/compilation_units/*.cu.in)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
foreach(FL_ITEM ${COMPILATION_UNITS})
    genCompilation(${FL_ITEM})
endforeach()

# --- Build Library ---
set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})
add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)
set_property(TARGET ${OBJECT_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
set_target_properties(${SD_LIBRARY_NAME} PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

# --- Link Dependencies ---
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        ${CUDNN}
        flatbuffers_interface
)

install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)