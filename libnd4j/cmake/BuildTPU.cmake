# cmake/BuildTPU.cmake
# Contains all logic for building the TPU library using PJRT.

# Function to set up TPU build environment
function(setup_tpu_build)
    message(STATUS "Setting up TPU/PJRT build environment")

    add_definitions(-D__TPUBLAS__=true)
    add_definitions(-DSD_TPU=true)
    add_definitions(-DHAVE_PJRT=1)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" PARENT_SCOPE)

# --- Source File Collection ---
# Core sources (same as CPU)
file(GLOB_RECURSE BLAS_SOURCES ./include/blas/*.h)
file(GLOB_RECURSE PERF_SOURCES ./include/performance/*.cpp ./include/performance/*.h)
file(GLOB_RECURSE EXCEPTIONS_SOURCES ./include/exceptions/*.cpp ./include/exceptions/*.h)
file(GLOB_RECURSE EXEC_SOURCES ./include/execution/*.cpp ./include/execution/*.h)
file(GLOB_RECURSE TYPES_SOURCES ./include/types/*.cpp ./include/types/*.h)
file(GLOB_RECURSE ARRAY_SOURCES ./include/array/*.cpp ./include/array/*.h)
file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/*.cpp ./include/memory/*.h)
file(GLOB_RECURSE GRAPH_SOURCES ./include/graph/*.cpp ./include/graph/*.h)
file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL_SOURCES ./include/ops/declarable/helpers/impl/*.cpp)
file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU_SOURCES ./include/ops/declarable/helpers/cpu/*.cpp)
file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp ./include/ops/*.h)
file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp ./include/indexing/*.h)
file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp ./include/helpers/cpu/*.cpp ./include/helpers/*.h)
file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/cpu/*.cpp ./include/legacy/*.h)
file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/*.cpp ./include/loops/*.h)

set(ALL_SOURCES "")
set(STATIC_SOURCES_TO_CHECK ${BLAS_SOURCES} ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${EXEC_SOURCES} ${TYPES_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES} ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES})

# --- TPU/PJRT Platform Sources ---
if(HAVE_PJRT)
    file(GLOB_RECURSE CUSTOMOPS_PJRT_SOURCES
        ./include/ops/declarable/platform/pjrt/*.cpp
        ./include/ops/declarable/platform/pjrt/*.h)
    list(APPEND STATIC_SOURCES_TO_CHECK ${CUSTOMOPS_PJRT_SOURCES})
    message(STATUS "Added PJRT platform sources: ${CUSTOMOPS_PJRT_SOURCES}")
endif()

# --- TPU Graph Backend Sources ---
file(GLOB_RECURSE TPU_GRAPH_SOURCES
    ./include/graph/tpu/*.cpp
    ./include/graph/tpu/*.h)
list(APPEND STATIC_SOURCES_TO_CHECK ${TPU_GRAPH_SOURCES})
message(STATUS "Added TPU graph backend sources: ${TPU_GRAPH_SOURCES}")

# Add no-PLT stub functions for functrace builds (GCC and Clang)
if(SD_GCC_FUNCTRACE)
    list(APPEND STATIC_SOURCES_TO_CHECK ./include/platform/noplt_libc_stubs.c)
    message(STATUS "Added noplt_libc_stubs.c for functrace build")
endif()

list(APPEND ALL_SOURCES ${STATIC_SOURCES_TO_CHECK})

# --- Exclude CUDA/HIP/Metal/Vulkan/L0/Hexagon specific sources ---
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*cuda.*")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*hip/.*")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*metal/.*")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*vulkan/.*")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*levelzero/.*")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*hexagon/.*")

list(REMOVE_DUPLICATES ALL_SOURCES)

# --- Generate Template Instantiations ---
file(GLOB_RECURSE COMPILATION_UNITS
        ./include/ops/impl/compilation_units/*.cpp.in
        ./include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in
        ./include/loops/cpu/compilation_units/*.cpp.in
        ./include/helpers/cpu/loops/*.cpp.in)

foreach(FL_ITEM ${COMPILATION_UNITS})
    genCompilation(${FL_ITEM})
endforeach()

# --- Build Library ---
set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})
add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)
target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS})

# Add PJRT include directories
if(HAVE_PJRT AND PJRT_INCLUDE_DIR)
    target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${PJRT_INCLUDE_DIR})
endif()

set_property(TARGET ${OBJECT_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
set_target_properties(${SD_LIBRARY_NAME} PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

# --- Link Dependencies ---
target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
        ${PJRT}
        ${OPENBLAS_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${JVM_LIBRARY}
        flatbuffers_interface
)

# --- OpenMP Configuration ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found, linking OpenMP::OpenMP_CXX")
        target_link_libraries(${SD_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found, falling back to manual configuration")
        target_compile_options(${SD_LIBRARY_NAME} INTERFACE "-fopenmp")
        target_link_libraries(${SD_LIBRARY_NAME} PUBLIC "-fopenmp")
    endif()
endif()

# --- TPU-specific compile definitions ---
target_compile_definitions(${SD_LIBRARY_NAME} PUBLIC
    SD_TPU=1
    DEFAULT_ENGINE=samediff::ENGINE_TPU
)

if(HAVE_PJRT)
    target_compile_definitions(${SD_LIBRARY_NAME} PUBLIC HAVE_PJRT=1)
endif()

# --- Install ---
install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)

message(STATUS "TPU build configured:")
message(STATUS "   Library name: ${SD_LIBRARY_NAME}")
message(STATUS "   PJRT support: ${HAVE_PJRT}")
message(STATUS "   TPU version: ${TPU_VERSION}")

endfunction()
