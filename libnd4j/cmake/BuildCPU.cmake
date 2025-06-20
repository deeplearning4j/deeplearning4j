# cmake/BuildCPU.cmake
# Contains all logic for building the CPU library.

add_definitions(-D__CPUBLAS__=true)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# --- Source File Collection ---
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
file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp  ./include/helpers/cpu/*.cpp ./include/helpers/*.h)
file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/cpu/*.cpp ./include/legacy/*.h)
file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/*.cpp ./include/loops/*.h)

set(ALL_SOURCES "")
set(STATIC_SOURCES_TO_CHECK ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${EXEC_SOURCES} ${TYPES_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES} ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES})

if(HAVE_ONEDNN)
    file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES ./include/ops/declarable/platform/mkldnn/*.cpp ./include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
    list(APPEND STATIC_SOURCES_TO_CHECK ${CUSTOMOPS_ONEDNN_SOURCES})
endif()
if(HAVE_ARMCOMPUTE)
    file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES ./include/ops/declarable/platform/armcompute/*.cpp ./include/ops/declarable/platform/armcompute/*.h)
    list(APPEND STATIC_SOURCES_TO_CHECK ${CUSTOMOPS_ARMCOMPUTE_SOURCES})
endif()

list(APPEND ALL_SOURCES ${STATIC_SOURCES_TO_CHECK})
list(REMOVE_DUPLICATES ALL_SOURCES)

# --- Generate CPU Template Instantiations ---
file(GLOB_RECURSE COMPILATION_UNITS
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
set_property(TARGET ${OBJECT_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
set_target_properties(${SD_LIBRARY_NAME} PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

# --- Link Dependencies ---
target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
        ${ONEDNN}
        ${ARMCOMPUTE_LIBRARIES}
        ${OPENBLAS_LIBRARIES}
        ${BLAS_LIBRARIES}
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

install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)