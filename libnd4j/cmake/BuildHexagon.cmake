# cmake/BuildHexagon.cmake
# Contains all logic for building the Hexagon NPU library using hexagon-mlir.

function(setup_hexagon_build)
    message(STATUS "Setting up Hexagon NPU build environment")

    add_definitions(-DSD_HEXAGON=true)
    add_definitions(-DHAVE_HEXAGON_MLIR=1)
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
    file(GLOB_RECURSE SYSTEM_CONFIG_SOURCES ./include/system/config/impl/*.cpp)

    set(ALL_SOURCES "")
    set(STATIC_SOURCES_TO_CHECK ${BLAS_SOURCES} ${PERF_SOURCES} ${EXCEPTIONS_SOURCES}
        ${EXEC_SOURCES} ${TYPES_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES}
        ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES}
        ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES}
        ${HELPERS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES} ${SYSTEM_CONFIG_SOURCES})

    # --- Hexagon Graph Backend Sources ---
    file(GLOB_RECURSE HEXAGON_GRAPH_SOURCES
        ./include/graph/hexagon/*.cpp
        ./include/graph/hexagon/*.h)
    list(APPEND STATIC_SOURCES_TO_CHECK ${HEXAGON_GRAPH_SOURCES})
    message(STATUS "Added Hexagon graph backend sources: ${HEXAGON_GRAPH_SOURCES}")

    # Add no-PLT stub functions for functrace builds
    if(SD_GCC_FUNCTRACE)
        list(APPEND STATIC_SOURCES_TO_CHECK ./include/platform/noplt_libc_stubs.c)
    endif()

    list(APPEND ALL_SOURCES ${STATIC_SOURCES_TO_CHECK})

    # --- Exclude CUDA/HIP/Metal/Vulkan/L0 specific sources ---
    list(FILTER ALL_SOURCES EXCLUDE REGEX ".*cuda.*")
    list(FILTER ALL_SOURCES EXCLUDE REGEX ".*hip/.*")
    list(FILTER ALL_SOURCES EXCLUDE REGEX ".*metal/.*")
    list(FILTER ALL_SOURCES EXCLUDE REGEX ".*vulkan/.*")
    list(FILTER ALL_SOURCES EXCLUDE REGEX ".*levelzero/.*")
    list(FILTER ALL_SOURCES EXCLUDE REGEX ".*tpu/.*")

    message(STATUS "Hexagon source count: ${CMAKE_CURRENT_LIST_LINE}")

    # --- Build Object Library ---
    add_library(nd4jhexagon_object OBJECT ${ALL_SOURCES})
    target_compile_definitions(nd4jhexagon_object PRIVATE
        SD_HEXAGON=1
        HAVE_HEXAGON_MLIR=1
        DEFAULT_ENGINE=samediff::ENGINE_CPU
    )
    target_include_directories(nd4jhexagon_object PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ops"
    )

    # --- Build Shared Library ---
    add_library(nd4jhexagon SHARED $<TARGET_OBJECTS:nd4jhexagon_object>)
    target_compile_definitions(nd4jhexagon PRIVATE
        SD_HEXAGON=1
        HAVE_HEXAGON_MLIR=1
    )

    # --- Linking ---
    target_link_libraries(nd4jhexagon PRIVATE
        ${OPENBLAS_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${JVM_LIBRARY}
        flatbuffers_interface
        ${CMAKE_DL_LIBS}
    )

    # Link hexagon-mlir if found at compile time
    if(HEXAGON_MLIR_FOUND)
        target_include_directories(nd4jhexagon_object PRIVATE ${HEXAGON_MLIR_INCLUDE_DIRS})
        target_link_libraries(nd4jhexagon PRIVATE ${HEXAGON_MLIR_LIBRARIES})
    endif()

    set_target_properties(nd4jhexagon PROPERTIES
        OUTPUT_NAME "nd4jhexagon"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    message(STATUS "Hexagon build configured successfully")
endfunction()
