# cmake/BuildHexagon.cmake
# Configures the Hexagon NPU native library (nd4jhexagon).
#
# MainBuildFlow.cmake (included before this file) already creates the
# ${SD_LIBRARY_NAME}_object (OBJECT) and ${SD_LIBRARY_NAME} (SHARED) targets
# with the standard CPU source set. This file's job is to:
#   1. Add Hexagon-specific sources (graph/hexagon/*.cpp).
#   2. Set compile definitions on the existing targets.
#   3. Link dlopen support (dlopen/dlsym used in HexagonRuntimeManager.cpp).
#
# libhexagon_mlir_runtime.so is loaded at RUNTIME via dlopen() — it is NOT a
# link-time dependency, and all hexagon-mlir types are opaque void* so no SDK
# headers are required at build time (hexagon-mlir itself is BSD-3 open source,
# Qualcomm Dec 2025). This mirrors the BuildTPU.cmake / PjrtClientManager design.
#
# DEFAULT_ENGINE stays ENGINE_CPU: there is no ENGINE_HEXAGON in
# execution/Engine.h and no ops/declarable/platform/hexagon op set yet — the
# Hexagon path is a graph-level backend (HexagonGraphBackend / HexagonIRBuilder),
# not a per-op platform engine.

function(setup_hexagon_build)
    message(STATUS "=== HEXAGON BUILD CONFIGURATION ===")

    if(NOT SD_HEXAGON)
        message(STATUS "Hexagon build not enabled (SD_HEXAGON=OFF)")
        return()
    endif()

    # -----------------------------------------------------------------
    # 1. Collect Hexagon-specific source files to ADD to the existing
    #    target. MainBuildFlow already compiled the CPU/base sources.
    # -----------------------------------------------------------------
    file(GLOB_RECURSE HEXAGON_GRAPH_SOURCES
        ${CMAKE_SOURCE_DIR}/include/graph/hexagon/*.cpp
        ${CMAKE_SOURCE_DIR}/include/graph/hexagon/*.h)
    message(STATUS "Hexagon graph backend sources: ${HEXAGON_GRAPH_SOURCES}")

    # -----------------------------------------------------------------
    # 2. Determine existing target names (set by CMakeLists.txt before
    #    MainBuildFlow.cmake was included).
    # -----------------------------------------------------------------
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    set(MAIN_LIB_NAME   "${SD_LIBRARY_NAME}")

    if(NOT TARGET ${OBJECT_LIB_NAME})
        message(FATAL_ERROR "Expected target '${OBJECT_LIB_NAME}' not found. "
                            "MainBuildFlow.cmake must be included before BuildHexagon.cmake.")
    endif()

    # -----------------------------------------------------------------
    # 3. Add Hexagon sources to the existing OBJECT target.
    # -----------------------------------------------------------------
    if(HEXAGON_GRAPH_SOURCES)
        target_sources(${OBJECT_LIB_NAME} PRIVATE ${HEXAGON_GRAPH_SOURCES})
        list(LENGTH HEXAGON_GRAPH_SOURCES _hexagon_src_count)
        message(STATUS "Added ${_hexagon_src_count} Hexagon source(s) to target '${OBJECT_LIB_NAME}'")
    endif()

    # -----------------------------------------------------------------
    # 4. Compile definitions.
    # -----------------------------------------------------------------
    target_compile_definitions(${OBJECT_LIB_NAME} PUBLIC
        SD_HEXAGON=1
        HAVE_HEXAGON_MLIR=1
        DEFAULT_ENGINE=samediff::ENGINE_CPU
    )
    target_compile_definitions(${MAIN_LIB_NAME} PUBLIC
        SD_HEXAGON=1
        HAVE_HEXAGON_MLIR=1
    )

    # -----------------------------------------------------------------
    # 5. Optional: hexagon-mlir install detected at build time
    #    (HEXAGON_MLIR_PATH env). Purely additive — dlopen still rules.
    # -----------------------------------------------------------------
    if(DEFINED ENV{HEXAGON_MLIR_PATH} AND EXISTS "$ENV{HEXAGON_MLIR_PATH}/include")
        target_include_directories(${OBJECT_LIB_NAME} PUBLIC "$ENV{HEXAGON_MLIR_PATH}/include")
        message(STATUS "Added hexagon-mlir include dir: $ENV{HEXAGON_MLIR_PATH}/include")
    endif()

    # -----------------------------------------------------------------
    # 6. Link dependencies. CMAKE_DL_LIBS provides -ldl for dlopen/dlsym.
    # -----------------------------------------------------------------
    target_link_libraries(${MAIN_LIB_NAME} PUBLIC
        ${CMAKE_DL_LIBS}
    )

    message(STATUS "Hexagon build configured:")
    message(STATUS "   Library name: ${MAIN_LIB_NAME}")
    message(STATUS "=== HEXAGON BUILD CONFIGURATION COMPLETE ===")

endfunction()
