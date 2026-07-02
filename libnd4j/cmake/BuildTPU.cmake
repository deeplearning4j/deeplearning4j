# cmake/BuildTPU.cmake
# Configures the TPU/PJRT native library (nd4jtpu).
#
# MainBuildFlow.cmake (included before this file) already creates the
# ${SD_LIBRARY_NAME}_object (OBJECT) and ${SD_LIBRARY_NAME} (SHARED) targets
# with the standard CPU source set. This file's job is to:
#   1. Detect the PJRT C API header (vendored copy ships in-tree).
#   2. Add TPU-specific sources (graph/tpu/*.cpp + pjrt platform/*.cpp).
#   3. Set compile definitions and include directories on the existing targets.
#   4. Link dlopen support (dlopen/dlsym used in PjrtClientManager.cpp at runtime).
#
# libtpu.so is loaded at RUNTIME via dlopen() — it is NOT a link-time dependency.
# The in-tree vendored pjrt_c_api.h is sufficient for a no-python, no-hardware build.

function(setup_tpu_build)
    message(STATUS "=== TPU BUILD CONFIGURATION ===")

    if(NOT SD_TPU)
        message(STATUS "TPU build not enabled (SD_TPU=OFF)")
        return()
    endif()

    # -----------------------------------------------------------------
    # 1. Detect PJRT C API header (+ optional libtpu.so at build time)
    #    setup_pjrt_paths() is defined in TpuConfiguration.cmake.
    # -----------------------------------------------------------------
    setup_pjrt_paths()

    # -----------------------------------------------------------------
    # 2. Collect TPU-specific source files to ADD to the existing target.
    #    MainBuildFlow already compiled the CPU/base sources.
    # -----------------------------------------------------------------
    file(GLOB_RECURSE TPU_GRAPH_SOURCES
        ${CMAKE_SOURCE_DIR}/include/graph/tpu/*.cpp
        ${CMAKE_SOURCE_DIR}/include/graph/tpu/*.h)
    message(STATUS "TPU graph backend sources: ${TPU_GRAPH_SOURCES}")

    set(TPU_EXTRA_SOURCES ${TPU_GRAPH_SOURCES})

    if(HAVE_PJRT)
        file(GLOB_RECURSE TPU_PJRT_SOURCES
            ${CMAKE_SOURCE_DIR}/include/ops/declarable/platform/pjrt/*.cpp
            ${CMAKE_SOURCE_DIR}/include/ops/declarable/platform/pjrt/*.h)
        list(APPEND TPU_EXTRA_SOURCES ${TPU_PJRT_SOURCES})
        message(STATUS "PJRT platform sources: ${TPU_PJRT_SOURCES}")
    endif()

    # -----------------------------------------------------------------
    # 3. Determine existing target names (set by CMakeLists.txt before
    #    MainBuildFlow.cmake was included).
    # -----------------------------------------------------------------
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    set(MAIN_LIB_NAME   "${SD_LIBRARY_NAME}")

    if(NOT TARGET ${OBJECT_LIB_NAME})
        message(FATAL_ERROR "Expected target '${OBJECT_LIB_NAME}' not found. "
                            "MainBuildFlow.cmake must be included before BuildTPU.cmake.")
    endif()

    # -----------------------------------------------------------------
    # 4. Add TPU sources to the existing OBJECT and SHARED targets.
    # -----------------------------------------------------------------
    if(TPU_EXTRA_SOURCES)
        target_sources(${OBJECT_LIB_NAME} PRIVATE ${TPU_EXTRA_SOURCES})
        list(LENGTH TPU_EXTRA_SOURCES _tpu_src_count)
        message(STATUS "Added ${_tpu_src_count} TPU source(s) to target '${OBJECT_LIB_NAME}'")
    endif()

    # -----------------------------------------------------------------
    # 5. Compile definitions.
    # -----------------------------------------------------------------
    target_compile_definitions(${OBJECT_LIB_NAME} PUBLIC
        SD_TPU=1
        __TPUBLAS__=true
        DEFAULT_ENGINE=samediff::ENGINE_TPU
    )
    target_compile_definitions(${MAIN_LIB_NAME} PUBLIC
        SD_TPU=1
        DEFAULT_ENGINE=samediff::ENGINE_TPU
    )

    if(HAVE_PJRT)
        target_compile_definitions(${OBJECT_LIB_NAME} PUBLIC HAVE_PJRT=1)
        target_compile_definitions(${MAIN_LIB_NAME}   PUBLIC HAVE_PJRT=1)
    endif()

    # -----------------------------------------------------------------
    # 6. Include directories.
    # -----------------------------------------------------------------
    if(PJRT_INCLUDE_DIR)
        target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${PJRT_INCLUDE_DIR})
        message(STATUS "Added PJRT include dir: ${PJRT_INCLUDE_DIR}")
    endif()

    # -----------------------------------------------------------------
    # 7. Link dependencies.
    #    PJRT_LIBRARIES is empty in header-only mode (dlopen at runtime).
    #    CMAKE_DL_LIBS provides -ldl which dlopen/dlsym need.
    # -----------------------------------------------------------------
    target_link_libraries(${MAIN_LIB_NAME} PUBLIC
        ${PJRT_LIBRARIES}
        ${CMAKE_DL_LIBS}
    )

    message(STATUS "TPU build configured:")
    message(STATUS "   Library name:  ${MAIN_LIB_NAME}")
    message(STATUS "   PJRT support:  ${HAVE_PJRT}")
    message(STATUS "   PJRT include:  ${PJRT_INCLUDE_DIR}")
    message(STATUS "   PJRT library:  ${PJRT_LIBRARIES}")
    message(STATUS "=== TPU BUILD CONFIGURATION COMPLETE ===")

endfunction()
