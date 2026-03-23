# cmake/HexagonConfiguration.cmake
# Configuration for Hexagon NPU backend using hexagon-mlir.
#
# Finds hexagon-mlir headers and libraries for NPU compilation.
# Sets HEXAGON_MLIR_FOUND, HEXAGON_MLIR_INCLUDE_DIRS, HEXAGON_MLIR_LIBRARIES.

function(setup_hexagon_paths)
    message(STATUS "Searching for hexagon-mlir SDK...")

    # Search paths for hexagon-mlir
    set(_hexagon_search_paths
        "${HEXAGON_MLIR_ROOT}"
        "$ENV{HEXAGON_MLIR_ROOT}"
        "$ENV{HEXAGON_SDK_ROOT}"
        "/opt/hexagon-mlir"
        "/usr/local/hexagon-mlir"
        "/usr/local/lib/hexagon-mlir"
    )

    # Find headers
    find_path(HEXAGON_MLIR_INCLUDE_DIR
        NAMES hexagon_mlir_runtime.h
        HINTS ${_hexagon_search_paths}
        PATH_SUFFIXES include
    )

    # Find runtime library
    find_library(HEXAGON_MLIR_LIBRARY
        NAMES hexagon_mlir_runtime
        HINTS ${_hexagon_search_paths}
        PATH_SUFFIXES lib lib64
    )

    if(HEXAGON_MLIR_INCLUDE_DIR AND HEXAGON_MLIR_LIBRARY)
        set(HEXAGON_MLIR_FOUND TRUE PARENT_SCOPE)
        set(HEXAGON_MLIR_INCLUDE_DIRS ${HEXAGON_MLIR_INCLUDE_DIR} PARENT_SCOPE)
        set(HEXAGON_MLIR_LIBRARIES ${HEXAGON_MLIR_LIBRARY} PARENT_SCOPE)
        message(STATUS "  hexagon-mlir found: ${HEXAGON_MLIR_LIBRARY}")
        message(STATUS "  hexagon-mlir headers: ${HEXAGON_MLIR_INCLUDE_DIR}")
    else()
        set(HEXAGON_MLIR_FOUND FALSE PARENT_SCOPE)
        message(STATUS "  hexagon-mlir NOT found (will use dlopen at runtime)")
        message(STATUS "  Set HEXAGON_MLIR_ROOT to the SDK installation path")
    endif()
endfunction()

function(build_hexagon_compiler_flags)
    set(HEXAGON_COMPILE_FLAGS "-fPIC -O3" PARENT_SCOPE)
    message(STATUS "Hexagon compiler flags: -fPIC -O3")
endfunction()

function(configure_hexagon_linking target)
    if(HEXAGON_MLIR_FOUND)
        target_include_directories(${target} PRIVATE ${HEXAGON_MLIR_INCLUDE_DIRS})
        target_link_libraries(${target} PRIVATE ${HEXAGON_MLIR_LIBRARIES})
        message(STATUS "Hexagon linking configured for target ${target}")
    else()
        message(STATUS "Hexagon: no compile-time linking (runtime dlopen only)")
    endif()
endfunction()

function(debug_hexagon_configuration)
    message(STATUS "=== Hexagon Configuration ===")
    message(STATUS "  HEXAGON_MLIR_FOUND: ${HEXAGON_MLIR_FOUND}")
    if(HEXAGON_MLIR_FOUND)
        message(STATUS "  HEXAGON_MLIR_INCLUDE_DIRS: ${HEXAGON_MLIR_INCLUDE_DIRS}")
        message(STATUS "  HEXAGON_MLIR_LIBRARIES: ${HEXAGON_MLIR_LIBRARIES}")
    endif()
    message(STATUS "=============================")
endfunction()
