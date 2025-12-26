# cmake/CompileMemoryProfiling.cmake
# Enable per-file compilation memory profiling
#
# Usage:
#   cmake -DENABLE_COMPILE_MEMORY_PROFILING=ON ...
#
# This wraps the C++ compiler with a script that tracks peak/average memory
# usage for each compilation unit.

option(ENABLE_COMPILE_MEMORY_PROFILING "Enable per-file compilation memory profiling" OFF)

if(ENABLE_COMPILE_MEMORY_PROFILING)
    set(WRAPPER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/track_compile_memory.sh")

    if(NOT EXISTS "${WRAPPER_SCRIPT}")
        message(FATAL_ERROR "Memory profiling wrapper script not found: ${WRAPPER_SCRIPT}")
    endif()

    # Make sure script is executable
    execute_process(COMMAND chmod +x "${WRAPPER_SCRIPT}")

    # Store original compilers
    set(ORIGINAL_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
    if(CMAKE_CUDA_COMPILER)
        set(ORIGINAL_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
    endif()

    # Create wrapper scripts in build directory
    set(CXX_WRAPPER "${CMAKE_BINARY_DIR}/cxx_wrapper.sh")

    file(WRITE "${CXX_WRAPPER}"
"#!/bin/bash
export BUILD_DIR='${CMAKE_BINARY_DIR}'
exec '${WRAPPER_SCRIPT}' '${ORIGINAL_CXX_COMPILER}' \"$@\"
")

    execute_process(COMMAND chmod +x "${CXX_WRAPPER}")

    # Override CMake compiler variables
    set(CMAKE_CXX_COMPILER "${CXX_WRAPPER}" CACHE FILEPATH "C++ compiler with memory profiling" FORCE)

    # For CUDA if present
    if(CMAKE_CUDA_COMPILER)
        set(CUDA_WRAPPER "${CMAKE_BINARY_DIR}/cuda_wrapper.sh")
        file(WRITE "${CUDA_WRAPPER}"
"#!/bin/bash
export BUILD_DIR='${CMAKE_BINARY_DIR}'
exec '${WRAPPER_SCRIPT}' '${ORIGINAL_CUDA_COMPILER}' \"$@\"
")
        execute_process(COMMAND chmod +x "${CUDA_WRAPPER}")
        set(CMAKE_CUDA_COMPILER "${CUDA_WRAPPER}" CACHE FILEPATH "CUDA compiler with memory profiling" FORCE)
    endif()

    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS "COMPILATION MEMORY PROFILING: ENABLED")
    message(STATUS "========================================")
    message(STATUS "Original C++ compiler: ${ORIGINAL_CXX_COMPILER}")
    message(STATUS "Wrapper script:        ${WRAPPER_SCRIPT}")
    message(STATUS "Memory logs will be saved to: ${CMAKE_BINARY_DIR}/compile_memory_logs/")
    message(STATUS "")
    message(STATUS "After build, analyze results with:")
    message(STATUS "  cd ${CMAKE_BINARY_DIR}")
    message(STATUS "  python3 ${CMAKE_CURRENT_SOURCE_DIR}/analyze_compile_memory.py")
    message(STATUS "")
    message(STATUS "Or for top 30 files:")
    message(STATUS "  python3 ${CMAKE_CURRENT_SOURCE_DIR}/analyze_compile_memory.py --top 30")
    message(STATUS "")
    message(STATUS "Or filter by pattern:")
    message(STATUS "  python3 ${CMAKE_CURRENT_SOURCE_DIR}/analyze_compile_memory.py --pattern pairwise")
    message(STATUS "========================================")
    message(STATUS "")

    # Set compiler identification to bypass CMake's compiler check
    # (since we're wrapping the compiler)
    set(CMAKE_CXX_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}" CACHE STRING "CXX compiler ID" FORCE)
    set(CMAKE_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}" CACHE STRING "CXX compiler version" FORCE)

endif()
