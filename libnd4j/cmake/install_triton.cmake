# install_triton.cmake — Cross-platform Triton install script
#
# Usage: cmake -DBINARY_DIR=<build> -DSOURCE_DIR=<src> -DINSTALL_DIR=<install> -P install_triton.cmake
#
# Triton v3.2.0+ uses OBJECT libraries (no install() rules, no libtriton.a).
# This script archives all .o/.obj files into a static library and copies headers.
# Replaces the bash-only install_triton.sh with a portable CMake script.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED BINARY_DIR)
    message(FATAL_ERROR "BINARY_DIR must be defined")
endif()
if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR must be defined")
endif()
if(NOT DEFINED INSTALL_DIR)
    message(FATAL_ERROR "INSTALL_DIR must be defined")
endif()

message(STATUS "Triton custom install: BINARY_DIR=${BINARY_DIR} SOURCE_DIR=${SOURCE_DIR} INSTALL_DIR=${INSTALL_DIR}")

# Create output directories
file(MAKE_DIRECTORY "${INSTALL_DIR}/lib")
file(MAKE_DIRECTORY "${INSTALL_DIR}/include")

# === Step 1: Find all object files and archive into a static library ===
file(GLOB_RECURSE _OBJ_FILES
    "${BINARY_DIR}/lib/*.o"
    "${BINARY_DIR}/lib/*.obj"
    "${BINARY_DIR}/third_party/*.o"
    "${BINARY_DIR}/third_party/*.obj"
)
list(LENGTH _OBJ_FILES _obj_count)
message(STATUS "Triton: found ${_obj_count} object files to archive")

if(_obj_count EQUAL 0)
    message(FATAL_ERROR "No object files found in ${BINARY_DIR}/lib or ${BINARY_DIR}/third_party")
endif()

# Determine platform and create static library
if(WIN32)
    set(_LIB_NAME "triton.lib")
    set(_LIB_PATH "${INSTALL_DIR}/lib/${_LIB_NAME}")
    # Use lib.exe on Windows (MSVC)
    find_program(_LIB_EXE lib HINTS ENV PATH)
    if(NOT _LIB_EXE)
        find_program(_LIB_EXE llvm-lib HINTS ENV PATH)
    endif()
    if(NOT _LIB_EXE)
        message(FATAL_ERROR "Cannot find lib.exe or llvm-lib for creating static library on Windows")
    endif()
    execute_process(
        COMMAND "${_LIB_EXE}" "/OUT:${_LIB_PATH}" ${_OBJ_FILES}
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "lib.exe failed with exit code ${_result}")
    endif()
else()
    set(_LIB_NAME "libtriton.a")
    set(_LIB_PATH "${INSTALL_DIR}/lib/${_LIB_NAME}")
    # Use ar on Unix
    find_program(_AR_EXE ar HINTS ENV PATH)
    if(NOT _AR_EXE AND DEFINED CMAKE_AR)
        set(_AR_EXE "${CMAKE_AR}")
    endif()
    if(NOT _AR_EXE)
        message(FATAL_ERROR "Cannot find ar for creating static library")
    endif()
    # ar has argument length limits; use response file approach
    set(_OBJ_LIST_FILE "${BINARY_DIR}/_triton_obj_files.txt")
    string(REPLACE ";" "\n" _obj_file_content "${_OBJ_FILES}")
    file(WRITE "${_OBJ_LIST_FILE}" "${_obj_file_content}")
    # Some ar implementations don't support @file, so we use xargs-style batching via cmake
    # First remove any existing archive
    file(REMOVE "${_LIB_PATH}")
    # Create archive in batches (ar rcs with append)
    set(_batch_size 500)
    set(_batch_start 0)
    list(LENGTH _OBJ_FILES _total)
    while(_batch_start LESS _total)
        math(EXPR _batch_end "${_batch_start} + ${_batch_size} - 1")
        if(_batch_end GREATER_EQUAL _total)
            math(EXPR _batch_end "${_total} - 1")
        endif()
        list(SUBLIST _OBJ_FILES ${_batch_start} ${_batch_size} _batch)
        execute_process(
            COMMAND "${_AR_EXE}" rcs "${_LIB_PATH}" ${_batch}
            RESULT_VARIABLE _result
        )
        if(NOT _result EQUAL 0)
            message(FATAL_ERROR "ar rcs failed with exit code ${_result}")
        endif()
        math(EXPR _batch_start "${_batch_end} + 1")
    endwhile()
    file(REMOVE "${_OBJ_LIST_FILE}")
endif()

file(SIZE "${_LIB_PATH}" _lib_size)
math(EXPR _lib_size_mb "${_lib_size} / 1048576")
message(STATUS "Triton: created ${_LIB_PATH} (${_lib_size_mb} MB)")

# === Step 2: Copy core Triton headers ===
if(EXISTS "${SOURCE_DIR}/include/triton")
    file(COPY "${SOURCE_DIR}/include/triton" DESTINATION "${INSTALL_DIR}/include")
    message(STATUS "Triton: installed core source headers")
endif()

# Overlay generated headers from build dir
if(EXISTS "${BINARY_DIR}/include/triton")
    file(COPY "${BINARY_DIR}/include/triton/" DESTINATION "${INSTALL_DIR}/include/triton")
    message(STATUS "Triton: installed core generated headers")
endif()

# === Step 3: Copy NVIDIA backend headers ===
if(EXISTS "${BINARY_DIR}/third_party/nvidia/include")
    if(EXISTS "${SOURCE_DIR}/third_party/nvidia/include")
        file(GLOB _nvidia_src_dirs "${SOURCE_DIR}/third_party/nvidia/include/*")
        foreach(_dir ${_nvidia_src_dirs})
            file(COPY "${_dir}" DESTINATION "${INSTALL_DIR}/include")
        endforeach()
        message(STATUS "Triton: installed NVIDIA source headers")
    endif()
    file(GLOB _nvidia_build_dirs "${BINARY_DIR}/third_party/nvidia/include/*")
    foreach(_dir ${_nvidia_build_dirs})
        file(COPY "${_dir}" DESTINATION "${INSTALL_DIR}/include")
    endforeach()
    message(STATUS "Triton: installed NVIDIA generated headers")
endif()

# Create nvidia/include/ mirror for internal relative includes.
# Triton's NVIDIA headers use #include "nvidia/include/.../Passes.h.inc"
if(EXISTS "${INSTALL_DIR}/include/TritonNVIDIAGPUToLLVM")
    file(MAKE_DIRECTORY "${INSTALL_DIR}/include/nvidia/include")
    if(WIN32)
        # Windows: copy instead of symlink (symlinks require elevated privileges)
        if(EXISTS "${INSTALL_DIR}/include/TritonNVIDIAGPUToLLVM")
            file(COPY "${INSTALL_DIR}/include/TritonNVIDIAGPUToLLVM"
                 DESTINATION "${INSTALL_DIR}/include/nvidia/include")
        endif()
        if(EXISTS "${INSTALL_DIR}/include/NVGPUToLLVM")
            file(COPY "${INSTALL_DIR}/include/NVGPUToLLVM"
                 DESTINATION "${INSTALL_DIR}/include/nvidia/include")
        endif()
        if(EXISTS "${INSTALL_DIR}/include/Dialect")
            file(COPY "${INSTALL_DIR}/include/Dialect"
                 DESTINATION "${INSTALL_DIR}/include/nvidia/include")
        endif()
        message(STATUS "Triton: created nvidia/include/ copies for relative includes (Windows)")
    else()
        # Unix: use symlinks for efficiency
        # Remove any pre-existing directory/symlink to avoid CREATE_LINK failure
        if(EXISTS "${INSTALL_DIR}/include/nvidia/include/TritonNVIDIAGPUToLLVM")
            file(REMOVE_RECURSE "${INSTALL_DIR}/include/nvidia/include/TritonNVIDIAGPUToLLVM")
        endif()
        file(CREATE_LINK "../../TritonNVIDIAGPUToLLVM"
             "${INSTALL_DIR}/include/nvidia/include/TritonNVIDIAGPUToLLVM" SYMBOLIC)
        if(EXISTS "${INSTALL_DIR}/include/NVGPUToLLVM")
            if(EXISTS "${INSTALL_DIR}/include/nvidia/include/NVGPUToLLVM")
                file(REMOVE_RECURSE "${INSTALL_DIR}/include/nvidia/include/NVGPUToLLVM")
            endif()
            file(CREATE_LINK "../../NVGPUToLLVM"
                 "${INSTALL_DIR}/include/nvidia/include/NVGPUToLLVM" SYMBOLIC)
        endif()
        if(EXISTS "${INSTALL_DIR}/include/Dialect")
            if(EXISTS "${INSTALL_DIR}/include/nvidia/include/Dialect")
                file(REMOVE_RECURSE "${INSTALL_DIR}/include/nvidia/include/Dialect")
            endif()
            file(CREATE_LINK "../../Dialect"
                 "${INSTALL_DIR}/include/nvidia/include/Dialect" SYMBOLIC)
        endif()
        message(STATUS "Triton: created nvidia/include/ symlinks for relative includes")
    endif()
endif()

# === Step 4: Copy AMD backend headers ===
if(EXISTS "${BINARY_DIR}/third_party/amd/include")
    if(EXISTS "${SOURCE_DIR}/third_party/amd/include")
        file(GLOB _amd_src_dirs "${SOURCE_DIR}/third_party/amd/include/*")
        foreach(_dir ${_amd_src_dirs})
            file(COPY "${_dir}" DESTINATION "${INSTALL_DIR}/include")
        endforeach()
        message(STATUS "Triton: installed AMD source headers")
    endif()
    file(GLOB _amd_build_dirs "${BINARY_DIR}/third_party/amd/include/*")
    foreach(_dir ${_amd_build_dirs})
        file(COPY "${_dir}" DESTINATION "${INSTALL_DIR}/include")
    endforeach()
    message(STATUS "Triton: installed AMD generated headers")
endif()

# AMD headers contain relative includes rooted at third_party/amd/include/...
if(EXISTS "${INSTALL_DIR}/include/TritonAMDGPUToLLVM")
    file(MAKE_DIRECTORY "${INSTALL_DIR}/include/third_party/amd")
    if(WIN32)
        # Windows: copy the include dir contents
        file(GLOB _amd_install_dirs "${INSTALL_DIR}/include/TritonAMDGPU*")
        foreach(_dir ${_amd_install_dirs})
            file(COPY "${_dir}" DESTINATION "${INSTALL_DIR}/include/third_party/amd/include")
        endforeach()
        message(STATUS "Triton: created third_party/amd/include copy mirror (Windows)")
    else()
        file(CREATE_LINK "../../" "${INSTALL_DIR}/include/third_party/amd/include" SYMBOLIC)
        message(STATUS "Triton: created third_party/amd/include symlink mirror")
    endif()
endif()

# === Step 5: Optional Intel backend headers ===
if(EXISTS "${BINARY_DIR}/third_party/intel/include")
    if(EXISTS "${SOURCE_DIR}/third_party/intel/include")
        file(GLOB _intel_src_dirs "${SOURCE_DIR}/third_party/intel/include/*")
        foreach(_dir ${_intel_src_dirs})
            file(COPY "${_dir}" DESTINATION "${INSTALL_DIR}/include")
        endforeach()
        message(STATUS "Triton: installed Intel source headers")
    endif()
    file(GLOB _intel_build_dirs "${BINARY_DIR}/third_party/intel/include/*")
    foreach(_dir ${_intel_build_dirs})
        file(COPY "${_dir}" DESTINATION "${INSTALL_DIR}/include")
    endforeach()
    message(STATUS "Triton: installed Intel generated headers")
endif()

message(STATUS "Triton custom install complete: ${_LIB_PATH}")
