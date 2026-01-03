# cmake/Dependencies.cmake
# Manages all third-party dependencies. Logic is encapsulated in functions.

include(ExternalProject)
include(ProcessorCount)

# =============================================================================
# PARALLEL BUILD CONFIGURATION FOR DEPENDENCIES
# =============================================================================
# Compute optimal parallel jobs for dependency builds based on:
# 1. User-specified SD_PARALLEL_COMPILE_JOBS (if set)
# 2. Available memory (each compile uses ~1-2GB for deps, less than main build)
# 3. Available CPU cores

ProcessorCount(NPROC)
if(NPROC EQUAL 0)
    set(NPROC 4)  # Fallback
endif()

# Use SD_PARALLEL_COMPILE_JOBS if specified, otherwise compute from memory
if(DEFINED SD_PARALLEL_COMPILE_JOBS AND NOT SD_PARALLEL_COMPILE_JOBS STREQUAL "" AND NOT SD_PARALLEL_COMPILE_JOBS STREQUAL "0")
    set(DEP_PARALLEL_JOBS ${SD_PARALLEL_COMPILE_JOBS})
else()
    # Dependencies use less memory per compile (~1GB vs 2GB for main build)
    # So we can be more aggressive with parallelism
    cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
    math(EXPR MEM_BASED_JOBS "${AVAILABLE_MEMORY} / 1000")  # 1GB per job for deps

    # Cap at processor count and ensure at least 4
    if(MEM_BASED_JOBS GREATER NPROC)
        set(DEP_PARALLEL_JOBS ${NPROC})
    elseif(MEM_BASED_JOBS LESS 4)
        set(DEP_PARALLEL_JOBS 4)
    else()
        set(DEP_PARALLEL_JOBS ${MEM_BASED_JOBS})
    endif()
endif()

message(STATUS "🔧 Dependency builds will use ${DEP_PARALLEL_JOBS} parallel jobs (${NPROC} cores, ${AVAILABLE_MEMORY}MB available)")
function(setup_android_arm_openblas)
    set(is_android_or_arm FALSE)

    if(ANDROID OR SD_ANDROID_BUILD OR SD_ARM_BUILD OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
        set(is_android_or_arm TRUE)
    endif()

    if(NOT is_android_or_arm)
        return()
    endif()

    message(STATUS "🔧 Setting up OpenBLAS for Android/ARM platform")

    # Handle path normalization for Android/ARM
    if(OPENBLAS_PATH MATCHES "lib/[^/]+$")
        get_filename_component(OPENBLAS_PATH "${OPENBLAS_PATH}/../.." ABSOLUTE)
        message(STATUS "🔧 Normalized OPENBLAS_PATH: ${OPENBLAS_PATH}")
    endif()

    # Platform-specific OpenBLAS library name handling
    if(EXISTS "${OPENBLAS_PATH}/lib")
        # Define search patterns based on platform
        set(LIB_SEARCH_PATTERNS
                "${OPENBLAS_PATH}/lib/libopenblas.so"
                "${OPENBLAS_PATH}/lib/libopenblas.a"
        )

        # Add platform-specific patterns
        if(ANDROID OR SD_ANDROID_BUILD)
            if(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
                list(APPEND LIB_SEARCH_PATTERNS
                        "${OPENBLAS_PATH}/lib/android-x86_64/libopenblas.so"
                        "${OPENBLAS_PATH}/lib/android-x86_64/libopenblas.a"
                )
            elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
                list(APPEND LIB_SEARCH_PATTERNS
                        "${OPENBLAS_PATH}/lib/android-arm64/libopenblas.so"
                        "${OPENBLAS_PATH}/lib/android-arm64/libopenblas.a"
                        "${OPENBLAS_PATH}/lib/android-aarch64/libopenblas.so"
                        "${OPENBLAS_PATH}/lib/android-aarch64/libopenblas.a"
                )
            endif()
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
            list(APPEND LIB_SEARCH_PATTERNS
                    "${OPENBLAS_PATH}/lib/linux-arm64/libopenblas.so"
                    "${OPENBLAS_PATH}/lib/linux-arm64/libopenblas.a"
                    "${OPENBLAS_PATH}/lib/linux-aarch64/libopenblas.so"
                    "${OPENBLAS_PATH}/lib/linux-aarch64/libopenblas.a"
            )
        endif()

        # Search for libraries
        set(FOUND_OPENBLAS_LIBS "")
        foreach(pattern ${LIB_SEARCH_PATTERNS})
            file(GLOB matched_libs ${pattern})
            if(matched_libs)
                list(APPEND FOUND_OPENBLAS_LIBS ${matched_libs})
            endif()
        endforeach()

        if(FOUND_OPENBLAS_LIBS)
            message(STATUS "✅ Found OpenBLAS libraries: ${FOUND_OPENBLAS_LIBS}")
        else()
            message(WARNING "⚠️  No OpenBLAS libraries found in ${OPENBLAS_PATH}/lib")
        endif()
    endif()

    # Set platform-specific compiler flags for OpenBLAS
    if(ANDROID OR SD_ANDROID_BUILD)
        if(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64" PARENT_SCOPE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64" PARENT_SCOPE)
        elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a" PARENT_SCOPE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a" PARENT_SCOPE)
        endif()
    elseif(SD_ARM_BUILD OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a -mtune=cortex-a72" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -mtune=cortex-a72" PARENT_SCOPE)
    endif()

    # Set additional ARM-specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64" OR CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfix-cortex-a53-835769 -mfix-cortex-a53-843419" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfix-cortex-a53-835769 -mfix-cortex-a53-843419" PARENT_SCOPE)
    endif()
endfunction()


function(setup_blas)
    if(SD_CUDA)
        return()
    endif()

    if(NOT OPENBLAS_PATH)
        message(STATUS "❌ OPENBLAS_PATH not set")
        return()
    endif()

    # Handle Android path normalization at CMake level (in case shell script normalization didn't work)
    setup_android_arm_openblas()


    # Verify the path exists and has the required headers
    if(NOT EXISTS "${OPENBLAS_PATH}/include")
        message(STATUS "❌ OpenBLAS include directory not found: ${OPENBLAS_PATH}/include")
        return()
    endif()

    if(NOT EXISTS "${OPENBLAS_PATH}/include/cblas.h")
        message(STATUS "❌ OpenBLAS cblas.h not found: ${OPENBLAS_PATH}/include/cblas.h")
        return()
    endif()

    # Set up OpenBLAS
    message(STATUS "✅ Setting up OpenBLAS:")
    message(STATUS "   Path: ${OPENBLAS_PATH}")
    message(STATUS "   Include: ${OPENBLAS_PATH}/include")
    message(STATUS "   Library: ${OPENBLAS_PATH}/")

    # Use global include_directories for compatibility
    include_directories(${OPENBLAS_PATH}/include/)

    # Find the actual OpenBLAS library file
    set(OPENBLAS_LIB_FOUND "")
    foreach(lib_candidate
            "${OPENBLAS_PATH}/lib/libopenblas.so"
            "${OPENBLAS_PATH}/libopenblas.so"
            "${OPENBLAS_PATH}/lib/libopenblas.a"
            "${OPENBLAS_PATH}/libopenblas.a")
        if(EXISTS "${lib_candidate}" AND NOT OPENBLAS_LIB_FOUND)
            set(OPENBLAS_LIB_FOUND "${lib_candidate}")
            message(STATUS "   Found OpenBLAS library: ${OPENBLAS_LIB_FOUND}")
        endif()
    endforeach()

    if(NOT OPENBLAS_LIB_FOUND)
        message(WARNING "⚠️  Could not find libopenblas.so or libopenblas.a in ${OPENBLAS_PATH}")
        # Fallback to link_directories approach
        if(EXISTS "${OPENBLAS_PATH}/lib")
            link_directories(${OPENBLAS_PATH}/lib)
        endif()
        link_directories(${OPENBLAS_PATH})
        set(OPENBLAS_LIB_FOUND "openblas")
    endif()

    add_compile_definitions(HAVE_OPENBLAS=1)
    # Note: OpenBLAS bfloat16 conflict is handled in BlasHelper.h via:
    #   struct bfloat16;
    #   #define BFLOAT16 BFLOAT16
    # This self-referential macro prevents OpenBLAS from typedef'ing bfloat16
    # while not breaking libnd4j's BFLOAT16 enum value in DataType.h

    # Set parent scope variables
    set(HAVE_OPENBLAS 1 PARENT_SCOPE)
    set(OPENBLAS_LIBRARIES "${OPENBLAS_LIB_FOUND}" PARENT_SCOPE)

    set(OPENBLAS_PATH "${OPENBLAS_PATH}" PARENT_SCOPE)

    message(STATUS "✅ OpenBLAS setup complete")
endfunction()




function(setup_cudnn)
    set(HAVE_CUDNN false PARENT_SCOPE)
    set(CUDNN "" PARENT_SCOPE)

    if(NOT (HELPERS_cudnn STREQUAL "ON" AND SD_CUDA))
        message(STATUS "cuDNN helper is disabled (HELPERS_cudnn=${HELPERS_cudnn}, SD_CUDA=${SD_CUDA})")
        return()
    endif()
endfunction()
# =============================================================================
# FLATBUFFERS (Required) - Cross-compilation compatible version
# =============================================================================
function(setup_flatbuffers)
    set(FLATBUFFERS_VERSION "25.2.10")
    set(FLATBUFFERS_URL "https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VERSION}.tar.gz")

    # Determine if we should build flatc
    set(SHOULD_BUILD_FLATC FALSE)
    if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
        set(SHOULD_BUILD_FLATC TRUE)
    endif()

    if(CMAKE_CROSSCOMPILING AND SHOULD_BUILD_FLATC)
        # Cross-compilation scenario: build flatc for host, library for target
        message(STATUS "Cross-compiling FlatBuffers: building flatc for host, library for target")

        # Stage 1: Build flatc for host system
        set(FLATC_HOST_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-host")
        set(FLATC_HOST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-host-build")
        set(FLATC_EXECUTABLE "${FLATC_HOST_BUILD_DIR}/flatc")

        # Determine host system compilers
        find_program(HOST_C_COMPILER NAMES gcc clang cc)
        find_program(HOST_CXX_COMPILER NAMES g++ clang++ c++)

        if(NOT HOST_C_COMPILER OR NOT HOST_CXX_COMPILER)
            message(FATAL_ERROR "Could not find host system compilers for flatc build")
        endif()

        # Build CMAKE_ARGS without toolchain file for host build
        set(HOST_CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DFLATBUFFERS_BUILD_FLATC=ON
                -DFLATBUFFERS_BUILD_FLATLIB=OFF
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SAMPLES=OFF
                -DCMAKE_C_COMPILER=${HOST_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${HOST_CXX_COMPILER}
        )

        # Pass compiler launcher (ccache/sccache) to host build if available
        # Use quotes to handle paths with spaces and prevent list expansion issues
        if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}")
            list(APPEND HOST_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER}")
        endif()
        if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}")
            list(APPEND HOST_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER}")
        endif()

        ExternalProject_Add(flatbuffers_host
                URL               ${FLATBUFFERS_URL}
                SOURCE_DIR        "${FLATC_HOST_DIR}"
                BINARY_DIR        "${FLATC_HOST_BUILD_DIR}"
                CMAKE_ARGS        ${HOST_CMAKE_ARGS}
                BUILD_COMMAND     ${CMAKE_COMMAND} --build . --target flatc --config Release --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${FLATC_EXECUTABLE}"
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        # Stage 2: Build FlatBuffers library for target
        # Build CMAKE_ARGS for target build, only include variables that are set
        set(TARGET_CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_BUILD_TYPE=Release
                -DFLATBUFFERS_BUILD_FLATC=OFF
                -DFLATBUFFERS_BUILD_FLATLIB=ON
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SAMPLES=OFF
        )

        # Pass compiler launcher (ccache/sccache) to target build if available
        if(CMAKE_C_COMPILER_LAUNCHER)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER})
        endif()
        if(CMAKE_CXX_COMPILER_LAUNCHER)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER})
        endif()

        # Only add cross-compilation arguments if they are defined
        if(CMAKE_TOOLCHAIN_FILE)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
        endif()
        if(CMAKE_SYSTEM_NAME)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})
        endif()
        if(CMAKE_SYSTEM_VERSION)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION})
        endif()
        if(CMAKE_ANDROID_ARCH_ABI)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI})
        endif()
        if(CMAKE_ANDROID_NDK)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK})
        endif()
        if(CMAKE_ANDROID_STL_TYPE)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE})
        endif()
        if(ANDROID_ABI)
            list(APPEND TARGET_CMAKE_ARGS -DANDROID_ABI=${ANDROID_ABI})
        endif()
        if(ANDROID_PLATFORM)
            list(APPEND TARGET_CMAKE_ARGS -DANDROID_PLATFORM=${ANDROID_PLATFORM})
        endif()
        if(ANDROID_STL)
            list(APPEND TARGET_CMAKE_ARGS -DANDROID_STL=${ANDROID_STL})
        endif()

        ExternalProject_Add(flatbuffers_target
                URL               ${FLATBUFFERS_URL}
                SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src"
                BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-build"
                CMAKE_ARGS        ${TARGET_CMAKE_ARGS}
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-build/libflatbuffers.a"
                DEPENDS           flatbuffers_host
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        # DO NOT use include_directories() - use target_include_directories on flatbuffers_interface instead
        # Set up include directories and library
        # include_directories("${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src/include")
        set(FLATBUFFERS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-build/libflatbuffers.a")
        set(FLATBUFFERS_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src")

        # Create interface library for target
        add_library(flatbuffers_interface INTERFACE)
        target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})
        target_include_directories(flatbuffers_interface INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src/include")
        add_dependencies(flatbuffers_interface flatbuffers_target)

        # Check if flatbuffers.h already exists
        set(FLATBUFFERS_HEADER_DEST "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers/flatbuffers.h")
        if(EXISTS ${FLATBUFFERS_HEADER_DEST})
            message(STATUS "Found existing flatbuffers.h at ${FLATBUFFERS_HEADER_DEST}")
        endif()

        # Generate headers and copy Java files inline after ExternalProject builds
        # Copy ALL flatbuffers headers (modular structure in newer versions)
        ExternalProject_Add_Step(flatbuffers_host generate_headers_and_copy_java
                COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}"
                bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
                COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
                COMMAND ${CMAKE_COMMAND} -E remove_directory
                "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-host/include/flatbuffers"
                "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Generating FlatBuffers headers, copying Java files, and copying all flatbuffers headers using host flatc"
                DEPENDEES build
                BYPRODUCTS
                ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h
                ${CMAKE_CURRENT_SOURCE_DIR}/.java_files_copied
        )

    else()
        # Native build or cross-compilation without flatc generation
        message(STATUS "Native FlatBuffers build")

        if(SHOULD_BUILD_FLATC)
            set(FLATBUFFERS_BUILD_FLATC "ON")
        else()
            set(FLATBUFFERS_BUILD_FLATC "OFF")
        endif()

        # Build CMAKE_ARGS list for native build
        set(NATIVE_CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_BUILD_TYPE=Release
                -DFLATBUFFERS_BUILD_FLATC=${FLATBUFFERS_BUILD_FLATC}
                -DFLATBUFFERS_BUILD_FLATLIB=ON
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SAMPLES=OFF
        )

        # Pass compiler launcher (ccache/sccache) to native build if available
        if(CMAKE_C_COMPILER_LAUNCHER)
            list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER})
        endif()
        if(CMAKE_CXX_COMPILER_LAUNCHER)
            list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER})
        endif()

        ExternalProject_Add(flatbuffers_external
                URL               ${FLATBUFFERS_URL}
                SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src"
                BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build"
                CMAKE_ARGS        ${NATIVE_CMAKE_ARGS}
                # Use computed parallel jobs for faster dependency builds
                BUILD_COMMAND     ${CMAKE_COMMAND} --build . --config Release --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc"
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/libflatbuffers.a"
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        # DO NOT use include_directories() - use target_include_directories on flatbuffers_interface instead
        # include_directories("${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")
        set(FLATBUFFERS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/libflatbuffers.a")
        set(FLATBUFFERS_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src")

        # Check if flatbuffers.h already exists
        set(FLATBUFFERS_HEADER_DEST "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers/flatbuffers.h")
        if(EXISTS ${FLATBUFFERS_HEADER_DEST})
            message(STATUS "Found existing flatbuffers.h at ${FLATBUFFERS_HEADER_DEST}")
        endif()

        if(SHOULD_BUILD_FLATC)
            set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")

            # Generate headers and copy Java files inline after ExternalProject builds
            # Copy ALL flatbuffers headers (modular structure in newer versions)
            ExternalProject_Add_Step(flatbuffers_external generate_headers_and_copy_java
                    COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}"
                    bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
                    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
                    COMMAND ${CMAKE_COMMAND} -E remove_directory
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include/flatbuffers"
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating FlatBuffers headers, copying Java files, and copying all flatbuffers headers"
                    DEPENDEES build
                    BYPRODUCTS
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/.java_files_copied
            )
        else()
            # Even without flatc generation, copy ALL flatbuffers headers
            # Newer flatbuffers uses modular headers - flatbuffers.h includes array.h, base.h, vector.h, etc.
            ExternalProject_Add_Step(flatbuffers_external copy_flatbuffers_headers
                    COMMAND ${CMAKE_COMMAND} -E remove_directory
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include/flatbuffers"
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying all flatbuffers headers (modular structure)"
                    DEPENDEES build
            )
        endif()

        # Create interface library
        add_library(flatbuffers_interface INTERFACE)
        target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})
        target_include_directories(flatbuffers_interface INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")
        add_dependencies(flatbuffers_interface flatbuffers_external)
    endif()

    # Set global variables for parent scope
    set(FLATBUFFERS_LIBRARY ${FLATBUFFERS_LIBRARY} PARENT_SCOPE)
    set(FLATBUFFERS_SOURCE_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
    if(SHOULD_BUILD_FLATC)
        set(FLATC_EXECUTABLE ${FLATC_EXECUTABLE} PARENT_SCOPE)
    endif()

    message(STATUS "✅ FlatBuffers setup complete")
    if(CMAKE_CROSSCOMPILING AND SHOULD_BUILD_FLATC)
        message(STATUS "   Host flatc: ${FLATC_EXECUTABLE}")
        message(STATUS "   Target library: ${FLATBUFFERS_LIBRARY}")
    else()
        message(STATUS "   Library: ${FLATBUFFERS_LIBRARY}")
        if(SHOULD_BUILD_FLATC)
            message(STATUS "   flatc: ${FLATC_EXECUTABLE}")
        endif()
    endif()
endfunction()
# =============================================================================
# ONEDNN (Optional)
# =============================================================================
function(setup_onednn)
    if(NOT HELPERS_onednn STREQUAL "ON")
        message(STATUS "OneDNN helper is disabled (HELPERS_onednn=${HELPERS_onednn})")
        set(HAVE_ONEDNN OFF CACHE BOOL "OneDNN availability" FORCE)
        set(ONEDNN "" PARENT_SCOPE)
        return()
    endif()

    if(TARGET onednn_external)
        message(STATUS "OneDNN helper is enabled (target already exists)")
        set(HAVE_ONEDNN ON CACHE BOOL "OneDNN availability" FORCE)
        set(ONEDNN onednn_interface PARENT_SCOPE)
        return()
    endif()

    message(STATUS "OneDNN helper is enabled")
    set(HAVE_ONEDNN ON CACHE BOOL "OneDNN availability" FORCE)
    set(ONEDNN_INSTALL_DIR "${CMAKE_BINARY_DIR}/onednn_install")
    set(ONEDNN_PREFIX "${CMAKE_BINARY_DIR}/onednn_external")
    set(ONEDNN_VERSION "3.8.1")
    set(ONEDNN_STAMP_DIR "${ONEDNN_PREFIX}/stamp")

    # Ensure stamp directory exists to prevent "cmake -E touch: failed to update" errors
    # This can happen when parallel builds race on directory creation
    file(MAKE_DIRECTORY "${ONEDNN_STAMP_DIR}")
    file(MAKE_DIRECTORY "${ONEDNN_PREFIX}/src")
    file(MAKE_DIRECTORY "${ONEDNN_PREFIX}/build")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")

    # Clean up stale stamp files that can cause "Failed to copy script-last-run stamp file" errors
    if(EXISTS "${ONEDNN_STAMP_DIR}")
        file(GLOB STALE_STAMPS "${ONEDNN_STAMP_DIR}/*-lastrun.txt" "${ONEDNN_STAMP_DIR}/*.txt")
        foreach(stamp ${STALE_STAMPS})
            message(STATUS "Cleaning stale OneDNN stamp file: ${stamp}")
            file(REMOVE "${stamp}")
        endforeach()
    endif()

    # Use URL download instead of git clone for more robust downloads
    set(ONEDNN_URL "https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz")

    # Build CMAKE_ARGS list for OneDNN
    set(ONEDNN_CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DDNNL_LIBRARY_TYPE=STATIC
            -DDNNL_BUILD_TESTS=OFF
            -DDNNL_BUILD_EXAMPLES=OFF
            -DDNNL_VERBOSE=OFF
            -DONEDNN_BUILD_GRAPH=ON
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )

    # Pass compiler launcher (ccache/sccache) to OneDNN build if available
    # Use quotes to handle paths with spaces and prevent list expansion issues
    # Also verify the launcher exists (it could be the smart_ccache.sh wrapper)
    if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}")
        list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER}")
        message(STATUS "   Passing C compiler launcher to OneDNN: ${CMAKE_C_COMPILER_LAUNCHER}")
    endif()
    if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}")
        list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER}")
        message(STATUS "   Passing CXX compiler launcher to OneDNN: ${CMAKE_CXX_COMPILER_LAUNCHER}")
    endif()

    ExternalProject_Add(onednn_external
            PREFIX            "${ONEDNN_PREFIX}"
            URL               "${ONEDNN_URL}"
            DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
            SOURCE_DIR        "${ONEDNN_PREFIX}/src/oneDNN-${ONEDNN_VERSION}"
            BINARY_DIR        "${ONEDNN_PREFIX}/build"
            STAMP_DIR         "${ONEDNN_STAMP_DIR}"
            DOWNLOAD_NO_PROGRESS FALSE
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            CMAKE_ARGS        ${ONEDNN_CMAKE_ARGS}
            BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE} --parallel ${DEP_PARALLEL_JOBS}
            INSTALL_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config ${CMAKE_BUILD_TYPE}
            BUILD_BYPRODUCTS
                "${ONEDNN_INSTALL_DIR}/include/dnnl.h"
                "${ONEDNN_INSTALL_DIR}/lib64/libdnnl.a"
                "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib"
            TIMEOUT           900
            LOG_DOWNLOAD      OFF
            LOG_CONFIGURE     OFF
            LOG_BUILD         OFF
            LOG_INSTALL       OFF
    )

    add_library(onednn_interface INTERFACE)
    target_include_directories(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/include")
    if(WIN32)
        target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib")
    else()
        target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/lib64/libdnnl.a")
    endif()
    add_dependencies(onednn_interface onednn_external)
    set(ONEDNN onednn_interface PARENT_SCOPE)

    message(STATUS "✅ OneDNN ${ONEDNN_VERSION} setup complete (using URL download)")
endfunction()

# =============================================================================
# ARM COMPUTE LIBRARY (Optional)
# =============================================================================
function(setup_armcompute)
    set(HAVE_ARMCOMPUTE 0 PARENT_SCOPE)
    if(NOT HELPERS_armcompute STREQUAL "ON")
        message(STATUS "ARM Compute helper is disabled (HELPERS_armcompute=${HELPERS_armcompute})")
        return()
    endif()

    if(TARGET armcompute_external)
        set(HAVE_ARMCOMPUTE 1 PARENT_SCOPE)
        set(ARMCOMPUTE_LIBRARIES armcompute_interface PARENT_SCOPE)
        return()
    endif()

    if(LIBND4J_BUILD_WITH_ARMCOMPUTE AND (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64"))
        set(ARMCOMPUTE_INSTALL_DIR "${CMAKE_BINARY_DIR}/armcompute_install")
        set(ARMCOMPUTE_VERSION "v25.04")
        set(ARMCOMPUTE_ARCH "aarch64")
        set(ARMCOMPUTE_PLATFORM "linux")
        set(ARMCOMPUTE_FLAVOR "cpu")
        set(ARMCOMPUTE_PKG_NAME "arm_compute-${ARMCOMPUTE_VERSION}-${ARMCOMPUTE_PLATFORM}-${ARMCOMPUTE_ARCH}-${ARMCOMPUTE_FLAVOR}-bin")
        set(ARMCOMPUTE_URL "https://github.com/ARM-software/ComputeLibrary/releases/download/${ARMCOMPUTE_VERSION}/${ARMCOMPUTE_PKG_NAME}.tar.gz")

        ExternalProject_Add(armcompute_external
                PREFIX      "${CMAKE_BINARY_DIR}/armcompute_external"
                URL         "${ARMCOMPUTE_URL}"
                DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads"
                CONFIGURE_COMMAND ""
                BUILD_COMMAND     ""
                INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/${ARMCOMPUTE_PKG_NAME} ${ARMCOMPUTE_INSTALL_DIR}
                BUILD_BYPRODUCTS "${ARMCOMPUTE_INSTALL_DIR}/include/arm_compute/core/CL/CLKernelLibrary.h"
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        add_library(armcompute_interface INTERFACE)
        target_include_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/include")
        target_link_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/lib")
        target_link_libraries(armcompute_interface INTERFACE arm_compute arm_compute_graph)
        add_dependencies(armcompute_interface armcompute_external)

        set(ARMCOMPUTE_LIBRARIES armcompute_interface PARENT_SCOPE)
        set(HAVE_ARMCOMPUTE 1 PARENT_SCOPE)
    endif()
endfunction()

# =============================================================================
# CUDNN (Optional, for CUDA builds)
# =============================================================================
function(setup_cudnn)
    set(HAVE_CUDNN false PARENT_SCOPE)
    set(CUDNN "" PARENT_SCOPE)

    if(NOT (HELPERS_cudnn AND SD_CUDA))
        return()
    endif()

    find_package(CUDNN)
    if(CUDNN_FOUND)
        message(STATUS "✓ Found cuDNN: ${CUDNN_LIBRARY}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true PARENT_SCOPE)
        set(CUDNN ${CUDNN_LIBRARIES} PARENT_SCOPE)
        add_definitions(-DHAVE_CUDNN=1)
    else()
        message(WARNING "✗ cuDNN not found. Continuing without cuDNN support.")
    endif()
endfunction()

# =============================================================================
# MLIR / LLVM (Optional)
# Provides JIT compilation support via MLIR for optimized kernel execution
# =============================================================================
function(setup_mlir)
    if(NOT HELPERS_mlir STREQUAL "ON")
        message(STATUS "MLIR helper is disabled (HELPERS_mlir=${HELPERS_mlir})")
        set(HAVE_MLIR FALSE PARENT_SCOPE)
        set(MLIR "" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "MLIR/LLVM helper is enabled")

    # Include the FindMLIR module
    include(${CMAKE_CURRENT_LIST_DIR}/FindMLIR.cmake)

    if(NOT MLIR_FOUND)
        message(WARNING "MLIR/LLVM ${MLIR_VERSION}+ not found. MLIR support will be disabled.")
        message(WARNING "To enable MLIR support:")
        message(WARNING "  1. Install LLVM ${MLIR_VERSION}+ with MLIR enabled")
        message(WARNING "  2. Set LLVM_DIR or LLVM_ROOT to your LLVM installation")
        set(HAVE_MLIR FALSE PARENT_SCOPE)
        set(MLIR "" PARENT_SCOPE)
        return()
    endif()

    set(HAVE_MLIR TRUE PARENT_SCOPE)
    set(MLIR MLIR::MLIR PARENT_SCOPE)

    # Configure GPU support if requested and CUDA is enabled
    if(MLIR_ENABLE_GPU AND SD_CUDA)
        if(TARGET MLIR::GPU)
            message(STATUS "   MLIR GPU dialect: enabled")
        else()
            message(WARNING "   MLIR GPU libraries not found. GPU dialect will not be available.")
        endif()
    endif()

    # Add compile definitions for conditional compilation
    add_compile_definitions(HAVE_MLIR=1)
    if(MLIR_ENABLE_GPU AND SD_CUDA)
        add_compile_definitions(MLIR_ENABLE_GPU=1)
    endif()
    if(MLIR_JIT_CACHE)
        add_compile_definitions(MLIR_JIT_CACHE=1)
    endif()
    if(MLIR_DEBUG_DUMPS)
        add_compile_definitions(MLIR_DEBUG_DUMPS=1)
    endif()

    message(STATUS "✅ MLIR/LLVM setup complete")
    message(STATUS "   LLVM Version: ${LLVM_VERSION}")
    message(STATUS "   GPU Support: ${MLIR_ENABLE_GPU}")
    message(STATUS "   JIT Cache: ${MLIR_JIT_CACHE}")
endfunction()

# =============================================================================
# METAL PERFORMANCE SHADERS (Optional, for macOS/iOS builds)
# =============================================================================
# =============================================================================
# ZLUDA Transpiler (Optional, for AMD/Intel GPU support via CUDA translation)
# Downloads ZLUDA and optionally MIOpen for AMD targets
# =============================================================================
function(setup_zluda_download)
    if(NOT SD_ZLUDA)
        message(STATUS "ZLUDA is disabled (SD_ZLUDA=${SD_ZLUDA})")
        set(HAVE_ZLUDA FALSE PARENT_SCOPE)
        return()
    endif()

    # Check if ZLUDA is already available via environment
    if(DEFINED ENV{ZLUDA_PATH} AND EXISTS "$ENV{ZLUDA_PATH}")
        message(STATUS "Using existing ZLUDA installation: $ENV{ZLUDA_PATH}")
        set(ZLUDA_ROOT "$ENV{ZLUDA_PATH}")
        set(HAVE_ZLUDA TRUE PARENT_SCOPE)
        set(ZLUDA_PATH "${ZLUDA_ROOT}" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "ZLUDA automatic download enabled")

    # ZLUDA release configuration
    # ZLUDA v3 supports both AMD (ROCm/HIP) and Intel (Level Zero) GPUs
    set(ZLUDA_VERSION "3")
    set(ZLUDA_INSTALL_DIR "${CMAKE_BINARY_DIR}/zluda_install")

    # Determine platform-specific download
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            set(ZLUDA_PLATFORM "linux-x86_64")
            set(ZLUDA_ARCHIVE_EXT "tar.gz")
        else()
            message(WARNING "ZLUDA: Unsupported processor ${CMAKE_SYSTEM_PROCESSOR} on Linux")
            set(HAVE_ZLUDA FALSE PARENT_SCOPE)
            return()
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
            set(ZLUDA_PLATFORM "windows-x86_64")
            set(ZLUDA_ARCHIVE_EXT "zip")
        else()
            message(WARNING "ZLUDA: Unsupported processor ${CMAKE_SYSTEM_PROCESSOR} on Windows")
            set(HAVE_ZLUDA FALSE PARENT_SCOPE)
            return()
        endif()
    else()
        message(WARNING "ZLUDA: Unsupported platform ${CMAKE_SYSTEM_NAME}")
        set(HAVE_ZLUDA FALSE PARENT_SCOPE)
        return()
    endif()

    # ZLUDA GitHub releases URL
    # Note: ZLUDA releases may vary in naming convention - adjust as needed
    set(ZLUDA_URL "https://github.com/vosen/ZLUDA/releases/download/v${ZLUDA_VERSION}/zluda-${ZLUDA_PLATFORM}.${ZLUDA_ARCHIVE_EXT}")

    message(STATUS "ZLUDA download URL: ${ZLUDA_URL}")

    # Download and extract ZLUDA
    ExternalProject_Add(zluda_external
            PREFIX            "${CMAKE_BINARY_DIR}/zluda_external"
            URL               "${ZLUDA_URL}"
            DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
            SOURCE_DIR        "${ZLUDA_INSTALL_DIR}"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND     ""
            INSTALL_COMMAND   ""
            BUILD_BYPRODUCTS  "${ZLUDA_INSTALL_DIR}/lib/libcuda.so"
            TIMEOUT           300
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            LOG_DOWNLOAD      OFF
            LOG_CONFIGURE     OFF
            LOG_BUILD         OFF
            LOG_INSTALL       OFF
    )

    # Create interface library for ZLUDA
    add_library(zluda_interface INTERFACE)
    target_include_directories(zluda_interface INTERFACE "${ZLUDA_INSTALL_DIR}/include")
    if(WIN32)
        target_link_directories(zluda_interface INTERFACE "${ZLUDA_INSTALL_DIR}/lib")
    else()
        target_link_directories(zluda_interface INTERFACE "${ZLUDA_INSTALL_DIR}/lib")
    endif()
    add_dependencies(zluda_interface zluda_external)

    set(HAVE_ZLUDA TRUE PARENT_SCOPE)
    set(ZLUDA_PATH "${ZLUDA_INSTALL_DIR}" PARENT_SCOPE)
    set(ZLUDA zluda_interface PARENT_SCOPE)

    # Set environment variable for runtime
    set(ENV{ZLUDA_PATH} "${ZLUDA_INSTALL_DIR}")

    message(STATUS "ZLUDA setup complete")
    message(STATUS "   Install directory: ${ZLUDA_INSTALL_DIR}")
    message(STATUS "   Platform: ${ZLUDA_PLATFORM}")

    # Setup MIOpen for AMD targets
    if(SD_ZLUDA_TARGET STREQUAL "AMD" OR SD_ZLUDA_TARGET STREQUAL "amd")
        setup_miopen_download()
    endif()
endfunction()

# =============================================================================
# MIOpen Download (Optional, for AMD GPU DNN operations via ZLUDA)
# =============================================================================
function(setup_miopen_download)
    if(NOT HELPERS_miopen STREQUAL "ON")
        message(STATUS "MIOpen helper is disabled (HELPERS_miopen=${HELPERS_miopen})")
        set(HAVE_MIOPEN FALSE PARENT_SCOPE)
        return()
    endif()

    # Check if MIOpen is already available via ROCm
    set(ROCM_SEARCH_PATHS
        $ENV{ROCM_PATH}
        $ENV{ROCM_HOME}
        /opt/rocm
        /opt/rocm-6.0
        /opt/rocm-5.7
        /opt/rocm-5.6
    )

    foreach(rocm_path ${ROCM_SEARCH_PATHS})
        if(EXISTS "${rocm_path}/lib/libMIOpen.so")
            message(STATUS "Using existing MIOpen from ROCm: ${rocm_path}")
            set(HAVE_MIOPEN TRUE PARENT_SCOPE)
            set(MIOPEN_PATH "${rocm_path}" PARENT_SCOPE)
            set(MIOPEN_LIBRARY "${rocm_path}/lib/libMIOpen.so" PARENT_SCOPE)
            set(MIOPEN_INCLUDE_DIR "${rocm_path}/include" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    # MIOpen requires ROCm to be installed - we can't download it standalone easily
    # because it has complex dependencies on HIP runtime and other ROCm components
    message(STATUS "MIOpen automatic download:")
    message(STATUS "   MIOpen is part of ROCm and cannot be downloaded standalone.")
    message(STATUS "   Please install ROCm toolkit from: https://rocm.docs.amd.com/")
    message(STATUS "   After installation, MIOpen will be automatically detected.")

    # For now, we'll create a stub that warns about missing MIOpen
    set(HAVE_MIOPEN FALSE PARENT_SCOPE)

    # Alternative: Try to build MIOpen from source (complex due to dependencies)
    if(FALSE)  # Disabled for now - ROCm installation is recommended
        set(MIOPEN_VERSION "3.0.0")
        set(MIOPEN_INSTALL_DIR "${CMAKE_BINARY_DIR}/miopen_install")
        set(MIOPEN_URL "https://github.com/ROCm/MIOpen/archive/refs/tags/rocm-${MIOPEN_VERSION}.tar.gz")

        ExternalProject_Add(miopen_external
                PREFIX            "${CMAKE_BINARY_DIR}/miopen_external"
                URL               "${MIOPEN_URL}"
                DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
                SOURCE_DIR        "${CMAKE_BINARY_DIR}/miopen_external/src"
                BINARY_DIR        "${CMAKE_BINARY_DIR}/miopen_external/build"
                CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${MIOPEN_INSTALL_DIR}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DMIOPEN_BACKEND=HIP
                BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE} --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config ${CMAKE_BUILD_TYPE}
                TIMEOUT           1200
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        add_library(miopen_interface INTERFACE)
        target_include_directories(miopen_interface INTERFACE "${MIOPEN_INSTALL_DIR}/include")
        target_link_libraries(miopen_interface INTERFACE "${MIOPEN_INSTALL_DIR}/lib/libMIOpen.so")
        add_dependencies(miopen_interface miopen_external)

        set(HAVE_MIOPEN TRUE PARENT_SCOPE)
        set(MIOPEN miopen_interface PARENT_SCOPE)
    endif()
endfunction()

function(setup_mps)
    set(HAVE_MPS FALSE PARENT_SCOPE)

    if(NOT HELPERS_mps STREQUAL "ON")
        message(STATUS "MPS helper is disabled (HELPERS_mps=${HELPERS_mps})")
        return()
    endif()

    # Check if we're on Apple platform
    if(NOT APPLE)
        message(STATUS "MPS helper requires macOS/iOS (current platform: ${CMAKE_SYSTEM_NAME})")
        return()
    endif()

    # Check for Metal framework availability
    find_library(METAL_FRAMEWORK Metal)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)

    if(NOT METAL_FRAMEWORK OR NOT MPS_FRAMEWORK)
        message(WARNING "Metal or MetalPerformanceShaders framework not found")
        return()
    endif()

    message(STATUS "✅ Metal Performance Shaders setup:")
    message(STATUS "   Metal Framework: ${METAL_FRAMEWORK}")
    message(STATUS "   MPS Framework: ${MPS_FRAMEWORK}")

    set(HAVE_MPS TRUE PARENT_SCOPE)
    set(MPS_LIBRARIES ${METAL_FRAMEWORK} ${MPS_FRAMEWORK} ${FOUNDATION_FRAMEWORK} PARENT_SCOPE)

    add_compile_definitions(HAVE_MPS=1)

    message(STATUS "✅ MPS setup complete")
endfunction()

