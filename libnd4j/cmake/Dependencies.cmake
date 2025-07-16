# cmake/Dependencies.cmake
# Manages all third-party dependencies. Logic is encapsulated in functions.

include(ExternalProject)
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

    # Set up library directories
    if(EXISTS "${OPENBLAS_PATH}/lib")
        link_directories(${OPENBLAS_PATH}/)
    endif()

    add_compile_definitions(HAVE_OPENBLAS=1)

    # Set parent scope variables
    set(HAVE_OPENBLAS 1 PARENT_SCOPE)
    set(OPENBLAS_LIBRARIES openblas PARENT_SCOPE)

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
# FLATBUFFERS (Required)
# =============================================================================
function(setup_flatbuffers)
    # --- Condition Detection ---

    # All arguments passed to this function are treated as schema files.
    set(FLATBUFFERS_SCHEMA_FILES ${ARGN})

    # Determine if flatc code generation is needed. This is enabled automatically
    # if schema files are provided, or can be forced with an option/env var.
    option(GENERATE_FLATC "Enable FlatBuffers schema compilation" OFF)
    if(DEFINED ENV{GENERATE_FLATC})
        set(GENERATE_FLATC ON)
    endif()
    if(FLATBUFFERS_SCHEMA_FILES)
        set(GENERATE_FLATC ON) # Automatically enable if schemas are passed
    endif()

    if(GENERATE_FLATC AND NOT FLATBUFFERS_SCHEMA_FILES)
        message(FATAL_ERROR "GENERATE_FLATC is ON, but no schema files were passed to setup_flatbuffers().")
    endif()

    # Determine if we are cross-compiling for an ARM target, which requires
    # building the flatc compiler for the host and the library for the target.
    set(NEEDS_SEPARATE_HOST_BUILD FALSE)
    if(CMAKE_CROSSCOMPILING)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64|arm|ARM" OR
                CMAKE_ANDROID_ARCH_ABI MATCHES "arm64-v8a|armeabi-v7a")
            set(NEEDS_SEPARATE_HOST_BUILD TRUE)
        endif()
    endif()

    # --- Main Logic ---

    if(NEEDS_SEPARATE_HOST_BUILD AND GENERATE_FLATC)
        #
        # ===== CASE 1: CROSS-COMPILING FOR ARM (e.g., Linux Host -> Android Target) =====
        #
        message(STATUS "🔧 Cross-compiling for ARM: Configuring separate host/target FlatBuffers builds.")

        # Build flatc for the HOST system.
        ExternalProject_Add(flatbuffers_host
                GIT_REPOSITORY    https://github.com/google/flatbuffers.git
                GIT_TAG           v24.3.25
                SOURCE_DIR        "${CMAKE_BINARY_DIR}/flatbuffers-host-src"
                BINARY_DIR        "${CMAKE_BINARY_DIR}/flatbuffers-host-build"
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${CMAKE_BINARY_DIR}/flatbuffers-host-build/flatc"
                CMAKE_ARGS
                # Force a native build by clearing toolchain-specific variables
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_TOOLCHAIN_FILE=""
                -DCMAKE_C_COMPILER=""
                -DCMAKE_CXX_COMPILER=""
                -DCMAKE_SYSTEM_NAME=${CMAKE_HOST_SYSTEM_NAME}
                -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_HOST_SYSTEM_PROCESSOR}
                # Configure build options
                -DFLATBUFFERS_BUILD_FLATC=ON
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SHAREDLIB=OFF
        )

        # Build libflatbuffers.a for the TARGET system (using the NDK toolchain).
        ExternalProject_Add(flatbuffers_target
                GIT_REPOSITORY    https://github.com/google/flatbuffers.git
                GIT_TAG           v24.3.25
                SOURCE_DIR        "${CMAKE_BINARY_DIR}/flatbuffers-target-src"
                BINARY_DIR        "${CMAKE_BINARY_DIR}/flatbuffers-target-build"
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${CMAKE_BINARY_DIR}/flatbuffers-target-build/libflatbuffers.a"
                CMAKE_ARGS
                # Pass through all cross-compilation settings from the parent project
                -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
                -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
                -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK}
                -DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}
                -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                # Configure build options
                -DFLATBUFFERS_BUILD_FLATC=OFF # Do NOT build flatc for the target
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SHAREDLIB=OFF
        )

        # Define paths and create a custom target for code generation.
        set(FLATC_EXECUTABLE          "${CMAKE_BINARY_DIR}/flatbuffers-host-build/flatc")
        set(FLATBUFFERS_LIBRARY       "${CMAKE_BINARY_DIR}/flatbuffers-target-build/libflatbuffers.a" PARENT_SCOPE)
        set(FLATBUFFERS_INCLUDE_DIR   "${CMAKE_BINARY_DIR}/flatbuffers-target-src/include" PARENT_SCOPE)
        set(FLATBUFFERS_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated/flatbuffers" PARENT_SCOPE)

        # Add a step to the host build to generate headers after flatc is built.
        ExternalProject_Add_Step(flatbuffers_host generate_headers
                COMMAND           ${FLATC_EXECUTABLE} -o ${FLATBUFFERS_GENERATED_DIR} --cpp ${FLATBUFFERS_SCHEMA_FILES}
                COMMENT           "Generating C++ headers from schemas using host flatc..."
                DEPENDEES         build
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_custom_target(GenerateFlatbufferHeaders DEPENDS flatbuffers_host-generate_headers)

    else()
        #
        # ===== CASE 2: NATIVE COMPILATION or NO CODEGEN =====
        #
        message(STATUS "🔧 Standard FlatBuffers build configured.")

        ExternalProject_Add(flatbuffers_unified
                GIT_REPOSITORY    https://github.com/google/flatbuffers.git
                GIT_TAG           v24.3.25
                SOURCE_DIR        "${CMAKE_BINARY_DIR}/flatbuffers-src"
                BINARY_DIR        "${CMAKE_BINARY_DIR}/flatbuffers-build"
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${CMAKE_BINARY_DIR}/flatbuffers-build/libflatbuffers.a"
                CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DFLATBUFFERS_BUILD_FLATC=${GENERATE_FLATC}
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SHAREDLIB=OFF
                -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        )

        # Define paths
        set(FLATBUFFERS_LIBRARY       "${CMAKE_BINARY_DIR}/flatbuffers-build/libflatbuffers.a" PARENT_SCOPE)
        set(FLATBUFFERS_INCLUDE_DIR   "${CMAKE_BINARY_DIR}/flatbuffers-src/include" PARENT_SCOPE)

        if(GENERATE_FLATC)
            set(FLATC_EXECUTABLE          "${CMAKE_BINARY_DIR}/flatbuffers-build/flatc")
            set(FLATBUFFERS_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated/flatbuffers" PARENT_SCOPE)

            ExternalProject_Add_Step(flatbuffers_unified generate_headers
                    COMMAND           ${FLATC_EXECUTABLE} -o ${FLATBUFFERS_GENERATED_DIR} --cpp ${FLATBUFFERS_SCHEMA_FILES}
                    COMMENT           "Generating C++ headers from schemas..."
                    DEPENDEES         build
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
            add_custom_target(GenerateFlatbufferHeaders DEPENDS flatbuffers_unified-generate_headers)
        endif()
    endif()

    # --- Create a final INTERFACE library for easy consumption ---
    # This abstracts away the build details. Other targets just link to 'flatbuffers::flatbuffers'.
    add_library(flatbuffers_interface INTERFACE)
    target_include_directories(flatbuffers_interface INTERFACE
            ${FLATBUFFERS_INCLUDE_DIR}
            ${FLATBUFFERS_GENERATED_DIR}
    )
    target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})

    # Ensure the library is built before anything tries to link to it.
    if(NEEDS_SEPARATE_HOST_BUILD AND GENERATE_FLATC)
        add_dependencies(flatbuffers_interface flatbuffers_target)
    else()
        add_dependencies(flatbuffers_interface flatbuffers_unified)
    endif()

    # Create a global alias for easier linking in the parent project.
    add_library(flatbuffers::flatbuffers ALIAS flatbuffers_interface)

    message(STATUS "✅ Flatbuffers setup complete.")
endfunction()


# =============================================================================
# ONEDNN (Optional)
# =============================================================================
function(setup_onednn)
    if(NOT HELPERS_onednn STREQUAL "ON")
        message(STATUS "OneDNN helper is disabled (HELPERS_onednn=${HELPERS_onednn})")
        set(HAVE_ONEDNN FALSE PARENT_SCOPE)
        set(ONEDNN "" PARENT_SCOPE)
        return()
    endif()

    if(TARGET onednn_external)
        message(STATUS "OneDNN helper is enabled (target already exists)")
        set(HAVE_ONEDNN TRUE PARENT_SCOPE)
        set(ONEDNN onednn_interface PARENT_SCOPE)
        return()
    endif()

    message(STATUS "OneDNN helper is enabled")
    set(HAVE_ONEDNN TRUE PARENT_SCOPE)
    set(ONEDNN_INSTALL_DIR "${CMAKE_BINARY_DIR}/onednn_install")

    ExternalProject_Add(onednn_external
            PREFIX            "${CMAKE_BINARY_DIR}/onednn_external"
            GIT_REPOSITORY    "https://github.com/uxlfoundation/oneDNN.git"
            GIT_TAG           "v3.8.1"
            SOURCE_DIR        "${CMAKE_BINARY_DIR}/onednn_external/src"
            BINARY_DIR        "${CMAKE_BINARY_DIR}/onednn_external/build"
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DDNNL_LIBRARY_TYPE=STATIC -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_VERBOSE=OFF -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE} --parallel
            INSTALL_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config ${CMAKE_BUILD_TYPE}
            BUILD_BYPRODUCTS "${ONEDNN_INSTALL_DIR}/include/dnnl.h" "${ONEDNN_INSTALL_DIR}/lib/libdnnl.a" "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib"
            TIMEOUT           600
    )

    add_library(onednn_interface INTERFACE)
    target_include_directories(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/include")
    if(WIN32)
        target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib")
    else()
        target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/lib/libdnnl.a")
    endif()
    add_dependencies(onednn_interface onednn_external)
    set(ONEDNN onednn_interface PARENT_SCOPE)
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

