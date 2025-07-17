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

        ExternalProject_Add(flatbuffers_host
                URL               ${FLATBUFFERS_URL}
                SOURCE_DIR        "${FLATC_HOST_DIR}"
                BINARY_DIR        "${FLATC_HOST_BUILD_DIR}"
                CMAKE_ARGS        ${HOST_CMAKE_ARGS}
                BUILD_COMMAND     ${CMAKE_COMMAND} --build . --target flatc --config Release
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${FLATC_EXECUTABLE}"
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
        )

        # Set up include directories and library
        include_directories("${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src/include")
        set(FLATBUFFERS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-build/libflatbuffers.a")
        set(FLATBUFFERS_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src")

        # Create interface library for target
        add_library(flatbuffers_interface INTERFACE)
        target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})
        add_dependencies(flatbuffers_interface flatbuffers_target)

        # Generate headers and copy Java files inline after ExternalProject builds
        ExternalProject_Add_Step(flatbuffers_host generate_headers_and_copy_java
                COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}"
                bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
                COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-host/include/flatbuffers/flatbuffers.h"
                "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers.h"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Generating FlatBuffers headers, copying Java files, and copying flatbuffers.h using host flatc"
                DEPENDEES build
                BYPRODUCTS
                ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h
                ${CMAKE_CURRENT_SOURCE_DIR}/.java_files_copied
                ${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers.h
        )

    else()
        # Native build or cross-compilation without flatc generation
        message(STATUS "Native FlatBuffers build")

        if(SHOULD_BUILD_FLATC)
            set(FLATBUFFERS_BUILD_FLATC "ON")
        else()
            set(FLATBUFFERS_BUILD_FLATC "OFF")
        endif()

        ExternalProject_Add(flatbuffers_external
                URL               ${FLATBUFFERS_URL}
                SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src"
                BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build"
                CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_BUILD_TYPE=Release
                -DFLATBUFFERS_BUILD_FLATC=${FLATBUFFERS_BUILD_FLATC}
                -DFLATBUFFERS_BUILD_FLATLIB=ON
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SAMPLES=OFF
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc"
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/libflatbuffers.a"
        )

        include_directories("${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")
        set(FLATBUFFERS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/libflatbuffers.a")
        set(FLATBUFFERS_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src")

        if(SHOULD_BUILD_FLATC)
            set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")

            # Generate headers and copy Java files inline after ExternalProject builds
            ExternalProject_Add_Step(flatbuffers_external generate_headers_and_copy_java
                    COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}"
                    bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
                    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include/flatbuffers/flatbuffers.h"
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers.h"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating FlatBuffers headers, copying Java files, and copying flatbuffers.h"
                    DEPENDEES build
                    BYPRODUCTS
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/.java_files_copied
                    ${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers.h
            )
        else()
            # Even without flatc generation, copy the flatbuffers.h header
            ExternalProject_Add_Step(flatbuffers_external copy_flatbuffers_header
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include/flatbuffers/flatbuffers.h"
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers.h"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying flatbuffers.h header"
                    DEPENDEES build
                    BYPRODUCTS
                    ${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers.h
            )
        endif()

        # Create interface library
        add_library(flatbuffers_interface INTERFACE)
        target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})
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

