# cmake/Dependencies.cmake
# Manages all third-party dependencies. Logic is encapsulated in functions.

include(ExternalProject)


function(setup_blas)
    if(SD_CUDA)
        return()
    endif()
    include_directories(${OPENBLAS_PATH}/include/)
    link_directories(${OPENBLAS_PATH}/lib/)
    add_compile_definitions(HAVE_OPENBLAS=1)
    set(HAVE_OPENBLAS 1 PARENT_SCOPE)
    set(OPENBLAS_LIBRARIES openblas PARENT_SCOPE)
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
    # Target Guard: Prevents this from ever running twice.
    if(TARGET flatbuffers_external)
        return()
    endif()

    if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
        set(FLATBUFFERS_BUILD_FLATC "ON" CACHE STRING "Enable flatc build" FORCE)
    else()
        set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Disable flatc build" FORCE)
    endif()

    ExternalProject_Add(flatbuffers_external
            GIT_REPOSITORY    https://github.com/google/flatbuffers/
            GIT_TAG           v25.2.10
            SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src"
            BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build"
            CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_BUILD_TYPE=Release
            -DFLATBUFFERS_BUILD_FLATC=${FLATBUFFERS_BUILD_FLATC}
            INSTALL_COMMAND   ""
            BUILD_BYPRODUCTS  "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc"
            "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/libflatbuffers.a"
    )

    include_directories("${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")

    add_library(flatbuffers_interface INTERFACE)
    set(FLATBUFFERS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/libflatbuffers.a")
    target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})
    add_dependencies(flatbuffers_interface flatbuffers_external)

    if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
        set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")
        set(MAIN_GENERATED_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h")

        add_custom_command(
                OUTPUT ${MAIN_GENERATED_HEADER}
                COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}" bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
                DEPENDS flatbuffers_external
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Running flatc to generate C++ headers"
                VERBATIM
        )
        add_custom_target(generate_flatbuffers_headers DEPENDS ${MAIN_GENERATED_HEADER})
        add_custom_command(
                TARGET generate_flatbuffers_headers POST_BUILD
                COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Copying generated Java files"
                VERBATIM
        )
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

