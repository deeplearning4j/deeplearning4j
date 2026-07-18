# cmake/ZludaConfiguration.cmake
# ZLUDA Transpiler Configuration for AMD/Intel GPU Support
#
# ZLUDA is a drop-in CUDA replacement that translates CUDA API calls to:
#   - HIP (for AMD GPUs via ROCm)
#   - Level Zero (for Intel GPUs via oneAPI)
#
# This module provides:
#   - ZLUDA runtime detection
#   - MIOpen configuration for AMD (cuDNN alternative)
#   - oneDNN configuration for Intel (cuDNN alternative)
#   - CUDA compatibility flags for ZLUDA

################################################################################
# Main ZLUDA Setup Function
################################################################################

function(setup_zluda)
    set(HAVE_ZLUDA FALSE PARENT_SCOPE)
    set(ZLUDA_TARGET_BACKEND "" PARENT_SCOPE)

    if(NOT SD_ZLUDA)
        message(STATUS "ZLUDA: Disabled (SD_ZLUDA=${SD_ZLUDA})")
        return()
    endif()

    print_status_colored("INFO" "=== ZLUDA Transpiler Configuration ===")

    # Detect ZLUDA installation
    set(ZLUDA_SEARCH_PATHS
        $ENV{ZLUDA_PATH}
        $ENV{ZLUDA_ROOT}
        /opt/zluda
        /usr/local/zluda
        ${CMAKE_PREFIX_PATH}/zluda
    )

    # Find ZLUDA library (provides libcuda.so replacement)
    find_library(ZLUDA_LIBRARY
        NAMES cuda nvcuda zluda
        HINTS ${ZLUDA_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
    )

    # Also check for ZLUDA in the environment
    if(NOT ZLUDA_LIBRARY AND DEFINED ENV{ZLUDA_PATH})
        set(ZLUDA_LIBRARY "$ENV{ZLUDA_PATH}/lib/libcuda.so")
        if(NOT EXISTS "${ZLUDA_LIBRARY}")
            set(ZLUDA_LIBRARY "$ENV{ZLUDA_PATH}/lib/libnvcuda.so")
        endif()
        if(NOT EXISTS "${ZLUDA_LIBRARY}")
            set(ZLUDA_LIBRARY "")
        endif()
    endif()

    if(NOT ZLUDA_LIBRARY)
        # Try to download ZLUDA automatically
        message(STATUS "ZLUDA not found locally, attempting automatic download...")
        setup_zluda_download()

        # Check if download succeeded and search again
        if(TARGET zluda_external)
            get_target_property(ZLUDA_INSTALL_DIR zluda_external INSTALL_DIR)
            if(EXISTS "${ZLUDA_INSTALL_DIR}/lib/libcuda.so")
                set(ZLUDA_LIBRARY "${ZLUDA_INSTALL_DIR}/lib/libcuda.so")
                message(STATUS "Using auto-downloaded ZLUDA: ${ZLUDA_LIBRARY}")
            elseif(EXISTS "${ZLUDA_INSTALL_DIR}/libnvcuda.so")
                set(ZLUDA_LIBRARY "${ZLUDA_INSTALL_DIR}/libnvcuda.so")
                message(STATUS "Using auto-downloaded ZLUDA: ${ZLUDA_LIBRARY}")
            endif()
        endif()
    endif()

    if(NOT ZLUDA_LIBRARY)
        print_status_colored("WARNING" "ZLUDA library not found. Set ZLUDA_PATH environment variable.")
        print_status_colored("WARNING" "Or ensure automatic download completes successfully.")
        print_status_colored("WARNING" "ZLUDA support disabled - falling back to standard CUDA if available.")
        return()
    endif()

    message(STATUS "Found ZLUDA library: ${ZLUDA_LIBRARY}")
    get_filename_component(ZLUDA_LIB_DIR "${ZLUDA_LIBRARY}" DIRECTORY)
    set(ZLUDA_LIB_DIR "${ZLUDA_LIB_DIR}" PARENT_SCOPE)

    # Determine target backend
    if(SD_ZLUDA_TARGET STREQUAL "AMD" OR SD_ZLUDA_TARGET STREQUAL "amd")
        set(ZLUDA_TARGET_BACKEND "AMD" PARENT_SCOPE)
        set(ZLUDA_TARGET_BACKEND "AMD")
        message(STATUS "ZLUDA target: AMD (ROCm/HIP)")
        setup_zluda_amd()
    elseif(SD_ZLUDA_TARGET STREQUAL "INTEL" OR SD_ZLUDA_TARGET STREQUAL "intel")
        set(ZLUDA_TARGET_BACKEND "INTEL" PARENT_SCOPE)
        set(ZLUDA_TARGET_BACKEND "INTEL")
        message(STATUS "ZLUDA target: Intel (Level Zero)")
        setup_zluda_intel()
    else()
        # Auto-detect based on available GPU
        detect_zluda_target()
        set(ZLUDA_TARGET_BACKEND "${ZLUDA_TARGET_BACKEND}" PARENT_SCOPE)
    endif()

    if(NOT ZLUDA_TARGET_BACKEND)
        print_status_colored("WARNING" "Could not determine ZLUDA target. Please set SD_ZLUDA_TARGET=AMD or SD_ZLUDA_TARGET=INTEL")
        return()
    endif()

    set(HAVE_ZLUDA TRUE PARENT_SCOPE)
    add_compile_definitions(HAVE_ZLUDA=1)
    add_compile_definitions(ZLUDA_TARGET_${ZLUDA_TARGET_BACKEND}=1)

    print_status_colored("SUCCESS" "ZLUDA configuration complete (target: ${ZLUDA_TARGET_BACKEND})")
endfunction()

################################################################################
# AMD-specific ZLUDA Configuration (ROCm/HIP backend)
################################################################################

function(setup_zluda_amd)
    message(STATUS "Configuring ZLUDA for AMD GPUs...")

    # Find ROCm installation
    set(ROCM_SEARCH_PATHS
        $ENV{ROCM_PATH}
        $ENV{ROCM_HOME}
        $ENV{HIP_PATH}
        /opt/rocm
        /opt/rocm-6.0
        /opt/rocm-5.7
        /opt/rocm-5.6
        /opt/rocm-5.5
        /opt/rocm-5.4
    )

    find_path(ROCM_PATH
        NAMES include/hip/hip_runtime.h
        HINTS ${ROCM_SEARCH_PATHS}
        NO_DEFAULT_PATH
    )

    if(ROCM_PATH)
        message(STATUS "Found ROCm: ${ROCM_PATH}")
        set(ROCM_PATH "${ROCM_PATH}" PARENT_SCOPE)
        set(ROCM_INCLUDE_DIR "${ROCM_PATH}/include" PARENT_SCOPE)
        set(ROCM_LIB_DIR "${ROCM_PATH}/lib" PARENT_SCOPE)

        # Set up MIOpen for DNN operations (cuDNN replacement)
        setup_miopen()
    else()
        print_status_colored("WARNING" "ROCm not found. AMD ZLUDA DNN operations may fall back to CPU.")
        print_status_colored("WARNING" "Install ROCm and set ROCM_PATH for full AMD GPU support.")
    endif()
endfunction()

################################################################################
# Intel-specific ZLUDA Configuration (Level Zero backend)
################################################################################

function(setup_zluda_intel)
    message(STATUS "Configuring ZLUDA for Intel GPUs...")

    # Find Level Zero / oneAPI installation
    set(ONEAPI_SEARCH_PATHS
        $ENV{ONEAPI_ROOT}
        $ENV{ONEAPI_PATH}
        $ENV{INTEL_ONEAPI_ROOT}
        /opt/intel/oneapi
        /usr/local/intel/oneapi
    )

    find_path(ONEAPI_PATH
        NAMES compiler/latest/include/sycl/sycl.hpp
        HINTS ${ONEAPI_SEARCH_PATHS}
        NO_DEFAULT_PATH
    )

    # Also check for Level Zero directly. The replay handle needs both the
    # headers and loader; a header-only detection must not advertise executable
    # Level Zero support in the portable replay matrix.
    set(HAVE_LEVELZERO FALSE CACHE BOOL "Level Zero replay handle available" FORCE)
    find_path(LEVEL_ZERO_INCLUDE
        NAMES level_zero/ze_api.h
        HINTS ${ONEAPI_SEARCH_PATHS} /usr /usr/local
        PATH_SUFFIXES include
    )
    find_library(LEVEL_ZERO_LIBRARY
        NAMES ze_loader
        HINTS ${ONEAPI_SEARCH_PATHS} /usr /usr/local
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu compiler/latest/lib
    )

    if(ONEAPI_PATH OR LEVEL_ZERO_INCLUDE)
        if(ONEAPI_PATH)
            message(STATUS "Found oneAPI: ${ONEAPI_PATH}")
            set(ONEAPI_PATH "${ONEAPI_PATH}" PARENT_SCOPE)
        endif()
        if(LEVEL_ZERO_INCLUDE AND LEVEL_ZERO_LIBRARY)
            message(STATUS "Found Level Zero headers: ${LEVEL_ZERO_INCLUDE}")
            message(STATUS "Found Level Zero loader: ${LEVEL_ZERO_LIBRARY}")
            set(HAVE_LEVELZERO TRUE CACHE BOOL "Level Zero replay handle available" FORCE)
            add_compile_definitions(HAVE_LEVELZERO=1)
        elseif(LEVEL_ZERO_INCLUDE)
            print_status_colored("WARNING" "Level Zero headers found but ze_loader is missing; native replay handle disabled.")
        endif()

        # For Intel ZLUDA, use oneDNN which has native Intel GPU support
        setup_onednn_for_zluda()
    else()
        print_status_colored("WARNING" "oneAPI/Level Zero not found. Intel ZLUDA DNN operations may fall back to CPU.")
        print_status_colored("WARNING" "Install oneAPI toolkit for full Intel GPU support.")
    endif()
endfunction()

################################################################################
# MIOpen Setup for AMD GPUs (cuDNN replacement)
################################################################################

function(setup_miopen)
    set(HAVE_MIOPEN FALSE PARENT_SCOPE)

    if(NOT ROCM_PATH)
        message(STATUS "MIOpen: Skipped (ROCm not found)")
        return()
    endif()

    find_library(MIOPEN_LIBRARY
        NAMES MIOpen miopen
        HINTS ${ROCM_PATH}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
    )

    find_path(MIOPEN_INCLUDE_DIR
        NAMES miopen/miopen.h
        HINTS ${ROCM_PATH}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
    )

    if(MIOPEN_LIBRARY AND MIOPEN_INCLUDE_DIR)
        message(STATUS "Found MIOpen:")
        message(STATUS "   Library: ${MIOPEN_LIBRARY}")
        message(STATUS "   Include: ${MIOPEN_INCLUDE_DIR}")

        set(HAVE_MIOPEN TRUE PARENT_SCOPE)
        set(MIOPEN_LIBRARY "${MIOPEN_LIBRARY}" PARENT_SCOPE)
        set(MIOPEN_INCLUDE_DIR "${MIOPEN_INCLUDE_DIR}" PARENT_SCOPE)
        add_compile_definitions(HAVE_MIOPEN=1)

        # Get MIOpen version
        if(EXISTS "${MIOPEN_INCLUDE_DIR}/miopen/version.h")
            file(READ "${MIOPEN_INCLUDE_DIR}/miopen/version.h" MIOPEN_VERSION_CONTENT)
            string(REGEX MATCH "MIOPEN_VERSION_MAJOR[ \t]+([0-9]+)" _ "${MIOPEN_VERSION_CONTENT}")
            set(MIOPEN_VERSION_MAJOR "${CMAKE_MATCH_1}")
            string(REGEX MATCH "MIOPEN_VERSION_MINOR[ \t]+([0-9]+)" _ "${MIOPEN_VERSION_CONTENT}")
            set(MIOPEN_VERSION_MINOR "${CMAKE_MATCH_1}")
            message(STATUS "   Version: ${MIOPEN_VERSION_MAJOR}.${MIOPEN_VERSION_MINOR}")
        endif()
    else()
        print_status_colored("WARNING" "MIOpen not found in ROCm installation.")
        print_status_colored("WARNING" "AMD GPU DNN operations will fall back to CPU implementations.")
        message(STATUS "   Searched in: ${ROCM_PATH}")
    endif()
endfunction()

################################################################################
# oneDNN Setup for Intel ZLUDA (cuDNN replacement)
# This function ensures ZLUDA Intel uses the existing oneDNN setup from
# Dependencies.cmake (setup_onednn) rather than configuring separately.
################################################################################

function(setup_onednn_for_zluda)
    message(STATUS "Configuring oneDNN for Intel ZLUDA...")

    # Check if oneDNN is already configured via the main build system
    if(HAVE_ONEDNN AND TARGET onednn_interface)
        message(STATUS "✅ Reusing existing oneDNN configuration for ZLUDA Intel")
        message(STATUS "   oneDNN target: onednn_interface")
        add_compile_definitions(ZLUDA_USE_ONEDNN=1)
        set(ZLUDA_ONEDNN_TARGET onednn_interface PARENT_SCOPE)
        return()
    endif()

    # Check if HAVE_ONEDNN is set but target doesn't exist yet
    # (might be configured but not yet built)
    if(HAVE_ONEDNN)
        message(STATUS "✅ oneDNN is enabled, ZLUDA Intel will use it when available")
        add_compile_definitions(ZLUDA_USE_ONEDNN=1)
        return()
    endif()

    # oneDNN not enabled - enable it via HELPERS_onednn
    # The main build system (setup_onednn in Dependencies.cmake) will handle
    # downloading and configuring oneDNN with auto-download support
    if(NOT HELPERS_onednn)
        message(STATUS "Enabling oneDNN helper for Intel ZLUDA support")
        set(HELPERS_onednn ON CACHE BOOL "Enable oneDNN for ZLUDA Intel" FORCE)
        message(STATUS "   HELPERS_onednn set to ON - oneDNN will be auto-downloaded")
        message(STATUS "   The main build system (setup_onednn) will configure oneDNN")
    else()
        message(STATUS "   HELPERS_onednn already enabled")
    endif()

    # Mark that ZLUDA will use oneDNN once it's configured
    add_compile_definitions(ZLUDA_USE_ONEDNN=1)

    # Note: The actual oneDNN setup happens in MainBuildFlow.cmake via setup_onednn()
    # which is called AFTER ZLUDA configuration. This just ensures the flag is set.
    message(STATUS "   Intel ZLUDA will link against project's oneDNN (onednn_interface)")
endfunction()

################################################################################
# Auto-detect ZLUDA Target GPU
################################################################################

function(detect_zluda_target)
    message(STATUS "Auto-detecting ZLUDA target GPU...")

    # Try to detect AMD GPU via rocminfo
    find_program(ROCMINFO_EXECUTABLE rocminfo
        HINTS $ENV{ROCM_PATH} /opt/rocm
        PATH_SUFFIXES bin
    )

    if(ROCMINFO_EXECUTABLE)
        execute_process(
            COMMAND ${ROCMINFO_EXECUTABLE}
            RESULT_VARIABLE ROCM_RESULT
            OUTPUT_VARIABLE ROCM_OUTPUT
            ERROR_QUIET
            TIMEOUT 10
        )

        if(ROCM_RESULT EQUAL 0 AND ROCM_OUTPUT MATCHES "gfx")
            message(STATUS "Detected AMD GPU via rocminfo")
            set(ZLUDA_TARGET_BACKEND "AMD" PARENT_SCOPE)
            setup_zluda_amd()
            return()
        endif()
    endif()

    # Try to detect Intel GPU via sycl-ls or clinfo
    find_program(SYCL_LS_EXECUTABLE sycl-ls
        HINTS $ENV{ONEAPI_ROOT} /opt/intel/oneapi
        PATH_SUFFIXES bin compiler/latest/bin
    )

    if(SYCL_LS_EXECUTABLE)
        execute_process(
            COMMAND ${SYCL_LS_EXECUTABLE}
            RESULT_VARIABLE SYCL_RESULT
            OUTPUT_VARIABLE SYCL_OUTPUT
            ERROR_QUIET
            TIMEOUT 10
        )

        if(SYCL_RESULT EQUAL 0 AND SYCL_OUTPUT MATCHES "[Ii]ntel")
            message(STATUS "Detected Intel GPU via sycl-ls")
            set(ZLUDA_TARGET_BACKEND "INTEL" PARENT_SCOPE)
            setup_zluda_intel()
            return()
        endif()
    endif()

    # Try clinfo as fallback
    find_program(CLINFO_EXECUTABLE clinfo)
    if(CLINFO_EXECUTABLE)
        execute_process(
            COMMAND ${CLINFO_EXECUTABLE}
            RESULT_VARIABLE CLINFO_RESULT
            OUTPUT_VARIABLE CLINFO_OUTPUT
            ERROR_QUIET
            TIMEOUT 10
        )

        if(CLINFO_RESULT EQUAL 0)
            if(CLINFO_OUTPUT MATCHES "[Aa][Mm][Dd]" OR CLINFO_OUTPUT MATCHES "gfx")
                message(STATUS "Detected AMD GPU via clinfo")
                set(ZLUDA_TARGET_BACKEND "AMD" PARENT_SCOPE)
                setup_zluda_amd()
                return()
            elseif(CLINFO_OUTPUT MATCHES "[Ii]ntel")
                message(STATUS "Detected Intel GPU via clinfo")
                set(ZLUDA_TARGET_BACKEND "INTEL" PARENT_SCOPE)
                setup_zluda_intel()
                return()
            endif()
        endif()
    endif()

    print_status_colored("WARNING" "Could not auto-detect GPU vendor.")
    print_status_colored("WARNING" "Please set SD_ZLUDA_TARGET=AMD or SD_ZLUDA_TARGET=INTEL explicitly.")
endfunction()

################################################################################
# Configure CUDA Flags for ZLUDA Compatibility
################################################################################

function(configure_zluda_cuda_flags)
    if(NOT HAVE_ZLUDA)
        return()
    endif()

    message(STATUS "Configuring CUDA flags for ZLUDA compatibility...")

    # =========================================================================
    # NOTE: Architecture flags (-arch=sm_50) and --relocatable-device-code=false
    # are now handled centrally in CudaConfiguration.cmake when SD_ZLUDA is set.
    # This function now only handles additional ZLUDA-specific flags that aren't
    # part of the core CUDA build configuration.
    # =========================================================================

    set(ZLUDA_CUDA_FLAGS "")

    # Report that architecture is handled by CudaConfiguration.cmake
    if(ZLUDA_TARGET_BACKEND STREQUAL "AMD")
        message(STATUS "   Target: AMD (ROCm/HIP) - sm_50 baseline set in CudaConfiguration")
    elseif(ZLUDA_TARGET_BACKEND STREQUAL "INTEL")
        message(STATUS "   Target: Intel (Level Zero) - sm_50 baseline set in CudaConfiguration")
    endif()

    # Additional ZLUDA-specific flags can be added here if needed in the future
    # For example, disabling specific PTX optimizations that don't translate well:
    # list(APPEND ZLUDA_CUDA_FLAGS "-Xptxas" "-dlcm=cg")

    # Only append if we have additional flags
    if(ZLUDA_CUDA_FLAGS)
        string(REPLACE ";" " " ZLUDA_CUDA_FLAGS_STR "${ZLUDA_CUDA_FLAGS}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ZLUDA_CUDA_FLAGS_STR}" PARENT_SCOPE)
        message(STATUS "   Additional ZLUDA CUDA flags: ${ZLUDA_CUDA_FLAGS_STR}")
    else()
        message(STATUS "   No additional ZLUDA flags needed (core flags in CudaConfiguration)")
    endif()
endfunction()

################################################################################
# Configure ZLUDA Linking for a Target
################################################################################

function(configure_zluda_linking target_name)
    if(NOT HAVE_ZLUDA)
        return()
    endif()

    message(STATUS "Configuring ZLUDA linking for ${target_name}...")

    # Add ZLUDA library path (takes precedence over system CUDA)
    if(ZLUDA_LIB_DIR)
        target_link_directories(${target_name} BEFORE PUBLIC ${ZLUDA_LIB_DIR})
        message(STATUS "   Added ZLUDA lib path: ${ZLUDA_LIB_DIR}")
    endif()

    if(ZLUDA_TARGET_BACKEND STREQUAL "AMD")
        # Link MIOpen instead of cuDNN for AMD
        if(HAVE_MIOPEN)
            target_link_libraries(${target_name} PUBLIC ${MIOPEN_LIBRARY})
            target_include_directories(${target_name} PUBLIC ${MIOPEN_INCLUDE_DIR})
            message(STATUS "   Linked MIOpen: ${MIOPEN_LIBRARY}")
        endif()

        # Add ROCm library path
        if(ROCM_LIB_DIR)
            target_link_directories(${target_name} PUBLIC ${ROCM_LIB_DIR})
        endif()

    elseif(ZLUDA_TARGET_BACKEND STREQUAL "INTEL")
        if(HAVE_LEVELZERO AND LEVEL_ZERO_LIBRARY)
            target_include_directories(${target_name} PUBLIC ${LEVEL_ZERO_INCLUDE})
            target_link_libraries(${target_name} PUBLIC ${LEVEL_ZERO_LIBRARY})
            message(STATUS "   Linked Level Zero loader: ${LEVEL_ZERO_LIBRARY}")
        endif()

        # Link against the project's existing oneDNN (onednn_interface target)
        # This target is created by setup_onednn() in Dependencies.cmake
        if(TARGET onednn_interface)
            target_link_libraries(${target_name} PUBLIC onednn_interface)
            message(STATUS "   Linked onednn_interface for Intel ZLUDA")
        elseif(HAVE_ONEDNN AND DEFINED ONEDNN)
            # Fallback to ONEDNN variable if set
            target_link_libraries(${target_name} PUBLIC ${ONEDNN})
            message(STATUS "   Linked ${ONEDNN} for Intel ZLUDA")
        elseif(HAVE_ONEDNN)
            message(STATUS "   oneDNN enabled but target not yet available - will be linked by main build")
        else()
            message(WARNING "   oneDNN not available for Intel ZLUDA - DNN ops will use fallback")
        endif()
    endif()

    # Add compile definition for ZLUDA backend
    target_compile_definitions(${target_name} PUBLIC ZLUDA_BACKEND=1)
    target_compile_definitions(${target_name} PUBLIC ZLUDA_TARGET_${ZLUDA_TARGET_BACKEND}=1)
endfunction()

################################################################################
# Print ZLUDA Configuration Summary
################################################################################

function(print_zluda_summary)
    if(NOT HAVE_ZLUDA)
        return()
    endif()

    message(STATUS "")
    message(STATUS "=== ZLUDA Configuration Summary ===")
    message(STATUS "Target Backend: ${ZLUDA_TARGET_BACKEND}")
    message(STATUS "ZLUDA Library: ${ZLUDA_LIBRARY}")

    if(ZLUDA_TARGET_BACKEND STREQUAL "AMD")
        message(STATUS "ROCm Path: ${ROCM_PATH}")
        if(HAVE_MIOPEN)
            message(STATUS "MIOpen: Enabled (${MIOPEN_LIBRARY})")
        else()
            message(STATUS "MIOpen: Not available (cuDNN ops will use fallback)")
        endif()
    elseif(ZLUDA_TARGET_BACKEND STREQUAL "INTEL")
        if(ONEAPI_PATH)
            message(STATUS "oneAPI Path: ${ONEAPI_PATH}")
        endif()
        if(TARGET onednn_interface)
            message(STATUS "oneDNN: Enabled (using project's onednn_interface)")
        elseif(HAVE_ONEDNN)
            message(STATUS "oneDNN: Enabled (via ONEDNN=${ONEDNN})")
        else()
            message(STATUS "oneDNN: Not available (cuDNN ops will use fallback)")
        endif()
    endif()

    message(STATUS "=====================================")
    message(STATUS "")
endfunction()
