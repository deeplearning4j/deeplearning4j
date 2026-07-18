# cmake/HelperConfiguration.cmake
# ============================================================================
# Multi-Backend Helper Configuration and Dynamic Kernel Selection
#
# This module provides:
# - Configuration of multiple helper libraries simultaneously
# - Dynamic kernel selection at runtime
# - Helper priority ordering
# - Platform-specific optimization selection
# ============================================================================

include_guard(GLOBAL)

# ============================================================================
# HELPER AVAILABILITY DETECTION
# ============================================================================

# Function to detect and configure available helpers based on platform
function(sd_detect_available_helpers)
    message(STATUS "")
    message(STATUS "=== Detecting Available Helpers ===")

    # OneDNN - Available on x86/x64 platforms
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386")
        message(STATUS "   OneDNN: Platform compatible (x86)")
        set(HELPER_ONEDNN_COMPATIBLE TRUE PARENT_SCOPE)
    else()
        message(STATUS "   OneDNN: Platform incompatible (requires x86)")
        set(HELPER_ONEDNN_COMPATIBLE FALSE PARENT_SCOPE)
    endif()

    # ARM Compute - Available on ARM64 platforms
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64|armv8")
        message(STATUS "   ARM Compute: Platform compatible (ARM64)")
        set(HELPER_ARMCOMPUTE_COMPATIBLE TRUE PARENT_SCOPE)
    else()
        message(STATUS "   ARM Compute: Platform incompatible (requires ARM64)")
        set(HELPER_ARMCOMPUTE_COMPATIBLE FALSE PARENT_SCOPE)
    endif()

    # MPS/Accelerate - Available on Apple platforms
    if(APPLE)
        message(STATUS "   MPS: Platform compatible (Apple)")
        message(STATUS "   Accelerate: Platform compatible (Apple)")
        set(HELPER_MPS_COMPATIBLE TRUE PARENT_SCOPE)
        set(HELPER_ACCELERATE_COMPATIBLE TRUE PARENT_SCOPE)
    else()
        message(STATUS "   MPS: Platform incompatible (requires macOS/iOS)")
        message(STATUS "   Accelerate: Platform incompatible (requires macOS/iOS)")
        set(HELPER_MPS_COMPATIBLE FALSE PARENT_SCOPE)
        set(HELPER_ACCELERATE_COMPATIBLE FALSE PARENT_SCOPE)
    endif()

    # cuDNN - Available with CUDA builds
    if(SD_CUDA)
        message(STATUS "   cuDNN: Platform compatible (CUDA build)")
        set(HELPER_CUDNN_COMPATIBLE TRUE PARENT_SCOPE)
    else()
        message(STATUS "   cuDNN: Platform incompatible (requires CUDA)")
        set(HELPER_CUDNN_COMPATIBLE FALSE PARENT_SCOPE)
    endif()

    # MIOpen - Available with ZLUDA AMD builds
    if(SD_ZLUDA AND SD_ZLUDA_TARGET STREQUAL "AMD")
        message(STATUS "   MIOpen: Platform compatible (ZLUDA AMD)")
        set(HELPER_MIOPEN_COMPATIBLE TRUE PARENT_SCOPE)
    else()
        message(STATUS "   MIOpen: Platform incompatible (requires ZLUDA AMD)")
        set(HELPER_MIOPEN_COMPATIBLE FALSE PARENT_SCOPE)
    endif()

    message(STATUS "===================================")
    message(STATUS "")
endfunction()

# ============================================================================
# MKL VML (Vector Math Library) SETUP
# ============================================================================

function(setup_mkl_vml)
    if(SD_VULKAN)
        # Vulkan kernels use Vulkan/MLIR accumulation and math implementations;
        # host MKL VML is neither a device abstraction nor a fallback.
        set(HAVE_MKL_VML FALSE PARENT_SCOPE)
        set(HAVE_MKL FALSE PARENT_SCOPE)
        set(MKL_VML_LIBRARIES "" PARENT_SCOPE)
        set(MKL_RUNTIME_LIBRARIES "" PARENT_SCOPE)
        return()
    endif()

    if(NOT HELPERS_onednn)
        message(STATUS "MKL VML: Skipped (requires onednn helper)")
        set(HAVE_MKL_VML FALSE PARENT_SCOPE)
        set(HAVE_MKL FALSE PARENT_SCOPE)
        return()
    endif()

    # Check if BLAS_IMPL is explicitly set to openblas
    if(DEFINED BLAS_IMPL AND BLAS_IMPL STREQUAL "openblas")
        message(STATUS "MKL VML: Skipped (BLAS_IMPL=openblas - using OpenBLAS instead)")
        set(HAVE_MKL_VML FALSE PARENT_SCOPE)
        set(HAVE_MKL FALSE PARENT_SCOPE)
        return()
    endif()

    message(STATUS "")
    message(STATUS "=== Setting up MKL (Following OpenBLAS pattern) ===")

    # =========================================================================
    # STEP 0: Auto-detect MKL from Maven repository if MKL_ROOT not provided
    # =========================================================================
    if(NOT DEFINED MKL_ROOT OR NOT EXISTS "${MKL_ROOT}")
        # Try to find MKL in Maven repository (JavaCPP extracted location)
        set(M2_MKL_BASE "$ENV{HOME}/.m2/repository/org/bytedeco/mkl")
        if(EXISTS "${M2_MKL_BASE}")
            # Find the latest MKL version
            file(GLOB MKL_VERSION_DIRS "${M2_MKL_BASE}/*-1.5.*")
            if(MKL_VERSION_DIRS)
                list(SORT MKL_VERSION_DIRS)
                list(GET MKL_VERSION_DIRS -1 MKL_LATEST_DIR)

                # Determine platform suffix
                if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
                    set(MKL_PLATFORM "linux-x86_64")
                elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
                    set(MKL_PLATFORM "macosx-x86_64")
                elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
                    set(MKL_PLATFORM "macosx-arm64")
                elseif(WIN32)
                    set(MKL_PLATFORM "windows-x86_64")
                endif()

                if(DEFINED MKL_PLATFORM)
                    set(MKL_EXTRACTED_PATH "${MKL_LATEST_DIR}/mkl-redist-extracted/org/bytedeco/mkl/${MKL_PLATFORM}")
                    if(EXISTS "${MKL_EXTRACTED_PATH}/include/mkl_service.h")
                        set(MKL_ROOT "${MKL_EXTRACTED_PATH}")
                        message(STATUS "   Auto-detected MKL from Maven: ${MKL_ROOT}")
                    endif()
                endif()
            endif()
        endif()
    endif()

    # =========================================================================
    # STEP 1: Check if MKL_ROOT was passed from build script (like OPENBLAS_PATH)
    # =========================================================================
    if(DEFINED MKL_ROOT AND EXISTS "${MKL_ROOT}")
        message(STATUS "   MKL_ROOT provided: ${MKL_ROOT}")

        # Verify include directory exists (like OpenBLAS pattern)
        if(NOT EXISTS "${MKL_ROOT}/include")
            message(STATUS "   ❌ MKL include directory not found: ${MKL_ROOT}/include")
            set(HAVE_MKL_VML FALSE PARENT_SCOPE)
            set(HAVE_MKL FALSE PARENT_SCOPE)
            return()
        endif()

        if(NOT EXISTS "${MKL_ROOT}/include/mkl.h")
            message(STATUS "   ❌ MKL mkl.h not found: ${MKL_ROOT}/include/mkl.h")
            set(HAVE_MKL_VML FALSE PARENT_SCOPE)
            set(HAVE_MKL FALSE PARENT_SCOPE)
            return()
        endif()

        message(STATUS "   ✓ MKL include directory: ${MKL_ROOT}/include")

        # =========================================================================
        # STEP 2: Find MKL libraries (JavaCPP uses versioned .so.2 files)
        # =========================================================================
        # JavaCPP MKL structure puts libs directly in the platform dir, not lib/
        # Libraries have .2 suffix: libmkl_rt.so.2, libmkl_core.so.2, etc.

        # Try to find libmkl_rt (single dynamic dispatch library - simplest option)
        set(MKL_RT_LIB "")
        foreach(lib_name "libmkl_rt.so.2" "libmkl_rt.so" "libmkl_rt.dylib" "mkl_rt.dll")
            if(EXISTS "${MKL_ROOT}/${lib_name}")
                set(MKL_RT_LIB "${MKL_ROOT}/${lib_name}")
                break()
            endif()
            if(EXISTS "${MKL_ROOT}/lib/${lib_name}")
                set(MKL_RT_LIB "${MKL_ROOT}/lib/${lib_name}")
                break()
            endif()
        endforeach()

        # If no single RT lib, find individual libraries
        set(MKL_CORE_LIB "")
        set(MKL_LP64_LIB "")
        set(MKL_THREAD_LIB "")

        if(NOT MKL_RT_LIB)
            # Find mkl_core
            foreach(lib_name "libmkl_core.so.2" "libmkl_core.so" "libmkl_core.dylib")
                if(EXISTS "${MKL_ROOT}/${lib_name}")
                    set(MKL_CORE_LIB "${MKL_ROOT}/${lib_name}")
                    break()
                endif()
            endforeach()

            # Find mkl_intel_lp64
            foreach(lib_name "libmkl_intel_lp64.so.2" "libmkl_intel_lp64.so" "libmkl_intel_lp64.dylib")
                if(EXISTS "${MKL_ROOT}/${lib_name}")
                    set(MKL_LP64_LIB "${MKL_ROOT}/${lib_name}")
                    break()
                endif()
            endforeach()

            # Find threading library (prefer gnu_thread for GCC compatibility)
            foreach(lib_name "libmkl_gnu_thread.so.2" "libmkl_intel_thread.so.2" "libmkl_sequential.so.2"
                            "libmkl_gnu_thread.so" "libmkl_intel_thread.so" "libmkl_sequential.so")
                if(EXISTS "${MKL_ROOT}/${lib_name}")
                    set(MKL_THREAD_LIB "${MKL_ROOT}/${lib_name}")
                    break()
                endif()
            endforeach()
        endif()

        # =========================================================================
        # STEP 3: Verify we have libraries and configure linking
        # =========================================================================
        if(MKL_RT_LIB)
            message(STATUS "   ✓ Found MKL RT library: ${MKL_RT_LIB}")
            set(MKL_LIBRARIES "${MKL_RT_LIB}")
        elseif(MKL_CORE_LIB)
            message(STATUS "   ✓ Found MKL core: ${MKL_CORE_LIB}")
            set(MKL_LIBRARIES "${MKL_CORE_LIB}")
            if(MKL_LP64_LIB)
                message(STATUS "   ✓ Found MKL LP64: ${MKL_LP64_LIB}")
                list(APPEND MKL_LIBRARIES "${MKL_LP64_LIB}")
            endif()
            if(MKL_THREAD_LIB)
                message(STATUS "   ✓ Found MKL thread: ${MKL_THREAD_LIB}")
                list(APPEND MKL_LIBRARIES "${MKL_THREAD_LIB}")
            endif()
        else()
            message(STATUS "   ❌ No MKL libraries found in ${MKL_ROOT}")
            message(STATUS "      Looked for: libmkl_rt.so.2, libmkl_core.so.2, etc.")
            set(HAVE_MKL_VML FALSE PARENT_SCOPE)
            set(HAVE_MKL FALSE PARENT_SCOPE)
            return()
        endif()

        # libmkl_rt is the link-time dispatcher, but it opens the core,
        # interface, threading, and ISA-specific oneMKL DSOs by loader name at
        # runtime. Keep the link set minimal while exporting the complete
        # downloaded redist set for backend classifier staging.
        set(MKL_RUNTIME_LIBRARIES "")
        foreach(_mkl_runtime_directory IN ITEMS "${MKL_ROOT}" "${MKL_ROOT}/lib")
            if(NOT IS_DIRECTORY "${_mkl_runtime_directory}")
                continue()
            endif()
            if(WIN32)
                file(GLOB _mkl_runtime_candidates
                    "${_mkl_runtime_directory}/*mkl*.dll"
                    "${_mkl_runtime_directory}/*iomp*.dll")
            elseif(APPLE)
                file(GLOB _mkl_runtime_candidates
                    "${_mkl_runtime_directory}/libmkl*.dylib"
                    "${_mkl_runtime_directory}/libiomp*.dylib")
            else()
                file(GLOB _mkl_runtime_candidates
                    "${_mkl_runtime_directory}/libmkl*.so*"
                    "${_mkl_runtime_directory}/libiomp*.so*")
            endif()
            list(APPEND MKL_RUNTIME_LIBRARIES ${_mkl_runtime_candidates})
        endforeach()
        list(APPEND MKL_RUNTIME_LIBRARIES ${MKL_LIBRARIES})
        list(REMOVE_DUPLICATES MKL_RUNTIME_LIBRARIES)

        set(_mkl_has_dispatcher FALSE)
        set(_mkl_has_core FALSE)
        foreach(_mkl_runtime_library IN LISTS MKL_RUNTIME_LIBRARIES)
            get_filename_component(_mkl_runtime_name
                "${_mkl_runtime_library}" NAME)
            if(_mkl_runtime_name MATCHES "mkl_rt")
                set(_mkl_has_dispatcher TRUE)
            elseif(_mkl_runtime_name MATCHES "mkl_core")
                set(_mkl_has_core TRUE)
            endif()
        endforeach()
        if(_mkl_has_dispatcher AND NOT _mkl_has_core)
            message(FATAL_ERROR
                "The oneMKL redist is incomplete: libmkl_rt requires libmkl_core at runtime")
        endif()

        # =========================================================================
        # STEP 4: Set up includes and linking (like OpenBLAS)
        # =========================================================================
        message(STATUS "   ✓ MKL Path: ${MKL_ROOT}")
        message(STATUS "   ✓ MKL Include: ${MKL_ROOT}/include")
        message(STATUS "   ✓ MKL Libraries: ${MKL_LIBRARIES}")

        # Add include directory
        include_directories(SYSTEM "${MKL_ROOT}/include")

        # Add library directory and link libraries
        link_directories("${MKL_ROOT}")
        foreach(mkl_lib ${MKL_LIBRARIES})
            link_libraries("${mkl_lib}")
        endforeach()

        # Set compile definitions
        add_compile_definitions(HAVE_MKL_VML=1)
        add_compile_definitions(HAVE_MKL=1)

        # Export variables
        set(HAVE_MKL_VML TRUE PARENT_SCOPE)
        set(HAVE_MKL TRUE PARENT_SCOPE)
        set(MKL_VML_INCLUDE_DIR "${MKL_ROOT}/include" PARENT_SCOPE)
        set(MKL_VML_LIBRARIES ${MKL_LIBRARIES} PARENT_SCOPE)
        set(MKL_RUNTIME_LIBRARIES ${MKL_RUNTIME_LIBRARIES} PARENT_SCOPE)
        set(MKL_PATH "${MKL_ROOT}" PARENT_SCOPE)

        message(STATUS "   ✓ MKL VML enabled for vectorized math (vsErf, vdErf, etc.)")
        message(STATUS "   ✓ MKL BLAS extensions enabled (bfloat16, batch GEMM)")
    else()
        # MKL_ROOT not provided or doesn't exist
        if(DEFINED BLAS_IMPL AND BLAS_IMPL STREQUAL "mkl")
            message(FATAL_ERROR "MKL explicitly requested (BLAS_IMPL=mkl) but MKL_ROOT not set or invalid. "
                "Ensure MKL is available via JavaCPP or use --blas openblas.")
        endif()
        message(STATUS "   MKL_ROOT not provided - MKL disabled (using OpenBLAS)")
        set(HAVE_MKL_VML FALSE PARENT_SCOPE)
        set(HAVE_MKL FALSE PARENT_SCOPE)
    endif()

    message(STATUS "=================================================")
    message(STATUS "")
endfunction()

# ============================================================================
# APPLE ACCELERATE FRAMEWORK SETUP
# ============================================================================

function(setup_accelerate)
    if(NOT HELPERS_accelerate)
        message(STATUS "Accelerate helper is disabled (HELPERS_accelerate=${HELPERS_accelerate})")
        set(HAVE_ACCELERATE FALSE PARENT_SCOPE)
        return()
    endif()

    if(NOT APPLE)
        message(STATUS "Accelerate framework requires macOS/iOS (current platform: ${CMAKE_SYSTEM_NAME})")
        set(HAVE_ACCELERATE FALSE PARENT_SCOPE)
        return()
    endif()

    # Find Accelerate framework
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    find_library(VECLIB_FRAMEWORK vecLib)

    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Found Accelerate framework: ${ACCELERATE_FRAMEWORK}")
        set(HAVE_ACCELERATE TRUE PARENT_SCOPE)
        set(ACCELERATE_LIBRARIES ${ACCELERATE_FRAMEWORK} PARENT_SCOPE)

        # Register the helper
        sd_register_helper("accelerate")

        # Add compile definitions
        add_compile_definitions(HAVE_ACCELERATE=1)

        message(STATUS "Accelerate framework setup complete")
    else()
        message(WARNING "Accelerate framework not found")
        set(HAVE_ACCELERATE FALSE PARENT_SCOPE)
    endif()
endfunction()

# ============================================================================
# VLM HELPER SETUP (Vision-Language Model Operations)
# ============================================================================

function(setup_vlm)
    if(NOT HELPERS_vlm)
        message(STATUS "VLM helper is disabled (HELPERS_vlm=${HELPERS_vlm})")
        set(HAVE_VLM FALSE PARENT_SCOPE)
        return()
    endif()

    # VLM is a source-only helper, no external library needed
    # Just check that the source files exist
    file(GLOB VLM_CHECK_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/vlm/*.cpp)
    if(VLM_CHECK_FILES)
        message(STATUS "VLM helper sources found")
        set(HAVE_VLM TRUE PARENT_SCOPE)

        # Register the helper
        sd_register_helper("vlm")

        # Add compile definitions
        add_compile_definitions(HAVE_VLM=1)

        message(STATUS "VLM helper setup complete")
    else()
        message(WARNING "VLM helper sources not found")
        set(HAVE_VLM FALSE PARENT_SCOPE)
    endif()
endfunction()

# ============================================================================
# HELPER INITIALIZATION
# ============================================================================

# Function to initialize all requested helpers
function(sd_initialize_helpers)
    message(STATUS "")
    message(STATUS "=== Initializing Requested Helpers ===")

    # Detect platform compatibility
    sd_detect_available_helpers()

    # Track enabled helpers
    set(enabled_helpers "")

    # OneDNN
    if(HELPERS_onednn)
        if(HELPER_ONEDNN_COMPATIBLE)
            setup_onednn()
            if(HAVE_ONEDNN)
                list(APPEND enabled_helpers "onednn")
                sd_register_helper("onednn")

                # Setup MKL VML when OneDNN is enabled
                # MKL VML provides optimized vectorized math functions (vsErf, vdErf, etc.)
                setup_mkl_vml()
                if(HAVE_MKL_VML)
                    message(STATUS "MKL VML enabled alongside OneDNN for vectorized math")
                endif()
            endif()
        else()
            message(WARNING "OneDNN requested but platform is not compatible")
        endif()
    endif()

    # ARM Compute
    if(HELPERS_armcompute)
        if(HELPER_ARMCOMPUTE_COMPATIBLE)
            setup_armcompute()
            if(HAVE_ARMCOMPUTE)
                list(APPEND enabled_helpers "armcompute")
                sd_register_helper("armcompute")
            endif()
        else()
            message(WARNING "ARM Compute requested but platform is not compatible")
        endif()
    endif()

    # cuDNN
    if(HELPERS_cudnn AND SD_CUDA)
        if(HELPER_CUDNN_COMPATIBLE)
            setup_modern_cudnn()
            if(HAVE_CUDNN)
                list(APPEND enabled_helpers "cudnn")
                sd_register_helper("cudnn")
            endif()
        else()
            message(WARNING "cuDNN requested but CUDA build is not enabled")
        endif()
    endif()

    # MPS (Metal Performance Shaders)
    if(HELPERS_mps)
        if(HELPER_MPS_COMPATIBLE)
            setup_mps()
            if(HAVE_MPS)
                list(APPEND enabled_helpers "mps")
                sd_register_helper("mps")
            endif()
        else()
            message(WARNING "MPS requested but platform is not Apple")
        endif()
    endif()

    # Accelerate
    if(HELPERS_accelerate)
        if(HELPER_ACCELERATE_COMPATIBLE)
            setup_accelerate()
            if(HAVE_ACCELERATE)
                list(APPEND enabled_helpers "accelerate")
            endif()
        else()
            message(WARNING "Accelerate requested but platform is not Apple")
        endif()
    endif()

    # MLIR
    if(HELPERS_mlir)
        setup_mlir()
        if(HAVE_MLIR)
            list(APPEND enabled_helpers "mlir")
            sd_register_helper("mlir")
        endif()
    endif()

    # MIOpen
    if(HELPERS_miopen)
        if(HELPER_MIOPEN_COMPATIBLE)
            setup_miopen_download()
            if(HAVE_MIOPEN)
                list(APPEND enabled_helpers "miopen")
                sd_register_helper("miopen")
            endif()
        else()
            message(WARNING "MIOpen requested but ZLUDA AMD is not configured")
        endif()
    endif()

    # VLM (Vision-Language Model operations)
    if(HELPERS_vlm)
        setup_vlm()
        if(HAVE_VLM)
            list(APPEND enabled_helpers "vlm")
        endif()
    endif()

    # CPU fallback is always available
    list(APPEND enabled_helpers "cpu")

    # Store the list of enabled helpers
    set(SD_ENABLED_HELPERS "${enabled_helpers}" CACHE INTERNAL "List of enabled helpers" FORCE)

    message(STATUS "")
    message(STATUS "Enabled helpers: ${enabled_helpers}")
    message(STATUS "======================================")
    message(STATUS "")
endfunction()

# ============================================================================
# DYNAMIC KERNEL SELECTION CONFIGURATION
# ============================================================================

# Function to configure dynamic kernel selection
function(sd_configure_kernel_selection target_name)
    if(NOT SD_DYNAMIC_KERNEL_SELECTION)
        message(STATUS "Dynamic kernel selection is disabled")
        return()
    endif()

    message(STATUS "Configuring dynamic kernel selection for ${target_name}")

    # Add compile definitions for runtime kernel selection
    target_compile_definitions(${target_name} PUBLIC SD_DYNAMIC_KERNEL_SELECTION=1)

    # Kernel selection strategy
    if(SD_KERNEL_STRATEGY)
        target_compile_definitions(${target_name} PUBLIC SD_KERNEL_STRATEGY="${SD_KERNEL_STRATEGY}")
    endif()

    # Auto-tuning
    if(SD_KERNEL_AUTOTUNING)
        target_compile_definitions(${target_name} PUBLIC SD_KERNEL_AUTOTUNING=1)
        message(STATUS "   Kernel auto-tuning: enabled")
    endif()

    # Benchmarking
    if(SD_KERNEL_BENCHMARKING)
        target_compile_definitions(${target_name} PUBLIC SD_KERNEL_BENCHMARKING=1)
        message(STATUS "   Kernel benchmarking: enabled")
    endif()

    # Caching
    if(SD_KERNEL_CACHING)
        target_compile_definitions(${target_name} PUBLIC SD_KERNEL_CACHING=1)
        message(STATUS "   Kernel caching: enabled")
    endif()

    # Helper priority
    if(SD_HELPER_PRIORITY)
        # Convert semicolon-separated list to comma-separated for C++ string
        string(REPLACE ";" "," priority_csv "${SD_HELPER_PRIORITY}")
        target_compile_definitions(${target_name} PUBLIC SD_HELPER_PRIORITY="${priority_csv}")
        message(STATUS "   Helper priority: ${priority_csv}")
    endif()

    # Pass list of enabled helpers as compile definition
    if(SD_ENABLED_HELPERS)
        string(REPLACE ";" "," enabled_csv "${SD_ENABLED_HELPERS}")
        target_compile_definitions(${target_name} PUBLIC SD_ENABLED_HELPERS="${enabled_csv}")
        message(STATUS "   Enabled helpers: ${enabled_csv}")
    endif()
endfunction()

# ============================================================================
# HELPER SUMMARY
# ============================================================================

# Function to print final helper configuration summary
function(sd_print_helper_summary)
    message(STATUS "")
    message(STATUS "╔══════════════════════════════════════════════════════════════════╗")
    message(STATUS "║              MULTI-BACKEND HELPER CONFIGURATION                  ║")
    message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
    message(STATUS "║ Dynamic Kernel Selection: ${SD_DYNAMIC_KERNEL_SELECTION}")
    message(STATUS "║ Kernel Strategy:          ${SD_KERNEL_STRATEGY}")
    message(STATUS "║ Auto-tuning:              ${SD_KERNEL_AUTOTUNING}")
    message(STATUS "║ Helper Priority:          ${SD_HELPER_PRIORITY}")
    message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
    message(STATUS "║ HELPER            │ REQUESTED │ AVAILABLE │ STATUS              ║")
    message(STATUS "╠══════════════════════════════════════════════════════════════════╣")

    # OneDNN
    if(HELPERS_onednn)
        if(HAVE_ONEDNN)
            message(STATUS "║ OneDNN            │    YES    │    YES    │ Active              ║")
            # Show MKL VML status when OneDNN is active
            if(HAVE_MKL_VML)
                message(STATUS "║   └─ MKL VML      │    YES    │    YES    │ Vectorized Math     ║")
            else()
                message(STATUS "║   └─ MKL VML      │    YES    │    NO     │ Scalar Fallback     ║")
            endif()
        else()
            message(STATUS "║ OneDNN            │    YES    │    NO     │ Not Found           ║")
        endif()
    else()
        message(STATUS "║ OneDNN            │    NO     │    -      │ Disabled            ║")
    endif()

    # cuDNN
    if(HELPERS_cudnn)
        if(HAVE_CUDNN)
            message(STATUS "║ cuDNN             │    YES    │    YES    │ Active              ║")
        else()
            message(STATUS "║ cuDNN             │    YES    │    NO     │ Not Found           ║")
        endif()
    else()
        message(STATUS "║ cuDNN             │    NO     │    -      │ Disabled            ║")
    endif()

    # ARM Compute
    if(HELPERS_armcompute)
        if(HAVE_ARMCOMPUTE)
            message(STATUS "║ ARM Compute       │    YES    │    YES    │ Active              ║")
        else()
            message(STATUS "║ ARM Compute       │    YES    │    NO     │ Not Found           ║")
        endif()
    else()
        message(STATUS "║ ARM Compute       │    NO     │    -      │ Disabled            ║")
    endif()

    # MPS
    if(HELPERS_mps)
        if(HAVE_MPS)
            message(STATUS "║ MPS               │    YES    │    YES    │ Active              ║")
        else()
            message(STATUS "║ MPS               │    YES    │    NO     │ Not Found           ║")
        endif()
    else()
        message(STATUS "║ MPS               │    NO     │    -      │ Disabled            ║")
    endif()

    # Accelerate
    if(HELPERS_accelerate)
        if(HAVE_ACCELERATE)
            message(STATUS "║ Accelerate        │    YES    │    YES    │ Active              ║")
        else()
            message(STATUS "║ Accelerate        │    YES    │    NO     │ Not Found           ║")
        endif()
    else()
        message(STATUS "║ Accelerate        │    NO     │    -      │ Disabled            ║")
    endif()

    # MLIR
    if(HELPERS_mlir)
        if(HAVE_MLIR)
            message(STATUS "║ MLIR              │    YES    │    YES    │ Active              ║")
        else()
            message(STATUS "║ MLIR              │    YES    │    NO     │ Not Found           ║")
        endif()
    else()
        message(STATUS "║ MLIR              │    NO     │    -      │ Disabled            ║")
    endif()

    # MIOpen
    if(HELPERS_miopen)
        if(HAVE_MIOPEN)
            message(STATUS "║ MIOpen            │    YES    │    YES    │ Active              ║")
        else()
            message(STATUS "║ MIOpen            │    YES    │    NO     │ Not Found           ║")
        endif()
    else()
        message(STATUS "║ MIOpen            │    NO     │    -      │ Disabled            ║")
    endif()

    # VLM
    if(HELPERS_vlm)
        if(HAVE_VLM)
            message(STATUS "║ VLM               │    YES    │    YES    │ Active              ║")
        else()
            message(STATUS "║ VLM               │    YES    │    NO     │ Not Found           ║")
        endif()
    else()
        message(STATUS "║ VLM               │    NO     │    -      │ Disabled            ║")
    endif()

    message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
    message(STATUS "║ CPU Fallback      │   AUTO    │    YES    │ Always Active       ║")
    message(STATUS "╚══════════════════════════════════════════════════════════════════╝")
    message(STATUS "")

    # Print enabled helpers summary
    if(SD_ENABLED_HELPERS)
        list(LENGTH SD_ENABLED_HELPERS helper_count)
        message(STATUS "Total active helpers: ${helper_count}")
        message(STATUS "Active helpers: ${SD_ENABLED_HELPERS}")
    endif()
    message(STATUS "")
endfunction()

# ============================================================================
# HELPER SOURCE FILE COLLECTION
# ============================================================================

# Function to collect all helper source files
function(sd_collect_helper_sources out_source_list)
    set(helper_sources "")

    # OneDNN sources
    if(HAVE_ONEDNN)
        file(GLOB_RECURSE ONEDNN_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/mkldnn/*.cpp
        )
        list(APPEND helper_sources ${ONEDNN_SOURCES})
    endif()

    # ARM Compute sources
    if(HAVE_ARMCOMPUTE)
        file(GLOB_RECURSE ARMCOMPUTE_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/armcompute/*.cpp
        )
        list(APPEND helper_sources ${ARMCOMPUTE_SOURCES})
    endif()

    # cuDNN sources
    if(HAVE_CUDNN)
        file(GLOB_RECURSE CUDNN_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/cudnn/*.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/cudnn/*.cpp
        )
        list(APPEND helper_sources ${CUDNN_SOURCES})
    endif()

    # MPS sources
    if(HAVE_MPS)
        file(GLOB_RECURSE MPS_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/mps/*.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/mps/*.cpp
        )
        list(APPEND helper_sources ${MPS_SOURCES})
    endif()

    # Accelerate sources
    if(HAVE_ACCELERATE)
        file(GLOB_RECURSE ACCELERATE_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/accelerate/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/accelerate/*.mm
        )
        list(APPEND helper_sources ${ACCELERATE_SOURCES})
    endif()

    # MLIR sources
    if(HAVE_MLIR)
        file(GLOB_RECURSE MLIR_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/mlir/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/mlir/**/*.cpp
        )
        list(APPEND helper_sources ${MLIR_SOURCES})
    endif()

    # MIOpen sources
    if(HAVE_MIOPEN)
        file(GLOB_RECURSE MIOPEN_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/miopen/*.cpp
        )
        list(APPEND helper_sources ${MIOPEN_SOURCES})
    endif()

    # VLM sources
    if(HAVE_VLM)
        file(GLOB_RECURSE VLM_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/vlm/*.cpp
        )
        list(APPEND helper_sources ${VLM_SOURCES})
    endif()

    # Multi-platform dispatcher sources (always included when dynamic selection is enabled)
    if(SD_DYNAMIC_KERNEL_SELECTION)
        file(GLOB_RECURSE DISPATCHER_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/impl/MultiPlatformDispatcher.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/KernelSelectionEnvironment.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/KernelAutoTuner.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/KernelPerformanceRegistry.cpp
        )
        list(APPEND helper_sources ${DISPATCHER_SOURCES})
    endif()

    # Return the collected sources
    set(${out_source_list} ${helper_sources} PARENT_SCOPE)

    # Log summary
    list(LENGTH helper_sources source_count)
    message(STATUS "Collected ${source_count} helper source files")
endfunction()
