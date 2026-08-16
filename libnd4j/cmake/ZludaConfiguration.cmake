# cmake/ZludaConfiguration.cmake
# ZLUDA Transpiler Configuration for AMD GPU Support
#
# ZLUDA is a drop-in CUDA replacement that translates CUDA API calls to HIP
# for AMD GPUs through ROCm.
#
# This module provides:
#   - CMake-owned, pinned ZLUDA acquisition and build-input resolution
#   - MIOpen configuration for AMD (cuDNN alternative)
#   - CUDA ABI compatibility flags for the AMD ZLUDA runtime

function(zluda_find_first_existing output_var)
    set(_zluda_match "")
    foreach(_zluda_candidate IN LISTS ARGN)
        if(NOT "${_zluda_candidate}" STREQUAL "" AND EXISTS "${_zluda_candidate}")
            set(_zluda_match "${_zluda_candidate}")
            break()
        endif()
    endforeach()
    set(${output_var} "${_zluda_match}" PARENT_SCOPE)
endfunction()

# Resolve the runtime independently from the link input. Official Windows
# packages contain nvcuda.dll but no import library: Windows builds link the
# CUDA SDK's ABI-compatible nvcuda.lib and ZLUDA replaces it at runtime through
# zluda.exe or the documented application-local DLL layout. Unix can link the
# ZLUDA shared object directly. Keeping this helper side-effect free also makes
# release layouts testable without configuring the complete native build.
function(resolve_zluda_runtime zluda_root windows_layout output_link output_runtime)
    set(_zluda_link "")
    if(windows_layout)
        set(_zluda_runtime "")
        foreach(_zluda_layout_root
                "${zluda_root}" "${zluda_root}/bin" "${zluda_root}/lib")
            if(EXISTS "${_zluda_layout_root}/nvcuda.dll"
                    AND EXISTS "${_zluda_layout_root}/nvcudart_hybrid64.dll"
                    AND EXISTS "${_zluda_layout_root}/zluda.exe"
                    AND EXISTS "${_zluda_layout_root}/zluda_redirect.dll")
                set(_zluda_runtime "${_zluda_layout_root}/nvcuda.dll")
                break()
            endif()
        endforeach()
    else()
        zluda_find_first_existing(_zluda_runtime
            "${zluda_root}/libcuda.so"
            "${zluda_root}/lib/libcuda.so"
            "${zluda_root}/lib64/libcuda.so"
            "${zluda_root}/libnvcuda.so"
            "${zluda_root}/lib/libnvcuda.so"
            "${zluda_root}/lib64/libnvcuda.so")
        set(_zluda_link "${_zluda_runtime}")
    endif()
    set(${output_link} "${_zluda_link}" PARENT_SCOPE)
    set(${output_runtime} "${_zluda_runtime}" PARENT_SCOPE)
endfunction()

# Locate the actual distribution directory below CMake's managed extraction
# root.  Release archives may add a top-level directory, so callers must not
# depend on archive layout details.
function(resolve_zluda_distribution_root extraction_root windows_layout output_root)
    if(windows_layout)
        file(GLOB_RECURSE _zluda_runtime_candidates LIST_DIRECTORIES FALSE
            "${extraction_root}/nvcuda.dll")
    else()
        file(GLOB_RECURSE _zluda_runtime_candidates LIST_DIRECTORIES FALSE
            "${extraction_root}/libcuda.so"
            "${extraction_root}/libnvcuda.so")
    endif()
    list(SORT _zluda_runtime_candidates)

    set(_zluda_distribution_root "")
    foreach(_zluda_runtime_candidate IN LISTS _zluda_runtime_candidates)
        get_filename_component(_zluda_candidate_root
            "${_zluda_runtime_candidate}" DIRECTORY)
        resolve_zluda_runtime("${_zluda_candidate_root}" "${windows_layout}"
            _zluda_candidate_link _zluda_candidate_runtime)
        if(_zluda_candidate_runtime)
            set(_zluda_distribution_root "${_zluda_candidate_root}")
            break()
        endif()
    endforeach()
    set(${output_root} "${_zluda_distribution_root}" PARENT_SCOPE)
endfunction()

# Return every shared library distributed beside the selected ZLUDA driver.
# The platform classifier must carry the complete pinned runtime, not merely the
# link input used while compiling libnd4j. Executables such as zluda.exe are not
# preloadable libraries and therefore remain outside this JavaCPP runtime list.
function(resolve_zluda_runtime_bundle zluda_root windows_layout
        output_libraries output_root)
    resolve_zluda_runtime("${zluda_root}" "${windows_layout}"
        _zluda_bundle_link _zluda_bundle_runtime)
    if(NOT _zluda_bundle_runtime)
        set(${output_libraries} "" PARENT_SCOPE)
        set(${output_root} "" PARENT_SCOPE)
        return()
    endif()

    get_filename_component(_zluda_bundle_root
        "${_zluda_bundle_runtime}" DIRECTORY)
    if(windows_layout)
        file(GLOB _zluda_bundle_libraries LIST_DIRECTORIES FALSE
            "${_zluda_bundle_root}/*.dll")
    else()
        file(GLOB _zluda_bundle_libraries LIST_DIRECTORIES FALSE
            "${_zluda_bundle_root}/*.so"
            "${_zluda_bundle_root}/*.so.*")
    endif()
    list(APPEND _zluda_bundle_libraries "${_zluda_bundle_runtime}")
    list(REMOVE_DUPLICATES _zluda_bundle_libraries)
    list(SORT _zluda_bundle_libraries)
    set(${output_libraries} "${_zluda_bundle_libraries}" PARENT_SCOPE)
    set(${output_root} "${_zluda_bundle_root}" PARENT_SCOPE)
endfunction()

# Select the CUDA ABI libraries implemented by the pinned ZLUDA distribution.
# Unix links these implementations directly. Windows uses CUDA SDK import
# libraries while the matching ZLUDA DLLs are staged application-local.
function(resolve_zluda_cuda_abi_libraries runtime_libraries
        output_libraries output_cudnn_libraries)
    # Link exactly one file per CUDA ABI family. Official ZLUDA archives carry
    # several compatibility symlinks for each implementation; passing every
    # alias to the linker is redundant and can make the selected SONAME depend
    # on archive ordering.
    set(_zluda_cublas "")
    set(_zluda_cublaslt "")
    set(_zluda_cusparse "")
    set(_zluda_cufft "")
    set(_zluda_cudnn_libraries "")
    if(NOT WIN32)
        foreach(_zluda_runtime IN LISTS runtime_libraries)
            get_filename_component(_zluda_runtime_name "${_zluda_runtime}" NAME)
            string(TOLOWER "${_zluda_runtime_name}" _zluda_runtime_lower)
            foreach(_zluda_family IN ITEMS cublas cublaslt cusparse cufft)
                if(_zluda_runtime_lower MATCHES
                        "^lib${_zluda_family}\\.so($|\\.)" AND
                   NOT _zluda_${_zluda_family})
                    set(_zluda_${_zluda_family} "${_zluda_runtime}")
                endif()
            endforeach()
            if(_zluda_runtime_lower MATCHES "^libcudnn\\.so($|\\.)")
                list(APPEND _zluda_cudnn_libraries "${_zluda_runtime}")
            endif()
        endforeach()
    endif()

    set(_zluda_abi_libraries "")
    foreach(_zluda_family IN ITEMS cublas cublaslt cusparse cufft)
        if(_zluda_${_zluda_family})
            list(APPEND _zluda_abi_libraries "${_zluda_${_zluda_family}}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _zluda_abi_libraries)
    list(REMOVE_DUPLICATES _zluda_cudnn_libraries)
    set(${output_libraries} "${_zluda_abi_libraries}" PARENT_SCOPE)
    set(${output_cudnn_libraries} "${_zluda_cudnn_libraries}" PARENT_SCOPE)
endfunction()

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

    # ZLUDA is a native dependency.  Dependencies.cmake owns the pinned asset,
    # checksum, download cache, and extraction; release orchestration supplies
    # no filesystem root and consumers need no loader environment variable.
    setup_zluda_download(ZLUDA_MANAGED_ROOT)
    resolve_zluda_distribution_root(
        "${ZLUDA_MANAGED_ROOT}" "${WIN32}" ZLUDA_DISTRIBUTION_ROOT)

    set(ZLUDA_LINK_LIBRARY "")
    set(ZLUDA_RUNTIME_LIBRARY "")
    set(ZLUDA_RUNTIME_LIBRARIES "")
    set(ZLUDA_RUNTIME_ROOT "")
    set(ZLUDA_CUDA_ABI_LIBRARIES "")
    set(ZLUDA_CUDNN_RUNTIME_LIBRARIES "")
    if(ZLUDA_DISTRIBUTION_ROOT)
        resolve_zluda_runtime("${ZLUDA_DISTRIBUTION_ROOT}" "${WIN32}"
            ZLUDA_LINK_LIBRARY ZLUDA_RUNTIME_LIBRARY)
        resolve_zluda_runtime_bundle(
            "${ZLUDA_DISTRIBUTION_ROOT}" "${WIN32}"
            ZLUDA_RUNTIME_LIBRARIES ZLUDA_RUNTIME_ROOT)
    endif()

    # Preserve the historic variable for consumers that report or inspect it.
    set(ZLUDA_LIBRARY "${ZLUDA_LINK_LIBRARY}")
    if(NOT ZLUDA_LIBRARY)
        set(ZLUDA_LIBRARY "${ZLUDA_RUNTIME_LIBRARY}")
    endif()

    if(NOT ZLUDA_RUNTIME_LIBRARY)
        message(FATAL_ERROR
            "Pinned ZLUDA ${SD_ZLUDA_VERSION} contains no valid ${CMAKE_SYSTEM_NAME} runtime")
    endif()

    resolve_zluda_cuda_abi_libraries("${ZLUDA_RUNTIME_LIBRARIES}"
        ZLUDA_CUDA_ABI_LIBRARIES ZLUDA_CUDNN_RUNTIME_LIBRARIES)
    if(NOT WIN32)
        foreach(_zluda_required_abi IN ITEMS cublas cublaslt cusparse)
            set(_zluda_required_found FALSE)
            foreach(_zluda_abi_library IN LISTS ZLUDA_CUDA_ABI_LIBRARIES)
                get_filename_component(_zluda_abi_name
                    "${_zluda_abi_library}" NAME)
                string(TOLOWER "${_zluda_abi_name}" _zluda_abi_lower)
                if(_zluda_abi_lower MATCHES
                        "^lib${_zluda_required_abi}\\.so($|\\.)")
                    set(_zluda_required_found TRUE)
                    break()
                endif()
            endforeach()
            if(NOT _zluda_required_found)
                message(FATAL_ERROR
                    "Pinned ZLUDA runtime is missing AMD-backed CUDA ABI library ${_zluda_required_abi}")
            endif()
        endforeach()
        if(NOT ZLUDA_CUDNN_RUNTIME_LIBRARIES)
            message(FATAL_ERROR
                "Pinned ZLUDA runtime is missing its AMD-backed cuDNN ABI libraries")
        endif()
    endif()

    message(STATUS "Found ZLUDA runtime: ${ZLUDA_RUNTIME_LIBRARY}")
    if(ZLUDA_LINK_LIBRARY)
        message(STATUS "Found ZLUDA link library: ${ZLUDA_LINK_LIBRARY}")
        get_filename_component(ZLUDA_LIB_DIR "${ZLUDA_LINK_LIBRARY}" DIRECTORY)
    else()
        message(STATUS "Windows will link the CUDA SDK driver import library")
        get_filename_component(ZLUDA_LIB_DIR "${ZLUDA_RUNTIME_LIBRARY}" DIRECTORY)
    endif()
    set(ZLUDA_LIB_DIR "${ZLUDA_LIB_DIR}" PARENT_SCOPE)
    set(ZLUDA_LIBRARY "${ZLUDA_LIBRARY}" PARENT_SCOPE)
    set(ZLUDA_LINK_LIBRARY "${ZLUDA_LINK_LIBRARY}" PARENT_SCOPE)
    set(ZLUDA_RUNTIME_LIBRARY "${ZLUDA_RUNTIME_LIBRARY}" PARENT_SCOPE)
    set(ZLUDA_RUNTIME_LIBRARIES "${ZLUDA_RUNTIME_LIBRARIES}" PARENT_SCOPE)
    set(ZLUDA_RUNTIME_ROOT "${ZLUDA_RUNTIME_ROOT}" PARENT_SCOPE)
    set(ZLUDA_CUDA_ABI_LIBRARIES
        "${ZLUDA_CUDA_ABI_LIBRARIES}" PARENT_SCOPE)
    set(ZLUDA_CUDNN_RUNTIME_LIBRARIES
        "${ZLUDA_CUDNN_RUNTIME_LIBRARIES}" PARENT_SCOPE)

    if(NOT (SD_ZLUDA_TARGET STREQUAL "AMD" OR SD_ZLUDA_TARGET STREQUAL "amd"))
        message(FATAL_ERROR
            "The published ZLUDA backend is AMD-only; set SD_ZLUDA_TARGET=AMD")
    endif()
    set(ZLUDA_TARGET_BACKEND "AMD" PARENT_SCOPE)
    set(ZLUDA_TARGET_BACKEND "AMD")
    message(STATUS "ZLUDA target: AMD (ROCm/HIP)")
    setup_zluda_amd()
    foreach(_zluda_amd_variable
            ROCM_PATH ROCM_INCLUDE_DIR ROCM_LIB_DIR ROCM_HIP_RUNTIME_LIBRARY
            ROCM_HSA_RUNTIME_LIBRARY ROCM_HSAKMT_RUNTIME_LIBRARY HAVE_MIOPEN MIOPEN_LIBRARY MIOPEN_INCLUDE_DIR)
        set(${_zluda_amd_variable} "${${_zluda_amd_variable}}" PARENT_SCOPE)
    endforeach()

    set(HAVE_ZLUDA TRUE PARENT_SCOPE)
    add_compile_definitions(HAVE_ZLUDA=1)
    add_compile_definitions(ZLUDA_TARGET_${ZLUDA_TARGET_BACKEND}=1)
    # A ZLUDA artifact remains a CUDA build. Never select a HIP platform at
    # directory scope: AMD-native translation units must opt in explicitly so
    # CUDA and ROCm cannot declare their vector types in the same source file.

    print_status_colored("SUCCESS" "ZLUDA configuration complete (target: ${ZLUDA_TARGET_BACKEND})")
endfunction()

################################################################################
# AMD-specific ZLUDA Configuration (ROCm/HIP backend)
################################################################################

function(setup_zluda_amd)
    message(STATUS "Configuring ZLUDA for AMD GPUs...")

    # Do not pre-seed find_library/find_path outputs with empty strings.
    # CMake treats any defined value other than *-NOTFOUND as resolved and
    # skips the filesystem search entirely.
    set(HAVE_MIOPEN FALSE)

    # Find ROCm installation
    # ROCM_PATH is the sole version selector. Keep the unversioned /opt/rocm
    # symlink only as a conventional fallback; release workers set ROCM_PATH to
    # the attested, versioned SDK root before configuring CMake.
    set(ROCM_SEARCH_PATHS
        $ENV{ROCM_PATH}
        $ENV{ROCM_HOME}
        $ENV{HIP_PATH}
        /opt/rocm
    )

    find_path(ROCM_PATH
        NAMES include/hip/hip_runtime.h
        HINTS ${ROCM_SEARCH_PATHS}
        NO_DEFAULT_PATH
    )

    if(ROCM_PATH)
        message(STATUS "Found ROCm: ${ROCM_PATH}")
        set(ROCM_INCLUDE_DIR "${ROCM_PATH}/include")
        set(ROCM_LIB_DIR "${ROCM_PATH}/lib")

        if(NOT WIN32)
            find_library(ROCM_HIP_RUNTIME_LIBRARY
                NAMES amdhip64
                HINTS ${ROCM_PATH}
                PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
                NO_DEFAULT_PATH
            )
            find_library(ROCM_HSA_RUNTIME_LIBRARY
                NAMES hsa-runtime64 libhsa-runtime64.so.1
                HINTS ${ROCM_PATH}
                PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
                NO_DEFAULT_PATH
            )
            find_library(ROCM_HSAKMT_RUNTIME_LIBRARY
                NAMES hsakmt libhsakmt.so.1
                HINTS ${ROCM_PATH}
                PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
                NO_DEFAULT_PATH
            )

            # HIP, HSA, and ROCt are one versioned ROCm user-space contract. Resolve
            # all three below the selected SDK so a classifier cannot combine
            # newer HIP/HSA libraries with an older host HSA or ROCt thunk.
            get_filename_component(_rocm_path_real "${ROCM_PATH}" REALPATH)
            foreach(_rocm_runtime_variable IN ITEMS
                    ROCM_HIP_RUNTIME_LIBRARY ROCM_HSA_RUNTIME_LIBRARY
                    ROCM_HSAKMT_RUNTIME_LIBRARY)
                if(${_rocm_runtime_variable})
                    get_filename_component(_rocm_runtime_real
                        "${${_rocm_runtime_variable}}" REALPATH)
                    file(RELATIVE_PATH _rocm_runtime_relative
                        "${_rocm_path_real}" "${_rocm_runtime_real}")
                    if(_rocm_runtime_relative STREQUAL ".." OR
                       _rocm_runtime_relative MATCHES "^[.][.]/" OR
                       IS_ABSOLUTE "${_rocm_runtime_relative}")
                        message(FATAL_ERROR
                            "${_rocm_runtime_variable} must resolve below the selected ROCM_PATH '${ROCM_PATH}', got '${${_rocm_runtime_variable}}'")
                    endif()
                endif()
            endforeach()
        endif()

        # ROCm headers are scoped to the MIOpen translation units in
        # MainBuildFlow.cmake. The primary nd4jcuda object target stays CUDA-only.

        # Set up MIOpen for DNN operations (cuDNN replacement)
        setup_miopen()

        # Only SDK-neutral bridge implementations see AMD HIP/MIOpen. All
        # other ZLUDA sources remain CUDA-only, so CUDA and AMD HIP declarations
        # can never collide in one translation unit.
        if(ROCM_HIP_RUNTIME_LIBRARY)
            set(_ZLUDA_HIP_MEMORY_BRIDGE_SOURCE
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/cuda/ZludaHipMemoryBridge.cpp")
            set_source_files_properties("${_ZLUDA_HIP_MEMORY_BRIDGE_SOURCE}" PROPERTIES
                COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1"
                INCLUDE_DIRECTORIES "${ROCM_INCLUDE_DIR}"
                SKIP_UNITY_BUILD_INCLUSION ON)
            add_compile_definitions(HAVE_ZLUDA_HIP_MEMORY_BRIDGE=1)
        endif()

        if(HAVE_MIOPEN)
            set(_ZLUDA_MIOPEN_BRIDGE_SOURCE
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/miopen/miopenBridge.cpp")
            set(_ZLUDA_MIOPEN_INCLUDE_DIRS "${ROCM_INCLUDE_DIR}" "${MIOPEN_INCLUDE_DIR}")
            list(REMOVE_DUPLICATES _ZLUDA_MIOPEN_INCLUDE_DIRS)
            set_source_files_properties("${_ZLUDA_MIOPEN_BRIDGE_SOURCE}" PROPERTIES
                COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1"
                INCLUDE_DIRECTORIES "${_ZLUDA_MIOPEN_INCLUDE_DIRS}"
                SKIP_UNITY_BUILD_INCLUSION ON)
        endif()

        if(DEFINED ENV{DL4J_ZLUDA_REQUIRE_ROCM}
                AND NOT "$ENV{DL4J_ZLUDA_REQUIRE_ROCM}" STREQUAL ""
                AND NOT "$ENV{DL4J_ZLUDA_REQUIRE_ROCM}" STREQUAL "0"
                AND NOT WIN32 AND
                (NOT ROCM_HIP_RUNTIME_LIBRARY OR
                 NOT ROCM_HSA_RUNTIME_LIBRARY OR
                 NOT ROCM_HSAKMT_RUNTIME_LIBRARY))
            message(FATAL_ERROR
                "Build-only ZLUDA contract requires version-matched libamdhip64, libhsa-runtime64, and libhsakmt below ${ROCM_PATH}")
        endif()
        if(DEFINED ENV{DL4J_ZLUDA_REQUIRE_MIOPEN}
                AND NOT "$ENV{DL4J_ZLUDA_REQUIRE_MIOPEN}" STREQUAL ""
                AND NOT "$ENV{DL4J_ZLUDA_REQUIRE_MIOPEN}" STREQUAL "0"
                AND NOT HAVE_MIOPEN)
            message(FATAL_ERROR
                "Build-only ZLUDA contract requires MIOpen headers and library below ${ROCM_PATH}")
        endif()

        foreach(_zluda_amd_variable
                ROCM_PATH ROCM_INCLUDE_DIR ROCM_LIB_DIR ROCM_HIP_RUNTIME_LIBRARY
                ROCM_HSA_RUNTIME_LIBRARY ROCM_HSAKMT_RUNTIME_LIBRARY HAVE_MIOPEN MIOPEN_LIBRARY MIOPEN_INCLUDE_DIR)
            set(${_zluda_amd_variable} "${${_zluda_amd_variable}}" PARENT_SCOPE)
        endforeach()
    else()
        if(DEFINED ENV{DL4J_ZLUDA_REQUIRE_ROCM}
                AND NOT "$ENV{DL4J_ZLUDA_REQUIRE_ROCM}" STREQUAL ""
                AND NOT "$ENV{DL4J_ZLUDA_REQUIRE_ROCM}" STREQUAL "0")
            message(FATAL_ERROR
                "Build-only ZLUDA contract requires ROCm headers below ROCM_PATH")
        endif()
        print_status_colored("WARNING" "ROCm not found. AMD ZLUDA DNN operations may fall back to CPU.")
        print_status_colored("WARNING" "Install ROCm and set ROCM_PATH for full AMD GPU support.")
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
# Configure CUDA Flags for ZLUDA Compatibility
################################################################################

function(configure_zluda_cuda_flags)
    if(NOT HAVE_ZLUDA)
        return()
    endif()

    message(STATUS "Configuring CUDA flags for ZLUDA compatibility...")

    # ZLUDA links the CUDA toolkit's static archives on Windows. Keep
    # the native CMake targets and JavaCPP-generated launcher on the same
    # static MSVC CRT so the linker does not mix libcpmt with msvcprt.
    if(MSVC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>" PARENT_SCOPE)
        message(STATUS "   Windows ZLUDA MSVC runtime: static (/MT)")
    endif()

    # =========================================================================
    # NOTE: Architecture flags (-arch=sm_50) and --relocatable-device-code=false
    # are now handled centrally in CudaConfiguration.cmake when SD_ZLUDA is set.
    # This function now only handles additional ZLUDA-specific flags that aren't
    # part of the core CUDA build configuration.
    # =========================================================================

    set(ZLUDA_CUDA_FLAGS "")

    # Report that architecture is handled by CudaConfiguration.cmake.
    message(STATUS "   Target: AMD (ROCm/HIP) - sm_50 baseline set in CudaConfiguration")

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

    if(ZLUDA_LINK_LIBRARY)
        target_link_libraries(${target_name} PUBLIC "${ZLUDA_LINK_LIBRARY}")
        message(STATUS "   Linked ZLUDA import/shared library: ${ZLUDA_LINK_LIBRARY}")
    endif()

    if(ZLUDA_CUDA_ABI_LIBRARIES)
        target_link_libraries(${target_name} PUBLIC
            ${ZLUDA_CUDA_ABI_LIBRARIES})
        message(STATUS
            "   Linked ZLUDA AMD CUDA ABI libraries: ${ZLUDA_CUDA_ABI_LIBRARIES}")
    endif()

    if(WIN32)
        message(STATUS
            "   Using build-only CUDA SDK import libraries; the classifier carries the matching ZLUDA DLLs")
    endif()

    # Add ZLUDA library path (takes precedence over system CUDA)
    if(ZLUDA_LIB_DIR)
        target_link_directories(${target_name} BEFORE PUBLIC ${ZLUDA_LIB_DIR})
        message(STATUS "   Added ZLUDA lib path: ${ZLUDA_LIB_DIR}")
    endif()

    # The stream-ordered memory bridge always calls the AMD HIP runtime;
    # MIOpen is an additional optional private dependency.
    if(ROCM_HIP_RUNTIME_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${ROCM_HIP_RUNTIME_LIBRARY})
        message(STATUS "   Linked isolated AMD HIP runtime: ${ROCM_HIP_RUNTIME_LIBRARY}")
    endif()
    if(HAVE_MIOPEN)
        target_link_libraries(${target_name} PRIVATE ${MIOPEN_LIBRARY})
        message(STATUS "   Linked isolated MIOpen runtime: ${MIOPEN_LIBRARY}")
    endif()

    # Add ROCm library path privately for the AMD-only runtime dependency.
    if(ROCM_LIB_DIR)
        target_link_directories(${target_name} PRIVATE ${ROCM_LIB_DIR})
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
    message(STATUS "ZLUDA Link Library: ${ZLUDA_LINK_LIBRARY}")
    message(STATUS "ZLUDA Runtime Library: ${ZLUDA_RUNTIME_LIBRARY}")

    message(STATUS "ROCm Path: ${ROCM_PATH}")
    if(HAVE_MIOPEN)
        message(STATUS "MIOpen: Enabled (${MIOPEN_LIBRARY})")
    else()
        message(STATUS "MIOpen: Not available (cuDNN ops will use fallback)")
    endif()

    message(STATUS "=====================================")
    message(STATUS "")
endfunction()
