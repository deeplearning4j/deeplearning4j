# cmake/Options.cmake
# Defines all user-configurable build options and helper functions.

# --- Build Feature Options ---
option(SD_NATIVE "Optimize for build machine (might not work on others)" OFF)
option(SD_CHECK_VECTORIZATION "checks for vectorization" OFF)
option(SD_STATIC_LIB "Build static library (ignored, only shared lib is built)" OFF)
option(SD_SHARED_LIB "Build shared library (ignored, this is the default)" ON)
option(SD_USE_LTO "Use link time optimization" OFF)
option(SD_SANITIZE "Enable Address Sanitizer" OFF)
option(SD_EXTRACT_INSTANTIATIONS "Extract template instantiations and exit" OFF)
option(SD_GENERATE_FIX_FILES "Generate fix files for missing instantiations" OFF)
option(SD_INSTANTIATION_VERBOSE "Verbose instantiation extraction" OFF)

# --- Backend Options ---
option(SD_CPU "Build CPU backend" ON)
option(SD_CUDA "Build CUDA/GPU backend" OFF)
option(SD_TPU "Build TPU backend using PJRT" OFF)
option(SD_HEXAGON "Build Hexagon backend" OFF)
option(SD_VULKAN "Build Vulkan backend" OFF)
set(TPU_VERSION "v5" CACHE STRING "TPU version (v4, v5)")

# Backend identity is exclusive. Shared build behavior belongs in common
# functions and source interfaces; it must never be obtained by enabling or
# routing through another backend.
function(sd_validate_primary_backend_selection)
    set(_sd_enabled_backends "")
    foreach(_sd_backend IN ITEMS SD_CPU SD_CUDA SD_TPU SD_HEXAGON SD_VULKAN)
        if(${_sd_backend})
            list(APPEND _sd_enabled_backends "${_sd_backend}")
        endif()
    endforeach()

    list(LENGTH _sd_enabled_backends _sd_backend_count)
    if(NOT _sd_backend_count EQUAL 1)
        list(JOIN _sd_enabled_backends ", " _sd_enabled_backend_text)
        if(NOT _sd_enabled_backend_text)
            set(_sd_enabled_backend_text "none")
        endif()
        message(FATAL_ERROR
            "Exactly one primary backend must be enabled; active backends: "
            "${_sd_enabled_backend_text}. Backend selection must be explicit "
            "and mutually exclusive.")
    endif()
endfunction()

sd_validate_primary_backend_selection()
# Apple Silicon unified-memory MLX/Metal stack.
# On macOS arm64 + -Dlibnd4j.triton=ON, buildnativeoperations.sh sets this to ON
# automatically via the MLX_CMAKE variable.  You can also set it directly:
#   cmake ... -DSD_MLX=ON
option(SD_MLX "Enable Apple MLX GPU-compute backend (macOS arm64 only; requires Metal)" OFF)

# --- Build Verbosity Options ---
option(SD_VERBOSE_CUDA "Enable verbose CUDA/nvcc output (increases memory usage)" OFF)

# --- Backend Namespace Isolation ---
# Backend DSOs are designed to coexist in one process. Give each artifact a
# distinct internal C++ namespace while preserving the shared exported C ABI and
# normal dynamic linkage of common dependencies.
#
# Default behavior:
#   - CPU build:    SD_BACKEND_NAMESPACE = sd_cpu
#   - CUDA build:   SD_BACKEND_NAMESPACE = sd_cuda
#   - TPU build:    SD_BACKEND_NAMESPACE = sd_tpu
#   - Vulkan build: SD_BACKEND_NAMESPACE = sd_vulkan
if(SD_CUDA)
    set(_sd_backend_namespace_default "sd_cuda")
elseif(SD_TPU)
    set(_sd_backend_namespace_default "sd_tpu")
elseif(SD_VULKAN)
    set(_sd_backend_namespace_default "sd_vulkan")
else()
    set(_sd_backend_namespace_default "sd_cpu")
endif()

# Standard sd_* values are CMake-owned defaults. Refresh them when a build
# directory is reconfigured for another chip so a stale CPU cache cannot label
# a Vulkan artifact as sd_cpu. Nonstandard values remain explicit overrides.
if(NOT DEFINED SD_BACKEND_NAMESPACE OR
   SD_BACKEND_NAMESPACE STREQUAL "" OR
   SD_BACKEND_NAMESPACE MATCHES "^sd_(cpu|cuda|tpu|vulkan)$")
    set(SD_BACKEND_NAMESPACE "${_sd_backend_namespace_default}" CACHE STRING
        "Backend-specific namespace for symbol isolation" FORCE)
    message(STATUS "🔧 Backend namespace: ${SD_BACKEND_NAMESPACE} (chip-derived)")
else()
    message(STATUS "🔧 Backend namespace: ${SD_BACKEND_NAMESPACE} (user-specified)")
endif()
unset(_sd_backend_namespace_default)

# Add the namespace and configured marker as compile definitions
add_definitions(-DSD_BACKEND_NAMESPACE=${SD_BACKEND_NAMESPACE})
add_definitions(-DSD_BACKEND_NAMESPACE_CONFIGURED=1)

# Publish exactly one backend identity for source-level capability routing.
if(SD_TPU)
    add_definitions(-DSD_BACKEND_TYPE_TPU=1)
elseif(SD_VULKAN)
    add_definitions(-DSD_BACKEND_TYPE_VULKAN=1)
elseif(NOT SD_CUDA)
    add_definitions(-DSD_BACKEND_TYPE_CPU=1)
endif()

# --- ZLUDA Transpiler Options ---
# The published ZLUDA backend translates the CUDA ABI to AMD HIP/ROCm.
option(SD_ZLUDA "Enable the AMD ZLUDA CUDA-compatibility backend" OFF)
set(SD_ZLUDA_TARGET "AMD" CACHE STRING "ZLUDA target backend (AMD only)")
set_property(CACHE SD_ZLUDA_TARGET PROPERTY STRINGS AMD)
set(SD_ZLUDA_VERSION "v7-preview.8" CACHE STRING "Pinned ZLUDA release used by the AMD backend")
set_property(CACHE SD_ZLUDA_VERSION PROPERTY STRINGS v7-preview.8 v6)

# --- COMPILATION OPTIMIZATION OPTIONS (NEW) ---
# These dramatically affect template compilation time
option(SD_FAST_BUILD "Enable fast build mode with minimal templates" OFF)
option(SD_UNITY_BUILD "Enable Unity build for faster compilation" OFF)
set(SD_PARALLEL_COMPILE_JOBS "0" CACHE STRING "Number of parallel compile jobs (0 = auto)")

# --- Helper Library Toggles ---
# Individual helper enable flags (can enable multiple simultaneously)
option(HELPERS_armcompute "Enable ARM Compute Library helper" OFF)
option(HELPERS_onednn "Enable OneDNN helper" OFF)
option(HELPERS_cudnn "Enable cuDNN helper" OFF)
option(HELPERS_mlir "Enable MLIR/LLVM JIT compilation helper" OFF)
option(HELPERS_mps "Enable Metal Performance Shaders helper (macOS/iOS)" OFF)
option(HELPERS_pjrt "Enable PJRT/TPU helper" OFF)
option(HELPERS_miopen "Enable MIOpen helper (AMD GPUs via ZLUDA)" OFF)
option(HELPERS_accelerate "Enable Apple Accelerate framework helper (macOS/iOS)" OFF)
option(HELPERS_vlm "Enable VLM (Vision-Language Model) helper" OFF)
option(HELPERS_cutlass "Enable CUTLASS helper for optimized GEMM operations" OFF)
option(HELPERS_nccl "Enable NCCL helper for multi-GPU collective communications" OFF)
option(HELPERS_nnapi "Enable Android NNAPI helper for hardware acceleration (Android only)" OFF)
# NOTE: OpenVINO is triggered by SD_TRITON=ON (via -Dlibnd4j.triton=ON), not via helpers.
# --- Multi-Backend Helper Configuration ---
# HELPERS_LIST: Semicolon-separated list of helpers to enable (alternative to individual flags)
# Example: -DHELPERS_LIST="onednn;cudnn" or from pom.xml: -Dlibnd4j.helpers=onednn,cudnn
set(HELPERS_LIST "" CACHE STRING "Semicolon-separated list of helpers to enable (onednn;cudnn;armcompute;mlir;mps;pjrt;miopen;accelerate;vlm;nnapi)")

# Parse HELPERS_LIST and set individual HELPERS_* flags
if(HELPERS_LIST)
    message(STATUS "🔧 Processing HELPERS_LIST: ${HELPERS_LIST}")
    foreach(helper ${HELPERS_LIST})
        string(TOUPPER "${helper}" helper_upper)
        string(TOLOWER "${helper}" helper_lower)
        # Normalize common variations
        if(helper_lower STREQUAL "mkldnn" OR helper_lower STREQUAL "dnnl")
            set(helper_lower "onednn")
        endif()
        set(HELPERS_${helper_lower} ON CACHE BOOL "Enable ${helper_lower} helper (from HELPERS_LIST)" FORCE)
        message(STATUS "   ✓ Enabled helper: ${helper_lower}")
    endforeach()
endif()

# --- Dynamic Kernel Selection Configuration ---
option(SD_DYNAMIC_KERNEL_SELECTION "Enable dynamic kernel selection at runtime" ON)
option(SD_KERNEL_AUTOTUNING "Enable kernel auto-tuning for performance optimization" OFF)
option(SD_KERNEL_BENCHMARKING "Enable kernel benchmarking during execution" OFF)
option(SD_KERNEL_CACHING "Enable caching of kernel selection decisions" ON)

# Kernel selection strategy: fastest, first, roundrobin, memory, power
set(SD_KERNEL_STRATEGY "fastest" CACHE STRING "Default kernel selection strategy")
set_property(CACHE SD_KERNEL_STRATEGY PROPERTY STRINGS fastest first roundrobin memory power)

# Helper priority order for kernel dispatch (semicolon-separated, highest priority first)
# Default priority: platform-specific optimized helpers first, then generic
set(SD_HELPER_PRIORITY "" CACHE STRING "Helper priority order for kernel dispatch (e.g., 'cudnn;onednn;cpu')")

# Derive the default from the artifact backend, not host hardware availability.
# Refresh CMake-owned defaults when a build directory is reconfigured for a
# different chip; preserve nonstandard values as explicit user policy.
if(SD_CUDA)
    set(_sd_helper_priority_default "cudnn;cpu")
    set(_sd_helper_priority_description "Default CUDA helper priority")
elseif(SD_VULKAN)
    set(_sd_helper_priority_default "vulkan")
    set(_sd_helper_priority_description "Default Vulkan helper priority")
elseif(APPLE)
    set(_sd_helper_priority_default "mps;accelerate;cpu")
    set(_sd_helper_priority_description "Default macOS helper priority")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        set(_sd_helper_priority_default "nnapi;armcompute;mlir;cpu")
        set(_sd_helper_priority_description "Default Android ARM64 helper priority")
    else()
        set(_sd_helper_priority_default "armcompute;onednn;cpu")
        set(_sd_helper_priority_description "Default ARM64 helper priority")
    endif()
else()
    set(_sd_helper_priority_default "onednn;cpu")
    set(_sd_helper_priority_description "Default x86 helper priority")
endif()

if(NOT DEFINED SD_HELPER_PRIORITY OR
   SD_HELPER_PRIORITY STREQUAL "" OR
   "${SD_HELPER_PRIORITY}" STREQUAL "cudnn;cpu" OR
   "${SD_HELPER_PRIORITY}" STREQUAL "vulkan" OR
   "${SD_HELPER_PRIORITY}" STREQUAL "mps;accelerate;cpu" OR
   "${SD_HELPER_PRIORITY}" STREQUAL "nnapi;armcompute;mlir;cpu" OR
   "${SD_HELPER_PRIORITY}" STREQUAL "armcompute;onednn;cpu" OR
   "${SD_HELPER_PRIORITY}" STREQUAL "onednn;cpu")
    set(SD_HELPER_PRIORITY "${_sd_helper_priority_default}" CACHE STRING
        "${_sd_helper_priority_description}" FORCE)
endif()
unset(_sd_helper_priority_default)
unset(_sd_helper_priority_description)

# --- MLIR Configuration Options ---
set(MLIR_VERSION "18" CACHE STRING "MLIR/LLVM minimum version (18+)")
option(MLIR_ENABLE_GPU "Enable MLIR GPU dialect and NVVM backend" OFF)
option(MLIR_ENABLE_VULKAN "Enable the vendor-neutral MLIR Vulkan/SPIR-V backend" OFF)
# Vulkan MLIR is a backend contract: whenever the Vulkan runtime requests the
# MLIR helper, enable the SPIR-V dialect and validated MLIR::SPIRV target.
# Keeping this derived from SD_VULKAN avoids platform-specific Maven/CMake
# flags drifting apart between Linux and MinGW Windows builds.
if(SD_VULKAN AND HELPERS_mlir)
    set(MLIR_ENABLE_VULKAN ON CACHE BOOL
        "Enable the vendor-neutral MLIR Vulkan/SPIR-V backend" FORCE)
endif()
option(MLIR_ENABLE_AARCH64 "Enable LLVM AArch64 backend for ARM cross-compilation" OFF)
set(MLIR_AOT_TARGET "HOST" CACHE STRING "AOT compilation target (HOST, AARCH64_LINUX, AARCH64_ANDROID, X86_64_LINUX)")
set_property(CACHE MLIR_AOT_TARGET PROPERTY STRINGS HOST AARCH64_LINUX AARCH64_ANDROID X86_64_LINUX)
option(MLIR_JIT_CACHE "Enable caching of JIT-compiled kernels" ON)
option(MLIR_DEBUG_DUMPS "Enable MLIR IR dump during lowering (debug)" OFF)

# Configure optional compiler and runtime helpers for Android ARM64.
if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    if(HELPERS_mlir)
        set(MLIR_ENABLE_AARCH64 ON CACHE BOOL "Auto-enabled for Android ARM64 compiler builds" FORCE)
        set(MLIR_AOT_TARGET "AARCH64_ANDROID" CACHE STRING "Auto-set for Android ARM64 compiler builds" FORCE)
        message(STATUS "MLIR: AArch64 backend enabled for Android cross-compilation")
    else()
        message(STATUS "MLIR: compiler helper disabled; Android runtime will load AOT artifacts only")
    endif()

    # Derive capabilities from the configured Android target API; never
    # manufacture an API level that differs from the NDK toolchain target.
    # The official NDK toolchain may expose CMAKE_SYSTEM_VERSION=1, so prefer
    # its Android-specific variables.
    if(DEFINED ANDROID_PLATFORM)
        string(REGEX REPLACE "^android-" "" SD_ANDROID_TARGET_API_LEVEL "${ANDROID_PLATFORM}")
    elseif(DEFINED ANDROID_NATIVE_API_LEVEL)
        set(SD_ANDROID_TARGET_API_LEVEL "${ANDROID_NATIVE_API_LEVEL}")
    else()
        set(SD_ANDROID_TARGET_API_LEVEL "${CMAKE_SYSTEM_VERSION}")
    endif()

    if(SD_ANDROID_TARGET_API_LEVEL MATCHES "^[0-9]+$" AND
       SD_ANDROID_TARGET_API_LEVEL GREATER_EQUAL 27 AND NOT SD_VULKAN)
        set(HELPERS_nnapi ON CACHE BOOL "Auto-enabled for Android API 27+" FORCE)
        message(STATUS "NNAPI: requested for Android target API ${SD_ANDROID_TARGET_API_LEVEL}")
    elseif(SD_ANDROID_TARGET_API_LEVEL MATCHES "^[0-9]+$" AND
           SD_ANDROID_TARGET_API_LEVEL GREATER_EQUAL 27 AND SD_VULKAN)
        message(STATUS "NNAPI: not auto-enabled for the dedicated Vulkan runtime")
    else()
        message(STATUS "NNAPI: disabled for Android target API ${SD_ANDROID_TARGET_API_LEVEL}; requires API 27+")
    endif()
endif()

# Configure the compiler backend on native ARM64 only when MLIR is requested.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64" AND
   NOT CMAKE_CROSSCOMPILING AND HELPERS_mlir)
    set(MLIR_ENABLE_AARCH64 ON CACHE BOOL "Auto-enabled for native ARM64 compiler builds" FORCE)
    message(STATUS "MLIR: AArch64 backend enabled for native ARM64 host")
endif()

# Note: Helper defaults are already set by option() above (lines 34-40)
# Do NOT force them OFF here - that would override command-line settings like -DHELPERS_onednn=ON
# The option() command already defaults them to OFF but allows override

# Set corresponding HAVE_* variables (will be updated by helper setup functions)
set(HAVE_ARMCOMPUTE OFF CACHE BOOL "ARM Compute Library availability" FORCE)
set(HAVE_ONEDNN OFF CACHE BOOL "OneDNN availability" FORCE)
set(HAVE_CUDNN OFF CACHE BOOL "cuDNN availability" FORCE)
set(HAVE_MLIR OFF CACHE BOOL "MLIR availability" FORCE)
set(HAVE_MPS OFF CACHE BOOL "MPS availability" FORCE)
set(HAVE_PJRT OFF CACHE BOOL "PJRT/TPU availability" FORCE)
set(HAVE_MIOPEN OFF CACHE BOOL "MIOpen availability" FORCE)
set(HAVE_ZLUDA OFF CACHE BOOL "ZLUDA availability" FORCE)
set(HAVE_ACCELERATE OFF CACHE BOOL "Apple Accelerate availability" FORCE)
set(HAVE_VLM OFF CACHE BOOL "VLM availability" FORCE)
set(HAVE_TRITON OFF CACHE BOOL "Triton availability" FORCE)
set(HAVE_MLX OFF CACHE BOOL "MLX availability" FORCE)
set(HAVE_CUTLASS OFF CACHE BOOL "CUTLASS availability" FORCE)
set(HAVE_NCCL OFF CACHE BOOL "NCCL availability" FORCE)
if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND HELPERS_nnapi AND
   SD_ANDROID_TARGET_API_LEVEL MATCHES "^[0-9]+$" AND
   SD_ANDROID_TARGET_API_LEVEL GREATER_EQUAL 27)
    set(HAVE_NNAPI ON CACHE BOOL "Android NNAPI availability" FORCE)
else()
    set(HAVE_NNAPI OFF CACHE BOOL "Android NNAPI availability" FORCE)
endif()
set(HAVE_OPENVINO OFF CACHE BOOL "OpenVINO availability" FORCE)
set(HAVE_VULKAN OFF CACHE BOOL "Vulkan availability" FORCE)

# --- Vulkan (standalone device backend, distinct from MLIR_ENABLE_VULKAN) ---
# Vulkan discovery and linkage belong only to an SD_VULKAN artifact. CPU and
# CUDA builds must not probe, bootstrap, compile, or link the Vulkan backend.
if(SD_VULKAN)
    set(_libnd4j_vulkan_default ON)
else()
    set(_libnd4j_vulkan_default OFF)
endif()
option(LIBND4J_ENABLE_VULKAN "Enable the standalone Vulkan device backend" ${_libnd4j_vulkan_default})
unset(_libnd4j_vulkan_default)

# --- Triton GPU Compiler Options ---
set(TRITON_GPU_TARGET "AUTO" CACHE STRING "Triton target: AUTO, NVIDIA, AMD, INTEL")
set_property(CACHE TRITON_GPU_TARGET PROPERTY STRINGS AUTO NVIDIA AMD INTEL)

# --- Enabled Helpers Tracking ---
# This list is populated by helper setup functions and used for dynamic kernel selection
set(SD_ENABLED_HELPERS "" CACHE INTERNAL "List of successfully configured helpers")

# Function to register an enabled helper
function(sd_register_helper helper_name)
    set(current_helpers "${SD_ENABLED_HELPERS}")
    list(FIND current_helpers "${helper_name}" helper_index)
    if(helper_index EQUAL -1)
        list(APPEND current_helpers "${helper_name}")
        set(SD_ENABLED_HELPERS "${current_helpers}" CACHE INTERNAL "List of successfully configured helpers" FORCE)
        message(STATUS "✅ Registered helper: ${helper_name}")
    endif()
endfunction()

# Function to print helper configuration summary
function(print_helper_configuration)
    message(STATUS "")
    message(STATUS "=== Helper Configuration Summary ===")
    message(STATUS "Dynamic Kernel Selection: ${SD_DYNAMIC_KERNEL_SELECTION}")
    message(STATUS "Kernel Strategy: ${SD_KERNEL_STRATEGY}")
    message(STATUS "Kernel Auto-tuning: ${SD_KERNEL_AUTOTUNING}")
    message(STATUS "Helper Priority: ${SD_HELPER_PRIORITY}")
    message(STATUS "")
    message(STATUS "Individual Helper Status:")
    message(STATUS "  OneDNN:       REQUESTED=${HELPERS_onednn}, AVAILABLE=${HAVE_ONEDNN}")
    message(STATUS "  cuDNN:        REQUESTED=${HELPERS_cudnn}, AVAILABLE=${HAVE_CUDNN}")
    message(STATUS "  ARM Compute:  REQUESTED=${HELPERS_armcompute}, AVAILABLE=${HAVE_ARMCOMPUTE}")
    message(STATUS "  MPS:          REQUESTED=${HELPERS_mps}, AVAILABLE=${HAVE_MPS}")
    message(STATUS "  Accelerate:   REQUESTED=${HELPERS_accelerate}, AVAILABLE=${HAVE_ACCELERATE}")
    message(STATUS "  MLIR:         REQUESTED=${HELPERS_mlir}, AVAILABLE=${HAVE_MLIR}, VULKAN=${MLIR_ENABLE_VULKAN}, AOT=${MLIR_AOT_TARGET}")
    message(STATUS "  PJRT:         REQUESTED=${HELPERS_pjrt}, AVAILABLE=${HAVE_PJRT}")
    message(STATUS "  MIOpen:       REQUESTED=${HELPERS_miopen}, AVAILABLE=${HAVE_MIOPEN}")
    message(STATUS "  VLM:          REQUESTED=${HELPERS_vlm}, AVAILABLE=${HAVE_VLM}")
    message(STATUS "  Triton:       AVAILABLE=${HAVE_TRITON}")
    message(STATUS "  CUTLASS:      REQUESTED=${HELPERS_cutlass}, AVAILABLE=${HAVE_CUTLASS}")
    message(STATUS "  NCCL:         REQUESTED=${HELPERS_nccl}, AVAILABLE=${HAVE_NCCL}")
    message(STATUS "  NNAPI:        REQUESTED=${HELPERS_nnapi}, AVAILABLE=${HAVE_NNAPI}")
    message(STATUS "  OpenVINO:     SD_TRITON=${SD_TRITON}, AVAILABLE=${HAVE_OPENVINO}")
    message(STATUS "")
    message(STATUS "Enabled Helpers: ${SD_ENABLED_HELPERS}")
    message(STATUS "====================================")
    message(STATUS "")
endfunction()
set(GENERATED_TYPE_COMBINATIONS "" CACHE INTERNAL "Generated type combinations")
set(PROCESSED_TEMPLATE_FILES "" CACHE INTERNAL "Processed template files")

# --- Debug and Trace Options ---
option(SD_GCC_FUNCTRACE "Use call traces" OFF)

# Enable compile_commands.json for functrace builds (helps with validation)
if(SD_GCC_FUNCTRACE)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands for validation" FORCE)
endif()

option(PRINT_INDICES "Print indices" OFF)
option(PRINT_MATH "Print math operations" OFF)
option(SD_PTXAS "Enable ptxas verbose output" OFF)
option(SD_KEEP_NVCC_OUTPUT "Keep NVCC output files" OFF)
option(SD_PREPROCESS "Enable preprocessing" OFF)

# --- CUDA Parallel Compilation Options (CUDA 11.2+) ---
# These options can dramatically reduce CUDA build times (up to 50% improvement)
# --threads and --split-compile are auto-enabled for CUDA 11.2+
# Device LTO is opt-in due to increased link time
option(SD_CUDA_DEVICE_LTO "Enable CUDA Device Link Time Optimization (better runtime perf, longer link)" OFF)
option(SD_CUDA_TIME_TRACE "Generate CUDA compilation time trace for build profiling (CUDA 12.8+)" OFF)
set(SD_CUDA_THREADS "0" CACHE STRING "NVCC --threads count (0=auto, max parallel arch compilation)")
set(SD_CUDA_SPLIT_COMPILE "0" CACHE STRING "NVCC --split-compile threads (0=auto, parallel optimization phase)")

# --- SDX Standalone Runtime ---
option(SD_BUILD_SDX_STANDALONE "Build standalone libsdx_cpu/libsdx_cuda (no JVM dependency)" OFF)
# Kernel backend toggles for SDX (all default ON when the parent build has them):
#   SDX_INCLUDE_TRITON, SDX_INCLUDE_ONEDNN, SDX_INCLUDE_MLIR, SDX_INCLUDE_OPENVINO
# Override any to OFF for a smaller deployment binary.  See BuildSDX.cmake.

# --- Build Target Options ---
option(SD_BUILD_TESTS "Build tests" OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" OFF)

# Hack to disable flatc build unless explicitly enabled
set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Hack to disable flatc build" FORCE)

# --- Type System and Template Options ---
option(SD_ENABLE_SEMANTIC_FILTERING "Enable semantic filtering to reduce template combinations" ON)
option(SD_ENABLE_SELECTIVE_RENDERING "Enable selective rendering system" ON)
option(SD_AGGRESSIVE_SEMANTIC_FILTERING "Use aggressive filtering rules" ON)

# CHANGED: Default to MINIMAL for faster builds
set(SD_TYPE_PROFILE "STANDARD_ALL_TYPES" CACHE STRING "Type profile for semantic filtering (MINIMAL, ESSENTIAL, QUANTIZATION, INFERENCE, TRAINING,STANDARD_ALL_TYPES)")
set_property(CACHE SD_TYPE_PROFILE PROPERTY STRINGS STANDARD_ALL_TYPES  MINIMAL ESSENTIAL QUANTIZATION INFERENCE TRAINING)

# CHANGED: Lower default to prevent explosion
set(SD_MAX_TEMPLATE_COMBINATIONS "1000" CACHE STRING "Maximum template combinations to generate (safety limit)")

# --- NEW: Template Chunking Configuration ---
# These control how template instantiations are split across files
# TLS relocation overflow is caused by TOO MANY FILES, not large files
# Solution: Use larger chunks with call tracing to minimize file count

# Detect build configuration and set appropriate defaults
if(DEFINED SD_GCC_FUNCTRACE AND SD_GCC_FUNCTRACE)
    # Call tracing enabled - need larger chunks to avoid TLS overflow
    if((DEFINED SD_SANITIZE AND SD_SANITIZE) OR (DEFINED SD_SANITIZERS AND NOT SD_SANITIZERS STREQUAL ""))
        # Call tracing + Sanitizers: Compromise between speed and RAM
        # Larger than sanitizers-only (prevents TLS overflow) but not too large (prevents OOM)
        set(CHUNK_TARGET_INSTANTIATIONS "30" CACHE STRING "Balanced chunks for call tracing + sanitizers" FORCE)
        set(MULTI_PASS_CHUNK_SIZE "70" CACHE STRING "Balanced direct chunks for call tracing + sanitizers" FORCE)
        message(STATUS "⚠️  Call tracing + Sanitizers: Using balanced chunks (30/70) - prevents TLS overflow and OOM")
    else()
        # Call tracing only: Very large chunks for maximum speed (no RAM constraints without sanitizers)
        # Functrace adds minimal memory overhead, so we can use much larger chunks than normal builds
        # Larger chunks = fewer files = faster linking + less TLS overhead
        set(CHUNK_TARGET_INSTANTIATIONS "80" CACHE STRING "Very large chunks for call tracing only" FORCE)
        set(MULTI_PASS_CHUNK_SIZE "200" CACHE STRING "Very large direct chunks for call tracing only" FORCE)
        message(STATUS "🔍 Call tracing only: Using very large chunks (80/200) for maximum build speed")
    endif()
elseif(DEFINED SD_SANITIZE AND SD_SANITIZE)
    # Sanitizers enabled: Small chunks to prevent OOM (sanitizer builds use massive RAM per file)
    # Note: Only trigger on SD_SANITIZE=ON, not just SD_SANITIZERS being set (it has a default value)
    set(CHUNK_TARGET_INSTANTIATIONS "6" CACHE STRING "Small chunks for sanitizers (prevents OOM)" FORCE)
    set(MULTI_PASS_CHUNK_SIZE "8" CACHE STRING "Small direct chunks for sanitizers (prevents OOM)" FORCE)
    message(STATUS "⚠️  Sanitizers: Using small chunks (6/8) to prevent out-of-memory during compilation")
else()
    # Normal builds: Use memory-based defaults (no special instrumentation)
    cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
    if(AVAILABLE_MEMORY LESS 4000)
        set(CHUNK_TARGET_INSTANTIATIONS "3" CACHE STRING "Conservative chunks for low memory" FORCE)
        set(MULTI_PASS_CHUNK_SIZE "25" CACHE STRING "Conservative direct chunks" FORCE)
        message(STATUS "💾 Low memory detected: Using small chunks (3/25)")
    elseif(AVAILABLE_MEMORY LESS 8000)
        set(CHUNK_TARGET_INSTANTIATIONS "6" CACHE STRING "Moderate chunks for medium memory" FORCE)
        set(MULTI_PASS_CHUNK_SIZE "35" CACHE STRING "Moderate direct chunks" FORCE)
        message(STATUS "💾 Medium memory detected: Using moderate chunks (6/35)")
    elseif(AVAILABLE_MEMORY LESS 16000)
        set(CHUNK_TARGET_INSTANTIATIONS "10" CACHE STRING "Balanced chunks for high memory" FORCE)
        set(MULTI_PASS_CHUNK_SIZE "50" CACHE STRING "Balanced direct chunks" FORCE)
        message(STATUS "💾 High memory detected: Using balanced chunks (10/50)")
    else()
        set(CHUNK_TARGET_INSTANTIATIONS "12" CACHE STRING "Optimized chunks for very high memory" FORCE)
        set(MULTI_PASS_CHUNK_SIZE "60" CACHE STRING "Optimized direct chunks" FORCE)
        message(STATUS "💾 Very high memory detected: Using large chunks (12/60)")
    endif()
endif()

set(CHUNK_MAX_INSTANTIATIONS "3" CACHE STRING "Maximum template instantiations per chunk file")
set(USE_MULTI_PASS_GENERATION "OFF" CACHE STRING "Use multi-pass generation (ON/OFF/AUTO)")

# --- NEW: Type Selection for Fast Builds ---
if(SD_FAST_BUILD)
    message(STATUS "🚀 Fast build mode enabled - using minimal type set")
    set(SD_TYPE_PROFILE "MINIMAL" CACHE STRING "Type profile for fast builds" FORCE)
    set(SD_TYPES_LIST "float;double" CACHE STRING "Minimal type set for fast builds" FORCE)
    set(SD_AGGRESSIVE_SEMANTIC_FILTERING ON CACHE BOOL "Use aggressive filtering" FORCE)
    set(CHUNK_TARGET_INSTANTIATIONS "100" CACHE STRING "Larger chunks for fast builds" FORCE)
    set(SD_MAX_TEMPLATE_COMBINATIONS "500" CACHE STRING "Strict limit for fast builds" FORCE)
endif()

# --- NEW: Configure Unity Build ---
if(SD_UNITY_BUILD)
    message(STATUS "🔧 Unity build enabled for faster compilation")
    set(CMAKE_UNITY_BUILD ON CACHE BOOL "Enable Unity builds" FORCE)
    set(CMAKE_UNITY_BUILD_BATCH_SIZE 20 CACHE STRING "Unity build batch size" FORCE)
endif()

# --- NEW: Configure Parallel Compilation ---
if(NOT SD_PARALLEL_COMPILE_JOBS STREQUAL "0")
    set(CMAKE_BUILD_PARALLEL_LEVEL ${SD_PARALLEL_COMPILE_JOBS} CACHE STRING "Parallel build level" FORCE)
    message(STATUS "🔧 Parallel compilation set to ${SD_PARALLEL_COMPILE_JOBS} jobs")
else()
    # Auto-detect number of cores
    include(ProcessorCount)
    ProcessorCount(N)
    if(NOT N EQUAL 0)
        set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Parallel build level" FORCE)
        message(STATUS "🔧 Auto-detected ${N} cores for parallel compilation")
    endif()
endif()

# --- NEW: Compiler-specific optimizations for template compilation ---
# Template depth: 512 for release (faster compilation), 1024 for debug (deeper nesting support)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-ftemplate-depth=1024)
    message(STATUS "Using template depth 1024 for ${CMAKE_BUILD_TYPE} build")
else()
    add_compile_options(-ftemplate-depth=512)
    message(STATUS "Using template depth 512 for ${CMAKE_BUILD_TYPE} build")
endif()

# --- CONSTEXPR LIMITS: Reduce memory usage during template instantiation ---
# These limits prevent excessive memory consumption during compile-time evaluation
# fconstexpr-depth: Maximum nesting depth of constexpr function calls (default: 512)
#   - Lowering to 256 reduces memory for deeply nested constexpr code
# fconstexpr-loop-limit: Maximum iterations for constexpr loops (default: 262144)
#   - 65536 is sufficient for most use cases while using less memory
# fconstexpr-ops-limit: Maximum operations in constexpr evaluation (default: 33554432)
#   - 4194304 (4M) is enough for practical constexpr while limiting memory
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
        add_compile_options(-fconstexpr-depth=256)
        add_compile_options(-fconstexpr-loop-limit=65536)
        add_compile_options(-fconstexpr-ops-limit=4194304)
        message(STATUS "💾 Applied GCC constexpr limits to reduce compilation memory")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fconstexpr-depth=256)
    add_compile_options(-fconstexpr-steps=4194304)
    message(STATUS "💾 Applied Clang constexpr limits to reduce compilation memory")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Enable faster template compilation in GCC 10+
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
        add_compile_options(-fconcepts-diagnostics-depth=2)
    endif()

    # For development builds, use faster but less optimized compilation
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR SD_FAST_BUILD)
        add_compile_options(-O0 -fno-inline-functions)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # For development builds, use faster but less optimized compilation
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR SD_FAST_BUILD)
        add_compile_options(-O0 -fno-inline-functions)
    endif()
endif()

# --- Dependency Cache Options ---
# Persistent cache for ExternalProject_Add dependencies (FlatBuffers, OneDNN, Triton/LLVM, etc.)
# so that 'mvn clean install' doesn't re-download and rebuild everything.
option(SD_DEPENDENCY_BOOTSTRAP_ONLY
       "Configure only the project-managed compiler dependency bootstrap targets" OFF)
mark_as_advanced(SD_DEPENDENCY_BOOTSTRAP_ONLY)
option(SD_DEP_CACHE "Enable dependency caching across clean builds" ON)
set(SD_DEP_CACHE_DIR "" CACHE STRING "Dependency cache directory (default: ~/.libnd4j/dep-cache)")
option(SD_DEP_CACHE_CLEAR "Clear all cached dependencies at configure time" OFF)
set(SD_DEP_CACHE_CLEAR_DEP "" CACHE STRING "Clear cache for a specific dependency (e.g. 'triton_llvm')")

# Set default cache directory if not specified
if(SD_DEP_CACHE AND NOT SD_DEP_CACHE_DIR)
    if(WIN32)
        set(SD_DEP_CACHE_DIR "$ENV{USERPROFILE}/.libnd4j/dep-cache" CACHE STRING "" FORCE)
    else()
        set(SD_DEP_CACHE_DIR "$ENV{HOME}/.libnd4j/dep-cache" CACHE STRING "" FORCE)
    endif()
endif()

option(BUILD_PPSTEP "Build ppstep preprocessor debugging tool" OFF)

# --- Helper function for colored status messages ---
function(print_status_colored type message)
    if(type STREQUAL "ERROR")
        message(FATAL_ERROR "❌ ${message}")
    elseif(type STREQUAL "WARNING")
        message(WARNING "⚠️  ${message}")
    elseif(type STREQUAL "SUCCESS")
        message(STATUS "✅ ${message}")
    elseif(type STREQUAL "INFO")
        message(STATUS "ℹ️  ${message}")
    elseif(type STREQUAL "DEBUG")
        message(STATUS "🔍 ${message}")
    elseif(type STREQUAL "NOTICE")
        message(NOTICE "📢 ${message}")
    else()
        message(STATUS "${message}")
    endif()
endfunction()

# --- NEW: Print build configuration summary ---
function(print_build_configuration)
    message(STATUS "")
    message(STATUS "=== Template Compilation Configuration ===")
    message(STATUS "Type Profile: ${SD_TYPE_PROFILE}")
    message(STATUS "Semantic Filtering: ${SD_ENABLE_SEMANTIC_FILTERING}")
    message(STATUS "Aggressive Filtering: ${SD_AGGRESSIVE_SEMANTIC_FILTERING}")
    message(STATUS "Max Template Combinations: ${SD_MAX_TEMPLATE_COMBINATIONS}")
    message(STATUS "Chunk Target Size: ${CHUNK_TARGET_INSTANTIATIONS}")
    message(STATUS "Unity Build: ${SD_UNITY_BUILD}")
    message(STATUS "Fast Build Mode: ${SD_FAST_BUILD}")
    
    if(DEFINED SD_TYPES_LIST)
        message(STATUS "Active Types: ${SD_TYPES_LIST}")
    endif()
    
    message(STATUS "==========================================")
    message(STATUS "")
endfunction()

# --- Macro for debugging all CMake variables ---
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# --- NEW: Validation and warnings ---
if(SD_TYPE_PROFILE STREQUAL "ESSENTIAL" OR SD_TYPE_PROFILE STREQUAL "TRAINING")
    message(WARNING "⚠️  Using '${SD_TYPE_PROFILE}' profile will generate many template instantiations and slow compilation. Consider 'MINIMAL' or 'INFERENCE' for faster builds.")
endif()

if(CHUNK_TARGET_INSTANTIATIONS LESS 20)
    message(WARNING "⚠️  CHUNK_TARGET_INSTANTIATIONS is very low (${CHUNK_TARGET_INSTANTIATIONS}). This will create many small files and slow compilation.")
endif()