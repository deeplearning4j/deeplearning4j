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
set(TPU_VERSION "v5" CACHE STRING "TPU version (v4, v5)")

# --- Build Verbosity Options ---
option(SD_VERBOSE_CUDA "Enable verbose CUDA/nvcc output (increases memory usage)" OFF)

# --- Backend Namespace Isolation ---
# When multiple backends (CPU, CUDA) are loaded in the same process, symbol conflicts
# can occur because both libraries export identically-named symbols (ops, helpers, singletons).
# This option sets a unique namespace prefix for all internal symbols.
#
# Default behavior:
#   - CPU build:  SD_BACKEND_NAMESPACE = sd_cpu
#   - CUDA build: SD_BACKEND_NAMESPACE = sd_cuda
#   - TPU build:  SD_BACKEND_NAMESPACE = sd_tpu
#
# This enables true multi-backend execution where both CPU and CUDA libraries
# can be loaded simultaneously without symbol conflicts.
if(NOT DEFINED SD_BACKEND_NAMESPACE OR SD_BACKEND_NAMESPACE STREQUAL "")
    if(SD_CUDA)
        set(SD_BACKEND_NAMESPACE "sd_cuda" CACHE STRING "Backend-specific namespace for symbol isolation")
    elseif(SD_TPU)
        set(SD_BACKEND_NAMESPACE "sd_tpu" CACHE STRING "Backend-specific namespace for symbol isolation")
    else()
        set(SD_BACKEND_NAMESPACE "sd_cpu" CACHE STRING "Backend-specific namespace for symbol isolation")
    endif()
    message(STATUS "🔧 Backend namespace: ${SD_BACKEND_NAMESPACE} (auto-detected)")
else()
    message(STATUS "🔧 Backend namespace: ${SD_BACKEND_NAMESPACE} (user-specified)")
endif()

# Add the namespace as a compile definition
add_definitions(-DSD_BACKEND_NAMESPACE=${SD_BACKEND_NAMESPACE})

# Also define which backend type this is (for conditional compilation)
if(SD_CUDA)
    add_definitions(-DSD_BACKEND_TYPE_CUDA=1)
elseif(SD_TPU)
    add_definitions(-DSD_BACKEND_TYPE_TPU=1)
else()
    add_definitions(-DSD_BACKEND_TYPE_CPU=1)
endif()

# --- ZLUDA Transpiler Options ---
# ZLUDA enables running CUDA code on AMD/Intel GPUs via runtime translation
option(SD_ZLUDA "Enable ZLUDA transpiler support for AMD/Intel GPUs" OFF)
set(SD_ZLUDA_TARGET "AUTO" CACHE STRING "ZLUDA target backend (AMD, INTEL, AUTO)")
set_property(CACHE SD_ZLUDA_TARGET PROPERTY STRINGS AUTO AMD INTEL)

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
option(HELPERS_llamacpp "Enable LlamaCpp helper for LLM operations" OFF)
option(HELPERS_vlm "Enable VLM (Vision-Language Model) helper" OFF)

# --- Multi-Backend Helper Configuration ---
# HELPERS_LIST: Semicolon-separated list of helpers to enable (alternative to individual flags)
# Example: -DHELPERS_LIST="onednn;cudnn" or from pom.xml: -Dlibnd4j.helpers=onednn,cudnn
set(HELPERS_LIST "" CACHE STRING "Semicolon-separated list of helpers to enable (onednn;cudnn;armcompute;mlir;mps;pjrt;miopen;accelerate;llamacpp;vlm)")

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

# Auto-detect and set default helper priority based on platform
if(NOT SD_HELPER_PRIORITY)
    if(SD_CUDA)
        set(SD_HELPER_PRIORITY "cudnn;cpu" CACHE STRING "Default CUDA helper priority" FORCE)
    elseif(APPLE)
        set(SD_HELPER_PRIORITY "mps;accelerate;cpu" CACHE STRING "Default macOS helper priority" FORCE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(SD_HELPER_PRIORITY "armcompute;onednn;cpu" CACHE STRING "Default ARM64 helper priority" FORCE)
    else()
        set(SD_HELPER_PRIORITY "onednn;cpu" CACHE STRING "Default x86 helper priority" FORCE)
    endif()
endif()

# --- MLIR Configuration Options ---
set(MLIR_VERSION "18" CACHE STRING "MLIR/LLVM minimum version (18+)")
option(MLIR_ENABLE_GPU "Enable MLIR GPU dialect and NVVM backend" OFF)
option(MLIR_JIT_CACHE "Enable caching of JIT-compiled kernels" ON)
option(MLIR_DEBUG_DUMPS "Enable MLIR IR dump during lowering (debug)" OFF)

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
set(HAVE_LLAMACPP OFF CACHE BOOL "LlamaCpp availability" FORCE)
set(HAVE_VLM OFF CACHE BOOL "VLM availability" FORCE)

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
    message(STATUS "  MLIR:         REQUESTED=${HELPERS_mlir}, AVAILABLE=${HAVE_MLIR}")
    message(STATUS "  PJRT:         REQUESTED=${HELPERS_pjrt}, AVAILABLE=${HAVE_PJRT}")
    message(STATUS "  MIOpen:       REQUESTED=${HELPERS_miopen}, AVAILABLE=${HAVE_MIOPEN}")
    message(STATUS "  LlamaCpp:     REQUESTED=${HELPERS_llamacpp}, AVAILABLE=${HAVE_LLAMACPP}")
    message(STATUS "  VLM:          REQUESTED=${HELPERS_vlm}, AVAILABLE=${HAVE_VLM}")
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