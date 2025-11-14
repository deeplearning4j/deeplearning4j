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

# --- COMPILATION OPTIMIZATION OPTIONS (NEW) ---
# These dramatically affect template compilation time
option(SD_FAST_BUILD "Enable fast build mode with minimal templates" OFF)
option(SD_UNITY_BUILD "Enable Unity build for faster compilation" OFF)
set(SD_PARALLEL_COMPILE_JOBS "0" CACHE STRING "Number of parallel compile jobs (0 = auto)")

# --- Helper Library Toggles ---
option(HELPERS_armcompute "Enable ARM Compute Library helper" OFF)
option(HELPERS_onednn "Enable OneDNN helper" OFF)
option(HELPERS_cudnn "Enable cuDNN helper" OFF)

# Force all helpers OFF by default to prevent compilation issues
set(HELPERS_armcompute OFF CACHE BOOL "Force disable ARM Compute Library helper" FORCE)
set(HELPERS_onednn OFF CACHE BOOL "Force disable OneDNN helper" FORCE)
set(HELPERS_cudnn OFF CACHE BOOL "Force disable cuDNN helper" FORCE)

# Set corresponding HAVE_* variables
set(HAVE_ARMCOMPUTE OFF CACHE BOOL "ARM Compute Library availability" FORCE)
set(HAVE_ONEDNN OFF CACHE BOOL "OneDNN availability" FORCE)
set(HAVE_CUDNN OFF CACHE BOOL "cuDNN availability" FORCE)
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
# CRITICAL FOR CLEAN BUILDS: Set correct defaults based on build flags FROM THE START
# TLS relocation overflow is caused by TOO MANY FILES, not large files
# Solution: Use larger chunks with call tracing to minimize file count

# Detect build configuration and set appropriate defaults
if(DEFINED SD_GCC_FUNCTRACE AND SD_GCC_FUNCTRACE)
    # Call tracing enabled - need larger chunks to avoid TLS overflow
    if((DEFINED SD_SANITIZE AND SD_SANITIZE) OR (DEFINED SD_SANITIZERS AND NOT SD_SANITIZERS STREQUAL ""))
        # Call tracing + Sanitizers: Extreme bloat, need very large chunks
        set(CHUNK_TARGET_INSTANTIATIONS "30" CACHE STRING "Large chunks for call tracing + sanitizers")
        set(MULTI_PASS_CHUNK_SIZE "70" CACHE STRING "Large direct chunks for call tracing + sanitizers")
        message(STATUS "⚠️  Call tracing + Sanitizers: Using large chunks (30/70) to minimize file count")
    else()
        # Call tracing only: Moderate bloat, need large chunks
        set(CHUNK_TARGET_INSTANTIATIONS "40" CACHE STRING "Large chunks for call tracing")
        set(MULTI_PASS_CHUNK_SIZE "100" CACHE STRING "Large direct chunks for call tracing")
        message(STATUS "🔍 Call tracing: Using large chunks (40/100) to minimize file count")
    endif()
elseif((DEFINED SD_SANITIZE AND SD_SANITIZE) OR (DEFINED SD_SANITIZERS AND NOT SD_SANITIZERS STREQUAL ""))
    # Sanitizers only: Some bloat, use moderate chunks
    set(CHUNK_TARGET_INSTANTIATIONS "6" CACHE STRING "Moderate chunks for sanitizers")
    set(MULTI_PASS_CHUNK_SIZE "8" CACHE STRING "Moderate direct chunks for sanitizers")
else()
    # Normal builds: Use memory-based defaults
    cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
    if(AVAILABLE_MEMORY LESS 4000)
        set(CHUNK_TARGET_INSTANTIATIONS "3" CACHE STRING "Conservative chunks for low memory")
        set(MULTI_PASS_CHUNK_SIZE "25" CACHE STRING "Conservative direct chunks")
    elseif(AVAILABLE_MEMORY LESS 8000)
        set(CHUNK_TARGET_INSTANTIATIONS "6" CACHE STRING "Moderate chunks for medium memory")
        set(MULTI_PASS_CHUNK_SIZE "35" CACHE STRING "Moderate direct chunks")
    elseif(AVAILABLE_MEMORY LESS 16000)
        set(CHUNK_TARGET_INSTANTIATIONS "10" CACHE STRING "Balanced chunks for high memory")
        set(MULTI_PASS_CHUNK_SIZE "50" CACHE STRING "Balanced direct chunks")
    else()
        set(CHUNK_TARGET_INSTANTIATIONS "12" CACHE STRING "Optimized chunks for very high memory")
        set(MULTI_PASS_CHUNK_SIZE "60" CACHE STRING "Optimized direct chunks")
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