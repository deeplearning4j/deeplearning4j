# cmake/Options.cmake
# Defines all user-configurable build options and helper functions.

# --- Build Feature Options ---
option(SD_NATIVE "Optimize for build machine (might not work on others)" OFF)
option(SD_CHECK_VECTORIZATION "checks for vectorization" OFF)
option(SD_STATIC_LIB "Build static library (ignored, only shared lib is built)" OFF)
option(SD_SHARED_LIB "Build shared library (ignored, this is the default)" ON)
option(SD_USE_LTO "Use link time optimization" OFF)
option(SD_SANITIZE "Enable Address Sanitizer" OFF)

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

# --- Debug and Trace Options ---
option(SD_GCC_FUNCTRACE "Use call traces" OFF)
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
set(SD_TYPE_PROFILE "ESSENTIAL" CACHE STRING "Type profile for semantic filtering (MINIMAL, ESSENTIAL, QUANTIZATION, etc.)")
set(SD_MAX_TEMPLATE_COMBINATIONS "10000" CACHE STRING "Maximum template combinations to generate (safety limit)")

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

# --- Macro for debugging all CMake variables ---
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()