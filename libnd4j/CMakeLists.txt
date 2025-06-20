# =============================================================================
# libnd4j Main CMakeLists.txt
# =============================================================================

cmake_minimum_required(VERSION 3.15)
project(libnd4j)

# =============================================================================
# 1. SETUP CMAKE MODULE PATH & BASIC CONFIG
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    set(CMAKE_VERBOSE_MAKEFILE OFF)
endif()

# =============================================================================
# 2. INCLUDE ALL MODULAR DEFINITION FILES
# This makes all functions available in the global scope.
# The order is based on your README.
# =============================================================================

# Utility and Option modules first
include(PrintingUtilities)
include(Options)
include(PlatformDetection)
include(TypeSystem)
# Core function definition modules
include(TypeCombinationEngine)
include(TemplateProcessing)
include(TypeValidation)
include(TypeProfiles)
include(SemanticTypeFiltering)
include(Dependencies)
include(CompilerOptimizations)
include(PlatformOptimizations)

# Conditional definition includes
if(SD_CUDA)
    include(CudaConfiguration)
endif()

# =============================================================================
# 3. EXECUTE THE BUILD
# =============================================================================

# By including this file last, we execute the entire build process in the
# top-level scope, perfectly replicating the flow of the original monolithic file
# and avoiding all variable scoping issues.
include(cmake/MainBuildFlow.cmake)

# =============================================================================
# 4. OPTIONAL POST-BUILD AND ANALYSIS TARGETS
# =============================================================================

if(SD_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests_cpu)
endif()

# Development and analysis targets
if(NOT TARGET analyze_types)
    add_custom_target(analyze_types
            COMMAND ${CMAKE_COMMAND} -E echo "Analyzing type usage..."
            COMMENT "Analyzing type usage patterns in codebase"
            VERBATIM
    )
endif()