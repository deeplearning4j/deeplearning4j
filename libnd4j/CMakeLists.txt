# =============================================================================
# libnd4j Main CMakeLists.txt
# =============================================================================

cmake_minimum_required(VERSION 3.15)
if(SD_CUDA)
project(libnd4j LANGUAGES CXX CUDA)
else()
    project(libnd4j LANGUAGES CXX)

endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    set(CMAKE_VERBOSE_MAKEFILE OFF)
endif()

# =============================================================================
# 1. EARLY SETUP - Set DEFAULT_ENGINE before anything else
# =============================================================================

# Ensure SD_CPU is TRUE if neither SD_CUDA nor SD_CPU is set
if(NOT SD_CUDA)
    if(NOT SD_CPU)
        set(SD_CUDA FALSE)
        set(SD_CPU TRUE)
    endif()
endif()

# Set DEFAULT_ENGINE early - this MUST happen before config.h generation
if(SD_CUDA)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CUDA)
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CPU)
endif()

# Set library name
if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
    endif()
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
include(TypeValidation)
include(TypeProfiles)
include(SemanticTypeFiltering)
include(SelectiveRenderingIntegration)
include(Dependencies)
include(CompilerOptimizations)
include(PlatformOptimizations)

# =============================================================================
# 3. EARLY CUDA SETUP (if needed)
# =============================================================================

# =============================================================================
# 4. EXECUTE THE BUILD
# =============================================================================
# By including this file last, we execute the entire build process in the
# top-level scope, perfectly replicating the flow of the original monolithic file
# and avoiding all variable scoping issues.
include(cmake/MainBuildFlow.cmake)
if(SD_CUDA)
    # Include CUDA configuration early for architecture setup
    include(cmake/CudaConfiguration.cmake)
    setup_cuda_architectures_early()
    setup_cuda_build()
else()
    setup_cpu_environment()
endif()

# =============================================================================
# 5. OPTIONAL POST-BUILD AND ANALYSIS TARGETS
# =============================================================================
# Development and analysis targets
if(NOT TARGET analyze_types)
    add_custom_target(analyze_types
            COMMAND ${CMAKE_COMMAND} -E echo "Analyzing type usage..."
            COMMENT "Analyzing type usage patterns in codebase"
            VERBATIM
    )
endif()
