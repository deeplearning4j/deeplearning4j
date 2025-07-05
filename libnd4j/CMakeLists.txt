# =============================================================================
# libnd4j Main CMakeLists.txt - FIXED VERSION
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



# Call the type validation setup macro - THIS WAS MISSING!
LIBND4J_SETUP_TYPE_VALIDATION()

# Alternative: If the macro doesn't work, call the function directly
# libnd4j_validate_and_setup_types()

# Debug: Show what types were configured
message(STATUS "=== TYPE SETUP VERIFICATION ===")
if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST)
    message(STATUS "SD_TYPES_LIST: ${SD_TYPES_LIST}")
    list(LENGTH SD_TYPES_LIST type_count)
    message(STATUS "Type count: ${type_count}")
else()
    message(STATUS "SD_TYPES_LIST: Not defined (ALL TYPES mode)")
endif()

if(DEFINED SD_TYPES_LIST_COUNT)
    message(STATUS "SD_TYPES_LIST_COUNT: ${SD_TYPES_LIST_COUNT}")
endif()

# Show some key compile definitions that should be set
get_directory_property(COMPILE_DEFS COMPILE_DEFINITIONS)
message(STATUS "Some compile definitions: ${COMPILE_DEFS}")
message(STATUS "==============================")

# =============================================================================
# 4. EARLY CUDA SETUP (if needed)
# =============================================================================


# =============================================================================
# 5. EXECUTE THE BUILD
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
# 6. OPTIONAL POST-BUILD AND ANALYSIS TARGETS
# =============================================================================
# Development and analysis targets
if(NOT TARGET analyze_types)
    add_custom_target(analyze_types
            COMMAND ${CMAKE_COMMAND} -E echo "Analyzing type usage..."
            COMMENT "Analyzing type usage patterns in codebase"
            VERBATIM
    )
endif()

# Add a target to verify type configuration
add_custom_target(verify_types
        COMMAND ${CMAKE_COMMAND} -E echo "Verifying type configuration..."
        COMMAND ${CMAKE_COMMAND} -E echo "Check build logs above for type setup verification"
        COMMENT "Verify that type definitions are properly set"
        VERBATIM
)