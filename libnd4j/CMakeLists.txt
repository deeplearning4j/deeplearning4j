cmake_minimum_required(VERSION 3.15)

# ============================================================================
# STALE MAKEFILE DETECTION AND CLEANUP
# ============================================================================
# CMake's progress tracking can become inconsistent when template-generated
# files change between builds, causing build failures with "No rule to make target"
# This check forces a clean reconfiguration if stale Makefiles are detected
if(EXISTS "${CMAKE_BINARY_DIR}/CMakeFiles/progress.marks")
    file(READ "${CMAKE_BINARY_DIR}/CMakeFiles/progress.marks" PROGRESS_MARKS)
    string(STRIP "${PROGRESS_MARKS}" PROGRESS_MARKS)

    # If progress marks is suspiciously low (99 or less), force reconfiguration
    # Real builds should have much higher counts with template instantiations
    if(PROGRESS_MARKS LESS 200)
        message(STATUS "⚠️  Detected stale CMake progress tracking (only ${PROGRESS_MARKS} marks)")
        message(STATUS "   Forcing clean reconfiguration to fix Makefile consistency...")

        # Delete all CMake-generated Makefiles and progress files
        file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/CMakeFiles/Makefile2")
        file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/CMakeFiles/progress.marks")
        file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/CMakeFiles/nd4jcpu_object.dir/progress.make")
        file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/CMakeFiles/nd4jcpu_object.dir/build.make")
        file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/CMakeCache.txt")

        message(FATAL_ERROR "Stale Makefiles deleted. Please re-run cmake to regenerate build system.")
    endif()
endif()

# ============================================================================
# CCACHE SETUP - Must be before project() for proper compiler launcher setup
# ============================================================================
# NOTE: ccache is DISABLED for debug and functrace builds due to poor hit rates
# Debug/functrace flags (-finstrument-functions, -fdebug-info-for-profiling, etc.)
# cause constant cache invalidation, making ccache overhead > benefit
# See: Build 265 analysis - 32% hit rate with 55% slower builds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    # Check if this is a debug or functrace build
    set(CCACHE_SKIP_REASON "")

    # Note: CMAKE_BUILD_TYPE might not be set yet, so check cache variables too
    if(DEFINED SD_GCC_FUNCTRACE AND SD_GCC_FUNCTRACE)
        set(CCACHE_SKIP_REASON "functrace build (instrumentation flags hurt cache hit rate)")
    elseif(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
        set(CCACHE_SKIP_REASON "debug build (debug info flags hurt cache hit rate)")
    endif()

    if(CCACHE_SKIP_REASON)
        message(STATUS "⚠️  Ccache DISABLED for ${CCACHE_SKIP_REASON}")
        message(STATUS "   Reason: Debug/functrace flags cause 60-70% cache miss rate, making ccache slower than no cache")
        set(CMAKE_C_COMPILER_LAUNCHER "" CACHE STRING "C compiler launcher" FORCE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "C++ compiler launcher" FORCE)
    else()
        message(STATUS "✅ Found ccache: ${CCACHE_PROGRAM}")

        # Configure ccache to avoid race conditions with directory creation
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compiler_check=mtime ERROR_QUIET)
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=sloppiness=include_file_mtime,include_file_ctime ERROR_QUIET)

        # Use FORCE to ensure cache is always updated and avoid inconsistent builds
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher" FORCE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C++ compiler launcher" FORCE)
        # CUDA launcher will be set after project() if needed
        message(STATUS "✅ Ccache enabled - builds will be significantly faster after first compilation")
    endif()
else()
    message(STATUS "ℹ️  Ccache not found - install with: sudo apt-get install ccache (or brew install ccache)")
    # Explicitly unset launcher if ccache not found
    set(CMAKE_C_COMPILER_LAUNCHER "" CACHE STRING "C compiler launcher" FORCE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "C++ compiler launcher" FORCE)
endif()

if(WIN32 AND SD_CUDA)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    set(CMAKE_VERBOSE_MAKEFILE OFF)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Disable response files and dependencies
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_DEPFILE_FORMAT "" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_DEPENDS_USE_COMPILER OFF CACHE BOOL "" FORCE)

    set(CMAKE_CUDA_COMPILE_OBJECT
            "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE> --compiler-options \"/bigobj /EHsc /MD /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH\""
            CACHE STRING "" FORCE)
endif()

if(SD_CUDA)
    project(libnd4j LANGUAGES C CXX CUDA)
else()
    project(libnd4j LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


# Configure ccache for CUDA if available
if(CCACHE_PROGRAM AND SD_CUDA AND CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher" FORCE)
    message(STATUS "✅ Enabled ccache for CUDA compilation")

    # Configure ccache settings for optimal performance
    execute_process(COMMAND ${CCACHE_PROGRAM} --max-size=10G ERROR_QUIET)
    execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compression=true ERROR_QUIET)
    execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compression_level=6 ERROR_QUIET)

    # Show ccache statistics
    execute_process(
        COMMAND ${CCACHE_PROGRAM} --show-stats
        OUTPUT_VARIABLE CCACHE_STATS
        ERROR_QUIET
    )
    message(STATUS "📊 Ccache statistics:\n${CCACHE_STATS}")
endif()


if(NOT SD_CUDA)
    if(NOT SD_CPU)
        set(SD_CUDA FALSE)
        set(SD_CPU TRUE)
    endif()
endif()

if(SD_CUDA)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CUDA)
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CPU)
endif()

if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
    endif()
endif()

include(PrintingUtilities)
include(Options)
include(PlatformDetection)
include(TypeSystem)
include(TypeCombinationEngine)
include(TypeValidation)
include(TypeProfiles)
include(SemanticTypeFiltering)
include(SelectiveRenderingIntegration)
include(Dependencies)
include(CompilerOptimizations)
include(PlatformOptimizations)

LIBND4J_SETUP_TYPE_VALIDATION()

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

get_directory_property(COMPILE_DEFS COMPILE_DEFINITIONS)
message(STATUS "Some compile definitions: ${COMPILE_DEFS}")
message(STATUS "==============================")

# Include JNI configuration BEFORE MainBuildFlow so JVM_LIBRARY is available for linking
include(cmake/JNIConfiguration.cmake)

include(cmake/MainBuildFlow.cmake)

if(SD_CUDA)
    include(cmake/CudaConfiguration.cmake)
    setup_cuda_build()
else()
    setup_cpu_environment()
endif()

if(WIN32 AND SD_CUDA)
    function(apply_cuda_target_fixes)
        get_property(all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target ${all_targets})
            if(TARGET ${target})
                get_target_property(target_type ${target} TYPE)
                if(target_type STREQUAL "OBJECT_LIBRARY" OR target_type STREQUAL "SHARED_LIBRARY")
                    get_target_property(compile_options ${target} COMPILE_OPTIONS)
                    if(compile_options)
                        list(FILTER compile_options EXCLUDE REGEX "/FS")
                        list(FILTER compile_options EXCLUDE REGEX "/Fd")
                        set_target_properties(${target} PROPERTIES COMPILE_OPTIONS "${compile_options}")
                    endif()
                    set_property(TARGET ${target} PROPERTY CUDA_FLAGS "")
                    set_target_properties(${target} PROPERTIES
                            CUDA_RESOLVE_DEVICE_SYMBOLS ON
                            CUDA_SEPARABLE_COMPILATION ON
                            MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
                    )
                endif()
            endif()
        endforeach()
    endfunction()
    apply_cuda_target_fixes()
endif()

if(NOT TARGET analyze_types)
    add_custom_target(analyze_types
            COMMAND ${CMAKE_COMMAND} -E echo "Analyzing type usage..."
            COMMENT "Analyzing type usage patterns in codebase"
            VERBATIM
    )
endif()

add_custom_target(verify_types
        COMMAND ${CMAKE_COMMAND} -E echo "Verifying type configuration..."
        COMMAND ${CMAKE_COMMAND} -E echo "Check build logs above for type setup verification"
        COMMENT "Verify that type definitions are properly set"
        VERBATIM
)

if(BUILD_PPSTEP)
include(cmake/Ppstep.cmake)
endif()