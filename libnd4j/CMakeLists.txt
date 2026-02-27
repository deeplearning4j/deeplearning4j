cmake_minimum_required(VERSION 3.15)

# ============================================================================
# CCACHE SETUP - Must be before project() for proper compiler launcher setup
# ============================================================================
# NOTE: ccache is DISABLED for debug and functrace builds due to poor hit rates
# Debug/functrace flags (-finstrument-functions, -fdebug-info-for-profiling, etc.)
# cause constant cache invalidation, making ccache overhead > benefit
# See: Build 265 analysis - 32% hit rate with 55% slower builds
find_program(CCACHE_PROGRAM ccache)
# Fallback for environments where Homebrew locations are not on PATH
if(NOT CCACHE_PROGRAM)
    set(_CCACHE_FALLBACK_PATHS
        "/home/linuxbrew/.linuxbrew/bin"  # Linuxbrew default
        "/opt/homebrew/bin"               # macOS Apple Silicon
        "/usr/local/Homebrew/bin"         # macOS Intel (alt)
        "/usr/local/bin"
    )
    find_program(CCACHE_PROGRAM ccache PATHS ${_CCACHE_FALLBACK_PATHS} NO_DEFAULT_PATH)
    unset(_CCACHE_FALLBACK_PATHS)

    if(CCACHE_PROGRAM)
        message(STATUS "ℹ️  Found ccache via fallback path search: ${CCACHE_PROGRAM}")
    endif()
endif()
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

        # IMPORTANT: Don't use sloppiness settings - they cause stale cache issues
        # Instead, we use depend_mode for accurate dependency tracking
        # and file_clone for faster cache operations on supported filesystems
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compiler_check=content ERROR_QUIET)
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=depend_mode=true ERROR_QUIET)
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=file_clone=true ERROR_QUIET)
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=inode_cache=false ERROR_QUIET)
        # Remove any old sloppiness settings that might cause stale caches
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=sloppiness= ERROR_QUIET)

        # Use FORCE to ensure cache is always updated and avoid inconsistent builds
        # On Windows CUDA builds: skip C/CXX caching to reserve all cache space for
        # expensive .cu compilations. GitHub Actions has limited cache storage and .cu
        # files take 10-100x longer to compile than .cpp files with MinGW.
        if(WIN32 AND SD_CUDA)
            set(CMAKE_C_COMPILER_LAUNCHER "" CACHE STRING "C compiler launcher" FORCE)
            set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "C++ compiler launcher" FORCE)
            message(STATUS "Windows CUDA: C/CXX ccache disabled to prioritize .cu file caching")
        else()
            set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher" FORCE)
            set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C++ compiler launcher" FORCE)
        endif()
        # CUDA launcher will be set after project() if needed
        message(STATUS "Ccache enabled with depend_mode for accurate change detection")
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

    # Disable MSVC debug information format for CUDA to prevent -Fd/-FS flags.
    # CMake wraps MSVC PDB flags in -Xcompiler=-FdPath\,-FS for CUDA objects.
    # nvcc misparses this (backslash escapes comma, -FS becomes input file).
    # CMP0141 (CMake 3.25+) enables CMAKE_MSVC_DEBUG_INFORMATION_FORMAT control.
    if(POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
    endif()
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "" CACHE STRING "" FORCE)
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

# On Windows + CUDA, CMake generates -Xcompiler=-FdPath\,-FS for CUDA objects.
# nvcc misparses this (backslash escapes comma, -FS becomes input file),
# failing with "A single input file is required".
# Fix: Override CMAKE_CUDA_COMPILE_OBJECT AFTER project() to use comma-separated
# -Xcompiler flags (no quotes needed, no path issues). The pre-project override
# gets reset by project(... LANGUAGES CUDA). Note: we check WIN32, not MSVC,
# because the C/CXX compiler is MinGW g++ but nvcc's host compiler is MSVC.
if(WIN32 AND SD_CUDA)
    # 1. Clear MSVC debug info format variables to prevent -Fd/-FS generation
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_Embedded "" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_ProgramDatabase "" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_EditAndContinue "" CACHE STRING "" FORCE)

    # 1b. Clear MSVC runtime library compile options for CUDA
    # These are set by CMake during project(LANGUAGES CUDA) and inject -Xcompiler=-MD etc.
    # configure_windows_cuda_build() clears them via PARENT_SCOPE but that only reaches
    # setup_cuda_build() scope, not the global scope. Use CACHE FORCE here to ensure
    # they're cleared globally and don't inject extra -Xcompiler flags into <FLAGS>.
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded "" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug "" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "" CACHE STRING "" FORCE)

    # 2. Disable compile PDB output directory to prevent -Fd path generation
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY "" CACHE PATH "" FORCE)
    set(CMAKE_PDB_OUTPUT_DIRECTORY "" CACHE PATH "" FORCE)

    # 2b. Disable response files for CUDA compilation
    # Ninja uses response files (.rsp) for long command lines. These can contain
    # problematic -Xcompiler flags that nvcc misparses.
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0 CACHE STRING "" FORCE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0 CACHE STRING "" FORCE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0 CACHE STRING "" FORCE)

    # 3. Clear per-config CUDA flags that may contain MSVC debug flags
    set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=/Od" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_FLAGS_RELEASE "-DNDEBUG" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-DNDEBUG" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-DNDEBUG" CACHE STRING "" FORCE)

    # 4. Fix CUDA flag variables: strip -Fd/-FS PDB flags and fix -Xcompiler quoting.
    #
    # Two problems with CMake-generated CUDA flags on Windows:
    # (a) -Xcompiler=-FdPath\,-FS: backslash-comma causes nvcc to misparse -FS as input file
    # (b) -Xcompiler=" /GR /EHsc": space inside quotes breaks Ninja .rsp file parsing —
    #     nvcc sees -Xcompiler=", /GR, /EHsc" as three tokens, interpreting /GR as input file
    # Both cause: "A single input file is required for a non-link phase"
    foreach(_cuda_flags_var
            CMAKE_CUDA_FLAGS
            CMAKE_CUDA_FLAGS_DEBUG
            CMAKE_CUDA_FLAGS_RELEASE
            CMAKE_CUDA_FLAGS_RELWITHDEBINFO
            CMAKE_CUDA_FLAGS_MINSIZEREL)
        if(DEFINED ${_cuda_flags_var})
            # Remove -Xcompiler flags containing -Fd or -FS (various formats)
            string(REGEX REPLACE "-Xcompiler[=,][^ ]*-Fd[^ ]*" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
            string(REGEX REPLACE "-Xcompiler[=,][^ ]*-FS[^ ]*" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
            string(REGEX REPLACE "-Xcompiler[=,][^ ]*/Fd[^ ]*" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
            string(REGEX REPLACE "-Xcompiler[=,][^ ]*/FS[^ ]*" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
            # Also remove bare /Fd and /FS (in case passed without -Xcompiler)
            string(REGEX REPLACE "/Fd[^ ]*" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
            string(REGEX REPLACE "/FS" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
            # Fix quoted -Xcompiler with spaces: -Xcompiler=" /GR /EHsc" → -Xcompiler=/GR,/EHsc
            # The quotes+spaces break Ninja rsp file parsing (space splits into multiple tokens)
            # IMPORTANT: Just REMOVE the entire -Xcompiler="..." since we re-add /EHsc via
            # the custom compile template (-Xcompiler=/bigobj,/EHsc,/MD).
            # /GR is the default on MSVC and doesn't need to be explicitly set.
            string(REGEX REPLACE "-Xcompiler=\" [^\"]*\"" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
            # Also try without space after =
            string(REGEX REPLACE "-Xcompiler=\"[^\"]*\"" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
            # Clean up multiple spaces
            string(REGEX REPLACE "  +" " " ${_cuda_flags_var} "${${_cuda_flags_var}}")
            string(STRIP "${${_cuda_flags_var}}" ${_cuda_flags_var})
            set(${_cuda_flags_var} "${${_cuda_flags_var}}" CACHE STRING "" FORCE)
        endif()
    endforeach()

    # 5. Custom compile rule with comma-separated -Xcompiler flags
    set(CMAKE_CUDA_COMPILE_OBJECT
            "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -Xcompiler=/bigobj,/EHsc,/MD -o <OBJECT> -c <SOURCE>"
            CACHE STRING "" FORCE)

    # Debug: dump ALL variables that could contribute to CUDA compile command
    message(STATUS "=== Windows CUDA Flag Diagnostics ===")
    message(STATUS "CMAKE_CUDA_FLAGS: [${CMAKE_CUDA_FLAGS}]")
    message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: [${CMAKE_CUDA_FLAGS_RELEASE}]")
    message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: [${CMAKE_CUDA_FLAGS_DEBUG}]")
    message(STATUS "CMAKE_CUDA_COMPILE_OBJECT: [${CMAKE_CUDA_COMPILE_OBJECT}]")
    message(STATUS "CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL: [${CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL}]")
    message(STATUS "CMAKE_CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_Embedded: [${CMAKE_CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_Embedded}]")
    message(STATUS "CMAKE_CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_ProgramDatabase: [${CMAKE_CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_ProgramDatabase}]")
    message(STATUS "CMAKE_MSVC_DEBUG_INFORMATION_FORMAT: [${CMAKE_MSVC_DEBUG_INFORMATION_FORMAT}]")
    message(STATUS "CMAKE_MSVC_RUNTIME_LIBRARY: [${CMAKE_MSVC_RUNTIME_LIBRARY}]")
    # Check all CUDA compile options vars
    get_cmake_property(_allvars VARIABLES)
    foreach(_v IN LISTS _allvars)
        if(_v MATCHES "^CMAKE_CUDA_COMPILE_OPTIONS")
            message(STATUS "${_v}: [${${_v}}]")
        endif()
    endforeach()
    message(STATUS "=== End Windows CUDA Flag Diagnostics ===")
endif()


# On Windows CUDA: use nvcc_filter.py wrapper to strip MSVC -Fd/-FS flags from
# response files. CMake's Ninja generator injects -Xcompiler=-Fd<path>\,-FS into
# .rsp files for CUDA compilation, and the backslash-comma causes nvcc to misparse
# -FS as an input file ("A single input file is required"). No CMake variable override
# can prevent this; we must filter at invocation time.
if(WIN32 AND SD_CUDA AND CMAKE_CUDA_COMPILER)
    # Try find_package first, then fall back to find_program for MSYS2/MinGW
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(NOT Python3_FOUND)
        # find_package may fail in MSYS2/MinGW — search PATH directly
        find_program(_PYTHON3_FALLBACK NAMES python3 python
            HINTS
                "C:/hostedtoolcache/windows/Python"
                "$ENV{USERPROFILE}/AppData/Local/Programs/Python"
            PATH_SUFFIXES "3.12.10/x64" "3.11/x64" "3.10/x64" ""
        )
        if(_PYTHON3_FALLBACK)
            set(Python3_EXECUTABLE "${_PYTHON3_FALLBACK}")
            set(Python3_FOUND TRUE)
            message(STATUS "Python3 found via fallback: ${Python3_EXECUTABLE}")
        endif()
    endif()

    if(Python3_FOUND)
        set(_NVCC_FILTER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc_filter.py")
        if(EXISTS "${_NVCC_FILTER}")
            if(CCACHE_PROGRAM AND NOT CCACHE_SKIP_REASON)
                # Chain ccache with nvcc_filter: filter cleans MSVC flags, ccache caches compilation
                set(CMAKE_CUDA_COMPILER_LAUNCHER "${Python3_EXECUTABLE}" "${_NVCC_FILTER}" "--ccache=${CCACHE_PROGRAM}"
                    CACHE STRING "CUDA compiler launcher (nvcc filter + ccache)" FORCE)
                message(STATUS "Windows CUDA: nvcc_filter.py + ccache enabled (${CCACHE_PROGRAM})")
            else()
                set(CMAKE_CUDA_COMPILER_LAUNCHER "${Python3_EXECUTABLE}" "${_NVCC_FILTER}"
                    CACHE STRING "CUDA compiler launcher (nvcc flag filter)" FORCE)
                message(STATUS "Windows CUDA: nvcc_filter.py wrapper enabled via ${Python3_EXECUTABLE}")
            endif()
        else()
            message(WARNING "nvcc_filter.py not found at ${_NVCC_FILTER}")
        endif()
    else()
        message(WARNING "Python3 not found - cannot enable nvcc flag filter wrapper. "
            "CUDA compilation may fail with 'A single input file is required'.")
    endif()
endif()

# Configure ccache for CUDA if available
# On Windows: ccache is chained through nvcc_filter.py (set above) to avoid
# -Xcompiler flag issues. On Linux/macOS: ccache is used directly as the launcher.
if(CCACHE_PROGRAM AND SD_CUDA AND CMAKE_CUDA_COMPILER AND NOT CCACHE_SKIP_REASON)
    if(NOT WIN32)
        # Linux/macOS: use ccache directly as CUDA compiler launcher
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher" FORCE)
        message(STATUS "Enabled ccache for CUDA compilation")
    else()
        # Windows: ccache is already chained via nvcc_filter.py --ccache=<path> above
        message(STATUS "Windows CUDA ccache: configured via nvcc_filter.py chain")
    endif()

    # Configure ccache settings for optimal CUDA caching (both platforms)
    execute_process(COMMAND ${CCACHE_PROGRAM} --max-size=10G ERROR_QUIET)
    execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compression=true ERROR_QUIET)
    execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compression_level=6 ERROR_QUIET)

    # Show ccache statistics
    execute_process(
        COMMAND ${CCACHE_PROGRAM} --show-stats
        OUTPUT_VARIABLE CCACHE_STATS
        ERROR_QUIET
    )
    message(STATUS "Ccache statistics:\n${CCACHE_STATS}")
endif()

# Enable smart ccache wrapper that forces recompile for modified files
# This solves stale cache issues when source files change
include(SmartCcache)

# Verify ccache configuration for external projects
if(CMAKE_C_COMPILER_LAUNCHER)
    if(EXISTS "${CMAKE_C_COMPILER_LAUNCHER}")
        message(STATUS "   Compiler launcher (for external projects): ${CMAKE_C_COMPILER_LAUNCHER}")
    else()
        message(WARNING "   Compiler launcher path does not exist: ${CMAKE_C_COMPILER_LAUNCHER}")
    endif()
endif()

if(NOT SD_CUDA AND NOT SD_TPU)
    if(NOT SD_CPU)
        set(SD_CUDA FALSE)
        set(SD_TPU FALSE)
        set(SD_CPU TRUE)
    endif()
endif()

if(SD_CUDA)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CUDA)
elseif(SD_TPU)
    set(DEFAULT_ENGINE "samediff::ENGINE_TPU")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_TPU)
    add_compile_definitions(SD_TPU=1)
    add_compile_definitions(HAVE_PJRT=1)
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CPU)
endif()

if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    elseif(SD_TPU)
        set(SD_LIBRARY_NAME nd4jtpu)
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
    # CUDA cleanup hooks - clean temp files at configure time and create build targets
    include(cmake/CudaCleanup.cmake)
    cuda_create_cleanup_target()
elseif(SD_TPU)
    include(cmake/TpuConfiguration.cmake)
    include(cmake/BuildTPU.cmake)
    setup_tpu_build()
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

# =============================================================================
# CUDA CLEANUP HOOKS - Connect pre-build cleanup to main library target
# This ensures CUDA temp files are cleaned before each build starts
# =============================================================================
if(SD_CUDA AND TARGET cuda_cleanup_pre_build AND TARGET ${SD_LIBRARY_NAME})
    # Add pre-build cleanup as a dependency of the main library
    add_dependencies(${SD_LIBRARY_NAME} cuda_cleanup_pre_build)
    message(STATUS "✅ CUDA cleanup hook connected to ${SD_LIBRARY_NAME}")
endif()
