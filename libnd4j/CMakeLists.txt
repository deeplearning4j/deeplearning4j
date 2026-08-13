cmake_minimum_required(VERSION 3.15)

# ============================================================================
# COMPILER CACHE SETUP - Must be before project() for proper launcher setup
# ============================================================================
option(SD_REQUIRE_COMPILER_CACHE
       "Fail configuration when the requested compiler cache is unavailable"
       OFF)

# Supports two cache backends:
#   1. sccache (preferred for CI) — per-object distributed caching via GHA backend
#   2. ccache  (fallback/local)  — directory-based caching
#
# When SD_USE_SCCACHE=1 is set (by setup-sccache-* CI actions), sccache is used.
# Otherwise falls back to ccache detection.

# Check for sccache first (set by CI actions via SD_USE_SCCACHE env var)
set(SD_USE_SCCACHE_VAL "$ENV{SD_USE_SCCACHE}")
if(SD_USE_SCCACHE_VAL)
    find_program(SCCACHE_PROGRAM sccache)
    if(NOT SCCACHE_PROGRAM AND (WIN32 OR CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows"))
        find_program(SCCACHE_PROGRAM sccache.exe PATHS "$ENV{USERPROFILE}/sccache" NO_DEFAULT_PATH)
    endif()
endif()

if(SCCACHE_PROGRAM)
    message(STATUS "✅ Found sccache: ${SCCACHE_PROGRAM} (distributed per-object caching)")

    # sccache handles all compiler types uniformly — no special Windows CUDA carve-out needed.
    # Unlike ccache, sccache natively handles nvcc response files and doesn't need
    # the nvcc_filter.py wrapper or cache size partitioning between C++ and CUDA.
    set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "C compiler launcher" FORCE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "C++ compiler launcher" FORCE)
    # CUDA launcher will be set after project() if needed
    set(CCACHE_SKIP_REASON "")  # Not skipped — sccache is active
    set(SD_COMPILER_CACHE "sccache")
else()
    # Fall back to ccache
    # NOTE: ccache is DISABLED for debug and functrace builds due to poor hit rates
    # Debug/functrace flags (-finstrument-functions, -fdebug-info-for-profiling, etc.)
    # cause constant cache invalidation, making ccache overhead > benefit
    # See: Build 265 analysis - 32% hit rate with 55% slower builds
    find_program(CCACHE_PROGRAM ccache)
    # Windows: also try ccache.exe (find_program may need explicit name on some generators)
    if(NOT CCACHE_PROGRAM AND (WIN32 OR CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows"))
        find_program(CCACHE_PROGRAM ccache.exe)
    endif()
    # Fallback for environments where ccache is not on PATH
    if(NOT CCACHE_PROGRAM)
        set(_CCACHE_FALLBACK_PATHS
            "/home/linuxbrew/.linuxbrew/bin"  # Linuxbrew default
            "/opt/homebrew/bin"               # macOS Apple Silicon
            "/usr/local/Homebrew/bin"         # macOS Intel (alt)
            "/usr/local/bin"
        )
        # Windows: GitHub Actions setup-ccache-windows installs to %USERPROFILE%\ccache
        if(WIN32 OR CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
            list(APPEND _CCACHE_FALLBACK_PATHS "$ENV{USERPROFILE}/ccache")
        endif()
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

            # Use compiler_check=content for reliable cache hits across different build dirs.
            # file_clone speeds up cache operations on supported filesystems.
            # IMPORTANT: Do NOT clear sloppiness or override depend_mode here — CI actions
            # configure these settings before CMake runs. Overriding them causes poor hit rates
            # (43% instead of 80%+) because nvcc doesn't support depend_mode well.
            execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compiler_check=content ERROR_QUIET)
            # NOTE: Do NOT set file_clone here — CI actions configure it to false.
            # file_clone=true causes issues on CI runners with different filesystem layouts.

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
            message(STATUS "Ccache enabled (sloppiness/depend_mode deferred to CI action config)")
        endif()
        set(SD_COMPILER_CACHE "ccache")
    else()
        message(STATUS "ℹ️  No compiler cache found - install ccache or sccache for faster builds")
        # Explicitly unset launcher if no cache found
        set(CMAKE_C_COMPILER_LAUNCHER "" CACHE STRING "C compiler launcher" FORCE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "C++ compiler launcher" FORCE)
        set(SD_COMPILER_CACHE "none")
    endif()
endif()

if(SD_REQUIRE_COMPILER_CACHE AND
   (NOT CMAKE_C_COMPILER_LAUNCHER OR NOT CMAKE_CXX_COMPILER_LAUNCHER))
    message(FATAL_ERROR
        "SD_REQUIRE_COMPILER_CACHE=ON, but C/CXX compiler caching is unavailable")
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
elseif(CMAKE_HOST_APPLE OR APPLE)
    # Objective-C++ must be enabled at directory scope. Enabling it later from
    # setup_mps() confines CMake's internal compile rules to the function scope.
    project(libnd4j LANGUAGES C CXX OBJCXX)
else()
    project(libnd4j LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Keep parser-facing generated headers inside this build tree. CPU, CUDA, Vulkan,
# and cross-compiled builds may configure concurrently and must never overwrite a
# shared source-tree include_ops.h.
set(OP_OUTPUT_FILE "${CMAKE_BINARY_DIR}/include/generated/include_ops.h" CACHE FILEPATH
    "Per-build generated operation-selection header")

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
    set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=-Od" CACHE STRING "" FORCE)
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
            # IMPORTANT: Just REMOVE the entire -Xcompiler="..." since we re-add -EHsc via
            # the custom compile template (-Xcompiler=-bigobj -Xcompiler=-EHsc -Xcompiler=-MD).
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

    # 5. Custom compile rule with hyphen-prefixed MSVC flags.
    # IMPORTANT: Use -bigobj (not /bigobj) for all -Xcompiler flags.
    # sccache wraps nvcc and decomposes the command line. Slash-prefixed MSVC flags
    # like /bigobj get interpreted as drive-relative paths (D:\bigobj) on Windows.
    # MSVC cl.exe accepts both / and - as option prefixes, so -bigobj == /bigobj.
    # Hyphen prefix is sccache-safe and works with plain nvcc too.
    set(CMAKE_CUDA_COMPILE_OBJECT
            "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -Xcompiler=-bigobj -Xcompiler=-EHsc -Xcompiler=-MD -o <OBJECT> -c <SOURCE>"
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


# ============================================================================
# CUDA COMPILER LAUNCHER SETUP (after project())
# ============================================================================
# sccache handles nvcc natively — no nvcc_filter.py wrapper needed.
# ccache needs nvcc_filter.py on Windows to expand response files.
if(SCCACHE_PROGRAM AND SD_CUDA AND CMAKE_CUDA_COMPILER AND NOT WIN32)
    # sccache handles nvcc natively on Linux/macOS, including response files.
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher (sccache)" FORCE)
    message(STATUS "sccache enabled for CUDA compilation (no nvcc_filter.py needed)")

elseif(WIN32 AND SD_CUDA AND CMAKE_CUDA_COMPILER AND SCCACHE_PROGRAM)
    # On Windows, sccache wrapping nvcc directly breaks tmpxft_* stderr redirect.
    # Route through nvcc_filter.py (same approach as ccache on Windows) which expands
    # response files into direct args, preventing the temp file path issue.
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(NOT Python3_FOUND)
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

    set(_NVCC_FILTER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc_filter.py")
    if(Python3_FOUND AND EXISTS "${_NVCC_FILTER}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${Python3_EXECUTABLE}" "${_NVCC_FILTER}" "--sccache=${SCCACHE_PROGRAM}"
            CACHE STRING "CUDA compiler launcher (nvcc filter + sccache)" FORCE)
        set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
        set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
        message(STATUS "Windows CUDA: nvcc_filter.py + sccache enabled (${SCCACHE_PROGRAM})")
    else()
        # Fallback: use sccache directly (may have tmpxft issues but still better than no caching)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher (sccache)" FORCE)
        message(STATUS "Windows CUDA: sccache direct (nvcc_filter.py not available)")
    endif()

elseif(WIN32 AND SD_CUDA AND CMAKE_CUDA_COMPILER AND NOT SCCACHE_PROGRAM)
    # On Windows CUDA: use nvcc_filter.py wrapper to strip MSVC -Fd/-FS flags from
    # response files. CMake's Ninja generator injects -Xcompiler=-Fd<path>\,-FS into
    # .rsp files for CUDA compilation, and the backslash-comma causes nvcc to misparse
    # -FS as an input file ("A single input file is required"). No CMake variable override
    # can prevent this; we must filter at invocation time.

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
            # Chain ccache with nvcc via nvcc_filter.py. The filter expands response
            # files into direct args when ccache is present, so ccache can properly
            # identify source files for caching.
            if(CCACHE_PROGRAM AND NOT CCACHE_SKIP_REASON)
                set(CMAKE_CUDA_COMPILER_LAUNCHER "${Python3_EXECUTABLE}" "${_NVCC_FILTER}" "--ccache=${CCACHE_PROGRAM}"
                    CACHE STRING "CUDA compiler launcher (nvcc filter + ccache)" FORCE)
                # Disable response files for CUDA includes — ccache can't parse them.
                # This forces CMake to put include dirs on the command line directly.
                set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
                set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
                # Tell ccache this is nvcc (helps with arg parsing)
                set(ENV{CCACHE_COMPILERTYPE} "nvcc")
                message(STATUS "Windows CUDA: nvcc_filter.py + ccache enabled (${CCACHE_PROGRAM})")
            else()
                set(CMAKE_CUDA_COMPILER_LAUNCHER "${Python3_EXECUTABLE}" "${_NVCC_FILTER}"
                    CACHE STRING "CUDA compiler launcher (nvcc flag filter)" FORCE)
                message(STATUS "Windows CUDA: nvcc_filter.py wrapper enabled (no ccache)")
            endif()
        else()
            message(WARNING "nvcc_filter.py not found at ${_NVCC_FILTER}")
        endif()
    else()
        message(WARNING "Python3 not found - cannot enable nvcc flag filter wrapper. "
            "CUDA compilation may fail with 'A single input file is required'.")
    endif()
endif()

# Configure ccache for CUDA if available (non-Windows, non-sccache path)
# On Windows: ccache is chained through nvcc_filter.py (set above) to avoid
# -Xcompiler flag issues. On Linux/macOS: ccache is used directly as the launcher.
if(CCACHE_PROGRAM AND NOT SCCACHE_PROGRAM AND SD_CUDA AND CMAKE_CUDA_COMPILER AND NOT CCACHE_SKIP_REASON)
    if(NOT WIN32)
        # Linux/macOS: use ccache directly as CUDA compiler launcher
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher" FORCE)
        message(STATUS "Enabled ccache for CUDA compilation")
    else()
        # Windows: ccache for CUDA is configured via nvcc_filter.py --ccache= above.
        # nvcc_filter.py expands response files so ccache can identify source files.
        message(STATUS "Windows CUDA ccache: configured via nvcc_filter.py chain")
    endif()

    # NOTE: Do NOT override max_size or compression_level here.
    # CI actions (setup-ccache-linux/windows/macos) configure these before CMake runs.
    # Previously this set max_size=30G (conflicting with CI's 8G) and compression_level=6
    # (conflicting with CI's level 1). GitHub Actions cache storage is limited, and the
    # CI action settings are tuned for that environment.
    execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compression=true ERROR_QUIET)

    # Show ccache statistics
    execute_process(
        COMMAND ${CCACHE_PROGRAM} --show-stats
        OUTPUT_VARIABLE CCACHE_STATS
        ERROR_QUIET
    )
    message(STATUS "Ccache statistics:\n${CCACHE_STATS}")
endif()

# Enable smart ccache wrapper that forces recompile for modified files
# This solves stale cache issues when source files change (only for ccache, not sccache)
if(NOT SCCACHE_PROGRAM)
    include(SmartCcache)
endif()

# Verify compiler cache configuration for external projects
if(CMAKE_C_COMPILER_LAUNCHER)
    message(STATUS "   Compiler cache: ${SD_COMPILER_CACHE}")
    message(STATUS "   C/CXX launcher: ${CMAKE_C_COMPILER_LAUNCHER}")
    if(SD_CUDA AND CMAKE_CUDA_COMPILER_LAUNCHER)
        message(STATUS "   CUDA launcher: ${CMAKE_CUDA_COMPILER_LAUNCHER}")
    endif()
endif()

if(NOT SD_CUDA AND NOT SD_TPU AND NOT SD_HEXAGON AND NOT SD_VULKAN)
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
elseif(SD_HEXAGON)
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CPU)
    add_compile_definitions(SD_HEXAGON=1)
    add_compile_definitions(HAVE_HEXAGON_MLIR=1)
elseif(SD_VULKAN)
    # Vulkan is a device backend. DSP native plans dispatch admitted operations
    # through VulkanReplayHandle; an operation without a Vulkan device path must
    # fail instead of entering the generic CPU implementation.
    set(DEFAULT_ENGINE "samediff::ENGINE_VULKAN")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_VULKAN)
    add_compile_definitions(SD_VULKAN=1)
    # The MLIR SPIR-V compiler is optional. Development builds opt in with
    # HELPERS_mlir=ON and MLIR_ENABLE_VULKAN=ON; mobile/runtime builds load
    # bundled AOT SPIR-V and must remain compiler-free.
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CPU)
endif()

if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    elseif(SD_TPU)
        set(SD_LIBRARY_NAME nd4jtpu)
    elseif(SD_HEXAGON)
        set(SD_LIBRARY_NAME nd4jhexagon)
    elseif(SD_VULKAN)
        set(SD_LIBRARY_NAME nd4jvulkan)
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

# A fresh Triton+MLIR build cannot discover MLIR package exports until the
# project-managed, patched compiler stack has been built. The wrapper performs
# this bootstrap configure/build before its normal configure, using the same
# target toolchain and cache identity as the final build.
if(SD_DEPENDENCY_BOOTSTRAP_ONLY)
    if(NOT SD_TRITON OR NOT HELPERS_mlir)
        message(FATAL_ERROR
            "SD_DEPENDENCY_BOOTSTRAP_ONLY requires SD_TRITON=ON and HELPERS_mlir=ON")
    endif()
    setup_triton()
    if(NOT TARGET triton_external)
        message(FATAL_ERROR
            "Compiler dependency bootstrap did not create triton_external")
    endif()
    add_custom_target(sd_compiler_dependencies_bootstrap
        DEPENDS triton_external)
    message(STATUS
        "Configured project-managed Triton/LLVM/MLIR dependency bootstrap only")
    return()
endif()

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
elseif(SD_HEXAGON)
    include(cmake/HexagonConfiguration.cmake)
    setup_hexagon_paths()
    include(cmake/BuildHexagon.cmake)
    setup_hexagon_build()
elseif(SD_VULKAN)
    include(cmake/VulkanConfiguration.cmake)
    verify_vulkan_chip_requirements()
    include(cmake/BuildVulkan.cmake)
    setup_vulkan_build()
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
# GOOGLE TEST INTEGRATION
# =============================================================================
if(SD_BUILD_TESTS)
    message(STATUS "Building libnd4j tests (SD_BUILD_TESTS=ON)")
    enable_testing()
    find_program(SD_TEST_PATCHELF_EXECUTABLE NAMES patchelf)
    add_test(
        NAME zluda_windows_runtime_contract
        COMMAND ${CMAKE_COMMAND}
            "-DLIBND4J_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/ZludaWindowsRuntimeContractTest.cmake")
    add_test(
        NAME shared_runtime_alias_contract
        COMMAND ${CMAKE_COMMAND}
            "-DLIBND4J_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DTEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DTEST_READELF=${CMAKE_READELF}"
            "-DTEST_PATCHELF=${SD_TEST_PATCHELF_EXECUTABLE}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/SharedRuntimeAliasContractTest.cmake")
    add_test(
        NAME cublas_lazy_solver_contract
        COMMAND ${CMAKE_COMMAND}
            "-DLIBND4J_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/CublasLazySolverContractTest.cmake")
    add_subdirectory(tests_cpu)
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
