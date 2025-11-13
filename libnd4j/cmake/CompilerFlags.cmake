# cmake/CompilerFlags.cmake
# Configures compiler and linker flags for optimization and correctness.

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # CRITICAL: -fno-plt is incompatible with large binaries using -mcmodel=large
    # When sanitizers or lifecycle tracking are enabled, we use large code model which can cause PLT overflow
    # Skip -fno-plt for these builds to avoid "PC-relative offset overflow" linker errors
    if(NOT ((DEFINED SD_SANITIZERS AND NOT SD_SANITIZERS STREQUAL "") OR SD_GCC_FUNCTRACE))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-plt")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-plt")
    else()
        if(DEFINED SD_SANITIZERS AND NOT SD_SANITIZERS STREQUAL "")
            message(STATUS "Skipping -fno-plt for sanitizer build (incompatible with large code model)")
        elseif(SD_GCC_FUNCTRACE)
            message(STATUS "Skipping -fno-plt for lifecycle tracking build (incompatible with large code model)")
        endif()
    endif()
    #This is to avoid jemalloc crashes where c++ uses sized deallocations
    add_compile_options(-fno-sized-deallocation)
endif()


# Template depth limits - standardized based on build type
# Release builds: 512 (sufficient for production, faster compilation)
# Debug builds: 1024 (deeper nesting for development)
# NOTE: Actual values set in Options.cmake based on build type

# GCC-specific error limiting
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fmax-errors=3)        # Stop on first few errors
endif()

# Clang-specific workarounds for source location limits
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Work around Clang's source location limit with heavily templated code
    add_compile_options(-Wno-error)
    add_compile_options(-ferror-limit=0)
    # Reduce macro expansion tracking overhead
    add_compile_options(-fmacro-backtrace-limit=0)
    # Reduce template error backtrace noise
    add_compile_options(-ftemplate-backtrace-limit=10)
    # Use newer source manager if available (Clang 15+)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "15.0")
        message(STATUS "Clang 15+ detected, using optimizations for large translation units")
    endif()
endif()



# --- Link Time Optimization (LTO) ---
if(SD_USE_LTO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Using Link Time Optimization")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

# --- Memory Model for large binaries ---
# Note: With sanitizers or lifecycle tracking enabled, we need large model to avoid PLT entry overflow
# Lifecycle tracking (SD_GCC_FUNCTRACE) adds significant instrumentation code, increasing binary size
# Without these features, medium model is sufficient
if(SD_X86_BUILD AND NOT WIN32)
    if(SD_SANITIZE OR SD_GCC_FUNCTRACE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=large -fPIC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=large")
        if(SD_SANITIZE)
            message(STATUS "Applied large memory model for x86-64 architecture (sanitizers enabled)")
        elseif(SD_GCC_FUNCTRACE)
            message(STATUS "Applied large memory model for x86-64 architecture (lifecycle tracking enabled)")
        endif()
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium -fPIC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=medium")
        message(STATUS "Applied medium memory model for x86-64 architecture")
    endif()
else()
    if(SD_ARM_BUILD OR SD_ANDROID_BUILD)
        message(STATUS "Skipping large memory model for ARM/Android architecture (not supported)")
    elseif(WIN32)
        message(STATUS "Skipping large memory model for Windows (using alternative approach)")
    endif()
endif()

# --- Section splitting for better linker handling ---
# Note: Memory optimization params (ggc-min-*) are set below in GCC-specific section to avoid duplicates
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
endif()

# --- Allow duplicate instantiations for template folding ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC: Use -fpermissive to allow duplicate instantiations
    add_compile_options(-fpermissive)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang: Suppress all warnings and errors for template instantiation issues
    add_compile_options(-w)
    add_compile_options(-Wno-error)
    add_compile_options(-Wno-everything)
        message(STATUS "✅ Clang: Enabled template folding with all warnings suppressed and dead strip")
endif()

# --- MSVC-specific optimizations ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/Gy)  # Function-level linking
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /OPT:REF /OPT:ICF")
    add_compile_options(/bigobj /EHsc /Zc:preprocessor)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# --- Windows Specific Configurations ---
if(WIN32 AND NOT ANDROID)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()
    set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE INTERNAL "")
endif()

# --- GCC/Clang Specific Flags ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT SD_CUDA)
    message(STATUS "Adding GCC memory optimization flags: --param ggc-min-expand=100 --param ggc-min-heapsize=131072")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=100 --param ggc-min-heapsize=131072 ${INFORMATIVE_FLAGS} -std=c++17 -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --param ggc-min-expand=100 --param ggc-min-heapsize=131072 -fPIC")
    if(UNIX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose")
    endif()
endif()

# --- Build Type Specific Flags ---
if(SD_ANDROID_BUILD)
    # ... flags for android ...
elseif(APPLE)
    # ... flags for apple ...
elseif(WIN32)
    # ... flags for windows ...
else() # Generic Linux/Unix
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -g")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")
endif()

# --- Sanitizer Configuration ---
# In CompilerFlags.cmake, change the sanitizer section:
# --- Sanitizer Configuration ---
if(SD_SANITIZE)
    # For large binaries with MSan: must use large code model
    # Use LLD linker for better memory handling (gold OOMs on 23GB+ builds)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # MEMORY OPTIMIZATION: Use -gline-tables-only instead of full debug info
        # This reduces memory by 40-60% for template-heavy code while maintaining stack traces
        set(SANITIZE_FLAGS " -fPIC -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all -fuse-ld=gold -gline-tables-only")

        # MemorySanitizer-specific: Use ignorelist to skip instrumentation of external libraries
        if(SD_SANITIZERS MATCHES "memory")
            set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fsanitize-ignorelist=${CMAKE_CURRENT_SOURCE_DIR}/msan_ignorelist.txt")
            message(STATUS "Applied MemorySanitizer ignorelist for external libraries")
        endif()

        # CRITICAL: For shared libraries with MSan, use initial-exec TLS model
        # Default local-exec model creates R_X86_64_TPOFF32 relocations incompatible with -shared
        # initial-exec uses R_X86_64_GOTTPOFF which works with shared libraries
        if(SD_SANITIZERS MATCHES "memory")
            set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -ftls-model=initial-exec")
            message(STATUS "Applied initial-exec TLS model for MemorySanitizer shared library compatibility")
        endif()

        # Additional memory optimizations for template-heavy instantiation files
        set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fmerge-all-constants")
        set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fno-unique-section-names")

        message(STATUS "Applied memory-optimized sanitizer flags (-gline-tables-only)")
    else()
        set(SANITIZE_FLAGS " -Wall -Wextra -fPIC -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all")
    endif()
    # Gold linker + sanitizers: Must explicitly pass -fsanitize to linker
    # The -fsanitize flag tells clang driver to link the sanitizer runtime
    # Gold linker handles MSan's TLS correctly for shared libraries (LLD has issues)
    set(SANITIZE_LINK_FLAGS "-fsanitize=${SD_SANITIZERS} -fuse-ld=gold")

    # CRITICAL: Pass code model to linker to match compiler flags
    # Without this, linker uses wrong relocation types → "relocation truncated to fit" errors
    if(SD_X86_BUILD AND NOT WIN32)
        if(DEFINED SD_SANITIZERS AND NOT SD_SANITIZERS STREQUAL "")
            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -mcmodel=large")
            message(STATUS "Applied large code model to linker flags (matches compiler -mcmodel=large)")
        else()
            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -mcmodel=medium")
        endif()
    endif()

    # Linker memory optimizations for large template-heavy builds
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Detect which linker is being used
        set(DETECTED_LINKER "unknown")
        if(SANITIZE_LINK_FLAGS MATCHES "-fuse-ld=gold")
            set(DETECTED_LINKER "gold")
        elseif(SANITIZE_LINK_FLAGS MATCHES "-fuse-ld=lld")
            set(DETECTED_LINKER "lld")
        elseif(SANITIZE_LINK_FLAGS MATCHES "-fuse-ld=bfd")
            set(DETECTED_LINKER "bfd")
        else()
            # Try to detect from compiler default
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -Wl,--version
                OUTPUT_VARIABLE LINKER_VERSION_OUTPUT
                ERROR_VARIABLE LINKER_VERSION_OUTPUT
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(LINKER_VERSION_OUTPUT MATCHES "GNU gold")
                set(DETECTED_LINKER "gold")
            elseif(LINKER_VERSION_OUTPUT MATCHES "LLD")
                set(DETECTED_LINKER "lld")
            elseif(LINKER_VERSION_OUTPUT MATCHES "GNU ld")
                set(DETECTED_LINKER "bfd")
            endif()
        endif()

        message(STATUS "Detected linker: ${DETECTED_LINKER}")

        # Apply linker-specific memory optimizations
        if(DETECTED_LINKER STREQUAL "gold")
            # GNU gold linker flags
            # Memory sanitizer needs reduced caching to avoid OOM, but leak sanitizer benefits from caching
            if(SD_SANITIZERS MATCHES "memory")
                set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--no-keep-memory")
                set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--no-map-whole-files")
            endif()
            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--icf=safe")

            # NOTE: --hash-size is NOT supported by GNU gold linker (only by GNU ld/bfd)
            # Gold uses its own internal hash table optimization that cannot be configured

            # Enable multi-threaded linking with gold (CRITICAL for performance + memory)
            # Gold by default is single-threaded. Multi-threading speeds up linking 2-4x
            # and can reduce peak memory by spreading work across time
            cmake_host_system_information(RESULT NUM_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
            math(EXPR LINKER_THREADS "${NUM_CORES} / 2")  # Use half of cores for linking
            if(LINKER_THREADS LESS 2)
                set(LINKER_THREADS 2)
            endif()
            if(LINKER_THREADS GREATER 8)
                set(LINKER_THREADS 8)  # Cap at 8 threads (gold's sweet spot)
            endif()

            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--threads")
            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--thread-count,${LINKER_THREADS}")

            message(STATUS "Applied GNU gold linker optimizations:")
            if(SD_SANITIZERS MATCHES "memory")
                message(STATUS "  - --no-keep-memory: Don't cache file contents (MemorySanitizer only)")
                message(STATUS "  - --no-map-whole-files: Map only needed file parts (MemorySanitizer only)")
            else()
                message(STATUS "  - Using default caching (better for leak/address sanitizers)")
            endif()
            message(STATUS "  - --icf=safe: Fold identical code sections")
            message(STATUS "  - --threads: Enable parallel linking with ${LINKER_THREADS} threads")
            message(STATUS "  Note: Gold linker manages hash table size internally (not configurable)")

        elseif(DETECTED_LINKER STREQUAL "lld")
            # LLVM lld linker flags (different syntax than gold)
            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--icf=all")

            # For very large builds (23GB+ object files), reduce memory usage
            # NOTE: --no-map-whole-files is gold-specific, NOT supported by lld
            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--gc-sections")

            # LLD has built-in threading, configure job count
            cmake_host_system_information(RESULT NUM_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
            math(EXPR LINKER_THREADS "${NUM_CORES} / 2")
            if(LINKER_THREADS LESS 2)
                set(LINKER_THREADS 2)
            endif()
            if(LINKER_THREADS GREATER 16)
                set(LINKER_THREADS 16)  # LLD scales better than gold
            endif()

            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--threads=${LINKER_THREADS}")

            message(STATUS "Applied LLVM lld linker memory optimizations:")
            message(STATUS "  - --icf=all: Aggressive identical code folding")
            message(STATUS "  - --gc-sections: Remove unused code sections")
            message(STATUS "  - --threads=${LINKER_THREADS}: Enable parallel linking")
            message(STATUS "  Note: LLD handles large builds (23GB+) better than gold")

        elseif(DETECTED_LINKER STREQUAL "bfd")
            # GNU ld (bfd) - older linker with limited optimization support
            set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -Wl,--no-keep-memory")
            # BFD doesn't support threading or advanced ICF

            message(STATUS "Applied GNU ld (bfd) linker memory optimizations:")
            message(STATUS "  - --no-keep-memory: Don't cache file contents in memory")
            message(WARNING "GNU ld (bfd) is single-threaded and slower than gold/lld. Consider using gold or lld for faster builds.")

        else()
            message(WARNING "Unknown linker detected. Skipping linker-specific optimizations.")
            message(STATUS "To specify linker explicitly, use: -DCMAKE_CXX_FLAGS=\"-fuse-ld=gold\" (or lld/bfd)")
        endif()
    endif()

    # For Homebrew clang or custom LLVM, also add explicit runtime library path
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
            OUTPUT_VARIABLE CLANG_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Determine OS-specific lib subdirectory
        if(APPLE)
            set(SANITIZER_LIB_SUBDIR "darwin")
        elseif(WIN32)
            set(SANITIZER_LIB_SUBDIR "windows")
        else()
            set(SANITIZER_LIB_SUBDIR "linux")
        endif()

        # Determine architecture for sanitizer library names
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            set(SANITIZER_ARCH "x86_64")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
            set(SANITIZER_ARCH "aarch64")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
            set(SANITIZER_ARCH "powerpc64le")
        else()
            set(SANITIZER_ARCH ${CMAKE_SYSTEM_PROCESSOR})
        endif()

        set(SANITIZER_LIB_PATH "${CLANG_RESOURCE_DIR}/lib/${SANITIZER_LIB_SUBDIR}")

        if(EXISTS "${SANITIZER_LIB_PATH}")
            # Add library search path and RPATH for sanitizer runtime
            if(APPLE)
                set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -L${SANITIZER_LIB_PATH} -Wl,-rpath,${SANITIZER_LIB_PATH}")
            elseif(WIN32)
                set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -L${SANITIZER_LIB_PATH}")
            else()
                set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -L${SANITIZER_LIB_PATH} -Wl,-rpath,${SANITIZER_LIB_PATH}")
            endif()

            # When using gold linker, -fsanitize=memory doesn't automatically link the runtime
            # We must explicitly add the MSan runtime libraries
            if(SD_SANITIZERS MATCHES "memory")
                set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -lclang_rt.msan-${SANITIZER_ARCH} -lclang_rt.msan_cxx-${SANITIZER_ARCH}")
                message(STATUS "Added explicit MSan runtime libraries for gold linker (${SANITIZER_ARCH})")
            endif()

            # CRITICAL: LeakSanitizer also needs explicit runtime linking for JNI usage
            # -fsanitize=leak uses the ASAN runtime (there's no separate LSAN .so)
            # For shared libraries loaded by JVM/JNI, the sanitizer runtime MUST be
            # dynamically linked, not statically linked. Without this, the sanitizer
            # runtime is not initialized and leaks are not detected.
            #
            # With gold linker, Clang defaults to static linking of sanitizer runtime.
            # We must explicitly link the shared .so file to make it a NEEDED dependency.
            if(SD_SANITIZERS MATCHES "leak" OR SD_SANITIZERS MATCHES "address")
                set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -lclang_rt.asan-${SANITIZER_ARCH}")
                message(STATUS "Added explicit ASAN shared runtime library for leak/address sanitizer (${SANITIZER_ARCH}, JNI compatibility)")
            endif()

            message(STATUS "Added clang sanitizer runtime library path and RPATH: ${SANITIZER_LIB_PATH}")
        endif()
    endif()

    message("Using sanitizers: ${SD_SANITIZERS}...")
    if(SD_CPU)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS}" CACHE STRING "C++ flags" FORCE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZE_FLAGS}" CACHE STRING "C flags" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZE_LINK_FLAGS}" CACHE STRING "Exe linker flags" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZE_LINK_FLAGS}" CACHE STRING "Shared linker flags" FORCE)
    endif()
    if(SD_CUDA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS} --relocatable-device-code=true" CACHE STRING "C++ flags" FORCE)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SANITIZE_FLAGS}" CACHE STRING "CUDA flags" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZE_LINK_FLAGS}" CACHE STRING "Exe linker flags" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZE_LINK_FLAGS}" CACHE STRING "Shared linker flags" FORCE)
    endif()

    # MEMORY OPTIMIZATION: Limit parallel compilation jobs for sanitizer builds
    # Template instantiation files with sanitizers can use 8-12GB per compiler process
    # Limiting parallel jobs prevents memory exhaustion
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Use CMake 3.0+ job pools to limit parallelism
        # Each instantiation file with MSan can peak at ~10GB
        # Safe limit: max(6, available_RAM_GB / 12)
        set(SANITIZER_MAX_JOBS 8)
        set_property(GLOBAL PROPERTY JOB_POOLS compile_pool=${SANITIZER_MAX_JOBS})
        set(CMAKE_JOB_POOL_COMPILE compile_pool)
        message(STATUS "⚠️  Limited parallel compilation to ${SANITIZER_MAX_JOBS} jobs for memory-intensive sanitizer build")
        message(STATUS "   Estimated peak memory: ~${SANITIZER_MAX_JOBS}0 GB")
    endif()
endif()

# --- Strict Linker Flags to Catch Undefined Symbols Early ---
# This helps catch missing template specializations at link time instead of runtime
# CRITICAL: This MUST be enabled for ALL builds to prevent shipping binaries with undefined symbols
# EXCEPTION: Do NOT use --no-undefined with sanitizers, as they require runtime symbol resolution
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT SD_SANITIZE AND NOT SD_GCC_FUNCTRACE)
    message(STATUS "⚠️  ENFORCING strict linker flags - build will FAIL on ANY undefined symbols")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined")

    # CRITICAL: Add code model to linker flags to match compiler flags
    # Without this, linker uses wrong relocation types → "relocation truncated to fit" errors
    if(SD_X86_BUILD AND NOT WIN32)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mcmodel=medium")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=medium")
        message(STATUS "Applied medium code model to linker flags (matches compiler -mcmodel=medium)")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND SD_GCC_FUNCTRACE AND NOT SD_SANITIZE)
    # GCC_FUNCTRACE (lifecycle tracking) adds significant instrumentation increasing binary size
    # Must use large code model to match compiler flags (set on lines 64-66)
    # CRITICAL FIX: Use gold linker for large code model + TLS compatibility
    # LLD linker CANNOT handle large code model (-mcmodel=large) with TLS relocations
    # Errors: "relocation R_X86_64_TLSGD out of range" when using lld with large model
    # Gold linker has proper support for large code model with TLS (Thread-Local Storage)

    # CRITICAL: Homebrew/Linuxbrew Clang has its own bundled lld that takes precedence
    # -fuse-ld=gold doesn't work because Clang searches its own bin directory first
    # Solution: Explicitly set CMAKE_LINKER to the full path of system gold linker
    find_program(GOLD_LINKER NAMES ld.gold PATHS /usr/bin /usr/local/bin NO_DEFAULT_PATH)
    if(GOLD_LINKER)
        set(CMAKE_LINKER "${GOLD_LINKER}" CACHE FILEPATH "Linker" FORCE)
        message(STATUS "✅ ENFORCING gold linker: ${GOLD_LINKER}")
    else()
        message(FATAL_ERROR "Gold linker (ld.gold) not found. Required for large code model + TLS + lifecycle tracking builds.")
    endif()

    message(STATUS "⚠️  ENFORCING strict linker flags - build will FAIL on ANY undefined symbols")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined -fuse-ld=gold")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined -fuse-ld=gold")

    # CRITICAL: Apply large code model to match compiler's -mcmodel=large
    # Without this, linker uses wrong relocation types → "relocation truncated to fit" errors
    if(SD_X86_BUILD AND NOT WIN32)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mcmodel=large")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=large")
        message(STATUS "Applied large code model to linker flags (matches compiler -mcmodel=large for lifecycle tracking)")
        message(STATUS "Using gold linker for TLS relocation support with large code model and call tracing")
    endif()
elseif(SD_SANITIZE)
    message(STATUS "ℹ️  Skipping --no-undefined for sanitizer build (sanitizer runtime symbols resolved at runtime)")
endif()

# Make build more verbose to see template instantiation issues
set(CMAKE_VERBOSE_MAKEFILE ON)

if(SD_GCC_FUNCTRACE)
    message(STATUS "✅ Applying SD_GCC_FUNCTRACE debug flags for line number information")

    # Override any optimization flags with debug-friendly ones
    set(CMAKE_CXX_FLAGS_RELEASE "-O0 -ggdb3 -fPIC -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb3 -fPIC")
    set(CMAKE_C_FLAGS_RELEASE "-O0 -ggdb3 -fPIC -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb3 -fPIC")

    # Add comprehensive debug flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Debug flags without function instrumentation (disabled due to TLS relocation overflow)
        # TLS optimizations still applied to reduce relocation overhead from thread-safe statics
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -gdwarf-4 -fno-eliminate-unused-debug-types -ftls-model=initial-exec -fno-threadsafe-statics")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb3 -fno-omit-frame-pointer -gdwarf-4 -ftls-model=initial-exec")
        message(STATUS "Applied debug flags for GCC (instrumentation disabled):")
        message(STATUS "  - initial-exec TLS model")
        message(STATUS "  - disabled thread-safe static guards")
        message(STATUS "  - disabled function instrumentation (prevents TLS overflow)")

        # Override any conflicting optimization
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Debug flags without function instrumentation (disabled due to TLS relocation overflow)
        # TLS optimizations still applied to reduce relocation overhead from thread-safe statics
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -ftls-model=initial-exec -fno-threadsafe-statics")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb3 -fno-omit-frame-pointer -ftls-model=initial-exec")
        message(STATUS "Applied debug flags for Clang (instrumentation disabled):")
        message(STATUS "  - initial-exec TLS model")
        message(STATUS "  - disabled thread-safe static guards")
        message(STATUS "  - disabled function instrumentation (prevents TLS overflow)")

        # Override any conflicting optimization
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    endif()

    # Ensure debug info is preserved in linker
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")

    # Prevent stripping
    set(CMAKE_STRIP "/bin/true")

    # Add the compiler definition
    add_compile_definitions(SD_GCC_FUNCTRACE=ON)

    # Function instrumentation has been disabled globally (above) to prevent TLS relocation overflow
    message(STATUS "ℹ️  Function instrumentation disabled globally - no per-file configuration needed")
endif()

# --- Flag Deduplication ---
# Remove duplicate flags that may have accumulated through multiple conditional blocks
# This ensures cleaner build logs and prevents potential flag conflicts

# Helper function to deduplicate flags in a space-separated string
function(deduplicate_flags FLAG_VAR)
    # Convert space-separated string to list
    string(REPLACE " " ";" FLAG_LIST "${${FLAG_VAR}}")
    # Remove duplicates while preserving order
    list(REMOVE_DUPLICATES FLAG_LIST)
    # Convert back to space-separated string
    string(REPLACE ";" " " FLAG_STRING "${FLAG_LIST}")
    # Set the parent scope variable
    set(${FLAG_VAR} "${FLAG_STRING}" PARENT_SCOPE)
endfunction()

# Deduplicate compiler flags
deduplicate_flags(CMAKE_CXX_FLAGS)
deduplicate_flags(CMAKE_C_FLAGS)
deduplicate_flags(CMAKE_CXX_FLAGS_RELEASE)
deduplicate_flags(CMAKE_CXX_FLAGS_DEBUG)
deduplicate_flags(CMAKE_C_FLAGS_RELEASE)
deduplicate_flags(CMAKE_C_FLAGS_DEBUG)

# Deduplicate linker flags
deduplicate_flags(CMAKE_SHARED_LINKER_FLAGS)
deduplicate_flags(CMAKE_EXE_LINKER_FLAGS)
deduplicate_flags(CMAKE_MODULE_LINKER_FLAGS)

message(STATUS "✅ Compiler and linker flags deduplicated")
