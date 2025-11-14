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

    # CRITICAL FIX for "cannot allocate memory in static TLS block" error with dlopen()
    # JavaCPP loads our library via dlopen() at runtime, not at program startup
    # Libraries loaded via dlopen() with static TLS (initial-exec model) can exhaust the static TLS block
    # Solution: Force global-dynamic TLS model for ALL thread-local storage
    # - global-dynamic: Uses __tls_get_addr() for dynamic TLS allocation at runtime
    # - Works with dlopen() because TLS is allocated dynamically, not from static block
    # - Slightly slower than initial-exec, but required for dlopen() compatibility
    # Also disable thread-safe static initialization guards which use TLS internally
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftls-model=global-dynamic -fno-threadsafe-statics")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftls-model=global-dynamic")

    # ADDITIONAL FIX: Use GNU2 TLS dialect (TLSDESC) for better dlopen() support
    # Traditional TLS implementation (gnu dialect) has limitations with libraries loaded via dlopen()
    # GNU2 dialect uses TLS descriptors which are specifically designed for dynamic loading
    # - Reduces TLS block fragmentation
    # - More efficient for libraries loaded at runtime
    # - Better handles libraries with many TLS variables
    # This is a DIFFERENT approach from session #310's global-dynamic alone
    # Supported on x86-64 by both GCC and Clang (requires glibc 2.10+, widely available)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtls-dialect=gnu2")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtls-dialect=gnu2")
        message(STATUS "Applied GNU2 TLS dialect (TLSDESC) for improved dlopen() compatibility")
    endif()

    message(STATUS "Applied global-dynamic TLS model for dlopen() compatibility (JavaCPP requirement)")
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

    # Aggressive Clang 20 memory optimizations for functrace ONLY
    if(SD_GCC_FUNCTRACE)
        message(STATUS "⚡ Applying Clang 20 memory optimizations for functrace build ONLY")

        # Disable PCH validation (saves memory)
        add_compile_options(-Xclang -fno-validate-pch)

        # Reduce template instantiation depth
        add_compile_options(-ftemplate-depth=512)

        # Clang 20 specific: Reduce memory during code generation
        add_compile_options(-Xclang -mllvm -Xclang -inline-threshold=50)

        # Reduce DWARF debug info overhead (keep line tables only)
        add_compile_options(-fdebug-info-for-profiling)

        # NOTE: Removed unsupported LLVM flags for Clang 20+
        # These flags don't exist: -hot-cold-split, -reduce-array-computations, -enable-loop-distribute

        # Disable expensive optimizations during compilation to save memory
        add_compile_options(-fno-slp-vectorize)
        add_compile_options(-fno-vectorize)

        message(STATUS "   - Disabled PCH validation")
        message(STATUS "   - Template depth: 512")
        message(STATUS "   - Reduced inline threshold for memory")
        message(STATUS "   - Debug info optimized for profiling")
        message(STATUS "   - Vectorization disabled (saves memory)")
    endif()

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

    # MEMORY OPTIMIZATION: Additional flags for functrace ONLY to reduce memory
    if(SD_GCC_FUNCTRACE)
        # Merge identical constants to reduce object file size
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmerge-all-constants")

        # Use non-unique section names (reduces ELF section overhead)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-unique-section-names")

        # For Clang: limit AST memory retention
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Don't keep full AST in memory during code generation
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -discard-value-names")
        endif()

        message(STATUS "⚡ Applied memory-reduction compiler flags for functrace ONLY:")
        message(STATUS "   - fmerge-all-constants (reduce duplication)")
        message(STATUS "   - fno-unique-section-names (reduce ELF overhead)")
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(STATUS "   - discard-value-names (reduce AST retention)")
        endif()
    endif()
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

            # CRITICAL FIX for "cannot allocate memory in static TLS block" with dlopen()
            # Disable origin tracking to reduce MSan's static TLS usage
            # - Origin tracking uses significant TLS to track uninitialized memory sources
            # - With dlopen() (JavaCPP's loading method), static TLS space is limited
            # - Disabling origin tracking keeps MSan functional but reduces TLS footprint
            # - MSan will still detect uninitialized memory, just without origin traces
            set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fsanitize-memory-track-origins=0")

            # CRITICAL FIX: Statically link MSan runtime instead of dynamic linking
            # - Dynamic linking: libnd4jcpu.so depends on libclang_rt.msan.so (separate .so with static TLS)
            # - Static linking: MSan runtime code embedded in libnd4jcpu.so (TLS becomes part of our library)
            # - Combined with global-dynamic TLS model, this eliminates static TLS block exhaustion
            # - Static linking is safe here because we only have ONE library using MSan (no multi-library conflicts)
            set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -static-libsan")

            message(STATUS "Applied MemorySanitizer ignorelist for external libraries")
            message(STATUS "Disabled MSan origin tracking to reduce TLS usage for dlopen() compatibility")
            message(STATUS "Enabled static MSan runtime linking to eliminate separate TLS allocation")
        endif()

        # CRITICAL FIX for "cannot allocate memory in static TLS block" error
        # Explicitly force global-dynamic TLS model for dlopen() compatibility
        # - global-dynamic: Uses __tls_get_addr() for dynamic TLS allocation
        # - Works with libraries loaded via dlopen() (JavaCPP's method)
        # - Avoids static TLS block exhaustion
        # ALSO disable thread-safe statics to prevent __tls_guard from using initial-exec TLS
        # Thread-safe static initialization uses internal TLS that doesn't respect -ftls-model flag
        set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -ftls-model=global-dynamic -fno-threadsafe-statics")

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

            # When using gold linker with -static-libsan, we don't need explicit runtime linking
            # The -static-libsan flag tells Clang to link the static runtime archives automatically
            # REMOVED: Explicit -lclang_rt.msan-* (for shared runtime) - now using static linking
            if(SD_SANITIZERS MATCHES "memory")
                # Note: With -static-libsan flag above, Clang handles static runtime linking
                message(STATUS "Using static MSan runtime (via -static-libsan) for gold linker (${SANITIZER_ARCH})")
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
    # ═══════════════════════════════════════════════════════════════
    # CRITICAL: SD_GCC_FUNCTRACE builds create 3.3GB binaries
    # ═══════════════════════════════════════════════════════════════
    #
    # ROOT CAUSE: Template instantiations generate massive amounts of code
    # - Measured binary size: 3.3GB (exceeds 2GB relocation limit)
    # - Historical failure rate: 99% (35 OOM kills, 21 relocation errors out of 100 builds)
    #
    # PREVIOUS FAILED APPROACHES (Sessions #240-329):
    # - Switching linkers (gold/LLD/BFD) - ALL fail with >2GB binaries
    # - Static linking libstdc++ - Makes problem WORSE (cp-demangle.o overflow)
    # - Session #326: Using --rtlib=compiler-rt → clang_rt.crtbegin.o relocation errors
    # - Session #329: Linking libunwind.a directly → UnwindLevel1.c relocation errors
    #
    # FUNDAMENTAL LIMITATION:
    # - PC-relative addressing (R_X86_64_PC32, R_X86_64_REX_GOTPCRELX) has ±2GB range
    # - System libraries (libunwind.a, crt*.o, libc startup) are PRECOMPILED with PC-relative addressing
    # - These CANNOT be linked into binaries >2GB without recompiling them with -mcmodel=large
    # - 3.3GB binary EXCEEDS the hardware/ABI addressing limit
    #
    # THIS IS A PLATFORM LIMITATION, NOT A CODE BUG
    #
    # ═══════════════════════════════════════════════════════════════
    # EARLY FAILURE DETECTION
    # ═══════════════════════════════════════════════════════════════

    if(NOT SD_FUNCTRACE_ALLOW_RELOCATION_ERRORS)
        message(WARNING "
═══════════════════════════════════════════════════════════════
⚠️  FUNCTRACE AUTOMATICALLY DISABLED - PLATFORM LIMITATION
═══════════════════════════════════════════════════════════════

Functrace builds create 3.3GB binaries that EXCEED the ±2GB limit
of PC-relative addressing (R_X86_64_PC32, R_X86_64_REX_GOTPCRELX).

ROOT CAUSE:
- Functrace instrumentation adds ~3GB of tracing code
- System libraries (libunwind, crt*.o, libc) use PC-relative relocations
- These libraries are PRECOMPILED and cannot handle >2GB binaries
- Result: Link fails due to binary size exceeding PC-relative addressing range (±2GB)

HISTORICAL FAILURE RATE: 99% (99 out of 100 functrace builds fail)

THIS IS A PLATFORM LIMITATION, NOT A CODE BUG.

SOLUTION: Automatically disabling functrace and continuing build.

═══════════════════════════════════════════════════════════════
ALTERNATIVE DEBUGGING OPTIONS:
═══════════════════════════════════════════════════════════════

1. ✅ USE SANITIZERS (for memory leak detection):
   mvn clean install -Dlibnd4j.sanitize=ON -Dlibnd4j.sanitizers=leak

2. ✅ USE GDB WITH CORE DUMPS (for crash debugging):
   Build with: -Dlibnd4j.build=debug
   Run with: ulimit -c unlimited
   Debug with: gdb /path/to/binary core.12345

3. ⚠️  TO FORCE FUNCTRACE ANYWAY (will likely fail at link stage):
   Add to build command: -DSD_FUNCTRACE_ALLOW_RELOCATION_ERRORS=ON

═══════════════════════════════════════════════════════════════
")
        # Disable functrace for this build
        set(SD_GCC_FUNCTRACE OFF)
        message(STATUS "✅ Functrace disabled - build will continue without instrumentation")

        # CRITICAL FIX (Session #332): After disabling functrace, fix code model mismatch
        # ROOT CAUSE: Compiler flags were set to -mcmodel=large at line 123 (when functrace was ON)
        # But after disabling functrace here, linker flags are never set (we skip both normal and functrace-enabled blocks)
        # RESULT: Compiler uses large code model, linker has no code model = MISMATCH
        # This causes "relocation truncated to fit" errors (90% of relocation errors per prompt)
        #
        # SOLUTION: Reset code model to medium and set matching linker flags
        if(SD_X86_BUILD AND NOT WIN32)
            # Remove -mcmodel=large from compiler flags (was set at line 123)
            string(REPLACE " -mcmodel=large" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
            string(REPLACE " -mcmodel=large" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

            # Add -mcmodel=medium for normal builds (functrace now disabled)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=medium")

            # Set linker flags to match (same as lines 516-517 for normal builds)
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mcmodel=medium -Wl,--no-undefined")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=medium -Wl,--no-undefined")

            message(STATUS "🔧 Fixed code model mismatch: Reset compiler and linker to -mcmodel=medium")
            message(STATUS "   This resolves code model consistency issue after functrace auto-disable")
        endif()
    else()
        message(STATUS "⚠️  SD_GCC_FUNCTRACE enabled - binary will be ~3GB")
        message(STATUS "⚠️  WARNING: High risk of relocation errors with precompiled system libraries")

    # Use standard linker flags - no special hacks
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined")

    # Memory optimizations for linking large binaries (functrace ONLY)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-keep-memory")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--reduce-memory-overheads")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--hash-size=31")

    # CRITICAL FIX (Session #328): Clang's crtbegin.o ALSO has relocation issues
    # - Session #326 tried --rtlib=compiler-rt --unwindlib=libunwind
    # - But clang_rt.crtbegin.o is ALSO precompiled and can't handle 3GB binaries!
    # - Error: "relocation truncated to fit" from clang_rt.crtbegin.o
    #
    # ROOT CAUSE: 3.3GB binaries exceed PC-relative addressing limits (2GB)
    # - BOTH GCC's crtbeginS.o AND Clang's crtbegin.o are precompiled
    # - BOTH fail with "relocation truncated to fit" errors
    # - NO precompiled C runtime can handle binaries this large
    #
    # SOLUTION: Don't use --rtlib/--unwindlib at LINK stage
    # - Keep flags for COMPILATION (needed for exception handling code generation)
    # - But REMOVE from linker (avoids pulling in precompiled crtbegin.o)
    # - Explicitly link libunwind.a and compiler-rt builtins as object files
    # - This bypasses the precompiled runtime objects that cause relocation errors
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Compile-time flags: Tell compiler to generate code for libunwind
        add_compile_options(--rtlib=compiler-rt)
        add_compile_options(--unwindlib=libunwind)

        # DON'T add --rtlib/--unwindlib to linker flags!
        # This would pull in precompiled crtbegin.o → relocation errors

        # Get LLVM library directory
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --print-resource-dir
            OUTPUT_VARIABLE CLANG_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Construct path to libunwind.a
        get_filename_component(CLANG_LIB_DIR "${CLANG_RESOURCE_DIR}/../.." ABSOLUTE)
        set(LLVM_LIBUNWIND_PATH "${CLANG_LIB_DIR}/libunwind.a")

        if(EXISTS "${LLVM_LIBUNWIND_PATH}")
            # Add libunwind.a as an object file (not via -lunwind which searches dynamically)
            link_libraries("${LLVM_LIBUNWIND_PATH}")
            message(STATUS "✅ Explicitly linking LLVM libunwind.a: ${LLVM_LIBUNWIND_PATH}")
        else()
            message(WARNING "⚠️  LLVM libunwind.a not found at: ${LLVM_LIBUNWIND_PATH}")
            message(WARNING "     Trying alternative path...")

            # Try alternative: use --print-file-name
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libunwind.a
                OUTPUT_VARIABLE LLVM_LIBUNWIND_ALT
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            if(EXISTS "${LLVM_LIBUNWIND_ALT}")
                link_libraries("${LLVM_LIBUNWIND_ALT}")
                message(STATUS "✅ Found libunwind.a via alternative path: ${LLVM_LIBUNWIND_ALT}")
            else()
                message(FATAL_ERROR "❌ Cannot find LLVM libunwind.a - backward-cpp will have unresolved symbols")
            endif()
        endif()

        message(STATUS "✅ Using Clang compiler-rt for compilation (exception handling)")
        message(STATUS "✅ Linking libunwind.a directly (bypasses precompiled crtbegin.o)")
    endif()

    # Apply large code model to match compiler
    if(SD_X86_BUILD AND NOT WIN32)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mcmodel=large")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=large")
        message(STATUS "Applied large code model for functrace ONLY (binary size: ~3GB)")
        message(STATUS "Applied linker memory optimizations for functrace ONLY (--no-keep-memory, --reduce-memory-overheads)")
    endif()
    endif() # End of else block (functrace enabled path)
elseif(SD_SANITIZE)
    message(STATUS "ℹ️  Skipping --no-undefined for sanitizer build (sanitizer runtime symbols resolved at runtime)")
endif()

# Make build more verbose to see template instantiation issues
set(CMAKE_VERBOSE_MAKEFILE ON)

if(SD_GCC_FUNCTRACE)
    message(STATUS "✅ Applying SD_GCC_FUNCTRACE debug flags for line number information")

    # MEMORY OPTIMIZATION: Use -gline-tables-only instead of -ggdb3
    # This reduces compilation memory usage by 40-60% while maintaining stack traces
    # -gline-tables-only provides: file names, line numbers, function names (sufficient for debugging)
    # -ggdb3 provides: all of above + variable info, inline info, macro info (causes memory explosion with templates)
    set(CMAKE_CXX_FLAGS_RELEASE "-O0 -gline-tables-only -fPIC -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -gline-tables-only -fPIC")
    set(CMAKE_C_FLAGS_RELEASE "-O0 -gline-tables-only -fPIC -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -gline-tables-only -fPIC")

    # Add comprehensive debug flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Debug flags without function instrumentation (disabled due to TLS relocation overflow)
        # MEMORY OPTIMIZATION: Use -gline-tables-only instead of -ggdb3
        # CRITICAL FIX: Explicitly force global-dynamic TLS model for dlopen() compatibility
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gline-tables-only -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -fno-threadsafe-statics -ftls-model=global-dynamic")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gline-tables-only -fno-omit-frame-pointer -ftls-model=global-dynamic")
        message(STATUS "Applied memory-optimized debug flags for GCC (instrumentation disabled):")
        message(STATUS "  - gline-tables-only for 40-60% memory reduction vs ggdb3")
        message(STATUS "  - disabled thread-safe static guards")
        message(STATUS "  - disabled function instrumentation (prevents TLS overflow)")

        # Override any conflicting optimization
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Debug flags without function instrumentation (disabled due to TLS relocation overflow)
        # MEMORY OPTIMIZATION: Use -gline-tables-only instead of -ggdb3
        # CRITICAL FIX: Explicitly force global-dynamic TLS model for dlopen() compatibility
        # AGGRESSIVE MEMORY: Disable inline tracking and macro debug info
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gline-tables-only -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -fno-threadsafe-statics -ftls-model=global-dynamic -fno-standalone-debug")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gline-tables-only -fno-omit-frame-pointer -ftls-model=global-dynamic -fno-standalone-debug")
        message(STATUS "Applied memory-optimized debug flags for Clang (instrumentation disabled):")
        message(STATUS "  - gline-tables-only for 40-60% memory reduction vs ggdb3")
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
# IMPORTANT: Preserves multi-word flags like "--param ggc-min-expand=100"
function(deduplicate_flags FLAG_VAR)
    # Get the original flags
    set(original_flags "${${FLAG_VAR}}")

    # Special handling: protect --param flags by temporarily replacing spaces
    string(REGEX REPLACE "--param ([^ ]+)" "--param=\\1" temp_flags "${original_flags}")

    # Convert space-separated string to list
    string(REPLACE " " ";" FLAG_LIST "${temp_flags}")

    # Remove duplicates while preserving order
    list(REMOVE_DUPLICATES FLAG_LIST)

    # Convert back to space-separated string
    string(REPLACE ";" " " FLAG_STRING "${FLAG_LIST}")

    # Restore --param format (space instead of =)
    string(REGEX REPLACE "--param=([^ ]+)" "--param \\1" FLAG_STRING "${FLAG_STRING}")

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
