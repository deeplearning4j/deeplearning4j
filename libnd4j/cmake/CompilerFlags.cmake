# cmake/CompilerFlags.cmake
# Configures compiler and linker flags for optimization and correctness.

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
# CRITICAL: -mcmodel=large is INCOMPATIBLE with system CRT libraries (crtbeginS.o, crti.o)
# System libraries are compiled with -mcmodel=small and cannot be linked into -mcmodel=large binaries
# This causes "relocation truncated to fit: R_X86_64_PC32" errors (see session #959, #1008)
# SOLUTION: Use -mcmodel=medium for both sanitizers AND functrace builds
# Medium model: Code can be anywhere, data/GOT in lowest 2GB (compatible with system libraries)
if(SD_X86_BUILD AND NOT WIN32)
    if(SD_SANITIZE OR SD_GCC_FUNCTRACE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium -fPIC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=medium")
        if(SD_SANITIZE)
            message(STATUS "Applied medium memory model for x86-64 architecture (sanitizers enabled)")
        elseif(SD_GCC_FUNCTRACE)
            message(STATUS "Applied medium memory model for x86-64 architecture (lifecycle tracking enabled)")
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

        # Use non-unique section names (reduces ELF section overhead) - Clang only
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-unique-section-names")
            # Don't keep full AST in memory during code generation
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -discard-value-names")
        endif()

        message(STATUS "⚡ Applied memory-reduction compiler flags for functrace ONLY:")
        message(STATUS "   - fmerge-all-constants (reduce duplication)")
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(STATUS "   - fno-unique-section-names (reduce ELF overhead, Clang only)")
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
        # MEMORY OPTIMIZATION: Use -gline-tables-only instead of full debug info (Clang-specific)
        # This reduces memory by 40-60% for template-heavy code while maintaining stack traces
        set(SANITIZE_FLAGS " -fPIC -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all -fuse-ld=gold -gline-tables-only")

        # MemorySanitizer-specific: Use ignorelist to skip instrumentation of external libraries
        if(SD_SANITIZERS MATCHES "memory")
            set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fsanitize-ignorelist=${CMAKE_CURRENT_SOURCE_DIR}/msan_ignorelist.txt")

            # Disable origin tracking to reduce MSan's static TLS usage
            # - Origin tracking uses significant TLS to track uninitialized memory sources
            # - With dlopen() (JavaCPP's loading method), static TLS space is limited
            # - Disabling origin tracking keeps MSan functional but reduces TLS footprint
            # - MSan will still detect uninitialized memory, just without origin traces
            set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fsanitize-memory-track-origins=0")

            # - Dynamic linking: libnd4jcpu.so depends on libclang_rt.msan.so (separate .so with static TLS)
            # - Static linking: MSan runtime code embedded in libnd4jcpu.so (TLS becomes part of our library)
            # - Combined with global-dynamic TLS model, this eliminates static TLS block exhaustion
            # - Static linking is safe here because we only have ONE library using MSan (no multi-library conflicts)
            set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -static-libsan")

            message(STATUS "Applied MemorySanitizer ignorelist for external libraries")
            message(STATUS "Disabled MSan origin tracking to reduce TLS usage for dlopen() compatibility")
            message(STATUS "Enabled static MSan runtime linking to eliminate separate TLS allocation")
        endif()

        # Explicitly force global-dynamic TLS model for dlopen() compatibility
        # - global-dynamic: Uses __tls_get_addr() for dynamic TLS allocation
        # - Works with libraries loaded via dlopen() (JavaCPP's method)
        # - Avoids static TLS block exhaustion
        # ALSO disable thread-safe statics to prevent __tls_guard from using initial-exec TLS
        # Thread-safe static initialization uses internal TLS that doesn't respect -ftls-model flag
        set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -ftls-model=global-dynamic -fno-threadsafe-statics")

        # Additional memory optimizations for template-heavy instantiation files
        set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fmerge-all-constants")
        set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fno-unique-section-names")  # Clang-specific

        message(STATUS "Applied memory-optimized sanitizer flags (-gline-tables-only, Clang-specific)")
    else()
        set(SANITIZE_FLAGS " -Wall -Wextra -fPIC -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all")
    endif()
    # Gold linker + sanitizers: Must explicitly pass -fsanitize to linker
    # The -fsanitize flag tells clang driver to link the sanitizer runtime
    # Gold linker handles MSan's TLS correctly for shared libraries (LLD has issues)
    set(SANITIZE_LINK_FLAGS "-fsanitize=${SD_SANITIZERS} -fuse-ld=gold")

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

            # Gold uses its own internal hash table optimization that cannot be configured

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
# EXCEPTION: Do NOT use --no-undefined with sanitizers, as they require runtime symbol resolution
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT SD_SANITIZE AND NOT SD_GCC_FUNCTRACE)
    message(STATUS "⚠️  ENFORCING strict linker flags - build will FAIL on ANY undefined symbols")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined")

    # Without this, linker uses wrong relocation types → "relocation truncated to fit" errors
    if(SD_X86_BUILD AND NOT WIN32)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mcmodel=medium")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=medium")
        message(STATUS "Applied medium code model to linker flags (matches compiler -mcmodel=medium)")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND SD_GCC_FUNCTRACE AND NOT SD_SANITIZE)
    #
    #
    # THIS IS A PLATFORM LIMITATION, NOT A CODE BUG
    #

        message(STATUS "⚠️  SD_GCC_FUNCTRACE enabled - binary will be ~3GB")
        message(STATUS "⚠️  WARNING: High risk of relocation errors with precompiled system libraries")

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined")

    # Aggressive size reduction for functrace builds
    add_compile_options(-ffunction-sections -fdata-sections)
    add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections,--as-needed")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections,--as-needed")

    # Memory optimizations for linking large binaries
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-keep-memory")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--reduce-memory-overheads")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--hash-size=31")

    # NOTE: We use compiler-rt for runtime builtins but NOT libunwind for exception handling
    # (Session #1045 fix: libunwind conflicts with JVM's libgcc_s, causing _Unwind_SetGR crashes)
    # The system's libgcc_s handles exception unwinding, which is compatible with JVM
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # CRITICAL FIX: Do NOT use --unwindlib=libunwind for JNI libraries!
        #
        # Problem (discovered in session #1045):
        # - When --unwindlib=libunwind is used, Clang generates code assuming LLVM's libunwind ABI
        # - We also statically link libunwind.a
        # - BUT: At runtime, the JVM has libgcc_s.so loaded (for its own exception handling)
        # - Due to dynamic symbol interposition, _Unwind* symbols from libgcc_s.so override
        #   our statically linked libunwind symbols
        # - Result: ABI mismatch between exception context format → CRASH in _Unwind_SetGR
        #
        # The crash manifests as: SIGSEGV in _Unwind_SetGR+0x3e trying to write to read-only
        # memory in libnd4jcpu.so during exception handling
        #
        # Solution: Use the system's default exception handling (libgcc_s), which is
        # compatible with the JVM environment. Keep compiler-rt only for builtins.
        #
        # Use compiler-rt for runtime builtins (math, etc.) but NOT for exception handling
        add_compile_options(--rtlib=compiler-rt)
        # REMOVED: add_compile_options(--unwindlib=libunwind)
        # Let the system use libgcc_s for exception unwinding - compatible with JVM

        message(STATUS "✅ Using Clang compiler-rt for runtime builtins")
        message(STATUS "✅ Using system libgcc_s for exception handling (JNI/JVM compatible)")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # With GCC, -finstrument-functions may pull in gprof startup code (__gmon_start__)
        # but this symbol is weakly defined and doesn't cause actual linker errors.
        # No special linker flags needed - GCC runtime handles this correctly.
        message(STATUS "✅ Using GCC with functrace (no special profiling symbols needed)")
    endif()

    # Apply medium code model to match compiler (CRITICAL: large model incompatible with system CRT)
    if(SD_X86_BUILD AND NOT WIN32)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mcmodel=medium")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=medium")
        # Allow text relocations as escape hatch if binary is still too large
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,notext")
        # Disable linker relaxation to prevent GOTPCREL relocation failures (required for medium code model)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-relax")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-relax")
        message(STATUS "Applied medium code model for functrace ONLY (binary size: ~3GB)")
        message(STATUS "Applied linker memory optimizations for functrace ONLY (--no-keep-memory, --reduce-memory-overheads)")
        message(STATUS "Added -Wl,-z,notext to allow text relocations if needed")
        message(STATUS "Added -Wl,--no-relax to prevent GOT relocation issues with medium code model")
    endif()
elseif(SD_SANITIZE)
    message(STATUS "ℹ️  Skipping --no-undefined for sanitizer build (sanitizer runtime symbols resolved at runtime)")
endif()

# Make build more verbose to see template instantiation issues
set(CMAKE_VERBOSE_MAKEFILE ON)

if(SD_GCC_FUNCTRACE)
    message(STATUS "✅ Applying SD_GCC_FUNCTRACE debug flags for line number information")

    # MEMORY OPTIMIZATION: Use minimal debug info instead of full debug info
    # This reduces compilation memory usage by 40-60% while maintaining stack traces
    # Clang: -gline-tables-only provides file names, line numbers, function names
    # GCC: -g1 provides minimal debug info (line numbers and functions only)
    # Full debug info (-ggdb3) causes memory explosion with templates
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "-O0 -gline-tables-only -fPIC -DNDEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -gline-tables-only -fPIC")
        set(CMAKE_C_FLAGS_RELEASE "-O0 -gline-tables-only -fPIC -DNDEBUG")
        set(CMAKE_C_FLAGS_DEBUG "-O0 -gline-tables-only -fPIC")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O0 -g1 -fPIC -DNDEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g1 -fPIC")
        set(CMAKE_C_FLAGS_RELEASE "-O0 -g1 -fPIC -DNDEBUG")
        set(CMAKE_C_FLAGS_DEBUG "-O0 -g1 -fPIC")
    endif()

    # Add comprehensive debug flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Debug flags with function instrumentation
        # MEMORY OPTIMIZATION: Use -g1 (minimal debug info) instead of -ggdb3
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g1 -finstrument-functions -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -fno-threadsafe-statics -ftls-model=global-dynamic")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g1 -fno-omit-frame-pointer -ftls-model=global-dynamic")
        message(STATUS "Applied memory-optimized debug flags for GCC (instrumentation enabled):")
        message(STATUS "  - g1 (minimal debug info) for 40-60% memory reduction vs ggdb3")
        message(STATUS "  - disabled thread-safe static guards")
        message(STATUS "  - enabled function instrumentation")

        # Override any conflicting optimization
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Debug flags with function instrumentation
        # MEMORY OPTIMIZATION: Use -gline-tables-only instead of -ggdb3 (Clang-specific flag)
        # AGGRESSIVE MEMORY: Disable inline tracking and macro debug info
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gline-tables-only -finstrument-functions -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -fno-threadsafe-statics -ftls-model=global-dynamic -fno-standalone-debug")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gline-tables-only -fno-omit-frame-pointer -ftls-model=global-dynamic -fno-standalone-debug")
        message(STATUS "Applied memory-optimized debug flags for Clang (instrumentation enabled):")
        message(STATUS "  - gline-tables-only (Clang-specific) for 40-60% memory reduction vs ggdb3")
        message(STATUS "  - disabled thread-safe static guards")
        message(STATUS "  - enabled function instrumentation")

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

    # Enable function instrumentation
    message(STATUS "ℹ️  Function instrumentation enabled. This will significantly increase binary size.")
endif()

# --- Flag Deduplication ---
# Remove duplicate flags that may have accumulated through multiple conditional blocks
# This ensures cleaner build logs and prevents potential flag conflicts

# Helper function to deduplicate flags in a space-separated string
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
