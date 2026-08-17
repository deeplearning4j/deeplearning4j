# cmake/Dependencies.cmake
# Manages all third-party dependencies. Logic is encapsulated in functions.

include(ExternalProject)
include(ProcessorCount)
include("${CMAKE_CURRENT_LIST_DIR}/ExternalProjectCompatibility.cmake")

# =============================================================================
# PARALLEL BUILD CONFIGURATION FOR DEPENDENCIES
# =============================================================================
# Compute optimal parallel jobs for dependency builds based on:
# 1. User-specified SD_PARALLEL_COMPILE_JOBS (if set)
# 2. Available memory (each compile uses ~1-2GB for deps, less than main build)
# 3. Available CPU cores

ProcessorCount(NPROC)
if(NPROC EQUAL 0)
    set(NPROC 4)  # Fallback
endif()

# Dependencies (LLVM, flatbuffers, etc.) use less memory per compile (~1GB vs 2-4GB for CUDA).
# The top-level Maven/native launcher publishes its explicit -j value as
# SD_PARALLEL_COMPILE_JOBS; that caller choice must govern every nested build.
cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
math(EXPR MEM_BASED_JOBS "${AVAILABLE_MEMORY} / 1000")  # 1GB per job for deps

if(DEFINED SD_PARALLEL_COMPILE_JOBS AND
   "${SD_PARALLEL_COMPILE_JOBS}" MATCHES "^[1-9][0-9]*$")
    set(DEP_PARALLEL_JOBS ${SD_PARALLEL_COMPILE_JOBS})
    if(DEP_PARALLEL_JOBS GREATER NPROC)
        set(DEP_PARALLEL_JOBS ${NPROC})
    endif()
    set(_DEP_PARALLEL_SOURCE "explicit SD_PARALLEL_COMPILE_JOBS")
else()
    # Auto mode is capped by both available memory and processor count.
    if(MEM_BASED_JOBS GREATER NPROC)
        set(DEP_PARALLEL_JOBS ${NPROC})
    elseif(MEM_BASED_JOBS LESS 1)
        set(DEP_PARALLEL_JOBS 1)
    else()
        set(DEP_PARALLEL_JOBS ${MEM_BASED_JOBS})
    endif()
    set(_DEP_PARALLEL_SOURCE "automatic memory/core limit")
endif()

message(STATUS "🔧 Dependency builds will use ${DEP_PARALLEL_JOBS} parallel jobs (${_DEP_PARALLEL_SOURCE}; ${NPROC} cores, ${AVAILABLE_MEMORY}MB available)")

# =============================================================================
# DEPENDENCY CACHE INFRASTRUCTURE
# =============================================================================
# Persistent cache layer that survives 'mvn clean'. Dependencies are cached
# in a directory outside the build tree (default: ~/.libnd4j/dep-cache).
# On cache hit, ExternalProject_Add is skipped entirely.

# Handle cache clearing at configure time
if(SD_DEP_CACHE AND SD_DEP_CACHE_CLEAR)
    if(SD_DEP_CACHE_CLEAR_DEP)
        set(_clear_path "${SD_DEP_CACHE_DIR}/${SD_DEP_CACHE_CLEAR_DEP}")
        if(EXISTS "${_clear_path}")
            message(STATUS "DEP-CACHE: Clearing cache for '${SD_DEP_CACHE_CLEAR_DEP}' at ${_clear_path}")
            file(REMOVE_RECURSE "${_clear_path}")
        else()
            message(STATUS "DEP-CACHE: No cache to clear for '${SD_DEP_CACHE_CLEAR_DEP}'")
        endif()
    else()
        if(EXISTS "${SD_DEP_CACHE_DIR}")
            message(STATUS "DEP-CACHE: Clearing ALL cached dependencies at ${SD_DEP_CACHE_DIR}")
            file(REMOVE_RECURSE "${SD_DEP_CACHE_DIR}")
        else()
            message(STATUS "DEP-CACHE: No cache directory to clear")
        endif()
    endif()
endif()

# Compute a cache key for a dependency.
# Key format: {version}-{8-char-md5-hash}
# The hash covers: version, system name, processor, compiler ID/version, build type, extra config.
function(sd_dep_cache_key dep_name version extra_config out_var)
    set(_key_input "${version};${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR};${CMAKE_C_COMPILER_ID};${CMAKE_C_COMPILER_VERSION};${CMAKE_BUILD_TYPE};${extra_config}")
    string(MD5 _hash "${_key_input}")
    string(SUBSTRING "${_hash}" 0 8 _hash8)
    set(${out_var} "${version}-${_hash8}" PARENT_SCOPE)
endfunction()

# Host generators used by a cross build are independent of the Android target
# toolchain. Key them from the pinned source/recipe plus the actual native host
# compilers so changing an NDK or target ABI never rebuilds llvm-tblgen, mlir-tblgen,
# or SLEEF's generators.
function(sd_dep_cache_host_key dep_name version host_c_compiler host_cxx_compiler extra_config out_var)
    execute_process(
        COMMAND "${host_c_compiler}" --version
        OUTPUT_VARIABLE _host_c_version
        ERROR_VARIABLE _host_c_version_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND "${host_cxx_compiler}" --version
        OUTPUT_VARIABLE _host_cxx_version
        ERROR_VARIABLE _host_cxx_version_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_host_c_version STREQUAL "")
        set(_host_c_version "${_host_c_version_error}")
    endif()
    if(_host_cxx_version STREQUAL "")
        set(_host_cxx_version "${_host_cxx_version_error}")
    endif()
    string(SHA256 _host_c_identity
        "${host_c_compiler};${_host_c_version}")
    string(SHA256 _host_cxx_identity
        "${host_cxx_compiler};${_host_cxx_version}")
    set(_key_input
        "${version};${CMAKE_HOST_SYSTEM_NAME};${CMAKE_HOST_SYSTEM_PROCESSOR};"
        "${_host_c_identity};${_host_cxx_identity};cmake=${CMAKE_VERSION};${extra_config}")
    string(MD5 _hash "${_key_input}")
    string(SUBSTRING "${_hash}" 0 8 _hash8)
    set(${out_var} "${version}-${_hash8}" PARENT_SCOPE)
endfunction()

# Check if a cached dependency exists and is complete.
# Sets out_hit to TRUE/FALSE and out_cache_path to the cache install directory.
function(sd_dep_cache_check dep_name cache_key out_hit out_cache_path)
    set(_cache_dir "${SD_DEP_CACHE_DIR}/${dep_name}/${cache_key}")
    set(_marker "${_cache_dir}/.cache_complete")
    if(EXISTS "${_marker}")
        # Count files for logging
        file(GLOB_RECURSE _cached_files "${_cache_dir}/install/*")
        list(LENGTH _cached_files _file_count)
        # Read marker timestamp
        file(TIMESTAMP "${_marker}" _cache_date "%Y-%m-%d %H:%M" UTC)
        message(STATUS "DEP-CACHE [${dep_name}] HIT - key=${cache_key} date=${_cache_date} files=${_file_count}")
        set(${out_hit} TRUE PARENT_SCOPE)
        set(${out_cache_path} "${_cache_dir}/install" PARENT_SCOPE)
    else()
        message(STATUS "DEP-CACHE [${dep_name}] MISS - key=${cache_key}")
        set(${out_hit} FALSE PARENT_SCOPE)
        set(${out_cache_path} "" PARENT_SCOPE)
    endif()
endfunction()

# Restore cached dependency artifacts into the build directory.
function(sd_dep_cache_restore dep_name cache_path install_dir)
    message(STATUS "DEP-CACHE [${dep_name}] Restoring from ${cache_path} -> ${install_dir}")
    file(MAKE_DIRECTORY "${install_dir}")
    # A routine configure must not refresh every cached header timestamp: those
    # headers are prerequisites of consumer objects. CMake 3.26 added the
    # directory-wide content-stable copy; retain the legacy command only for
    # older CMake versions supported by the project.
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.26")
        set(_copy_directory_command copy_directory_if_different)
    else()
        set(_copy_directory_command copy_directory)
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E ${_copy_directory_command} "${cache_path}" "${install_dir}"
        RESULT_VARIABLE _copy_result
    )
    if(NOT _copy_result EQUAL 0)
        message(WARNING "DEP-CACHE [${dep_name}] Restore failed (exit code ${_copy_result}), will rebuild")
    endif()
endfunction()

# Store dependency artifacts into the persistent cache after build.
# Adds an ExternalProject_Add_Step that runs post-install.
function(sd_dep_cache_store dep_name cache_key install_dir ep_target)
    set(_cache_dir "${SD_DEP_CACHE_DIR}/${dep_name}/${cache_key}")
    # Normalize paths to forward slashes for Windows compatibility in generated scripts
    string(REPLACE "\\" "/" _cache_dir "${_cache_dir}")
    string(REPLACE "\\" "/" install_dir "${install_dir}")
    # A configure can stage multiple variants of the same dependency. Give every
    # producer a unique script so a later setup_triton() call cannot overwrite
    # the cache publication command of an earlier target.
    string(MD5 _store_script_identity
        "${dep_name};${cache_key};${install_dir};${ep_target}")
    set(_store_script
        "${CMAKE_BINARY_DIR}/dep_cache_store_${dep_name}_${_store_script_identity}.cmake")
    # Write a cmake script that copies install_dir to cache and writes the marker
    file(WRITE "${_store_script}" "
        # Store dependency artifacts into cache
        set(_cache_install \"${_cache_dir}/install\")
        set(_marker \"${_cache_dir}/.cache_complete\")
        # Remove old cache for this key if partial
        if(EXISTS \"\${_cache_install}\" AND NOT EXISTS \"\${_marker}\")
            file(REMOVE_RECURSE \"\${_cache_install}\")
        endif()
        # Only store if marker doesn't exist (avoid redundant copies)
        if(NOT EXISTS \"\${_marker}\")
            message(STATUS \"DEP-CACHE [${dep_name}] Storing artifacts to cache (key=${cache_key})\")
            file(MAKE_DIRECTORY \"\${_cache_install}\")
            execute_process(
                COMMAND \${CMAKE_COMMAND} -E copy_directory \"${install_dir}\" \"\${_cache_install}\"
                RESULT_VARIABLE _copy_result
            )
            if(_copy_result EQUAL 0)
                # Write marker LAST to ensure atomicity
                file(WRITE \"\${_marker}\" \"cached by libnd4j at configure time\")
                message(STATUS \"DEP-CACHE [${dep_name}] Cache stored successfully\")
            else()
                message(WARNING \"DEP-CACHE [${dep_name}] Failed to store cache (exit code \${_copy_result})\")
                # Clean up partial cache
                file(REMOVE_RECURSE \"\${_cache_install}\")
            endif()
        endif()
    ")
    ExternalProject_Add_Step(${ep_target} dep_cache_store
        COMMAND ${CMAKE_COMMAND} -P "${_store_script}"
        DEPENDEES install
        COMMENT "DEP-CACHE [${dep_name}] Storing build artifacts to persistent cache"
    )
endfunction()

# Print dependency cache summary at configure time
function(sd_dep_cache_summary)
    if(NOT SD_DEP_CACHE)
        message(STATUS "")
        message(STATUS "=== Dependency Cache: DISABLED ===")
        message(STATUS "  Enable with -DSD_DEP_CACHE=ON")
        message(STATUS "==================================")
        message(STATUS "")
        return()
    endif()

    set(_cached_count 0)
    set(_dep_summary "")

    if(EXISTS "${SD_DEP_CACHE_DIR}")
        file(GLOB _dep_dirs "${SD_DEP_CACHE_DIR}/*")
        foreach(_dep_dir ${_dep_dirs})
            if(IS_DIRECTORY "${_dep_dir}")
                get_filename_component(_dep_name "${_dep_dir}" NAME)
                file(GLOB _version_dirs "${_dep_dir}/*")
                list(LENGTH _version_dirs _ver_count)
                if(_ver_count GREATER 0)
                    math(EXPR _cached_count "${_cached_count} + 1")
                    string(APPEND _dep_summary "    ${_dep_name}: ${_ver_count} cached version(s)\n")
                endif()
            endif()
        endforeach()
    endif()

    message(STATUS "")
    message(STATUS "=== Dependency Cache Configuration ===")
    message(STATUS "  Enabled:   ON")
    message(STATUS "  Directory: ${SD_DEP_CACHE_DIR}")
    message(STATUS "  Cached dependencies: ${_cached_count}")
    if(_dep_summary)
        message(STATUS "${_dep_summary}")
    endif()
    message(STATUS "======================================")
    message(STATUS "")
endfunction()

# Print cache summary at configure time
sd_dep_cache_summary()

function(setup_android_arm_openblas)
    set(is_android_or_arm FALSE)

    if(ANDROID OR SD_ANDROID_BUILD OR SD_ARM_BUILD OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
        set(is_android_or_arm TRUE)
    endif()

    if(NOT is_android_or_arm)
        return()
    endif()

    message(STATUS "🔧 Setting up OpenBLAS for Android/ARM platform")

    # Handle path normalization for Android/ARM
    if(OPENBLAS_PATH MATCHES "lib/[^/]+$")
        get_filename_component(OPENBLAS_PATH "${OPENBLAS_PATH}/../.." ABSOLUTE)
        message(STATUS "🔧 Normalized OPENBLAS_PATH: ${OPENBLAS_PATH}")
    endif()

    # Platform-specific OpenBLAS library name handling
    if(EXISTS "${OPENBLAS_PATH}/lib")
        # Define search patterns based on platform
        set(LIB_SEARCH_PATTERNS
                "${OPENBLAS_PATH}/lib/libopenblas.so"
                "${OPENBLAS_PATH}/lib/libopenblas.so.0"
                "${OPENBLAS_PATH}/lib/libopenblas.a"
                "${OPENBLAS_PATH}/libopenblas.so"
                "${OPENBLAS_PATH}/libopenblas.so.0"
                "${OPENBLAS_PATH}/libopenblas.a"
        )

        # Add platform-specific patterns
        if(ANDROID OR SD_ANDROID_BUILD)
            if(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
                list(APPEND LIB_SEARCH_PATTERNS
                        "${OPENBLAS_PATH}/lib/android-x86_64/libopenblas.so"
                        "${OPENBLAS_PATH}/lib/android-x86_64/libopenblas.a"
                )
            elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
                list(APPEND LIB_SEARCH_PATTERNS
                        "${OPENBLAS_PATH}/lib/android-arm64/libopenblas.so"
                        "${OPENBLAS_PATH}/lib/android-arm64/libopenblas.a"
                        "${OPENBLAS_PATH}/lib/android-aarch64/libopenblas.so"
                        "${OPENBLAS_PATH}/lib/android-aarch64/libopenblas.a"
                )
            endif()
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
            list(APPEND LIB_SEARCH_PATTERNS
                    "${OPENBLAS_PATH}/lib/linux-arm64/libopenblas.so"
                    "${OPENBLAS_PATH}/lib/linux-arm64/libopenblas.a"
                    "${OPENBLAS_PATH}/lib/linux-aarch64/libopenblas.so"
                    "${OPENBLAS_PATH}/lib/linux-aarch64/libopenblas.a"
            )
        endif()

        # Search for libraries
        set(FOUND_OPENBLAS_LIBS "")
        foreach(pattern ${LIB_SEARCH_PATTERNS})
            file(GLOB matched_libs ${pattern})
            if(matched_libs)
                list(APPEND FOUND_OPENBLAS_LIBS ${matched_libs})
            endif()
        endforeach()

        if(FOUND_OPENBLAS_LIBS)
            message(STATUS "✅ Found OpenBLAS libraries: ${FOUND_OPENBLAS_LIBS}")
        else()
            message(WARNING "⚠️  No OpenBLAS libraries found in ${OPENBLAS_PATH}/lib")
        endif()
    endif()

    # Set platform-specific compiler flags for OpenBLAS
    if(ANDROID OR SD_ANDROID_BUILD)
        if(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64" PARENT_SCOPE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64" PARENT_SCOPE)
        elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a" PARENT_SCOPE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a" PARENT_SCOPE)
        endif()
    elseif(SD_ARM_BUILD OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a -mtune=cortex-a72" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -mtune=cortex-a72" PARENT_SCOPE)
    endif()

    # Set additional ARM-specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64" OR CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfix-cortex-a53-835769 -mfix-cortex-a53-843419" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfix-cortex-a53-835769 -mfix-cortex-a53-843419" PARENT_SCOPE)
    endif()
endfunction()


function(setup_blas)
    if(SD_CUDA)
        return()
    endif()
    if(SD_VULKAN OR SD_HEXAGON OR SD_TPU OR SD_GOOGLE_TENSOR_TPU)
        # Primary accelerator builds are device-only. Host BLAS would create a
        # second execution path, enlarge mobile packages, and make unsupported
        # device segments look successful through CPU fallback. Force the cache
        # clean as well so a reused build directory cannot retain an ambient
        # OpenBLAS configuration.
        set(BLAS FALSE CACHE BOOL
            "Host BLAS is disabled for device-only accelerator builds" FORCE)
        unset(OPENBLAS_PATH CACHE)
        unset(BLAS_IMPL CACHE)
        set(HAVE_OPENBLAS FALSE PARENT_SCOPE)
        set(OPENBLAS_LIBRARIES "" PARENT_SCOPE)
        set(OPENBLAS_PATH "" PARENT_SCOPE)
        return()
    endif()

    if(NOT OPENBLAS_PATH)
        # Try Homebrew's openblas as fallback on macOS
        if(APPLE)
            if(EXISTS "/opt/homebrew/opt/openblas")
                set(OPENBLAS_PATH "/opt/homebrew/opt/openblas" PARENT_SCOPE)
                set(OPENBLAS_PATH "/opt/homebrew/opt/openblas")
                message(STATUS "🔧 Using Homebrew OpenBLAS: ${OPENBLAS_PATH}")
            elseif(EXISTS "/usr/local/opt/openblas")
                set(OPENBLAS_PATH "/usr/local/opt/openblas" PARENT_SCOPE)
                set(OPENBLAS_PATH "/usr/local/opt/openblas")
                message(STATUS "🔧 Using Homebrew OpenBLAS: ${OPENBLAS_PATH}")
            endif()
        endif()
        if(NOT OPENBLAS_PATH)
            # System OpenBLAS fallback. There is no bytedeco openblas jar in
            # ~/.javacpp/cache or ~/.m2 at libnd4j-build time on the glibc-2.28
            # compat container (openblas is a dependency of nd4j-native, which
            # builds AFTER libnd4j), so the jar-based auto-detect in
            # buildnativeoperations.sh leaves OPENBLAS_PATH empty. openblas-devel
            # installs libopenblas.so under /usr/lib64 (or the Debian multiarch
            # dir) and cblas.h under /usr/include or /usr/include/openblas. This
            # link is for symbol resolution only; the portable bytedeco .so is
            # still bundled and loaded at runtime via the classpath.
            find_library(SD_SYSTEM_OPENBLAS
                NAMES openblas openblaso openblasp
                PATHS /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu
                      /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/local/lib64)
            find_path(SD_SYSTEM_CBLAS_INCLUDE
                NAMES cblas.h
                PATHS /usr/include /usr/include/openblas /usr/local/include)
            if(SD_SYSTEM_OPENBLAS AND SD_SYSTEM_CBLAS_INCLUDE)
                message(STATUS "🔧 Using system OpenBLAS: ${SD_SYSTEM_OPENBLAS}")
                message(STATUS "   System cblas.h dir: ${SD_SYSTEM_CBLAS_INCLUDE}")
                include_directories(${SD_SYSTEM_CBLAS_INCLUDE})
                add_compile_definitions(HAVE_OPENBLAS=1)
                set(HAVE_OPENBLAS 1 PARENT_SCOPE)
                set(OPENBLAS_LIBRARIES "${SD_SYSTEM_OPENBLAS}" PARENT_SCOPE)
                message(STATUS "✅ OpenBLAS setup complete (system)")
                return()
            endif()
            message(STATUS "❌ OPENBLAS_PATH not set")
            return()
        endif()
    endif()

    # Handle Android path normalization at CMake level (in case shell script normalization didn't work)
    setup_android_arm_openblas()

    # config.h exposes this value as a quoted C macro and JavaCPP copies it into
    # generated Java sources. Normalize Windows backslashes before either parser
    # sees the path so sequences such as \t and \U cannot become escapes. Push
    # the normalized value to directory scope immediately so both config-header
    # generators see it even if a later OpenBLAS validation returns early.
    string(REPLACE "\\" "/" OPENBLAS_PATH "${OPENBLAS_PATH}")
    set(OPENBLAS_PATH "${OPENBLAS_PATH}" PARENT_SCOPE)

    # Verify the path exists and has the required headers
    if(NOT EXISTS "${OPENBLAS_PATH}/include")
        message(STATUS "❌ OpenBLAS include directory not found: ${OPENBLAS_PATH}/include")
        return()
    endif()

    if(NOT EXISTS "${OPENBLAS_PATH}/include/cblas.h")
        message(STATUS "❌ OpenBLAS cblas.h not found: ${OPENBLAS_PATH}/include/cblas.h")
        return()
    endif()

    # Set up OpenBLAS
    message(STATUS "✅ Setting up OpenBLAS:")
    message(STATUS "   Path: ${OPENBLAS_PATH}")
    message(STATUS "   Include: ${OPENBLAS_PATH}/include")
    message(STATUS "   Library: ${OPENBLAS_PATH}/")

    # Use global include_directories for compatibility
    include_directories(${OPENBLAS_PATH}/include/)

    # Find the actual OpenBLAS library file
    set(OPENBLAS_LIB_FOUND "")
    foreach(lib_candidate
            "${OPENBLAS_PATH}/lib/libopenblas.so"
            "${OPENBLAS_PATH}/lib/libopenblas.so.0"
            "${OPENBLAS_PATH}/libopenblas.so"
            "${OPENBLAS_PATH}/libopenblas.so.0"
            "${OPENBLAS_PATH}/lib/libopenblas.dylib"
            "${OPENBLAS_PATH}/libopenblas.dylib"
            "${OPENBLAS_PATH}/lib/libopenblas.a"
            "${OPENBLAS_PATH}/libopenblas.a")
        if(EXISTS "${lib_candidate}" AND NOT OPENBLAS_LIB_FOUND)
            set(OPENBLAS_LIB_FOUND "${lib_candidate}")
            message(STATUS "   Found OpenBLAS library: ${OPENBLAS_LIB_FOUND}")
        endif()
    endforeach()

    if(NOT OPENBLAS_LIB_FOUND)
        message(WARNING "⚠️  Could not find libopenblas.so or libopenblas.a in ${OPENBLAS_PATH}")
        # Fallback to link_directories approach
        if(EXISTS "${OPENBLAS_PATH}/lib")
            link_directories(${OPENBLAS_PATH}/lib)
        endif()
        link_directories(${OPENBLAS_PATH})
        set(OPENBLAS_LIB_FOUND "openblas")
    endif()

    # A same-architecture Linux ELF can satisfy the Android link but will
    # fail on-device because it references glibc and the Linux program loader.
    # Fail closed here so packaging can never turn that into a plausible AAR.
    if((ANDROID OR SD_ANDROID_BUILD) AND OPENBLAS_LIB_FOUND MATCHES "\\.so(\\.[0-9]+)*$")
        if(NOT CMAKE_READELF)
            message(FATAL_ERROR
                "Android OpenBLAS validation requires CMAKE_READELF from the NDK toolchain")
        endif()
        execute_process(
            COMMAND "${CMAKE_READELF}" -d "${OPENBLAS_LIB_FOUND}"
            RESULT_VARIABLE _sd_openblas_readelf_status
            OUTPUT_VARIABLE _sd_openblas_dynamic
            ERROR_VARIABLE _sd_openblas_readelf_error)
        if(NOT _sd_openblas_readelf_status EQUAL 0)
            message(FATAL_ERROR
                "Could not inspect Android OpenBLAS ${OPENBLAS_LIB_FOUND}: "
                "${_sd_openblas_readelf_error}")
        endif()
        if(_sd_openblas_dynamic MATCHES
                "libc\\.so\\.6|libpthread\\.so\\.0|libgfortran|ld-linux")
            message(FATAL_ERROR
                "Resolved OpenBLAS is a Linux/glibc binary, not Android: "
                "${OPENBLAS_LIB_FOUND}")
        endif()
    endif()

    add_compile_definitions(HAVE_OPENBLAS=1)
    # Note: OpenBLAS bfloat16 conflict is handled in BlasHelper.h via:
    #   struct bfloat16;
    #   #define BFLOAT16 BFLOAT16
    # This self-referential macro prevents OpenBLAS from typedef'ing bfloat16
    # while not breaking libnd4j's BFLOAT16 enum value in DataType.h

    # Set parent scope variables
    set(HAVE_OPENBLAS 1 PARENT_SCOPE)
    set(OPENBLAS_LIBRARIES "${OPENBLAS_LIB_FOUND}" PARENT_SCOPE)

    set(OPENBLAS_PATH "${OPENBLAS_PATH}" PARENT_SCOPE)

    message(STATUS "✅ OpenBLAS setup complete")
endfunction()




function(setup_cudnn)
    set(HAVE_CUDNN false PARENT_SCOPE)
    set(CUDNN "" PARENT_SCOPE)

    if(NOT (HELPERS_cudnn STREQUAL "ON" AND SD_CUDA))
        message(STATUS "cuDNN helper is disabled (HELPERS_cudnn=${HELPERS_cudnn}, SD_CUDA=${SD_CUDA})")
        return()
    endif()
endfunction()
# =============================================================================
# FLATBUFFERS (Required) - Cross-compilation compatible version
# =============================================================================
function(setup_flatbuffers)
    set(FLATBUFFERS_VERSION "25.2.10")
    # ExternalProject tries URL entries in order. Keep both canonical GitHub
    # archive forms so a transient edge failure does not abort a release shard.
    set(FLATBUFFERS_URL
        "https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VERSION}.tar.gz"
        "https://codeload.github.com/google/flatbuffers/tar.gz/refs/tags/v${FLATBUFFERS_VERSION}"
    )
    set(FLATBUFFERS_URL_HASH "SHA256=b9c2df49707c57a48fc0923d52b8c73beb72d675f9d44b2211e4569be40a7421")

    # MSVC produces flatbuffers.lib; GCC/MinGW/Clang produce libflatbuffers.a
    if(MSVC)
        set(FLATBUFFERS_LIB_NAME "flatbuffers.lib")
    else()
        set(FLATBUFFERS_LIB_NAME "libflatbuffers.a")
    endif()

    # --- Dependency cache check ---
    if(SD_DEP_CACHE AND NOT CMAKE_CROSSCOMPILING)
        set(_fb_flatc_flag "FLATC=OFF")
        if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
            set(_fb_flatc_flag "FLATC=ON")
        endif()
        sd_dep_cache_key("flatbuffers" "${FLATBUFFERS_VERSION}" "${_fb_flatc_flag}" _fb_cache_key)
        sd_dep_cache_check("flatbuffers" "${_fb_cache_key}" _fb_hit _fb_cache_path)
        if(_fb_hit)
            # Restore cached artifacts
            set(_fb_restore_dir "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-cached")
            sd_dep_cache_restore("flatbuffers" "${_fb_cache_path}" "${_fb_restore_dir}")
            # Set up the same interface as a normal build
            set(FLATBUFFERS_LIBRARY "${_fb_restore_dir}/lib/${FLATBUFFERS_LIB_NAME}")
            set(FLATBUFFERS_SOURCE_DIR "${_fb_restore_dir}")
            if(NOT TARGET flatbuffers_external)
                add_custom_target(flatbuffers_external)
            endif()
            add_library(flatbuffers_interface INTERFACE)
            target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})
            target_include_directories(flatbuffers_interface INTERFACE "${_fb_restore_dir}/include")
            add_dependencies(flatbuffers_interface flatbuffers_external)
            # Copy flatbuffers headers to project include dir if needed
            if(EXISTS "${_fb_restore_dir}/include/flatbuffers")
                execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${_fb_restore_dir}/include/flatbuffers"
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers")
            endif()
            set(FLATBUFFERS_LIBRARY ${FLATBUFFERS_LIBRARY} PARENT_SCOPE)
            set(FLATBUFFERS_SOURCE_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
            message(STATUS "✅ FlatBuffers setup complete (from cache)")
            return()
        endif()
    endif()

    # Determine if we should build flatc
    set(SHOULD_BUILD_FLATC FALSE)
    if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
        set(SHOULD_BUILD_FLATC TRUE)
    endif()

    if(CMAKE_CROSSCOMPILING AND SHOULD_BUILD_FLATC)
        # Cross-compilation scenario: build flatc for host, library for target
        message(STATUS "Cross-compiling FlatBuffers: building flatc for host, library for target")

        # Stage 1: Build flatc for host system
        set(FLATC_HOST_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-host")
        set(FLATC_HOST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-host-build")
        set(FLATC_EXECUTABLE "${FLATC_HOST_BUILD_DIR}/flatc")

        # Determine host system compilers
        find_program(HOST_C_COMPILER NAMES gcc clang cc)
        find_program(HOST_CXX_COMPILER NAMES g++ clang++ c++)

        if(NOT HOST_C_COMPILER OR NOT HOST_CXX_COMPILER)
            message(FATAL_ERROR "Could not find host system compilers for flatc build")
        endif()

        # Build CMAKE_ARGS without toolchain file for host build
        set(HOST_CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DFLATBUFFERS_BUILD_FLATC=ON
                -DFLATBUFFERS_BUILD_FLATLIB=OFF
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SAMPLES=OFF
                -DCMAKE_C_COMPILER=${HOST_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${HOST_CXX_COMPILER}
        )

        # Pass compiler launcher (ccache/sccache) to host build if available
        # Smart ccache uses a multi-element list (python + script + args) which can't
        # be passed as FILEPATH. Fall back to SD_PLAIN_CCACHE_PATH for ExternalProject.
        if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}" AND NOT CMAKE_C_COMPILER_LAUNCHER MATCHES "\\.sh$")
            list(APPEND HOST_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER}")
        elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
            list(APPEND HOST_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
        endif()
        if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}" AND NOT CMAKE_CXX_COMPILER_LAUNCHER MATCHES "\\.sh$")
            list(APPEND HOST_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER}")
        elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
            list(APPEND HOST_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
        endif()

        ExternalProject_Add(flatbuffers_host
                URL               ${FLATBUFFERS_URL}
                URL_HASH          ${FLATBUFFERS_URL_HASH}
                SOURCE_DIR        "${FLATC_HOST_DIR}"
                BINARY_DIR        "${FLATC_HOST_BUILD_DIR}"
                CMAKE_ARGS        ${HOST_CMAKE_ARGS}
                BUILD_COMMAND     ${CMAKE_COMMAND} --build . --target flatc --config Release --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${FLATC_EXECUTABLE}"
                ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        # Stage 2: Build FlatBuffers library for target
        # Build CMAKE_ARGS for target build, only include variables that are set
        set(TARGET_CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_BUILD_TYPE=Release
                -DFLATBUFFERS_BUILD_FLATC=OFF
                -DFLATBUFFERS_BUILD_FLATLIB=ON
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SAMPLES=OFF
        )

        # Pass compiler launcher (ccache/sccache) to target build if available
        # Smart ccache multi-element lists can't be passed as FILEPATH; fall back to plain ccache.
        if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}" AND NOT CMAKE_C_COMPILER_LAUNCHER MATCHES "\\.sh$")
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER})
        elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH})
        endif()
        if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}" AND NOT CMAKE_CXX_COMPILER_LAUNCHER MATCHES "\\.sh$")
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER})
        elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH})
        endif()

        # Only add cross-compilation arguments if they are defined
        if(CMAKE_TOOLCHAIN_FILE)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
        endif()
        if(CMAKE_SYSTEM_NAME)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})
        endif()
        if(CMAKE_SYSTEM_VERSION)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION})
        endif()
        if(CMAKE_ANDROID_ARCH_ABI)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI})
        endif()
        if(CMAKE_ANDROID_NDK)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK})
        endif()
        if(CMAKE_ANDROID_STL_TYPE)
            list(APPEND TARGET_CMAKE_ARGS -DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE})
        endif()
        if(ANDROID_ABI)
            list(APPEND TARGET_CMAKE_ARGS -DANDROID_ABI=${ANDROID_ABI})
        endif()
        if(ANDROID_PLATFORM)
            list(APPEND TARGET_CMAKE_ARGS -DANDROID_PLATFORM=${ANDROID_PLATFORM})
        endif()
        if(ANDROID_STL)
            list(APPEND TARGET_CMAKE_ARGS -DANDROID_STL=${ANDROID_STL})
        endif()

        ExternalProject_Add(flatbuffers_target
                URL               ${FLATBUFFERS_URL}
                URL_HASH          ${FLATBUFFERS_URL_HASH}
                SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src"
                BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-build"
                CMAKE_ARGS        ${TARGET_CMAKE_ARGS}
                BUILD_COMMAND     ${CMAKE_COMMAND} --build . --config Release --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-build/libflatbuffers.a"
                DEPENDS           flatbuffers_host
                ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        # DO NOT use include_directories() - use target_include_directories on flatbuffers_interface instead
        # Set up include directories and library
        # include_directories("${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src/include")
        set(FLATBUFFERS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-build/libflatbuffers.a")
        set(FLATBUFFERS_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src")

        # Create interface library for target
        add_library(flatbuffers_interface INTERFACE)
        target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})
        target_include_directories(flatbuffers_interface INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-target-src/include")
        add_dependencies(flatbuffers_interface flatbuffers_target)

        # Check if flatbuffers.h already exists
        set(FLATBUFFERS_HEADER_DEST "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers/flatbuffers.h")
        if(EXISTS ${FLATBUFFERS_HEADER_DEST})
            message(STATUS "Found existing flatbuffers.h at ${FLATBUFFERS_HEADER_DEST}")
        endif()

        # Generate headers and copy Java files inline after ExternalProject builds
        # Copy ALL flatbuffers headers (modular structure in newer versions)
        ExternalProject_Add_Step(flatbuffers_host generate_headers_and_copy_java
                COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}"
                bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
                COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
                COMMAND ${CMAKE_COMMAND} -E remove_directory
                "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-host/include/flatbuffers"
                "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Generating FlatBuffers headers, copying Java files, and copying all flatbuffers headers using host flatc"
                DEPENDEES build
                BYPRODUCTS
                ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h
                ${CMAKE_CURRENT_SOURCE_DIR}/.java_files_copied
        )

    else()
        # Native build or cross-compilation without flatc generation
        message(STATUS "Native FlatBuffers build")

        if(SHOULD_BUILD_FLATC)
            set(FLATBUFFERS_BUILD_FLATC "ON")
        else()
            set(FLATBUFFERS_BUILD_FLATC "OFF")
        endif()

        # Build CMAKE_ARGS list for native build
        set(NATIVE_CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_BUILD_TYPE=Release
                -DFLATBUFFERS_BUILD_FLATC=${FLATBUFFERS_BUILD_FLATC}
                -DFLATBUFFERS_BUILD_FLATLIB=ON
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_SAMPLES=OFF
        )

        # Pass compiler launcher (ccache/sccache) to native build if available
        # Smart ccache multi-element lists can't be passed as FILEPATH; fall back to plain ccache.
        if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}" AND NOT CMAKE_C_COMPILER_LAUNCHER MATCHES "\\.sh$")
            list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER})
        elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
            list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH})
        endif()
        if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}" AND NOT CMAKE_CXX_COMPILER_LAUNCHER MATCHES "\\.sh$")
            list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER})
        elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
            list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH})
        endif()

        # Cross-compilation without flatc generation still needs the target
        # toolchain. Passing only clang/clang++ loses the NDK target triple and
        # silently produces a host archive that cannot be linked into an AAR.
        if(CMAKE_CROSSCOMPILING)
            if(CMAKE_TOOLCHAIN_FILE)
                list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
            endif()
            if(CMAKE_SYSTEM_NAME)
                list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})
            endif()
            if(CMAKE_SYSTEM_VERSION)
                list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION})
            endif()
            if(CMAKE_ANDROID_ARCH_ABI)
                list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI})
            endif()
            if(CMAKE_ANDROID_NDK)
                list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK})
            endif()
            if(CMAKE_ANDROID_STL_TYPE)
                list(APPEND NATIVE_CMAKE_ARGS -DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE})
            endif()
            if(ANDROID_ABI)
                list(APPEND NATIVE_CMAKE_ARGS -DANDROID_ABI=${ANDROID_ABI})
            endif()
            if(ANDROID_PLATFORM)
                list(APPEND NATIVE_CMAKE_ARGS -DANDROID_PLATFORM=${ANDROID_PLATFORM})
            endif()
            if(ANDROID_STL)
                list(APPEND NATIVE_CMAKE_ARGS -DANDROID_STL=${ANDROID_STL})
            endif()
        endif()

        ExternalProject_Add(flatbuffers_external
                URL               ${FLATBUFFERS_URL}
                URL_HASH          ${FLATBUFFERS_URL_HASH}
                SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src"
                BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build"
                CMAKE_ARGS        ${NATIVE_CMAKE_ARGS}
                # Use computed parallel jobs for faster dependency builds
                BUILD_COMMAND     ${CMAKE_COMMAND} --build . --config Release --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc"
                "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/${FLATBUFFERS_LIB_NAME}"
                ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        # DO NOT use include_directories() - use target_include_directories on flatbuffers_interface instead
        # include_directories("${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")
        set(FLATBUFFERS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/${FLATBUFFERS_LIB_NAME}")
        set(FLATBUFFERS_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src")

        # Check if flatbuffers.h already exists
        set(FLATBUFFERS_HEADER_DEST "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers/flatbuffers.h")
        if(EXISTS ${FLATBUFFERS_HEADER_DEST})
            message(STATUS "Found existing flatbuffers.h at ${FLATBUFFERS_HEADER_DEST}")
        endif()

        if(SHOULD_BUILD_FLATC)
            set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")
            if(DEFINED ENV{DL4J_FLATC_EXECUTABLE} AND EXISTS "$ENV{DL4J_FLATC_EXECUTABLE}")
                set(FLATC_EXECUTABLE "$ENV{DL4J_FLATC_EXECUTABLE}")
                message(STATUS "Using prebuilt host flatc: ${FLATC_EXECUTABLE}")
            endif()

            # Generate headers and copy Java files inline after ExternalProject builds
            # Copy ALL flatbuffers headers (modular structure in newer versions)
            ExternalProject_Add_Step(flatbuffers_external generate_headers_and_copy_java
                    COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}"
                    bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
                    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
                    COMMAND ${CMAKE_COMMAND} -E remove_directory
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include/flatbuffers"
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating FlatBuffers headers, copying Java files, and copying all flatbuffers headers"
                    DEPENDEES build
                    BYPRODUCTS
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/.java_files_copied
            )
        else()
            # Even without flatc generation, copy ALL flatbuffers headers
            # Newer flatbuffers uses modular headers - flatbuffers.h includes array.h, base.h, vector.h, etc.
            ExternalProject_Add_Step(flatbuffers_external copy_flatbuffers_headers
                    COMMAND ${CMAKE_COMMAND} -E remove_directory
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include/flatbuffers"
                    "${CMAKE_SOURCE_DIR}/libnd4j/include/flatbuffers"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying all flatbuffers headers (modular structure)"
                    DEPENDEES build
            )
        endif()

        # Create interface library
        add_library(flatbuffers_interface INTERFACE)
        target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})
        target_include_directories(flatbuffers_interface INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")
        add_dependencies(flatbuffers_interface flatbuffers_external)
    endif()

    # --- Cache store for native build ---
    if(SD_DEP_CACHE AND NOT CMAKE_CROSSCOMPILING AND DEFINED _fb_cache_key)
        # Create a store script that gathers flatbuffers artifacts into a staging dir, then caches
        set(_fb_cache_dir "${SD_DEP_CACHE_DIR}/flatbuffers/${_fb_cache_key}")
        # Normalize paths to forward slashes for Windows compatibility in generated scripts
        string(REPLACE "\\" "/" _fb_cache_dir "${_fb_cache_dir}")
        set(_fb_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")
        string(REPLACE "\\" "/" _fb_binary_dir "${_fb_binary_dir}")
        set(_fb_store_script "${CMAKE_BINARY_DIR}/dep_cache_store_flatbuffers.cmake")
        file(WRITE "${_fb_store_script}" "
            set(_cache_install \"${_fb_cache_dir}/install\")
            set(_marker \"${_fb_cache_dir}/.cache_complete\")
            if(NOT EXISTS \"\${_marker}\")
                message(STATUS \"DEP-CACHE [flatbuffers] Storing artifacts to cache\")
                file(MAKE_DIRECTORY \"\${_cache_install}/include\")
                file(MAKE_DIRECTORY \"\${_cache_install}/lib\")
                execute_process(COMMAND \${CMAKE_COMMAND} -E copy_directory
                    \"${_fb_binary_dir}/flatbuffers-src/include\"
                    \"\${_cache_install}/include\")
                execute_process(COMMAND \${CMAKE_COMMAND} -E copy_if_different
                    \"${_fb_binary_dir}/flatbuffers-build/${FLATBUFFERS_LIB_NAME}\"
                    \"\${_cache_install}/lib/${FLATBUFFERS_LIB_NAME}\")
                file(WRITE \"\${_marker}\" \"cached\")
                message(STATUS \"DEP-CACHE [flatbuffers] Cache stored successfully\")
            endif()
        ")
        ExternalProject_Add_Step(flatbuffers_external dep_cache_store
            COMMAND ${CMAKE_COMMAND} -P "${_fb_store_script}"
            DEPENDEES build
            COMMENT "DEP-CACHE [flatbuffers] Storing build artifacts to persistent cache"
        )
    endif()

    # Set global variables for parent scope
    set(FLATBUFFERS_LIBRARY ${FLATBUFFERS_LIBRARY} PARENT_SCOPE)
    set(FLATBUFFERS_SOURCE_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
    if(SHOULD_BUILD_FLATC)
        set(FLATC_EXECUTABLE ${FLATC_EXECUTABLE} PARENT_SCOPE)
    endif()

    message(STATUS "✅ FlatBuffers setup complete")
    if(CMAKE_CROSSCOMPILING AND SHOULD_BUILD_FLATC)
        message(STATUS "   Host flatc: ${FLATC_EXECUTABLE}")
        message(STATUS "   Target library: ${FLATBUFFERS_LIBRARY}")
    else()
        message(STATUS "   Library: ${FLATBUFFERS_LIBRARY}")
        if(SHOULD_BUILD_FLATC)
            message(STATUS "   flatc: ${FLATC_EXECUTABLE}")
        endif()
    endif()
endfunction()
# =============================================================================
# ONEDNN (Optional)
# =============================================================================
function(setup_onednn)
    if(NOT HELPERS_onednn STREQUAL "ON")
        message(STATUS "OneDNN helper is disabled (HELPERS_onednn=${HELPERS_onednn})")
        set(HAVE_ONEDNN OFF CACHE BOOL "OneDNN availability" FORCE)
        set(ONEDNN "" PARENT_SCOPE)
        return()
    endif()

    # OneDNN only supports x86/x64 platforms in our build configuration.
    # ARM64 builds with OneDNN fail because OneDNN's CMake adds x86-specific
    # compiler flags (e.g. -msse4.1) that are invalid on AArch64.
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386")
        message(STATUS "OneDNN helper requested but platform ${CMAKE_SYSTEM_PROCESSOR} is not x86 — skipping")
        set(HAVE_ONEDNN OFF CACHE BOOL "OneDNN availability" FORCE)
        set(ONEDNN "" PARENT_SCOPE)
        return()
    endif()

    if(TARGET onednn_external)
        message(STATUS "OneDNN helper is enabled (target already exists)")
        set(HAVE_ONEDNN ON CACHE BOOL "OneDNN availability" FORCE)
        set(ONEDNN onednn_interface PARENT_SCOPE)
        return()
    endif()

    message(STATUS "OneDNN helper is enabled")
    set(HAVE_ONEDNN ON CACHE BOOL "OneDNN availability" FORCE)
    set(ONEDNN_INSTALL_DIR "${CMAKE_BINARY_DIR}/onednn_install")

    # Use one platform-derived library directory for both fresh installs and
    # dependency-cache restores. MinGW installs oneDNN into lib, not lib64.
    if(WIN32)
        set(ONEDNN_LIB_DIR "lib")
    else()
        set(ONEDNN_LIB_DIR "lib64")
    endif()

    # --- Dependency cache check ---
    set(ONEDNN_VERSION "3.8.1")
    if(SD_DEP_CACHE)
        sd_dep_cache_key("onednn" "${ONEDNN_VERSION}" "${CMAKE_BUILD_TYPE};STATIC;GRAPH=ON" _onednn_cache_key)
        sd_dep_cache_check("onednn" "${_onednn_cache_key}" _onednn_hit _onednn_cache_path)
        if(_onednn_hit)
            sd_dep_cache_restore("onednn" "${_onednn_cache_path}" "${ONEDNN_INSTALL_DIR}")
            if(NOT TARGET onednn_external)
                add_custom_target(onednn_external)
            endif()
            add_library(onednn_interface INTERFACE)
            target_include_directories(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/include")
            if(MSVC)
                target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/${ONEDNN_LIB_DIR}/dnnl.lib")
            else()
                target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/${ONEDNN_LIB_DIR}/libdnnl.a")
            endif()
            add_dependencies(onednn_interface onednn_external)
            set(ONEDNN onednn_interface PARENT_SCOPE)
            message(STATUS "✅ OneDNN ${ONEDNN_VERSION} setup complete (from cache)")
            return()
        endif()
    endif()
    set(ONEDNN_PREFIX "${CMAKE_BINARY_DIR}/onednn_external")
    set(ONEDNN_VERSION "3.8.1")
    set(ONEDNN_STAMP_DIR "${ONEDNN_PREFIX}/stamp")

    # Ensure stamp directory exists to prevent "cmake -E touch: failed to update" errors
    # This can happen when parallel builds race on directory creation
    file(MAKE_DIRECTORY "${ONEDNN_STAMP_DIR}")
    file(MAKE_DIRECTORY "${ONEDNN_PREFIX}/src")
    file(MAKE_DIRECTORY "${ONEDNN_PREFIX}/build")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")

    # Clean up stale stamp files that can cause "Failed to copy script-last-run stamp file" errors
    if(EXISTS "${ONEDNN_STAMP_DIR}")
        file(GLOB STALE_STAMPS "${ONEDNN_STAMP_DIR}/*-lastrun.txt" "${ONEDNN_STAMP_DIR}/*.txt")
        foreach(stamp ${STALE_STAMPS})
            message(STATUS "Cleaning stale OneDNN stamp file: ${stamp}")
            file(REMOVE "${stamp}")
        endforeach()
    endif()

    # Use URL download instead of git clone for more robust downloads
    set(ONEDNN_URL "https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz")
    set(ONEDNN_URL_HASH "SHA256=4b0638061a789a1efbefdcd2e85eb257c7b432b3b6a71ba8909e19d75f50b163")

    # Build CMAKE_ARGS list for OneDNN
    set(ONEDNN_CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}
            -DCMAKE_INSTALL_LIBDIR=${ONEDNN_LIB_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DDNNL_LIBRARY_TYPE=STATIC
            -DDNNL_CPU_RUNTIME=OMP
            -DDNNL_BUILD_TESTS=OFF
            -DDNNL_BUILD_EXAMPLES=OFF
            -DDNNL_VERBOSE=OFF
            -DONEDNN_BUILD_GRAPH=ON
            -DDNNL_ENABLE_CONCURRENT_EXEC=ON
            -DDNNL_ENABLE_JIT_PROFILING=OFF
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )

    # Pass generator to OneDNN so it matches the parent build system
    if(CMAKE_GENERATOR)
        list(APPEND ONEDNN_CMAKE_ARGS "-G${CMAKE_GENERATOR}")
    endif()

    # On Windows/MSYS2, use SEQ runtime instead of OMP — oneDNN's OpenMP.cmake
    # can't find OpenMP on MinGW and fails at configure time.
    if(WIN32 OR MINGW OR MSYS)
        list(REMOVE_ITEM ONEDNN_CMAKE_ARGS "-DDNNL_CPU_RUNTIME=OMP")
        list(APPEND ONEDNN_CMAKE_ARGS "-DDNNL_CPU_RUNTIME=SEQ")
        message(STATUS "   OneDNN using SEQ runtime on Windows/MSYS2 (OpenMP not available)")
    endif()

    # Pass compiler launcher (ccache/sccache) to OneDNN build if available
    # Smart ccache multi-element lists can't be passed as FILEPATH; fall back to plain ccache.
    if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}" AND NOT CMAKE_C_COMPILER_LAUNCHER MATCHES "\\.sh$")
        list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER}")
        message(STATUS "   Passing C compiler launcher to OneDNN: ${CMAKE_C_COMPILER_LAUNCHER}")
    elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
        list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
        message(STATUS "   Passing plain ccache to OneDNN C: ${SD_PLAIN_CCACHE_PATH}")
    endif()
    if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}" AND NOT CMAKE_CXX_COMPILER_LAUNCHER MATCHES "\\.sh$")
        list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER}")
        message(STATUS "   Passing CXX compiler launcher to OneDNN: ${CMAKE_CXX_COMPILER_LAUNCHER}")
    elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
        list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
        message(STATUS "   Passing plain ccache to OneDNN CXX: ${SD_PLAIN_CCACHE_PATH}")
    endif()

    # Pass the complete Android NDK toolchain contract to oneDNN. CMake's
    # CMAKE_ANDROID_* variables alone are not sufficient for the NDK toolchain:
    # without ANDROID_ABI it defaults to a 32-bit ABI and oneDNN rejects it.
    if(CMAKE_TOOLCHAIN_FILE)
        list(APPEND ONEDNN_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    endif()
    if(CMAKE_SYSTEM_NAME)
        list(APPEND ONEDNN_CMAKE_ARGS -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})
    endif()
    if(CMAKE_SYSTEM_VERSION AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
        list(APPEND ONEDNN_CMAKE_ARGS -DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION})
    endif()
    if(CMAKE_ANDROID_ARCH_ABI)
        list(APPEND ONEDNN_CMAKE_ARGS -DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI})
    endif()
    if(CMAKE_ANDROID_NDK)
        list(APPEND ONEDNN_CMAKE_ARGS -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK})
    endif()
    if(CMAKE_ANDROID_STL_TYPE)
        list(APPEND ONEDNN_CMAKE_ARGS -DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE})
    endif()
    if(ANDROID_ABI)
        list(APPEND ONEDNN_CMAKE_ARGS -DANDROID_ABI=${ANDROID_ABI})
    elseif(CMAKE_ANDROID_ARCH_ABI)
        list(APPEND ONEDNN_CMAKE_ARGS -DANDROID_ABI=${CMAKE_ANDROID_ARCH_ABI})
    endif()
    if(ANDROID_PLATFORM)
        list(APPEND ONEDNN_CMAKE_ARGS -DANDROID_PLATFORM=${ANDROID_PLATFORM})
    endif()
    if(ANDROID_STL)
        list(APPEND ONEDNN_CMAKE_ARGS -DANDROID_STL=${ANDROID_STL})
    endif()

    ExternalProject_Add(onednn_external
            PREFIX            "${ONEDNN_PREFIX}"
            URL               "${ONEDNN_URL}"
            URL_HASH          "${ONEDNN_URL_HASH}"
            DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
            SOURCE_DIR        "${ONEDNN_PREFIX}/src/oneDNN-${ONEDNN_VERSION}"
            BINARY_DIR        "${ONEDNN_PREFIX}/build"
            STAMP_DIR         "${ONEDNN_STAMP_DIR}"
            DOWNLOAD_NO_PROGRESS FALSE
            ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
            CMAKE_ARGS        ${ONEDNN_CMAKE_ARGS}
            BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE} --parallel ${DEP_PARALLEL_JOBS}
            INSTALL_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config ${CMAKE_BUILD_TYPE}
            BUILD_BYPRODUCTS
                "${ONEDNN_INSTALL_DIR}/include/dnnl.h"
                "${ONEDNN_INSTALL_DIR}/${ONEDNN_LIB_DIR}/libdnnl.a"
                "${ONEDNN_INSTALL_DIR}/${ONEDNN_LIB_DIR}/dnnl.lib"
            TIMEOUT           900
            LOG_DOWNLOAD      OFF
            LOG_CONFIGURE     OFF
            LOG_BUILD         OFF
            LOG_INSTALL       OFF
    )

    add_library(onednn_interface INTERFACE)
    target_include_directories(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/include")
    if(MSVC)
        # MSVC produces dnnl.lib
        target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/${ONEDNN_LIB_DIR}/dnnl.lib")
    else()
        # GCC/MinGW/Clang all produce libdnnl.a (including MinGW on Windows where WIN32=true)
        target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/${ONEDNN_LIB_DIR}/libdnnl.a")
    endif()
    add_dependencies(onednn_interface onednn_external)
    set(ONEDNN onednn_interface PARENT_SCOPE)

    # --- Cache store ---
    if(SD_DEP_CACHE AND DEFINED _onednn_cache_key)
        sd_dep_cache_store("onednn" "${_onednn_cache_key}" "${ONEDNN_INSTALL_DIR}" "onednn_external")
    endif()

    message(STATUS "✅ OneDNN ${ONEDNN_VERSION} setup complete (using URL download)")
endfunction()

# =============================================================================
# ARM COMPUTE LIBRARY (Optional)
# =============================================================================
function(setup_armcompute)
    set(HAVE_ARMCOMPUTE 0 PARENT_SCOPE)
    if(NOT HELPERS_armcompute STREQUAL "ON")
        message(STATUS "ARM Compute helper is disabled (HELPERS_armcompute=${HELPERS_armcompute})")
        return()
    endif()

    # Android Tensor G3 must use the prevalidated Android/NDK libc++ ACL
    # artifact. The generic aarch64 package below is a Linux/libstdc++ DSO;
    # linking it into an Android consumer produces unresolved std::__ndk1
    # symbols at the final link (and can otherwise fail much later at load time).
    if(SD_NNAPI_TENSOR_G3_HYBRID AND
       (ANDROID OR SD_ANDROID_BUILD OR CMAKE_SYSTEM_NAME STREQUAL "Android"))
        set(SDX_TENSOR_G3_ARMCOMPUTE_VERSION "v25.04" CACHE STRING
            "Pinned ARM Compute release for the Tensor G3 Android provider")
        set(SDX_TENSOR_G3_ARMCOMPUTE_CACHE_KEY "v25.04-a0b2f80b" CACHE STRING
            "Dependency-cache key for the prevalidated Tensor G3 Android ACL artifact")
        if(NOT SDX_TENSOR_G3_ARMCOMPUTE_VERSION STREQUAL "v25.04")
            message(FATAL_ERROR
                "Tensor G3 Android requires ARM Compute v25.04; got "
                "${SDX_TENSOR_G3_ARMCOMPUTE_VERSION}")
        endif()

        if(TARGET armcompute_external OR TARGET armcompute_interface)
            get_property(_tensor_g3_configured GLOBAL PROPERTY
                SD_TENSOR_G3_ARMCOMPUTE_CONFIGURED)
            if(NOT _tensor_g3_configured)
                message(FATAL_ERROR
                    "Tensor G3 ARM Compute target names are already owned by a "
                    "non-Tensor configuration; refusing an ABI-ambiguous mix")
            endif()
            set(HAVE_ARMCOMPUTE 1 PARENT_SCOPE)
            set(ARMCOMPUTE_LIBRARIES armcompute_interface PARENT_SCOPE)
            return()
        endif()

        if(NOT SD_DEP_CACHE)
            message(FATAL_ERROR
                "Tensor G3 Android requires the prevalidated Android ARM Compute "
                "dependency cache; refusing the Linux ARM Compute fallback")
        endif()
        sd_dep_cache_check("tensor-g3-armcompute"
            "${SDX_TENSOR_G3_ARMCOMPUTE_CACHE_KEY}" _tensor_g3_cache_hit
            _tensor_g3_cache_path)
        if(NOT _tensor_g3_cache_hit)
            message(FATAL_ERROR
                "Tensor G3 Android ARM Compute cache miss for "
                "${SDX_TENSOR_G3_ARMCOMPUTE_CACHE_KEY}; refusing to download or "
                "link the Linux package. Populate the validated Android cache first.")
        endif()

        set(ARMCOMPUTE_INSTALL_DIR
            "${CMAKE_BINARY_DIR}/tensor_g3_armcompute_install")
        sd_dep_cache_restore("tensor-g3-armcompute"
            "${_tensor_g3_cache_path}" "${ARMCOMPUTE_INSTALL_DIR}")
        set(_tensor_g3_shared_library
            "${ARMCOMPUTE_INSTALL_DIR}/lib/armv8a-neon/libarm_compute.so")
        set(_tensor_g3_required_header
            "${ARMCOMPUTE_INSTALL_DIR}/arm_compute/runtime/NEON/NEFunctions.h")
        if(NOT EXISTS "${_tensor_g3_shared_library}" OR
           NOT EXISTS "${_tensor_g3_required_header}")
            message(FATAL_ERROR
                "Tensor G3 Android ARM Compute cache is incomplete: expected "
                "${_tensor_g3_shared_library} and ${_tensor_g3_required_header}")
        endif()

        add_custom_target(armcompute_external)
        add_library(armcompute_interface INTERFACE)
        target_include_directories(armcompute_interface INTERFACE
            "${ARMCOMPUTE_INSTALL_DIR}"
            "${ARMCOMPUTE_INSTALL_DIR}/include")
        # Link the exact Android DSO by path. Do not add the generic Linux
        # armcompute_install directory or libarm_compute_graph.so.
        target_link_libraries(armcompute_interface INTERFACE
            "${_tensor_g3_shared_library}")
        add_dependencies(armcompute_interface armcompute_external)

        set(ARMCOMPUTE_LINK_MODE "BUNDLED_SHARED" CACHE STRING
            "ARM Compute linkage mode" FORCE)
        set(ARMCOMPUTE_SHARED_LIBRARY "${_tensor_g3_shared_library}"
            CACHE FILEPATH "Pinned bundled ACL DSO used by Tensor G3" FORCE)
        set(ARMCOMPUTE_LIBRARIES armcompute_interface PARENT_SCOPE)
        set(HAVE_ARMCOMPUTE 1 PARENT_SCOPE)
        set_property(GLOBAL PROPERTY SD_TENSOR_G3_ARMCOMPUTE_CONFIGURED TRUE)
        message(STATUS
            "Tensor G3 ACL/NEON: pinned v25.04 Android AArch64 bundled DSO")
        return()
    endif()

    if(TARGET armcompute_external)
        set(HAVE_ARMCOMPUTE 1 PARENT_SCOPE)
        set(ARMCOMPUTE_LIBRARIES armcompute_interface PARENT_SCOPE)
        return()
    endif()

    if(LIBND4J_BUILD_WITH_ARMCOMPUTE AND (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64"))
        set(ARMCOMPUTE_INSTALL_DIR "${CMAKE_BINARY_DIR}/armcompute_install")
        set(ARMCOMPUTE_VERSION "v25.04")
        set(ARMCOMPUTE_ARCH "aarch64")
        set(ARMCOMPUTE_PLATFORM "linux")
        set(ARMCOMPUTE_FLAVOR "cpu")
        set(ARMCOMPUTE_PKG_NAME "arm_compute-${ARMCOMPUTE_VERSION}-${ARMCOMPUTE_PLATFORM}-${ARMCOMPUTE_ARCH}-${ARMCOMPUTE_FLAVOR}-bin")
        set(ARMCOMPUTE_URL "https://github.com/ARM-software/ComputeLibrary/releases/download/${ARMCOMPUTE_VERSION}/${ARMCOMPUTE_PKG_NAME}.tar.gz")
        set(ARMCOMPUTE_URL_HASH "SHA256=c7296ddb163a14da239b896b44dcb4b5d73d79623c4ba83c8ad1b8e653a99c92")

        # --- Dependency cache check ---
        if(SD_DEP_CACHE)
            sd_dep_cache_key("armcompute" "${ARMCOMPUTE_VERSION}" "${ARMCOMPUTE_PLATFORM}-${ARMCOMPUTE_ARCH}-${ARMCOMPUTE_FLAVOR}" _ac_cache_key)
            sd_dep_cache_check("armcompute" "${_ac_cache_key}" _ac_hit _ac_cache_path)
            if(_ac_hit)
                sd_dep_cache_restore("armcompute" "${_ac_cache_path}" "${ARMCOMPUTE_INSTALL_DIR}")
                if(NOT TARGET armcompute_external)
                    add_custom_target(armcompute_external)
                endif()
                add_library(armcompute_interface INTERFACE)
                target_include_directories(armcompute_interface INTERFACE
                    "${ARMCOMPUTE_INSTALL_DIR}"
                    "${ARMCOMPUTE_INSTALL_DIR}/include")
                target_link_directories(armcompute_interface INTERFACE
                    "${ARMCOMPUTE_INSTALL_DIR}/lib"
                    "${ARMCOMPUTE_INSTALL_DIR}/lib/armv8a-neon")
                target_link_libraries(armcompute_interface INTERFACE arm_compute arm_compute_graph)
                add_dependencies(armcompute_interface armcompute_external)
                set(ARMCOMPUTE_LIBRARIES armcompute_interface PARENT_SCOPE)
                set(HAVE_ARMCOMPUTE 1 PARENT_SCOPE)
                message(STATUS "✅ ARM Compute setup complete (from cache)")
                return()
            endif()
        endif()

        ExternalProject_Add(armcompute_external
                PREFIX      "${CMAKE_BINARY_DIR}/armcompute_external"
                URL         "${ARMCOMPUTE_URL}"
                URL_HASH    "${ARMCOMPUTE_URL_HASH}"
                DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads"
                CONFIGURE_COMMAND ""
                BUILD_COMMAND     ""
                INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR> ${ARMCOMPUTE_INSTALL_DIR}
                BUILD_BYPRODUCTS "${ARMCOMPUTE_INSTALL_DIR}/arm_compute/core/CL/CLKernelLibrary.h"
                ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        add_library(armcompute_interface INTERFACE)
        target_include_directories(armcompute_interface INTERFACE
                    "${ARMCOMPUTE_INSTALL_DIR}"
                    "${ARMCOMPUTE_INSTALL_DIR}/include")
        target_link_directories(armcompute_interface INTERFACE
                    "${ARMCOMPUTE_INSTALL_DIR}/lib"
                    "${ARMCOMPUTE_INSTALL_DIR}/lib/armv8a-neon")
        target_link_libraries(armcompute_interface INTERFACE arm_compute arm_compute_graph)
        add_dependencies(armcompute_interface armcompute_external)

        set(ARMCOMPUTE_LIBRARIES armcompute_interface PARENT_SCOPE)
        set(HAVE_ARMCOMPUTE 1 PARENT_SCOPE)

        # --- Cache store ---
        if(SD_DEP_CACHE AND DEFINED _ac_cache_key)
            sd_dep_cache_store("armcompute" "${_ac_cache_key}" "${ARMCOMPUTE_INSTALL_DIR}" "armcompute_external")
        endif()
    endif()
endfunction()

# =============================================================================
# CUDNN (Optional, for CUDA builds)
# =============================================================================
function(setup_cudnn)
    set(HAVE_CUDNN false PARENT_SCOPE)
    set(CUDNN "" PARENT_SCOPE)

    if(NOT (HELPERS_cudnn AND SD_CUDA))
        return()
    endif()

    find_package(CUDNN)
    if(CUDNN_FOUND)
        message(STATUS "✓ Found cuDNN: ${CUDNN_LIBRARY}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true PARENT_SCOPE)
        set(CUDNN ${CUDNN_LIBRARIES} PARENT_SCOPE)
        add_definitions(-DHAVE_CUDNN=1)
    else()
        message(WARNING "✗ cuDNN not found. Continuing without cuDNN support.")
    endif()
endfunction()

# =============================================================================
# MLIR / LLVM (Optional)
# Provides JIT compilation support via MLIR for optimized kernel execution
# =============================================================================
function(setup_mlir)
    if(NOT HELPERS_mlir STREQUAL "ON")
        message(STATUS "MLIR helper is disabled (HELPERS_mlir=${HELPERS_mlir})")
        set(HAVE_MLIR FALSE PARENT_SCOPE)
        set(MLIR "" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "MLIR/LLVM helper is enabled")

    # Include the FindMLIR module
    include(${CMAKE_CURRENT_LIST_DIR}/FindMLIR.cmake)

    if(NOT MLIR_FOUND)
        message(FATAL_ERROR
            "MLIR support was requested, but LLVM/MLIR ${MLIR_VERSION}+ was not found. "
            "Set LLVM_DIR, MLIR_DIR, LLVM_ROOT, or CMAKE_PREFIX_PATH to a matching shared installation.")
    endif()

    set(HAVE_MLIR TRUE PARENT_SCOPE)
    if(MLIR_ENABLE_VULKAN)
        if(NOT TARGET MLIR::SPIRV)
            message(FATAL_ERROR
                "MLIR Vulkan support was requested, but the validated MLIR::SPIRV target is unavailable.")
        endif()
        # MLIR::SPIRV carries the same shared MLIR/LLVM libraries as CUDA's
        # MLIR::MLIR target after enforcing the Vulkan lowering contract.
        set(MLIR MLIR::SPIRV PARENT_SCOPE)
    else()
        set(MLIR MLIR::MLIR PARENT_SCOPE)
    endif()

    # Configure GPU support if requested and CUDA is enabled
    if(MLIR_ENABLE_GPU AND SD_CUDA)
        if(TARGET MLIR::GPU)
            message(STATUS "   MLIR GPU dialect: enabled")
        else()
            message(FATAL_ERROR "MLIR GPU support was requested, but the MLIR::GPU target is unavailable.")
        endif()
    endif()

    # Add compile definitions for conditional compilation
    add_compile_definitions(HAVE_MLIR=1)
    if(MLIR_ENABLE_GPU AND SD_CUDA)
        add_compile_definitions(MLIR_ENABLE_GPU=1)
    endif()
    if(MLIR_ENABLE_VULKAN)
        add_compile_definitions(MLIR_ENABLE_VULKAN=1)
    endif()
    if(MLIR_ENABLE_AARCH64)
        add_compile_definitions(MLIR_ENABLE_AARCH64=1)
    endif()
    if(NOT MLIR_AOT_TARGET STREQUAL "HOST")
        add_compile_definitions(MLIR_AOT_TARGET="${MLIR_AOT_TARGET}")
    endif()
    if(MLIR_JIT_CACHE)
        add_compile_definitions(MLIR_JIT_CACHE=1)
    endif()
    if(MLIR_DEBUG_DUMPS)
        add_compile_definitions(MLIR_DEBUG_DUMPS=1)
    endif()

    message(STATUS "✅ MLIR/LLVM setup complete")
    message(STATUS "   LLVM Version: ${LLVM_VERSION}")
    message(STATUS "   GPU Support: ${MLIR_ENABLE_GPU}")
    message(STATUS "   Vulkan/SPIR-V: ${MLIR_ENABLE_VULKAN}")
    message(STATUS "   AArch64 Backend: ${MLIR_ENABLE_AARCH64}")
    message(STATUS "   AOT Target: ${MLIR_AOT_TARGET}")
    message(STATUS "   JIT Cache: ${MLIR_JIT_CACHE}")
endfunction()

# =============================================================================
# ZLUDA CUDA ABI runtime (AMD-only)
# =============================================================================
# ZLUDA is a configure-time input: its exact shared libraries are needed while
# CMake creates link targets and the JavaCPP runtime manifest.  ExternalProject
# would materialize them only at build time, which is too late.  Keep download,
# integrity validation, and extraction here with the other native dependencies.
function(setup_zluda_download output_root)
    if(NOT SD_ZLUDA)
        set(${output_root} "" PARENT_SCOPE)
        return()
    endif()

    if(NOT SD_ZLUDA_VERSION STREQUAL "v6")
        message(FATAL_ERROR
            "Unsupported ZLUDA release ${SD_ZLUDA_VERSION}; this source tree pins v6")
    endif()

    if(WIN32)
        set(_zluda_platform "windows-x86_64")
        set(_zluda_asset "zluda-windows-3fe1206.zip")
        set(_zluda_sha256 "fda8891c6fdfaba438f2eb0f9d749ffa2c1fddbdf225be2301f0d7a25e37208a")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
           CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        set(_zluda_platform "linux-x86_64")
        set(_zluda_asset "zluda-linux-3fe12063.tar.gz")
        set(_zluda_sha256 "d9fd9893abaf3206c56d3eb25f0475c6327aa8de8e77f21be8a24f275556c3e1")
    else()
        message(FATAL_ERROR
            "ZLUDA ${SD_ZLUDA_VERSION} is unsupported on ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(_zluda_url
        "https://github.com/vosen/ZLUDA/releases/download/${SD_ZLUDA_VERSION}/${_zluda_asset}")
    set(_zluda_download_dir "${CMAKE_BINARY_DIR}/downloads")
    set(_zluda_archive "${_zluda_download_dir}/${_zluda_asset}")
    set(_zluda_extract_root
        "${CMAKE_BINARY_DIR}/dependencies/zluda-${SD_ZLUDA_VERSION}-${_zluda_platform}")
    set(_zluda_marker "${_zluda_extract_root}/.dl4j-zluda-${_zluda_sha256}")

    file(MAKE_DIRECTORY "${_zluda_download_dir}")
    message(STATUS
        "Resolving pinned ZLUDA ${SD_ZLUDA_VERSION} for ${_zluda_platform}")
    file(DOWNLOAD "${_zluda_url}" "${_zluda_archive}"
        EXPECTED_HASH "SHA256=${_zluda_sha256}"
        STATUS _zluda_download_status
        SHOW_PROGRESS
        TLS_VERIFY ON
        TIMEOUT 1800)
    list(GET _zluda_download_status 0 _zluda_download_code)
    list(GET _zluda_download_status 1 _zluda_download_message)
    if(NOT _zluda_download_code EQUAL 0)
        message(FATAL_ERROR
            "Failed to download pinned ZLUDA ${SD_ZLUDA_VERSION}: ${_zluda_download_message}")
    endif()

    if(NOT EXISTS "${_zluda_marker}")
        file(REMOVE_RECURSE "${_zluda_extract_root}")
        file(MAKE_DIRECTORY "${_zluda_extract_root}")
        file(ARCHIVE_EXTRACT
            INPUT "${_zluda_archive}"
            DESTINATION "${_zluda_extract_root}")
        file(WRITE "${_zluda_marker}" "${_zluda_sha256}\n")
    endif()

    set(ZLUDA_MANAGED_VERSION "${SD_ZLUDA_VERSION}" PARENT_SCOPE)
    set(ZLUDA_MANAGED_ARCHIVE_SHA256 "${_zluda_sha256}" PARENT_SCOPE)
    set(${output_root} "${_zluda_extract_root}" PARENT_SCOPE)
endfunction()

# =============================================================================
# METAL PERFORMANCE SHADERS (Optional, for macOS/iOS builds)
# =============================================================================
# =============================================================================
# MIOpen Download (Optional, for AMD GPU DNN operations via ZLUDA)
# =============================================================================
function(setup_miopen_download)
    if(NOT HELPERS_miopen STREQUAL "ON")
        message(STATUS "MIOpen helper is disabled (HELPERS_miopen=${HELPERS_miopen})")
        set(HAVE_MIOPEN FALSE PARENT_SCOPE)
        return()
    endif()

    # Check if MIOpen is already available via ROCm
    set(ROCM_SEARCH_PATHS
        $ENV{ROCM_PATH}
        $ENV{ROCM_HOME}
        /opt/rocm
        /opt/rocm-6.0
        /opt/rocm-5.7
        /opt/rocm-5.6
    )

    foreach(rocm_path ${ROCM_SEARCH_PATHS})
        if(EXISTS "${rocm_path}/lib/libMIOpen.so")
            message(STATUS "Using existing MIOpen from ROCm: ${rocm_path}")
            set(HAVE_MIOPEN TRUE PARENT_SCOPE)
            set(MIOPEN_PATH "${rocm_path}" PARENT_SCOPE)
            set(MIOPEN_LIBRARY "${rocm_path}/lib/libMIOpen.so" PARENT_SCOPE)
            set(MIOPEN_INCLUDE_DIR "${rocm_path}/include" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    # MIOpen requires ROCm to be installed - we can't download it standalone easily
    # because it has complex dependencies on HIP runtime and other ROCm components
    message(STATUS "MIOpen automatic download:")
    message(STATUS "   MIOpen is part of ROCm and cannot be downloaded standalone.")
    message(STATUS "   Please install ROCm toolkit from: https://rocm.docs.amd.com/")
    message(STATUS "   After installation, MIOpen will be automatically detected.")

    # For now, we'll create a stub that warns about missing MIOpen
    set(HAVE_MIOPEN FALSE PARENT_SCOPE)

    # Alternative: Try to build MIOpen from source (complex due to dependencies)
    if(FALSE)  # Disabled for now - ROCm installation is recommended
        set(MIOPEN_VERSION "3.0.0")
        set(MIOPEN_INSTALL_DIR "${CMAKE_BINARY_DIR}/miopen_install")
        set(MIOPEN_URL "https://github.com/ROCm/MIOpen/archive/refs/tags/rocm-${MIOPEN_VERSION}.tar.gz")

        ExternalProject_Add(miopen_external
                PREFIX            "${CMAKE_BINARY_DIR}/miopen_external"
                URL               "${MIOPEN_URL}"
                DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
                SOURCE_DIR        "${CMAKE_BINARY_DIR}/miopen_external/src"
                BINARY_DIR        "${CMAKE_BINARY_DIR}/miopen_external/build"
                CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${MIOPEN_INSTALL_DIR}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DMIOPEN_BACKEND=HIP
                BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE} --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config ${CMAKE_BUILD_TYPE}
                TIMEOUT           1200
                ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        add_library(miopen_interface INTERFACE)
        target_include_directories(miopen_interface INTERFACE "${MIOPEN_INSTALL_DIR}/include")
        target_link_libraries(miopen_interface INTERFACE "${MIOPEN_INSTALL_DIR}/lib/libMIOpen.so")
        add_dependencies(miopen_interface miopen_external)

        set(HAVE_MIOPEN TRUE PARENT_SCOPE)
        set(MIOPEN miopen_interface PARENT_SCOPE)
    endif()
endfunction()

function(setup_mps)
    set(HAVE_MPS FALSE PARENT_SCOPE)

    if(NOT HELPERS_mps STREQUAL "ON")
        message(STATUS "MPS helper is disabled (HELPERS_mps=${HELPERS_mps})")
        return()
    endif()

    # Check if we're on Apple platform
    if(NOT APPLE)
        message(STATUS "MPS helper requires macOS/iOS (current platform: ${CMAKE_SYSTEM_NAME})")
        return()
    endif()

    # Check for Metal framework availability
    find_library(METAL_FRAMEWORK Metal)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)

    if(NOT METAL_FRAMEWORK OR NOT MPS_FRAMEWORK)
        message(WARNING "Metal or MetalPerformanceShaders framework not found")
        return()
    endif()

    message(STATUS "✅ Metal Performance Shaders setup:")
    message(STATUS "   Metal Framework: ${METAL_FRAMEWORK}")
    message(STATUS "   MPS Framework: ${MPS_FRAMEWORK}")

    set(HAVE_MPS TRUE PARENT_SCOPE)
    set(MPS_LIBRARIES ${METAL_FRAMEWORK} ${MPS_FRAMEWORK} ${FOUNDATION_FRAMEWORK} PARENT_SCOPE)

    add_compile_definitions(HAVE_MPS=1)

    # Enable SD_METAL for MetalReplayHandle in the DSP framework
    add_compile_definitions(SD_METAL=1)

    # The top-level project enables Objective-C++ at directory scope on Apple.
    # Keep language initialization out of this function so CMake's internal
    # OBJCXX compile rules remain visible while targets are generated.

    sd_register_helper("mps")
    message(STATUS "✅ MPS setup complete with Metal ICB replay (SD_METAL)")
endfunction()

# =============================================================================
# MANAGED LLVM/MLIR + TRITON DSP COMPILER (Optional)
# =============================================================================
function(setup_triton)
    # Triton is controlled only by explicit CMake options. Environment and
    # hardware discovery never change whether it is enabled.
    set(_triton_requested OFF)
    # Legacy explicit CMake opt-in path (kept for backward compatibility).
    if(HELPERS_triton)
        set(_triton_requested ON)
    endif()
    if(DEFINED SD_TRITON)
        set(_triton_requested ${SD_TRITON})
    endif()

    # JavaCPP's libnd4j.cmake property is applied to a nested configure. Keep
    # the worker-provided contract available through the inherited environment
    # as well, so setup_triton sees the restored package in that configure.
    if((NOT DEFINED SD_TRITON_MANAGED_LLVM_ROOT OR
        "${SD_TRITON_MANAGED_LLVM_ROOT}" STREQUAL "") AND
       DEFINED ENV{SD_TRITON_MANAGED_LLVM_ROOT} AND
       NOT "$ENV{SD_TRITON_MANAGED_LLVM_ROOT}" STREQUAL "")
        set(SD_TRITON_MANAGED_LLVM_ROOT "$ENV{SD_TRITON_MANAGED_LLVM_ROOT}")
    endif()
    if((NOT DEFINED SD_TRITON_CONSUMER_KIND OR
        "${SD_TRITON_CONSUMER_KIND}" STREQUAL "") AND
       DEFINED ENV{SD_TRITON_CONSUMER_KIND} AND
       NOT "$ENV{SD_TRITON_CONSUMER_KIND}" STREQUAL "")
        set(SD_TRITON_CONSUMER_KIND "$ENV{SD_TRITON_CONSUMER_KIND}")
    endif()

    # Release workers can restore the exact managed LLVM/MLIR package from the
    # cloud dependency cache. That package is also a valid MLIR-only consumer
    # when Triton itself is disabled (for example Android compile classifiers).
    # Preserve the injected root instead of clearing it below.
    set(_managed_llvm_root_from_config "")
    if(DEFINED SD_TRITON_MANAGED_LLVM_ROOT AND
       NOT "${SD_TRITON_MANAGED_LLVM_ROOT}" STREQUAL "" AND
       EXISTS "${SD_TRITON_MANAGED_LLVM_ROOT}/lib/cmake/llvm/LLVMConfig.cmake" AND
       EXISTS "${SD_TRITON_MANAGED_LLVM_ROOT}/lib/cmake/mlir/MLIRConfig.cmake")
        set(_managed_llvm_root_from_config "${SD_TRITON_MANAGED_LLVM_ROOT}")
    endif()

    # Vulkan replay consumes the pinned shared LLVM/MLIR package for native
    # MLIR-to-SPIR-V lowering, but it does not consume or enable the Triton DSP
    # compiler. Keep that infrastructure request independent of SD_TRITON.
    set(_managed_llvm_requested ${_triton_requested})
    if(NOT _managed_llvm_root_from_config STREQUAL "")
        set(_managed_llvm_requested ON)
    endif()
    if(SD_VULKAN AND HELPERS_mlir STREQUAL "ON" AND MLIR_ENABLE_VULKAN)
        set(_managed_llvm_requested ON)
    endif()

    if(NOT _managed_llvm_requested)
        message(STATUS "Triton disabled (use -DSD_TRITON=ON to enable)")
        set(HAVE_TRITON OFF CACHE BOOL "Triton availability" FORCE)
        set(HAVE_TRITON OFF PARENT_SCOPE)
        set(HAVE_MANAGED_LLVM_MLIR OFF CACHE BOOL
            "Project-managed shared LLVM/MLIR availability" FORCE)
        set(HAVE_MANAGED_LLVM_MLIR OFF PARENT_SCOPE)
        set(HAVE_TRITON_CPU OFF CACHE BOOL "Triton CPU backend" FORCE)
        set(HAVE_TRITON_CPU OFF PARENT_SCOPE)
        set(SD_TRITON_CONSUMER_KIND "" CACHE INTERNAL
            "Compiler package consumer selected by setup_triton" FORCE)
        set(SD_TRITON_MANAGED_LLVM_ROOT "" CACHE INTERNAL
            "Project-managed LLVM/MLIR package root selected by setup_triton" FORCE)
        set(TRITON "" PARENT_SCOPE)
        return()
    endif()
    if(SD_VULKAN AND NOT _triton_requested)
        message(STATUS
            "Triton DSP compiler disabled; provisioning managed LLVM/MLIR for Vulkan replay")
    endif()

    # The pinned LLVM sources currently require headers that are incompatible
    # with MSVC's STL (constexpr string_view::substr in TypeName.h). MinGW
    # uses GCC/libstdc++ and is a supported Windows Vulkan toolchain.
    if(WIN32 AND NOT MINGW)
        message(FATAL_ERROR
            "The project-managed LLVM/MLIR stack is unsupported with MSVC because "
            "its LLVM headers are incompatible with the MSVC STL.")
    endif()
    # Select the compiler package from the backend that consumes it. This is
    # deliberately not a GPU-vs-non-GPU shortcut: Vulkan consumes the same
    # project-managed shared LLVM/MLIR package as the GPU emitter stack, but it
    # lowers through MLIR/SPIR-V and therefore must not build either libtriton
    # implementation or select Triton CPU code generation.
    if(SD_VULKAN)
        set(_TRITON_CONSUMER_KIND "VULKAN_SPIRV")
        set(_TRITON_BUILDS_COMPILER FALSE)
    elseif(SD_CUDA OR SD_HIP OR SD_LEVEL_ZERO)
        set(_TRITON_CONSUMER_KIND "GPU_EMITTER")
        set(_TRITON_BUILDS_COMPILER TRUE)
    elseif(SD_CPU)
        set(_TRITON_CONSUMER_KIND "CPU_COMPILER")
        set(_TRITON_BUILDS_COMPILER TRUE)
    else()
        message(FATAL_ERROR
            "SD_TRITON=ON has no compiler-package routing for the selected backend. "
            "Expected SD_CPU, SD_CUDA/SD_HIP/SD_LEVEL_ZERO, or SD_VULKAN.")
    endif()
    # A restored package can satisfy MLIR discovery without asking setup_triton
    # to build or link the Triton compiler itself.
    if(NOT _triton_requested AND NOT _managed_llvm_root_from_config STREQUAL "")
        set(_TRITON_BUILDS_COMPILER FALSE)
    endif()
    set(SD_TRITON_CONSUMER_KIND "${_TRITON_CONSUMER_KIND}" CACHE INTERNAL
        "Compiler package consumer selected by setup_triton" FORCE)
    set(HAVE_TRITON_CPU OFF CACHE BOOL "Triton CPU backend" FORCE)
    set(HAVE_TRITON_CPU OFF PARENT_SCOPE)

    # External target builds must receive the same canonical toolchain contract
    # as libnd4j itself. Host-only generators are handled separately below.
    set(_TRITON_TARGET_CMAKE_ARGS "")
    if(CMAKE_CROSSCOMPILING)
        foreach(_triton_target_var
                CMAKE_TOOLCHAIN_FILE
                CMAKE_SYSTEM_NAME
                CMAKE_SYSTEM_VERSION
                CMAKE_SYSTEM_PROCESSOR
                CMAKE_SYSROOT
                CMAKE_ANDROID_NDK
                CMAKE_ANDROID_NDK_VERSION
                CMAKE_ANDROID_ARCH_ABI
                CMAKE_ANDROID_API
                CMAKE_ANDROID_STL_TYPE
                ANDROID_ABI
                ANDROID_PLATFORM
                ANDROID_STL
                CMAKE_C_COMPILER_TARGET
                CMAKE_CXX_COMPILER_TARGET)
            if(DEFINED ${_triton_target_var} AND
               NOT "${${_triton_target_var}}" STREQUAL "")
                list(APPEND _TRITON_TARGET_CMAKE_ARGS
                    "-D${_triton_target_var}=${${_triton_target_var}}")
            endif()
        endforeach()
        message(STATUS
            "Triton target toolchain: system=${CMAKE_SYSTEM_NAME} "
            "processor=${CMAKE_SYSTEM_PROCESSOR} androidAbi=${CMAKE_ANDROID_ARCH_ABI} "
            "api=${CMAKE_SYSTEM_VERSION}")
    endif()

    # Check if the selected compiler package is already built from a previous
    # run. CPU Triton has its own LLVM ABI. GPU emitters and Vulkan SPIR-V use
    # the same patched shared LLVM/MLIR package and revision.
    if(NOT _managed_llvm_root_from_config STREQUAL "")
        # The release worker already restored this exact package. Keep all
        # CMake/package paths pointed at it and avoid a source rebuild.
        set(TRITON_LLVM_INSTALL_DIR "${_managed_llvm_root_from_config}")
        if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
            set(TRITON_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_cpu_install")
        elseif(_TRITON_CONSUMER_KIND STREQUAL "GPU_EMITTER")
            set(TRITON_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_install")
        endif()
    elseif(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
        set(TRITON_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_cpu_install")
        set(TRITON_LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_cpu_llvm_install")
    elseif(_TRITON_CONSUMER_KIND STREQUAL "GPU_EMITTER")
        set(TRITON_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_install")
        set(TRITON_LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_llvm_install")
    else()
        set(TRITON_LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_llvm_install")
    endif()
    set(SD_TRITON_MANAGED_LLVM_ROOT "${TRITON_LLVM_INSTALL_DIR}" CACHE INTERNAL
        "Project-managed LLVM/MLIR package root selected by setup_triton" FORCE)

    # Keep cache identities aligned with the artifacts they describe. LLVM/MLIR,
    # the Triton compiler, native LLVM TableGen tools, and SLEEF generators have
    # different source/configuration closures and must not invalidate one another.
    # The managed LLVM package now enables hidden visibility on MinGW so the
    # monolithic DLL exports only its annotated ABI surface. Bump both package
    # identities so dependency caches built with the pre-visibility contract are
    # never reused by a build that needs the corrected Windows runtime.
    set(_TRITON_LLVM_RECIPE_REVISION "managed-llvm-patches-v13")
    set(_TRITON_COMPILER_RECIPE_REVISION "managed-llvm-patches-v13")
    set(_TRITON_LLVM_HOST_TOOLS_RECIPE_REVISION "managed-llvm-host-tools-v1")
    set(_TRITON_SLEEF_HOST_TOOLS_RECIPE_REVISION "managed-sleef-host-tools-v1")
    set(_TRITON_LLVM_INSTALL_MARKER
        "${TRITON_LLVM_INSTALL_DIR}/.sd-${_TRITON_LLVM_RECIPE_REVISION}")
    if(_TRITON_BUILDS_COMPILER)
        set(_TRITON_INSTALL_MARKER
            "${TRITON_INSTALL_DIR}/.sd-${_TRITON_COMPILER_RECIPE_REVISION}")
    endif()

    # A generated MLIRConfig.cmake alone does not make a reusable install.
    # Dependency caches from the former static-link configuration contain the
    # package metadata and archives but not the monolithic LLVM/MLIR DSOs. If
    # that partial install is accepted, CMake creates imported shared targets
    # with no producer and the final link fails with "No rule to make target".
    if(WIN32)
        set(_TRITON_LLVM_DSO_DIR "${TRITON_LLVM_INSTALL_DIR}/bin")
    else()
        set(_TRITON_LLVM_DSO_DIR "${TRITON_LLVM_INSTALL_DIR}/lib")
    endif()
    set(_TRITON_LLVM_SHARED_LIBRARY
        "${_TRITON_LLVM_DSO_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}LLVM${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_TRITON_MLIR_SHARED_LIBRARY
        "${_TRITON_LLVM_DSO_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}MLIR${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_TRITON_MLIR_EXECUTION_ENGINE_SHARED_LIBRARY
        "${_TRITON_LLVM_DSO_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}MLIRExecutionEngineShared${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_TRITON_LLVM_INSTALL_COMPLETE FALSE)
    if(EXISTS "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake" AND
       EXISTS "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake" AND
       EXISTS "${_TRITON_LLVM_SHARED_LIBRARY}" AND
       EXISTS "${_TRITON_MLIR_SHARED_LIBRARY}" AND
       EXISTS "${_TRITON_MLIR_EXECUTION_ENGINE_SHARED_LIBRARY}" AND
       (EXISTS "${_TRITON_LLVM_INSTALL_MARKER}" OR
        NOT _managed_llvm_root_from_config STREQUAL ""))
        set(_TRITON_LLVM_INSTALL_COMPLETE TRUE)
    endif()

    set(_TRITON_LIB_EXISTS FALSE)
    if(_TRITON_BUILDS_COMPILER)
        if((EXISTS "${TRITON_INSTALL_DIR}/lib/libtriton.a" OR
            EXISTS "${TRITON_INSTALL_DIR}/lib/triton.lib") AND
           EXISTS "${_TRITON_INSTALL_MARKER}")
            set(_TRITON_LIB_EXISTS TRUE)
        endif()
    endif()
    set(_TRITON_PACKAGE_INSTALL_COMPLETE ${_TRITON_LLVM_INSTALL_COMPLETE})
    if(_TRITON_BUILDS_COMPILER AND NOT _TRITON_LIB_EXISTS)
        set(_TRITON_PACKAGE_INSTALL_COMPLETE FALSE)
    endif()
    if(_TRITON_PACKAGE_INSTALL_COMPLETE)
        if(_TRITON_BUILDS_COMPILER)
            message(STATUS
                "Triton ${_TRITON_CONSUMER_KIND}: reusing compiler install at ${TRITON_INSTALL_DIR}")
        else()
            message(STATUS
                "Triton ${_TRITON_CONSUMER_KIND}: reusing shared LLVM/MLIR at ${TRITON_LLVM_INSTALL_DIR}")
        endif()
        set(HAVE_MANAGED_LLVM_MLIR ON CACHE BOOL
            "Project-managed shared LLVM/MLIR availability" FORCE)
        set(HAVE_MANAGED_LLVM_MLIR ON PARENT_SCOPE)
        set(HAVE_TRITON ${_TRITON_BUILDS_COMPILER} CACHE BOOL
            "Triton availability" FORCE)
        set(HAVE_TRITON ${_TRITON_BUILDS_COMPILER} PARENT_SCOPE)
        if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
            set(HAVE_TRITON_CPU ON CACHE BOOL "Triton CPU backend" FORCE)
            set(HAVE_TRITON_CPU ON PARENT_SCOPE)
        endif()
        # Create a dummy triton_external target so MainBuildFlow's dependency block picks us up
        if(NOT TARGET triton_external)
            add_custom_target(triton_external)
        endif()
        if(NOT TARGET triton_interface)
            add_library(triton_interface INTERFACE)
            # These headers and the shared libraries below are one versioned
            # package. Keep their include roots ahead of ambient compiler search paths.
            target_include_directories(triton_interface INTERFACE
                "${TRITON_LLVM_INSTALL_DIR}/include"
            )
            if(_TRITON_BUILDS_COMPILER)
                target_include_directories(triton_interface INTERFACE
                    "${TRITON_INSTALL_DIR}/include")
                if(WIN32)
                    target_link_libraries(triton_interface INTERFACE "${TRITON_INSTALL_DIR}/lib/triton.lib")
                else()
                    target_link_libraries(triton_interface INTERFACE "${TRITON_INSTALL_DIR}/lib/libtriton.a")
                endif()
            endif()
            # Consume the monolithic shared libraries exported by the pinned
            # LLVM/MLIR install. This preserves LLVM/MLIR's DSO boundaries and lets
            # independent LLVM/MLIR versions coexist in the same process.
            set(LLVM_DIR "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/llvm" CACHE PATH
                "Project-managed target LLVM package" FORCE)
            set(MLIR_DIR "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH
                "Project-managed target MLIR package" FORCE)
            set(LLVM_LINK_LLVM_DYLIB ON)
            set(MLIR_LINK_MLIR_DYLIB ON)
            find_package(LLVM CONFIG REQUIRED PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)
            find_package(MLIR CONFIG REQUIRED PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)

            foreach(_triton_shared_target MLIR LLVM)
                if(NOT TARGET ${_triton_shared_target})
                    message(FATAL_ERROR
                        "Triton LLVM install at ${TRITON_LLVM_INSTALL_DIR} does not export "
                        "the required shared target '${_triton_shared_target}'. Remove that "
                        "install and reconfigure so it is rebuilt with LLVM_BUILD_LLVM_DYLIB, "
                        "LLVM_LINK_LLVM_DYLIB, MLIR_BUILD_MLIR_DYLIB, and "
                        "MLIR_LINK_MLIR_DYLIB enabled.")
                endif()
                get_target_property(_triton_shared_type ${_triton_shared_target} TYPE)
                get_target_property(_triton_shared_location ${_triton_shared_target} IMPORTED_LOCATION_RELEASE)
                if(NOT _triton_shared_location)
                    get_target_property(_triton_shared_location ${_triton_shared_target} IMPORTED_LOCATION)
                endif()
                if(NOT _triton_shared_type STREQUAL "SHARED_LIBRARY" OR
                   NOT _triton_shared_location OR
                   NOT EXISTS "${_triton_shared_location}")
                    message(FATAL_ERROR
                        "Triton requires an installed shared ${_triton_shared_target} target; "
                        "got type='${_triton_shared_type}', location='${_triton_shared_location}'.")
                endif()
                message(STATUS
                    "Triton shared ${_triton_shared_target}: ${_triton_shared_location}")
            endforeach()

            # Normalize package-exported MLIR/LLVM targets to the same project-owned
            # imported target names used by the fresh/cache-restored path. Downstream
            # linking and runtime packaging must not depend on which setup path ran.
            if(NOT TARGET triton_llvm_shared)
                add_library(triton_llvm_shared SHARED IMPORTED GLOBAL)
                if(WIN32)
                    set_target_properties(triton_llvm_shared PROPERTIES
                        IMPORTED_LOCATION "${_TRITON_LLVM_SHARED_LIBRARY}"
                        IMPORTED_IMPLIB "${TRITON_LLVM_INSTALL_DIR}/lib/libLLVM.dll.a")
                else()
                    set_target_properties(triton_llvm_shared PROPERTIES
                        IMPORTED_LOCATION "${_TRITON_LLVM_SHARED_LIBRARY}")
                endif()
            endif()
            if(NOT TARGET triton_mlir_shared)
                add_library(triton_mlir_shared SHARED IMPORTED GLOBAL)
                if(WIN32)
                    set_target_properties(triton_mlir_shared PROPERTIES
                        IMPORTED_LOCATION "${_TRITON_MLIR_SHARED_LIBRARY}"
                        IMPORTED_IMPLIB "${TRITON_LLVM_INSTALL_DIR}/lib/libMLIR.dll.a"
                        INTERFACE_LINK_LIBRARIES triton_llvm_shared)
                else()
                    set_target_properties(triton_mlir_shared PROPERTIES
                        IMPORTED_LOCATION "${_TRITON_MLIR_SHARED_LIBRARY}"
                        INTERFACE_LINK_LIBRARIES triton_llvm_shared)
                endif()
            endif()

            target_link_libraries(triton_interface INTERFACE
                triton_mlir_shared
                triton_llvm_shared)
            if(NOT WIN32)
                target_link_libraries(triton_interface INTERFACE -lz -lm)
                if(NOT APPLE AND NOT ANDROID)
                    target_link_libraries(triton_interface INTERFACE -lrt -ldl -lpthread)
                elseif(APPLE)
                    target_link_libraries(triton_interface INTERFACE -ldl -lpthread)
                else()
                    target_link_libraries(triton_interface INTERFACE -ldl)
                endif()
            endif()
            if(SD_CUDA)
                if(WIN32)
                    target_link_libraries(triton_interface INTERFACE nvrtc.lib cuda.lib)
                else()
                    target_link_libraries(triton_interface INTERFACE -lnvrtc -lcuda)
                endif()
            endif()
            message(STATUS
                "Triton interface: shared MLIR ${MLIR_PACKAGE_VERSION}, LLVM ${LLVM_PACKAGE_VERSION} "
                "(reused install)")
        endif()
        set(TRITON triton_interface PARENT_SCOPE)
        return()
    endif()
    if(TARGET triton_external)
        message(STATUS
            "Managed LLVM/MLIR helper is enabled for ${_TRITON_CONSUMER_KIND} (target already exists)")
        set(HAVE_MANAGED_LLVM_MLIR ON CACHE BOOL
            "Project-managed shared LLVM/MLIR availability" FORCE)
        set(HAVE_MANAGED_LLVM_MLIR ON PARENT_SCOPE)
        set(HAVE_TRITON ${_TRITON_BUILDS_COMPILER} CACHE BOOL
            "Triton availability" FORCE)
        set(HAVE_TRITON ${_TRITON_BUILDS_COMPILER} PARENT_SCOPE)
        if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
            set(HAVE_TRITON_CPU ON CACHE BOOL "Triton CPU backend" FORCE)
            set(HAVE_TRITON_CPU ON PARENT_SCOPE)
        endif()
        set(TRITON triton_interface PARENT_SCOPE)
        return()
    endif()

    message(STATUS "Managed LLVM/MLIR helper is enabled (consumer=${_TRITON_CONSUMER_KIND})")
    set(HAVE_MANAGED_LLVM_MLIR ON CACHE BOOL
        "Project-managed shared LLVM/MLIR availability" FORCE)
    set(HAVE_MANAGED_LLVM_MLIR ON PARENT_SCOPE)
    set(HAVE_TRITON ${_TRITON_BUILDS_COMPILER} CACHE BOOL
        "Triton availability" FORCE)
    set(HAVE_TRITON ${_TRITON_BUILDS_COMPILER} PARENT_SCOPE)
    if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
        set(HAVE_TRITON_CPU ON CACHE BOOL "Triton CPU backend" FORCE)
        set(HAVE_TRITON_CPU ON PARENT_SCOPE)
    endif()
    if(_TRITON_BUILDS_COMPILER)
        set(HELPERS_triton ON PARENT_SCOPE)
    endif()

    if(_TRITON_BUILDS_COMPILER)
        # The CPU and GPU emitter compilers have independent source identities.
        # Vulkan intentionally skips this entire source/build path.
        set(TRITON_CPU_COMMIT "c4ccb98970bfe0fa17548b5b32def8d0de2bdc53")
        if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
            set(TRITON_PREFIX
                "${CMAKE_BINARY_DIR}/triton_cpu_external_${_TRITON_COMPILER_RECIPE_REVISION}")
            set(TRITON_VERSION "cpu-${TRITON_CPU_COMMIT}")
        else()
            set(TRITON_PREFIX
                "${CMAKE_BINARY_DIR}/triton_external_${_TRITON_COMPILER_RECIPE_REVISION}")
            set(TRITON_VERSION "3.6.0")
        endif()
        set(TRITON_STAMP_DIR "${TRITON_PREFIX}/stamp")

        file(MAKE_DIRECTORY "${TRITON_STAMP_DIR}")
        file(MAKE_DIRECTORY "${TRITON_PREFIX}/src")
        file(MAKE_DIRECTORY "${TRITON_PREFIX}/build")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")

        if(EXISTS "${TRITON_STAMP_DIR}")
            file(GLOB STALE_STAMPS
                "${TRITON_STAMP_DIR}/*-lastrun.txt" "${TRITON_STAMP_DIR}/*.txt")
            foreach(stamp ${STALE_STAMPS})
                message(STATUS "Cleaning stale Triton stamp file: ${stamp}")
                file(REMOVE "${stamp}")
            endforeach()
        endif()

        if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
            set(TRITON_URL
                "https://github.com/triton-lang/triton-cpu/archive/${TRITON_CPU_COMMIT}.tar.gz")
            set(TRITON_URL_HASH
                "SHA256=29bbed27580d8785605216fe628759165b195cfcf81ee7bc2a3ef681c4f161b3")
        elseif(WIN32)
            set(TRITON_URL
                "https://github.com/triton-lang/triton-windows/archive/refs/heads/release/3.6.x-windows.tar.gz")
        else()
            set(TRITON_URL
                "https://github.com/triton-lang/triton/archive/refs/tags/v${TRITON_VERSION}.tar.gz")
            set(TRITON_URL_HASH
                "SHA256=be270ed11ca5a8fbd9d7941c5bbe9a23a9f6e2ffd372c8398346928bee464774")
        endif()
    endif()

    # Determine codegen backends based on TRITON_GPU_TARGET and build config.
    # This MUST be done before the LLVM build so LLVM_TARGETS_TO_BUILD can be set correctly.
    set(TRITON_CODEGEN_BACKENDS "")

    if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
        set(TRITON_CODEGEN_BACKENDS "cpu")
    elseif(_TRITON_CONSUMER_KIND STREQUAL "VULKAN_SPIRV")
        # Vulkan lowering is MLIR-to-SPIR-V; no Triton codegen backend applies.
        message(STATUS "   Triton codegen backends: none (Vulkan MLIR/SPIR-V)")
    elseif(TRITON_GPU_TARGET STREQUAL "AUTO")
        # NVIDIA backend: always for CUDA builds (pure or ZLUDA)
        if(SD_CUDA)
            list(APPEND TRITON_CODEGEN_BACKENDS "nvidia")
        endif()
        # AMD backend: only when ZLUDA targets AMD, or native HIP build
        if(SD_HIP OR (SD_ZLUDA AND ZLUDA_TARGET_BACKEND STREQUAL "AMD"))
            list(APPEND TRITON_CODEGEN_BACKENDS "amd")
        endif()
        # Intel codegen belongs only to the native Level Zero backend.
        if(SD_LEVEL_ZERO)
            list(APPEND TRITON_CODEGEN_BACKENDS "intel")
        endif()
        if(NOT TRITON_CODEGEN_BACKENDS)
            message(FATAL_ERROR
                "TRITON_GPU_TARGET=AUTO could not infer a codegen backend from the enabled hardware backend. "
                "Enable CUDA/HIP/Level Zero or set TRITON_GPU_TARGET explicitly.")
        endif()
        list(REMOVE_DUPLICATES TRITON_CODEGEN_BACKENDS)
    elseif(TRITON_GPU_TARGET STREQUAL "NVIDIA")
        set(TRITON_CODEGEN_BACKENDS "nvidia")
    elseif(TRITON_GPU_TARGET STREQUAL "AMD")
        set(TRITON_CODEGEN_BACKENDS "amd")
    elseif(TRITON_GPU_TARGET STREQUAL "INTEL")
        set(TRITON_CODEGEN_BACKENDS "intel")
    endif()

    string(REPLACE ";" "\\;" TRITON_BACKENDS_STR "${TRITON_CODEGEN_BACKENDS}")
    if(NOT _TRITON_CONSUMER_KIND STREQUAL "VULKAN_SPIRV")
        message(STATUS "   Triton codegen backends: ${TRITON_CODEGEN_BACKENDS}")
    endif()

    # Each Triton variant pins a specific LLVM commit for ABI compatibility.
    # We build LLVM/MLIR from source at that commit.
    if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
        # triton-cpu pins a different LLVM commit than GPU triton
        set(TRITON_LLVM_COMMIT "20902f0b721ba6cf2fb134362d27144bd8584d53")
        set(TRITON_LLVM_URL_HASH "SHA256=1736af3127e73eab0f2a2f489275c9509d5b60f80c050c42be3a1f85843993e2")
        set(TRITON_LLVM_PREFIX
            "${CMAKE_BINARY_DIR}/triton_cpu_llvm_${_TRITON_LLVM_RECIPE_REVISION}")
        set(_TRITON_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP OFF)
    else()
        # GPU Triton v3.6.0 pins f6ded0be. Vulkan deliberately consumes this
        # exact shared LLVM/MLIR package and the same download-time patch flow,
        # without consuming Triton emitter code.
        set(TRITON_LLVM_COMMIT "f6ded0be897e2878612dd903f7e8bb85448269e5")
        set(TRITON_LLVM_URL_HASH "SHA256=f63c624aa63eda73508b9df2be2a6945ea4fddbee58615fbe1cd747b6884dd5e")
        set(TRITON_LLVM_PREFIX
            "${CMAKE_BINARY_DIR}/triton_llvm_${_TRITON_LLVM_RECIPE_REVISION}")
        set(_TRITON_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP ON)
    endif()

    if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER" OR
       _TRITON_CONSUMER_KIND STREQUAL "VULKAN_SPIRV")
        if(CMAKE_CROSSCOMPILING)
            if(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a" OR
               CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|AARCH64|ARM64)$")
                set(TRITON_LLVM_TARGETS "AArch64")
            elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64" OR
                   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|X86_64|AMD64)$")
                set(TRITON_LLVM_TARGETS "X86")
            else()
                message(FATAL_ERROR
                    "No explicit LLVM target mapping for cross target "
                    "system='${CMAKE_SYSTEM_NAME}' processor='${CMAKE_SYSTEM_PROCESSOR}' "
                    "androidAbi='${CMAKE_ANDROID_ARCH_ABI}'.")
            endif()
            message(STATUS
                "   LLVM target: ${TRITON_LLVM_TARGETS} (cross target, host tools separate)")
        else()
            set(TRITON_LLVM_TARGETS "host")
            message(STATUS
                "   LLVM target: host (${_TRITON_CONSUMER_KIND} native package)")
        endif()
    else()
        # Emitter packages carry only the LLVM targets selected by their
        # configured codegen backends.
        set(TRITON_LLVM_TARGETS "host")
        if("nvidia" IN_LIST TRITON_CODEGEN_BACKENDS)
            set(TRITON_LLVM_TARGETS "${TRITON_LLVM_TARGETS}$<SEMICOLON>NVPTX")
        endif()
        if("amd" IN_LIST TRITON_CODEGEN_BACKENDS)
            set(TRITON_LLVM_TARGETS "${TRITON_LLVM_TARGETS}$<SEMICOLON>AMDGPU")
        endif()
        message(STATUS
            "   LLVM targets: ${TRITON_LLVM_TARGETS} (GPU emitter package)")
    endif()

    # Cross-target identity is part of every managed compiler cache key. Never
    # restore host, AArch64, or x86_64 Android libraries into another target.
    set(_TRITON_TARGET_CACHE_CONFIG "")
    if(CMAKE_CROSSCOMPILING)
        set(_TRITON_TARGET_CACHE_CONFIG
            "targetSystem=${CMAKE_SYSTEM_NAME};targetProcessor=${CMAKE_SYSTEM_PROCESSOR};"
            "androidAbi=${CMAKE_ANDROID_ARCH_ABI};androidApi=${CMAKE_SYSTEM_VERSION};"
            "ndkVersion=${CMAKE_ANDROID_NDK_VERSION};"
            "cTarget=${CMAKE_C_COMPILER_TARGET};cxxTarget=${CMAKE_CXX_COMPILER_TARGET}")
    endif()

    # --- Dependency cache: restore Triton LLVM and Triton into build dirs ---
    # The in-tree fast path above handles an already-populated build directory.
    # Fresh build directories restore here, then recompute producer completeness.
    if(SD_DEP_CACHE)
        # Check Triton LLVM cache
        string(SUBSTRING "${TRITON_LLVM_COMMIT}" 0 8 TRITON_LLVM_COMMIT_SHORT)
        sd_dep_cache_key("triton_llvm" "${TRITON_LLVM_COMMIT_SHORT}"
            "TARGETS=${TRITON_LLVM_TARGETS};llvm_mlir_dylib=1;recipe=${_TRITON_LLVM_RECIPE_REVISION};${_TRITON_TARGET_CACHE_CONFIG}"
            _tllvm_cache_key)
        sd_dep_cache_check("triton_llvm" "${_tllvm_cache_key}" _tllvm_hit _tllvm_cache_path)
        if(_tllvm_hit AND NOT _TRITON_LLVM_INSTALL_COMPLETE)
            sd_dep_cache_restore("triton_llvm" "${_tllvm_cache_path}" "${TRITON_LLVM_INSTALL_DIR}")
            if(EXISTS "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake" AND
               EXISTS "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake" AND
               EXISTS "${_TRITON_LLVM_SHARED_LIBRARY}" AND
               EXISTS "${_TRITON_MLIR_SHARED_LIBRARY}" AND
               EXISTS "${_TRITON_MLIR_EXECUTION_ENGINE_SHARED_LIBRARY}" AND
               (EXISTS "${_TRITON_LLVM_INSTALL_MARKER}" OR
                NOT _managed_llvm_root_from_config STREQUAL ""))
                set(_TRITON_LLVM_INSTALL_COMPLETE TRUE)
            endif()
        endif()

        if(_TRITON_BUILDS_COMPILER)
            # Cache shape includes the shared LLVM/MLIR ABI so installs created
            # by the former static-archive linkage are never reused.
            set(_triton_ver "${TRITON_VERSION}")
            string(REPLACE ";" "_" _triton_backends_str "${TRITON_CODEGEN_BACKENDS}")
            sd_dep_cache_key("triton" "${_triton_ver}"
                "BACKENDS=${_triton_backends_str};llvm_mlir_dylib=1;recipe=${_TRITON_COMPILER_RECIPE_REVISION};${_TRITON_TARGET_CACHE_CONFIG}"
                _triton_cache_key)
            sd_dep_cache_check("triton" "${_triton_cache_key}" _triton_hit _triton_cache_path)
            if(_triton_hit)
                set(_need_triton_restore TRUE)
                if((EXISTS "${TRITON_INSTALL_DIR}/lib/libtriton.a" OR
                    EXISTS "${TRITON_INSTALL_DIR}/lib/triton.lib") AND
                   EXISTS "${_TRITON_INSTALL_MARKER}")
                    set(_need_triton_restore FALSE)
                endif()
                if(_need_triton_restore)
                    sd_dep_cache_restore("triton" "${_triton_cache_path}" "${TRITON_INSTALL_DIR}")
                endif()
            endif()
        endif()
    endif()

    # Cache restoration happens after the initial in-tree fast-path check. Recompute
    # the compiler package state now so restored artifacts suppress every producer
    # that exists only to rebuild them.
    set(_TRITON_COMPILER_INSTALL_COMPLETE FALSE)
    if(_TRITON_BUILDS_COMPILER AND
       (EXISTS "${TRITON_INSTALL_DIR}/lib/libtriton.a" OR
        EXISTS "${TRITON_INSTALL_DIR}/lib/triton.lib") AND
       EXISTS "${_TRITON_INSTALL_MARKER}")
        set(_TRITON_COMPILER_INSTALL_COMPLETE TRUE)
        message(STATUS
            "   Reusing cache-restored Triton compiler at ${TRITON_INSTALL_DIR}")
    endif()

    # Build-time-only SmartCcache partition key for Triton LLVM external build.
    # This stays in CMake/source-tree scope; runtime extraction infra is separate.
    set(_TRITON_LLVM_SHAPE_KEY_RAW
        "${TRITON_LLVM_COMMIT}-${TRITON_LLVM_TARGETS}-Release-${_TRITON_LLVM_RECIPE_REVISION}-${_TRITON_TARGET_CACHE_CONFIG}")
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" TRITON_LLVM_SHAPE_KEY "${_TRITON_LLVM_SHAPE_KEY_RAW}")
    # Windows MinGW generator executables need both their in-tree DLLs and the
    # compiler runtime on PATH. Keep this host-specific; use CMake's structured
    # PATH modification syntax so semicolons never become ExternalProject command
    # separators on Windows.
    set(_TRITON_LLVM_BUILD_ENV)
    if(CMAKE_HOST_WIN32)
        set(_TRITON_LLVM_BUILD_ENV
            --modify "PATH=path_list_prepend:${TRITON_LLVM_PREFIX}/build/bin"
            --modify "PATH=path_list_prepend:${TRITON_LLVM_INSTALL_DIR}/bin")
    endif()
    set(TRITON_LLVM_BUILD_COMMAND
            ${CMAKE_COMMAND}
                "-DBUILD_DIR=<BINARY_DIR>"
                -P "${CMAKE_SOURCE_DIR}/cmake/ValidateCmakeDependencyFiles.cmake"
            COMMAND ${CMAKE_COMMAND} -E env
                "SD_SMART_CCACHE_SEGMENT=triton_llvm"
                "SD_SMART_CCACHE_SHAPE_KEY=${TRITON_LLVM_SHAPE_KEY}"
                # Native LLVM/MLIR generators are executed during this build. Keep
                # the in-tree build/install DLL directories ahead of the inherited
                # toolchain path so MinGW generator executables resolve the exact
                # runtime they were linked with (and do not silently exit before
                # producing generated .inc files).
                ${_TRITON_LLVM_BUILD_ENV}
                -- # End environment assignments before the build command.
                # MLIRExecutionEngineShared is EXCLUDE_FROM_LIBMLIR, so include it
            # explicitly in the same build invocation. Keeping both goals in one
            # generator call avoids rebuilding the complete dependency graph.
            ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
                --target all MLIRExecutionEngineShared --parallel ${DEP_PARALLEL_JOBS}
    )
    message(STATUS "   Triton LLVM smart ccache segment=triton_llvm shape=${TRITON_LLVM_SHAPE_KEY}")

    set(TRITON_LLVM_URL
        "https://github.com/llvm/llvm-project/archive/${TRITON_LLVM_COMMIT}.tar.gz")
    set(_TRITON_LLVM_NATIVE_TOOL_DIR "${TRITON_LLVM_INSTALL_DIR}/bin")
    set(_TRITON_MANAGED_HOST_TOOLS_READY FALSE)
    if(CMAKE_CROSSCOMPILING AND SD_TRITON_MANAGED_LLVM_HOST_TOOLS)
        get_filename_component(_TRITON_LLVM_NATIVE_TOOL_DIR
            "${SD_TRITON_MANAGED_LLVM_HOST_TOOLS}" ABSOLUTE)
        if(NOT EXISTS "${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-tblgen" OR
           NOT EXISTS "${_TRITON_LLVM_NATIVE_TOOL_DIR}/mlir-tblgen")
            message(FATAL_ERROR
                "SD_TRITON_MANAGED_LLVM_HOST_TOOLS must contain executable llvm-tblgen and mlir-tblgen")
        endif()
        set(_TRITON_MANAGED_HOST_TOOLS_READY TRUE)
        message(STATUS
            "   Reusing managed LLVM host tools: ${_TRITON_LLVM_NATIVE_TOOL_DIR}")
    endif()

    if(CMAKE_CROSSCOMPILING)
        # Every cold cross-build generator, including SLEEF, needs native host
        # compilers. Resolve them independently of the target LLVM cache state.
        find_program(_TRITON_HOST_C_COMPILER
            NAMES gcc cc clang NO_CMAKE_FIND_ROOT_PATH)
        find_program(_TRITON_HOST_CXX_COMPILER
            NAMES g++ c++ clang++ NO_CMAKE_FIND_ROOT_PATH)
        if(NOT _TRITON_HOST_C_COMPILER OR NOT _TRITON_HOST_CXX_COMPILER)
            message(FATAL_ERROR
                "Cross-building Triton requires native host C and C++ compilers.")
        endif()
        set(_TRITON_HOST_EXE_SUFFIX "")
        if(CMAKE_HOST_WIN32)
            set(_TRITON_HOST_EXE_SUFFIX ".exe")
        endif()
    endif()

    if(CMAKE_CROSSCOMPILING AND
       NOT _TRITON_MANAGED_HOST_TOOLS_READY AND
       (NOT _TRITON_LLVM_INSTALL_COMPLETE OR
        (_TRITON_BUILDS_COMPILER AND NOT _TRITON_COMPILER_INSTALL_COMPLETE)))
        # A cold target LLVM build and Triton's generated sources both require
        # native tablegen executables. Cache this native snapshot independently
        # from every target ABI/NDK because the pinned LLVM source and host
        # compiler are its complete platform identity.
        string(SUBSTRING "${TRITON_LLVM_COMMIT}" 0 8 _TRITON_LLVM_HOST_VERSION)
        if(SD_DEP_CACHE)
            sd_dep_cache_host_key(
                "triton_llvm_host_tools"
                "${_TRITON_LLVM_HOST_VERSION}"
                "${_TRITON_HOST_C_COMPILER}"
                "${_TRITON_HOST_CXX_COMPILER}"
                "recipe=${_TRITON_LLVM_HOST_TOOLS_RECIPE_REVISION};patchSpirv=${_TRITON_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP}"
                _tllvm_host_cache_key)
            sd_dep_cache_check(
                "triton_llvm_host_tools"
                "${_tllvm_host_cache_key}"
                _tllvm_host_hit
                _tllvm_host_cache_path)
            if(_tllvm_host_hit)
                set(_TRITON_LLVM_NATIVE_TOOL_DIR "${_tllvm_host_cache_path}/bin")
                if(EXISTS "${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-tblgen${_TRITON_HOST_EXE_SUFFIX}" AND
                   EXISTS "${_TRITON_LLVM_NATIVE_TOOL_DIR}/mlir-tblgen${_TRITON_HOST_EXE_SUFFIX}" AND
                   EXISTS "${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-config${_TRITON_HOST_EXE_SUFFIX}")
                    set(_TRITON_MANAGED_HOST_TOOLS_READY TRUE)
                    add_custom_target(triton_llvm_host_tools_external)
                    message(STATUS
                        "   Reusing cached LLVM host tools: ${_TRITON_LLVM_NATIVE_TOOL_DIR}")
                else()
                    message(FATAL_ERROR
                        "Cached LLVM host-tool snapshot is incomplete: ${_tllvm_host_cache_path}")
                endif()
            endif()
        endif()

        if(NOT _TRITON_MANAGED_HOST_TOOLS_READY)
            find_program(_TRITON_HOST_NINJA
                NAMES ninja ninja-build NO_CMAKE_FIND_ROOT_PATH)
            if(NOT _TRITON_HOST_NINJA)
                message(FATAL_ERROR
                    "A cold cross-build of Triton LLVM requires host Ninja.")
            endif()

            set(_TRITON_LLVM_HOST_PREFIX
                "${CMAKE_BINARY_DIR}/triton_llvm_host_tools_${_TRITON_LLVM_HOST_TOOLS_RECIPE_REVISION}")
            set(_TRITON_LLVM_HOST_BUILD_DIR "${_TRITON_LLVM_HOST_PREFIX}/build")
            set(_TRITON_LLVM_HOST_INSTALL_DIR "${_TRITON_LLVM_HOST_PREFIX}/install")
            set(_TRITON_LLVM_NATIVE_TOOL_DIR "${_TRITON_LLVM_HOST_INSTALL_DIR}/bin")
            set(_TRITON_LLVM_HOST_CMAKE_ARGS
                -DCMAKE_MAKE_PROGRAM=${_TRITON_HOST_NINJA}
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON
                -DLLVM_ENABLE_PROJECTS=mlir
                -DLLVM_TARGETS_TO_BUILD=host
                -DLLVM_ENABLE_ASSERTIONS=ON
                -DLLVM_ENABLE_RTTI=ON
                -DLLVM_INCLUDE_TESTS=OFF
                -DLLVM_INCLUDE_BENCHMARKS=OFF
                -DLLVM_BUILD_EXAMPLES=OFF
                -DLLVM_INCLUDE_EXAMPLES=OFF
                -DMLIR_INCLUDE_TESTS=OFF
                -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
                -DCMAKE_C_COMPILER=${_TRITON_HOST_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${_TRITON_HOST_CXX_COMPILER})
            if(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
                list(APPEND _TRITON_LLVM_HOST_CMAKE_ARGS
                    "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}"
                    "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
            endif()

            ExternalProject_Add(triton_llvm_host_tools_external
                PREFIX            "${_TRITON_LLVM_HOST_PREFIX}"
                URL               "${TRITON_LLVM_URL}"
                URL_HASH          "${TRITON_LLVM_URL_HASH}"
                DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
                ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
                PATCH_COMMAND     ${CMAKE_COMMAND}
                    -DSOURCE_DIR=<SOURCE_DIR>
                    -DSD_EXTERNAL_PROJECT=LLVM
                    -DSD_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP=${_TRITON_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP}
                    -P "${CMAKE_SOURCE_DIR}/cmake/patch_external_llvm_coexistence.cmake"
                SOURCE_SUBDIR     llvm
                BINARY_DIR        "${_TRITON_LLVM_HOST_BUILD_DIR}"
                CMAKE_GENERATOR   "Ninja"
                CMAKE_ARGS        ${_TRITON_LLVM_HOST_CMAKE_ARGS}
                BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR>
                    --config Release --target llvm-tblgen mlir-tblgen llvm-config
                    --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ${CMAKE_COMMAND} -E make_directory
                        "${_TRITON_LLVM_HOST_INSTALL_DIR}/bin"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/llvm-tblgen${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-tblgen${_TRITON_HOST_EXE_SUFFIX}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/mlir-tblgen${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_LLVM_NATIVE_TOOL_DIR}/mlir-tblgen${_TRITON_HOST_EXE_SUFFIX}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/llvm-config${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-config${_TRITON_HOST_EXE_SUFFIX}"
                BUILD_BYPRODUCTS
                    "${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-tblgen${_TRITON_HOST_EXE_SUFFIX}"
                    "${_TRITON_LLVM_NATIVE_TOOL_DIR}/mlir-tblgen${_TRITON_HOST_EXE_SUFFIX}"
                    "${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-config${_TRITON_HOST_EXE_SUFFIX}"
                TIMEOUT           7200
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF)
            if(SD_DEP_CACHE AND DEFINED _tllvm_host_cache_key)
                sd_dep_cache_store(
                    "triton_llvm_host_tools"
                    "${_tllvm_host_cache_key}"
                    "${_TRITON_LLVM_HOST_INSTALL_DIR}"
                    "triton_llvm_host_tools_external")
            endif()
            message(STATUS
                "   LLVM host tools: ${_TRITON_LLVM_NATIVE_TOOL_DIR}")
        endif()
    endif()

    if(NOT _TRITON_LLVM_INSTALL_COMPLETE)
        message(STATUS "   Building LLVM/MLIR from Triton-pinned commit ${TRITON_LLVM_COMMIT}...")
        message(STATUS "   This is a one-time build (~15-30 min). Install dir: ${TRITON_LLVM_INSTALL_DIR}")

        file(MAKE_DIRECTORY "${TRITON_LLVM_PREFIX}/stamp")

        # Build LLVM cmake args, including compiler launcher if available
        set(TRITON_LLVM_CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=${TRITON_LLVM_INSTALL_DIR}
                -DCMAKE_BUILD_TYPE=Release
                # LLVM enables PCH under Clang, but compiler launchers cannot
                # reliably validate .pch outputs as object files. Disable PCH
                # for the managed dependency so ccache remains deterministic.
                -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON
                -DLLVM_ENABLE_PROJECTS=mlir
                -DLLVM_TARGETS_TO_BUILD=${TRITON_LLVM_TARGETS}
                -DLLVM_ENABLE_ASSERTIONS=ON
                -DLLVM_ENABLE_RTTI=ON
                # Upstream-supported monolithic DSOs. Triton and libnd4j consume
                # the exported LLVM and MLIR shared targets instead of embedding
                # private copies from component archives.
                -DLLVM_BUILD_LLVM_DYLIB=ON
                -DLLVM_LINK_LLVM_DYLIB=ON
                -DLLVM_DYLIB_COMPONENTS=all
                -DMLIR_BUILD_MLIR_DYLIB=ON
                -DMLIR_LINK_MLIR_DYLIB=ON
                # Cross-compiling Android makes LLVM_NATIVE_ARCH differ from
                # the target list, which otherwise disables MLIR's execution
                # engine (and its shared runtime target) by default.
                -DMLIR_ENABLE_EXECUTION_ENGINE=ON
                -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
                # Disable tests — mlir-translate/mlir-query link against test dialect
                # libraries (TestDialect, SymbolOp) that fail to build.
                # LLVM_BUILD_TOOLS must stay ON because mlir-tblgen is needed by Triton.
                -DLLVM_INCLUDE_TESTS=OFF
                -DLLVM_INCLUDE_BENCHMARKS=OFF
                -DLLVM_BUILD_EXAMPLES=OFF
                -DLLVM_INCLUDE_EXAMPLES=OFF
                -DMLIR_INCLUDE_TESTS=OFF
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        )
        if(CMAKE_CROSSCOMPILING)
            list(APPEND TRITON_LLVM_CMAKE_ARGS
                ${_TRITON_TARGET_CMAKE_ARGS}
                -DLLVM_NATIVE_TOOL_DIR=${_TRITON_LLVM_NATIVE_TOOL_DIR}
                -DLLVM_TABLEGEN=${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-tblgen${_TRITON_HOST_EXE_SUFFIX}
                -DMLIR_TABLEGEN=${_TRITON_LLVM_NATIVE_TOOL_DIR}/mlir-tblgen${_TRITON_HOST_EXE_SUFFIX})
        endif()

        # MSVC-specific flags for LLVM build
        if(MSVC)
            list(APPEND TRITON_LLVM_CMAKE_ARGS
                -DLLVM_BUILD_SHARED_LIBS=OFF
                "-DCMAKE_C_FLAGS=/utf-8 /D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING"
                "-DCMAKE_CXX_FLAGS=/utf-8 /D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING"
            )
        elseif(MINGW)
            # GNU ld cannot represent more than 65535 PE export ordinals. LLVM's
            # upstream MinGW DSO path requests --export-all-symbols, so the
            # default GCC visibility would make the monolithic libLLVM DLL exceed
            # that hard format limit. Keep the shared ABI boundary, but hide
            # unannotated implementation symbols; LLVM_ABI/LLVM_EXPORT_TEMPLATE
            # annotations remain exported for MLIR and downstream consumers.
            # MinGW cannot link LLVM tools against the monolithic DLL after
            # the PE export table is bounded below 65,535 entries. Keep the shared
            # producer for runtime packaging, but link LLVM/MLIR tools and the MLIR
            # shared target against their component archives.
            list(APPEND TRITON_LLVM_CMAKE_ARGS
                -DCMAKE_C_VISIBILITY_PRESET=hidden
                -DCMAKE_CXX_VISIBILITY_PRESET=hidden
                -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
                -DLLVM_LINK_LLVM_DYLIB=OFF
                -DMLIR_LINK_MLIR_DYLIB=OFF
            )
        endif()


        # Pass SmartCcache / compiler launcher to LLVM build
        # Smart ccache multi-element lists can't be passed as FILEPATH; fall back to plain ccache.
        if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}")
            list(APPEND TRITON_LLVM_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER}")
            message(STATUS "   Passing C compiler launcher to LLVM build: ${CMAKE_C_COMPILER_LAUNCHER}")
        elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
            list(APPEND TRITON_LLVM_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
        endif()
        if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}")
            list(APPEND TRITON_LLVM_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER}")
            message(STATUS "   Passing CXX compiler launcher to LLVM build: ${CMAKE_CXX_COMPILER_LAUNCHER}")
        elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
            list(APPEND TRITON_LLVM_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
        endif()

        ExternalProject_Add(triton_llvm_external
                PREFIX            "${TRITON_LLVM_PREFIX}"
                URL               "${TRITON_LLVM_URL}"
                URL_HASH          "${TRITON_LLVM_URL_HASH}"
                DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
                ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
                PATCH_COMMAND     ${CMAKE_COMMAND}
                    -DSOURCE_DIR=<SOURCE_DIR>
                    -DSD_EXTERNAL_PROJECT=LLVM
                    -DSD_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP=${_TRITON_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP}
                    -P "${CMAKE_SOURCE_DIR}/cmake/patch_external_llvm_coexistence.cmake"
                SOURCE_SUBDIR     llvm
                BINARY_DIR        "${TRITON_LLVM_PREFIX}/build"
                STAMP_DIR         "${TRITON_LLVM_PREFIX}/stamp"
                CMAKE_ARGS        ${TRITON_LLVM_CMAKE_ARGS}
                BUILD_COMMAND     ${TRITON_LLVM_BUILD_COMMAND}
                # BUILD_COMMAND already establishes the complete dependency graph.
                # Install from that finished tree directly instead of re-entering
                # GNU Make's dependency scanner from a nested ExternalProject recipe.
                INSTALL_COMMAND   ${CMAKE_COMMAND} --install <BINARY_DIR> --config Release
                    COMMAND ${CMAKE_COMMAND} -E touch "${_TRITON_LLVM_INSTALL_MARKER}"
                BUILD_BYPRODUCTS
                    "${_TRITON_LLVM_INSTALL_MARKER}"
                    "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake"
                    "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake"
                    "${_TRITON_MLIR_EXECUTION_ENGINE_SHARED_LIBRARY}"
                    "${_TRITON_MLIR_SHARED_LIBRARY}"
                    "${_TRITON_LLVM_SHARED_LIBRARY}"
                TIMEOUT           7200
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        # --- Cache store for Triton LLVM ---
        if(SD_DEP_CACHE AND DEFINED _tllvm_cache_key)
            sd_dep_cache_store("triton_llvm" "${_tllvm_cache_key}" "${TRITON_LLVM_INSTALL_DIR}" "triton_llvm_external")
        endif()
    else()
        message(STATUS "   Reusing existing LLVM/MLIR at ${TRITON_LLVM_INSTALL_DIR}")
        # Create a dummy target so triton_external can depend on it
        add_custom_target(triton_llvm_external)

    endif()
    if(TARGET triton_llvm_host_tools_external)
        add_dependencies(triton_llvm_external triton_llvm_host_tools_external)
    endif()

    set(_TRITON_EXTERNAL_DEPENDENCIES triton_llvm_external)
    if(CMAKE_CROSSCOMPILING AND
       _TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER" AND
       NOT _TRITON_COMPILER_INSTALL_COMPLETE)
        # SLEEF generates target headers with native utilities such as mkrename.
        # They depend on pinned SLEEF sources and the host compiler, not the NDK.
        set(_TRITON_SLEEF_HOST_TOOLS_READY FALSE)
        if(SD_DEP_CACHE)
            sd_dep_cache_host_key(
                "triton_sleef_host_tools"
                "3.8"
                "${_TRITON_HOST_C_COMPILER}"
                "${_TRITON_HOST_CXX_COMPILER}"
                "recipe=${_TRITON_SLEEF_HOST_TOOLS_RECIPE_REVISION};sourceSha256=a12ccd50f57083c530e1c76f10d52865defbd19fc9e2c85b483493065709874a"
                _triton_sleef_host_cache_key)
            sd_dep_cache_check(
                "triton_sleef_host_tools"
                "${_triton_sleef_host_cache_key}"
                _triton_sleef_host_hit
                _triton_sleef_host_cache_path)
            if(_triton_sleef_host_hit)
                set(_TRITON_SLEEF_HOST_BUILD_DIR "${_triton_sleef_host_cache_path}")
                foreach(_triton_sleef_tool
                        mkrename mkrename_gnuabi mkmasked_gnuabi mkdisp mkalias addSuffix)
                    if(NOT EXISTS
                       "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/${_triton_sleef_tool}${_TRITON_HOST_EXE_SUFFIX}")
                        message(FATAL_ERROR
                            "Cached SLEEF host-tool snapshot is incomplete: ${_triton_sleef_host_cache_path}")
                    endif()
                endforeach()
                set(_TRITON_SLEEF_HOST_TOOLS_READY TRUE)
                add_custom_target(triton_cpu_sleef_host_tools_external)
                message(STATUS
                    "   Reusing cached SLEEF host tools: ${_TRITON_SLEEF_HOST_BUILD_DIR}/bin")
            endif()
        endif()

        if(NOT _TRITON_SLEEF_HOST_TOOLS_READY)
            set(_TRITON_SLEEF_HOST_PREFIX
                "${CMAKE_BINARY_DIR}/triton_cpu_sleef_host_tools_${_TRITON_SLEEF_HOST_TOOLS_RECIPE_REVISION}")
            set(_TRITON_SLEEF_HOST_COMPILE_DIR
                "${_TRITON_SLEEF_HOST_PREFIX}/build")
            set(_TRITON_SLEEF_HOST_BUILD_DIR
                "${_TRITON_SLEEF_HOST_PREFIX}/install")
            set(_TRITON_SLEEF_HOST_CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON
                -DCMAKE_C_COMPILER=${_TRITON_HOST_C_COMPILER}
                -DSLEEF_BUILD_SHARED_LIBS=OFF
                -DSLEEF_BUILD_DFT=OFF
                -DSLEEF_BUILD_QUAD=OFF
                -DSLEEF_BUILD_GNUABI_LIBS=OFF
                -DSLEEF_BUILD_SCALAR_LIB=OFF
                -DSLEEF_BUILD_TESTS=OFF
                -DSLEEF_BUILD_BENCH=OFF)
            if(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
                list(APPEND _TRITON_SLEEF_HOST_CMAKE_ARGS
                    "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
            endif()

            ExternalProject_Add(triton_cpu_sleef_host_tools_external
                PREFIX            "${_TRITON_SLEEF_HOST_PREFIX}"
                URL               "https://github.com/shibatch/sleef/archive/refs/tags/3.8.tar.gz"
                URL_HASH          "SHA256=a12ccd50f57083c530e1c76f10d52865defbd19fc9e2c85b483493065709874a"
                DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
                ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
                BINARY_DIR        "${_TRITON_SLEEF_HOST_COMPILE_DIR}"
                CMAKE_ARGS        ${_TRITON_SLEEF_HOST_CMAKE_ARGS}
                BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR>
                    --config Release
                    --target mkrename mkrename_gnuabi mkmasked_gnuabi mkdisp mkalias addSuffix
                    --parallel ${DEP_PARALLEL_JOBS}
                INSTALL_COMMAND   ${CMAKE_COMMAND} -E make_directory
                        "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/mkrename${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkrename${_TRITON_HOST_EXE_SUFFIX}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/mkrename_gnuabi${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkrename_gnuabi${_TRITON_HOST_EXE_SUFFIX}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/mkmasked_gnuabi${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkmasked_gnuabi${_TRITON_HOST_EXE_SUFFIX}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/mkdisp${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkdisp${_TRITON_HOST_EXE_SUFFIX}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/mkalias${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkalias${_TRITON_HOST_EXE_SUFFIX}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "<BINARY_DIR>/bin/addSuffix${_TRITON_HOST_EXE_SUFFIX}"
                        "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/addSuffix${_TRITON_HOST_EXE_SUFFIX}"
                BUILD_BYPRODUCTS
                    "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkrename${_TRITON_HOST_EXE_SUFFIX}"
                    "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkrename_gnuabi${_TRITON_HOST_EXE_SUFFIX}"
                    "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkmasked_gnuabi${_TRITON_HOST_EXE_SUFFIX}"
                    "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkdisp${_TRITON_HOST_EXE_SUFFIX}"
                    "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/mkalias${_TRITON_HOST_EXE_SUFFIX}"
                    "${_TRITON_SLEEF_HOST_BUILD_DIR}/bin/addSuffix${_TRITON_HOST_EXE_SUFFIX}"
                TIMEOUT           1800
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF)
            if(SD_DEP_CACHE AND DEFINED _triton_sleef_host_cache_key)
                sd_dep_cache_store(
                    "triton_sleef_host_tools"
                    "${_triton_sleef_host_cache_key}"
                    "${_TRITON_SLEEF_HOST_BUILD_DIR}"
                    "triton_cpu_sleef_host_tools_external")
            endif()
            message(STATUS
                "   SLEEF host tools: ${_TRITON_SLEEF_HOST_BUILD_DIR}/bin")
        endif()

        list(APPEND _TRITON_EXTERNAL_DEPENDENCIES
            triton_cpu_sleef_host_tools_external)
    endif()

    set(TRITON_MLIR_DIR "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir")
    set(TRITON_LLVM_DIR "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/llvm")
    # Publish one target package identity for both Triton and the standalone
    # MLIR consumer. Cross builds must never rediscover host package files.
    set(LLVM_DIR "${TRITON_LLVM_DIR}" CACHE PATH
        "Project-managed target LLVM package" FORCE)
    set(MLIR_DIR "${TRITON_MLIR_DIR}" CACHE PATH
        "Project-managed target MLIR package" FORCE)

    if(_TRITON_BUILDS_COMPILER)
    set(TRITON_CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${TRITON_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${TRITON_INSTALL_DIR}/lib
            -DTRITON_BUILD_PYTHON_MODULE=OFF
            -DTRITON_BUILD_TESTING=OFF
            -DTRITON_BUILD_TUTORIALS=OFF
            -DTRITON_BUILD_PROTON=OFF
            -DTRITON_BUILD_UT=OFF
            -DTRITON_CODEGEN_BACKENDS=${TRITON_BACKENDS_STR}
            -DTRITON_CACHE_PATH=${CMAKE_BINARY_DIR}/.triton_cache
            -DMLIR_DIR=${TRITON_MLIR_DIR}
            -DLLVM_DIR=${TRITON_LLVM_DIR}
            -DLLVM_NATIVE_TOOL_DIR=${_TRITON_LLVM_NATIVE_TOOL_DIR}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )
    if(CMAKE_CROSSCOMPILING)
        list(APPEND TRITON_CMAKE_ARGS
            ${_TRITON_TARGET_CMAKE_ARGS}
            -DLLVM_HOST_TABLEGEN=${_TRITON_LLVM_NATIVE_TOOL_DIR}/llvm-tblgen${_TRITON_HOST_EXE_SUFFIX}
            -DMLIR_HOST_TABLEGEN=${_TRITON_LLVM_NATIVE_TOOL_DIR}/mlir-tblgen${_TRITON_HOST_EXE_SUFFIX})
        if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
            list(APPEND TRITON_CMAKE_ARGS
                -DNATIVE_BUILD_DIR=${_TRITON_SLEEF_HOST_BUILD_DIR})
        endif()
    endif()

    # MSVC-specific flags for Triton build
    if(MSVC)
        list(APPEND TRITON_CMAKE_ARGS
            "-DCMAKE_C_FLAGS=-bigobj -Zc:preprocessor -permissive- -utf-8"
            "-DCMAKE_CXX_FLAGS=-bigobj -Zc:preprocessor -permissive- -utf-8"
        )
    endif()

    # Pass compiler launcher if available
    # Smart ccache multi-element lists can't be passed as FILEPATH; fall back to plain ccache.
    if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}")
        list(APPEND TRITON_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER}")
    elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
        list(APPEND TRITON_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
    endif()
    if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}")
        list(APPEND TRITON_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER}")
    elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
        list(APPEND TRITON_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
    endif()

    # Build-time-only SmartCcache partition key for Triton external build.
    # Runtime key extraction/caching infrastructure will be layered separately.
    string(REPLACE ";" "_" _TRITON_BACKENDS_KEY "${TRITON_CODEGEN_BACKENDS}")
    set(_TRITON_SHAPE_KEY_RAW
        "${TRITON_VERSION}-${_TRITON_BACKENDS_KEY}-${CMAKE_BUILD_TYPE}-${_TRITON_COMPILER_RECIPE_REVISION}-${_TRITON_TARGET_CACHE_CONFIG}")
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" TRITON_SHAPE_KEY "${_TRITON_SHAPE_KEY_RAW}")
    # Use DEP_PARALLEL_JOBS (memory-based) instead of hardcoded 8.
    # Higher parallelism can cause race conditions between TableGen-generated
    # .h.inc files and compilation of sources that include them.
    set(TRITON_BUILD_COMMAND
            ${CMAKE_COMMAND} -E env
            "SD_SMART_CCACHE_SEGMENT=triton"
            "SD_SMART_CCACHE_SHAPE_KEY=${TRITON_SHAPE_KEY}"
            ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE} --parallel ${DEP_PARALLEL_JOBS}
    )
    message(STATUS "   Triton smart ccache segment=triton shape=${TRITON_SHAPE_KEY}")

    # Source dir name depends on variant (GitHub tarball extraction naming)
    if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
        set(TRITON_SOURCE_DIR "${TRITON_PREFIX}/src/triton-cpu-${TRITON_CPU_COMMIT}")
    else()
        set(TRITON_SOURCE_DIR "${TRITON_PREFIX}/src/triton-${TRITON_VERSION}")
    endif()

    # Cross-platform patch script (replaces bash-only patch_triton_no_amd.sh).
    # Triton's bin/RegisterTritonDialects.h unconditionally includes AMD dialect
    # headers, even when only the nvidia backend is selected via TRITON_CODEGEN_BACKENDS.
    # When the AMD backend isn't built, the .h.inc files don't get generated, causing
    # compilation failures. The CMake patch script removes AMD-specific code and adds
    # NegFOp/TanhOp patterns needed by our IR.
    # For CPU triton-cpu builds, the source tree is different — skip GPU-specific patches.
    if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
        # triton-cpu has its own dialect structure; no GPU-specific patching needed
        set(_TRITON_PATCH_COMMAND "")
    else()
        set(TRITON_PATCH_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/patch_triton.cmake")
        set(_TRITON_PATCH_ARGS
            -DSOURCE_DIR=${TRITON_SOURCE_DIR}
        )
        if(NOT "amd" IN_LIST TRITON_CODEGEN_BACKENDS)
            list(APPEND _TRITON_PATCH_ARGS -DREMOVE_AMD=ON)
        endif()
        set(_TRITON_PATCH_COMMAND ${CMAKE_COMMAND} ${_TRITON_PATCH_ARGS} -P "${TRITON_PATCH_SCRIPT}")
    endif()

    # Cross-platform install script (replaces bash-only install_triton.sh).
    # Triton uses OBJECT libraries (no install() rules), so we archive .o/.obj files
    # into a static library and copy headers.
    set(TRITON_INSTALL_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/install_triton.cmake")

    # Build byproducts differ by platform
    if(WIN32)
        set(_TRITON_LIB_BYPRODUCT "${TRITON_INSTALL_DIR}/lib/triton.lib")
    else()
        set(_TRITON_LIB_BYPRODUCT "${TRITON_INSTALL_DIR}/lib/libtriton.a")
    endif()

    # For CPU triton-cpu, PATCH_COMMAND strips Python dependency from CMakeLists.txt.
    # triton-cpu unconditionally calls find_package(Python3 REQUIRED) to run build_helpers.py
    # which generates LLVM/JSON cmake vars. We pre-generate this file and patch CMakeLists.txt
    # to skip the Python-dependent block entirely.
    if(_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER")
        set(_TRITON_CPU_PATCH_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/patch_triton_cpu.cmake")
        set(_TRITON_EP_PATCH_CMD ${CMAKE_COMMAND}
            -DSOURCE_DIR=<SOURCE_DIR>
            -DLLVM_INSTALL_DIR=${TRITON_LLVM_INSTALL_DIR}
            -DDOWNLOAD_DIR=${CMAKE_BINARY_DIR}/downloads
            -P "${_TRITON_CPU_PATCH_SCRIPT}")
    else()
        set(_TRITON_EP_PATCH_CMD ${_TRITON_PATCH_COMMAND})
    endif()

    # URL_HASH for download caching (skip for Windows which uses unpinned branch)
    set(_TRITON_EP_URL_HASH "")
    if(DEFINED TRITON_URL_HASH)
        set(_TRITON_EP_URL_HASH "${TRITON_URL_HASH}")
    endif()

    if(_TRITON_COMPILER_INSTALL_COMPLETE)
        # Preserve the stable target consumed by MainBuildFlow without
        # downloading, configuring, or rebuilding an already-restored compiler.
        add_custom_target(triton_external)
        add_dependencies(triton_external triton_llvm_external)
    else()
        ExternalProject_Add(triton_external
            PREFIX            "${TRITON_PREFIX}"
            URL               "${TRITON_URL}"
            URL_HASH          "${_TRITON_EP_URL_HASH}"
            DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
            SOURCE_DIR        "${TRITON_SOURCE_DIR}"
            BINARY_DIR        "${TRITON_PREFIX}/build"
            STAMP_DIR         "${TRITON_STAMP_DIR}"
            DOWNLOAD_NO_PROGRESS FALSE
            ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
            PATCH_COMMAND     ${_TRITON_EP_PATCH_CMD}
            CMAKE_ARGS        ${TRITON_CMAKE_ARGS}
            BUILD_COMMAND     ${TRITON_BUILD_COMMAND}
            INSTALL_COMMAND   ${CMAKE_COMMAND} -DBINARY_DIR=<BINARY_DIR> -DSOURCE_DIR=<SOURCE_DIR> -DINSTALL_DIR=${TRITON_INSTALL_DIR} -P "${TRITON_INSTALL_SCRIPT}"
                COMMAND ${CMAKE_COMMAND} -E touch "${_TRITON_INSTALL_MARKER}"
            BUILD_BYPRODUCTS
                "${_TRITON_INSTALL_MARKER}"
                "${TRITON_INSTALL_DIR}/include/triton/Compiler/Compiler.h"
                "${_TRITON_LIB_BYPRODUCT}"
            TIMEOUT           1800
            LOG_DOWNLOAD      OFF
            LOG_CONFIGURE     OFF
            LOG_BUILD         OFF
            LOG_INSTALL       OFF
            DEPENDS           ${_TRITON_EXTERNAL_DEPENDENCIES}
        )
    endif()
    else()
        # MainBuildFlow depends on this stable compiler-package target. Vulkan
        # has no Triton emitter build; its target represents only the managed
        # shared LLVM/MLIR producer.
        add_custom_target(triton_external)
        add_dependencies(triton_external triton_llvm_external)
    endif()

    add_library(triton_interface INTERFACE)
    # These headers and the shared libraries below are one versioned
    # package. Keep their include roots ahead of ambient compiler search paths.
    target_include_directories(triton_interface INTERFACE
        "${TRITON_LLVM_INSTALL_DIR}/include"
    )

    if(_TRITON_BUILDS_COMPILER)
        target_include_directories(triton_interface INTERFACE
            "${TRITON_INSTALL_DIR}/include")
        if(WIN32)
            target_link_libraries(triton_interface INTERFACE "${TRITON_INSTALL_DIR}/lib/triton.lib")
        else()
            target_link_libraries(triton_interface INTERFACE "${TRITON_INSTALL_DIR}/lib/libtriton.a")
        endif()
    endif()

    # The pinned LLVM build produces the upstream monolithic LLVM and MLIR
    # shared libraries. For a fresh build their package exports do not exist at
    # configure time, so model the known build byproducts as imported targets.
    if(WIN32)
        set(_TRITON_LLVM_SHARED_LIBRARY "${TRITON_LLVM_INSTALL_DIR}/bin/libLLVM.dll")
        set(_TRITON_MLIR_SHARED_LIBRARY "${TRITON_LLVM_INSTALL_DIR}/bin/libMLIR.dll")
        set(_TRITON_LLVM_IMPORT_LIBRARY "${TRITON_LLVM_INSTALL_DIR}/lib/libLLVM.dll.a")
        set(_TRITON_MLIR_IMPORT_LIBRARY "${TRITON_LLVM_INSTALL_DIR}/lib/libMLIR.dll.a")
    elseif(APPLE)
        set(_TRITON_LLVM_SHARED_LIBRARY "${TRITON_LLVM_INSTALL_DIR}/lib/libLLVM.dylib")
        set(_TRITON_MLIR_SHARED_LIBRARY "${TRITON_LLVM_INSTALL_DIR}/lib/libMLIR.dylib")
    else()
        set(_TRITON_LLVM_SHARED_LIBRARY "${TRITON_LLVM_INSTALL_DIR}/lib/libLLVM.so")
        set(_TRITON_MLIR_SHARED_LIBRARY "${TRITON_LLVM_INSTALL_DIR}/lib/libMLIR.so")
    endif()

    add_library(triton_llvm_shared SHARED IMPORTED GLOBAL)
    if(WIN32)
        set_target_properties(triton_llvm_shared PROPERTIES
            IMPORTED_LOCATION "${_TRITON_LLVM_SHARED_LIBRARY}"
            IMPORTED_IMPLIB "${_TRITON_LLVM_IMPORT_LIBRARY}")
    else()
        set_target_properties(triton_llvm_shared PROPERTIES
            IMPORTED_LOCATION "${_TRITON_LLVM_SHARED_LIBRARY}")
    endif()
    add_dependencies(triton_llvm_shared triton_llvm_external)

    add_library(triton_mlir_shared SHARED IMPORTED GLOBAL)
    if(WIN32)
        set_target_properties(triton_mlir_shared PROPERTIES
            IMPORTED_LOCATION "${_TRITON_MLIR_SHARED_LIBRARY}"
            IMPORTED_IMPLIB "${_TRITON_MLIR_IMPORT_LIBRARY}"
            INTERFACE_LINK_LIBRARIES triton_llvm_shared)
    else()
        set_target_properties(triton_mlir_shared PROPERTIES
            IMPORTED_LOCATION "${_TRITON_MLIR_SHARED_LIBRARY}"
            INTERFACE_LINK_LIBRARIES triton_llvm_shared)
    endif()
    add_dependencies(triton_mlir_shared triton_llvm_external)

    target_link_libraries(triton_interface INTERFACE
        triton_mlir_shared
        triton_llvm_shared)
    if(NOT WIN32)
        target_link_libraries(triton_interface INTERFACE -lz -lm)
        if(NOT APPLE AND NOT ANDROID)
            target_link_libraries(triton_interface INTERFACE -lrt -ldl -lpthread)
        elseif(APPLE)
            target_link_libraries(triton_interface INTERFACE -ldl -lpthread)
        else()
            target_link_libraries(triton_interface INTERFACE -ldl)
        endif()
    endif()
    message(STATUS
        "Triton interface: shared MLIR=${_TRITON_MLIR_SHARED_LIBRARY}, "
        "LLVM=${_TRITON_LLVM_SHARED_LIBRARY} (fresh build)")

    # NVRTC and the CUDA driver belong only to the CUDA emitter consumer.
    if(SD_CUDA)
        if(WIN32)
            target_link_libraries(triton_interface INTERFACE nvrtc.lib cuda.lib)
        else()
            target_link_libraries(triton_interface INTERFACE -lnvrtc -lcuda)
        endif()
    endif()

    add_dependencies(triton_interface triton_external)
    set(TRITON triton_interface PARENT_SCOPE)

    # --- Cache store for Triton ---
    if(_TRITON_BUILDS_COMPILER AND NOT _TRITON_COMPILER_INSTALL_COMPLETE AND
       SD_DEP_CACHE AND DEFINED _triton_cache_key)
        sd_dep_cache_store("triton" "${_triton_cache_key}" "${TRITON_INSTALL_DIR}" "triton_external")
    endif()

    if(_TRITON_BUILDS_COMPILER)
        message(STATUS
            "Triton ${TRITON_VERSION} setup complete (${_TRITON_CONSUMER_KIND}, target: ${TRITON_GPU_TARGET})")
    else()
        message(STATUS
            "Triton DSP compiler package setup complete (${_TRITON_CONSUMER_KIND}, shared LLVM/MLIR only)")
    endif()
endfunction()

# =============================================================================
# CUTLASS (Header-only CUDA Templates for Linear Algebra)
# =============================================================================
function(setup_cutlass)
    set(HAVE_CUTLASS FALSE PARENT_SCOPE)

    if(NOT HELPERS_cutlass STREQUAL "ON")
        if(HAVE_TRITON)
            message(STATUS "CUTLASS auto-enabled (HAVE_TRITON=ON)")
        else()
            message(STATUS "CUTLASS helper is disabled (HELPERS_cutlass=${HELPERS_cutlass})")
            return()
        endif()
    endif()

    if(NOT SD_CUDA)
        message(STATUS "CUTLASS helper requires CUDA build (SD_CUDA=ON)")
        return()
    endif()

    if(TARGET cutlass_external)
        message(STATUS "CUTLASS helper is enabled (target already exists)")
        set(HAVE_CUTLASS TRUE PARENT_SCOPE)
        set(CUTLASS cutlass_interface PARENT_SCOPE)
        return()
    endif()

    message(STATUS "CUTLASS helper is enabled")
    set(HAVE_CUTLASS TRUE PARENT_SCOPE)
    set(HELPERS_cutlass ON PARENT_SCOPE)

    set(CUTLASS_VERSION "3.7.0")
    set(CUTLASS_INSTALL_DIR "${CMAKE_BINARY_DIR}/cutlass_install")
    set(CUTLASS_PREFIX "${CMAKE_BINARY_DIR}/cutlass_external")

    file(MAKE_DIRECTORY "${CUTLASS_PREFIX}/stamp")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")

    set(CUTLASS_URL "https://github.com/NVIDIA/cutlass/archive/refs/tags/v${CUTLASS_VERSION}.tar.gz")
    set(CUTLASS_URL_HASH "SHA256=dfcafb7435a1b114ce32faee4f3257e276caf08f55fea04fa8bf3efa3a83c814")

    # CUTLASS is header-only for templates — we only need to download and install headers
    ExternalProject_Add(cutlass_external
            PREFIX            "${CUTLASS_PREFIX}"
            URL               "${CUTLASS_URL}"
            URL_HASH          "${CUTLASS_URL_HASH}"
            DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
            ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND     ""
            INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${CUTLASS_INSTALL_DIR}/include
            TIMEOUT           300
            LOG_DOWNLOAD      OFF
    )

    add_library(cutlass_interface INTERFACE)
    target_include_directories(cutlass_interface INTERFACE
        "${CUTLASS_INSTALL_DIR}/include"
    )
    add_dependencies(cutlass_interface cutlass_external)

    set(CUTLASS cutlass_interface PARENT_SCOPE)
    # HAVE_CUTLASS is provided via generated config.h, not as a global -D flag.
    # Global -D flags change every file's compiler command line, breaking ccache.
    set(HAVE_CUTLASS ON CACHE BOOL "CUTLASS availability" FORCE)

    message(STATUS "✅ CUTLASS ${CUTLASS_VERSION} setup complete (header-only)")
endfunction()

# =============================================================================
# MLX (Apple Metal GPU Compute on Apple Silicon)
# =============================================================================
function(setup_mlx)
    # Only on macOS arm64
    if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(HAVE_MLX OFF CACHE BOOL "MLX availability" FORCE)
        message(STATUS "MLX disabled (requires macOS arm64)")
        return()
    endif()

    # Opt-in: -DSD_MLX=ON or SD_MLX=ON env var
    set(_mlx_requested OFF)
    if(DEFINED SD_MLX AND SD_MLX)
        set(_mlx_requested ON)
    endif()
    if(DEFINED ENV{SD_MLX} AND "$ENV{SD_MLX}" STREQUAL "ON")
        set(_mlx_requested ON)
    endif()
    # Explicit opt-out overrides
    if(DEFINED SD_MLX AND NOT SD_MLX)
        set(_mlx_requested OFF)
    endif()
    if(DEFINED ENV{SD_MLX} AND "$ENV{SD_MLX}" STREQUAL "OFF")
        set(_mlx_requested OFF)
    endif()

    if(NOT _mlx_requested)
        message(STATUS "MLX disabled (use -DSD_MLX=ON to enable)")
        set(HAVE_MLX OFF CACHE BOOL "MLX availability" FORCE)
        set(MLX "" PARENT_SCOPE)
        return()
    endif()

    # Try FindMLX first (pre-installed)
    include(${CMAKE_CURRENT_LIST_DIR}/FindMLX.cmake)
    if(MLX_FOUND)
        message(STATUS "MLX found at ${MLX_INCLUDE_DIRS}")
        if(NOT TARGET mlx_interface)
            add_library(mlx_interface INTERFACE)
            target_include_directories(mlx_interface INTERFACE ${MLX_INCLUDE_DIRS})
            target_link_libraries(mlx_interface INTERFACE ${MLX_LIBRARIES})
            # MLX requires Metal, Foundation, Accelerate frameworks
            target_link_libraries(mlx_interface INTERFACE
                "-framework Metal"
                "-framework Foundation"
                "-framework Accelerate"
            )
        endif()
        set(HAVE_MLX ON CACHE BOOL "MLX availability" FORCE)
        set(MLX mlx_interface PARENT_SCOPE)
        message(STATUS "MLX setup complete (pre-installed)")
        return()
    endif()

    # Build from source via ExternalProject_Add
    message(STATUS "MLX not found locally — building from source")
    set(MLX_VERSION "0.22.0")
    set(MLX_INSTALL_DIR "${CMAKE_BINARY_DIR}/mlx_install")
    set(MLX_PREFIX "${CMAKE_BINARY_DIR}/mlx_external")

    file(MAKE_DIRECTORY "${MLX_PREFIX}/stamp")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")

    set(MLX_URL "https://github.com/ml-explore/mlx/archive/refs/tags/v${MLX_VERSION}.tar.gz")
    set(MLX_URL_HASH "SHA256=c8c890f450a4c09704b2597c56e111fbd4eb2c75d66a9c8f1fb1096c3e2b2cbe")

    # Build ccache args for MLX
    set(MLX_CCACHE_ARGS "")
    if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}")
        list(APPEND MLX_CCACHE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER}")
    elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
        list(APPEND MLX_CCACHE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
    endif()
    if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}")
        list(APPEND MLX_CCACHE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER}")
    elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
        list(APPEND MLX_CCACHE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
    endif()

    include(ExternalProject)
    ExternalProject_Add(mlx_external
        PREFIX            "${MLX_PREFIX}"
        URL               "${MLX_URL}"
        URL_HASH          "${MLX_URL_HASH}"
        DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
        ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${MLX_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_CXX_STANDARD=20
            -DMLX_BUILD_TESTS=OFF
            -DMLX_BUILD_EXAMPLES=OFF
            -DMLX_BUILD_PYTHON_BINDINGS=OFF
            -DMLX_BUILD_BENCHMARKS=OFF
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            # Force shared library so libmlx.dylib is produced (CMake default is static)
            -DBUILD_SHARED_LIBS=ON
            ${MLX_CCACHE_ARGS}
        BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --parallel ${DEP_PARALLEL_JOBS}
        TIMEOUT           600
        LOG_DOWNLOAD      OFF
        LOG_CONFIGURE     OFF
        LOG_BUILD         OFF
        LOG_INSTALL       OFF
    )

    add_library(mlx_interface INTERFACE)
    target_include_directories(mlx_interface INTERFACE
        "${MLX_INSTALL_DIR}/include"
    )
    # Link the built MLX library
    target_link_libraries(mlx_interface INTERFACE
        "${MLX_INSTALL_DIR}/lib/libmlx.dylib"
    )
    # Apple frameworks
    target_link_libraries(mlx_interface INTERFACE
        "-framework Metal"
        "-framework Foundation"
        "-framework Accelerate"
    )
    add_dependencies(mlx_interface mlx_external)

    set(HAVE_MLX ON CACHE BOOL "MLX availability" FORCE)
    set(MLX mlx_interface PARENT_SCOPE)

    message(STATUS "MLX ${MLX_VERSION} setup complete (building from source)")
endfunction()

# =============================================================================
# NCCL (NVIDIA Collective Communications Library)
# =============================================================================
function(setup_nccl)
    set(HAVE_NCCL FALSE PARENT_SCOPE)

    if(NOT HELPERS_nccl STREQUAL "ON")
        message(STATUS "NCCL helper is disabled (HELPERS_nccl=${HELPERS_nccl})")
        return()
    endif()

    if(NOT SD_CUDA)
        message(STATUS "NCCL helper requires CUDA build (SD_CUDA=ON)")
        return()
    endif()

    # Try to find system-installed NCCL
    find_package(NCCL QUIET)

    if(NCCL_FOUND)
        message(STATUS "✅ Found system NCCL: version ${NCCL_VERSION}")
        message(STATUS "   Include: ${NCCL_INCLUDE_DIRS}")
        message(STATUS "   Library: ${NCCL_LIBRARIES}")

        set(HAVE_NCCL TRUE PARENT_SCOPE)
        set(NCCL_LIB NCCL::nccl PARENT_SCOPE)
        add_compile_definitions(HAVE_NCCL=1)
        sd_register_helper("nccl")
    else()
        message(WARNING "NCCL not found. Install NCCL or set NCCL_ROOT. Multi-GPU collective ops will be unavailable.")
        message(STATUS "   Install: apt-get install libnccl-dev  OR  set -DNCCL_ROOT=/path/to/nccl")
    endif()

    message(STATUS "✅ NCCL setup complete (HAVE_NCCL=${HAVE_NCCL})")
endfunction()

# =============================================================================
# cuSPARSE — per-device sparse BLAS handles (CUDA only)
# cuSPARSE ships as part of the CUDA Toolkit; no ExternalProject needed.
# CMake's FindCUDAToolkit (already called for cuBLAS/cuSolver) exposes the
# CUDA::cusparse imported target automatically.  We just set the compile
# definition so C++ code can guard on HAVE_CUSPARSE.
# =============================================================================
function(setup_cusparse)
    set(HAVE_CUSPARSE FALSE PARENT_SCOPE)

    if(NOT SD_CUDA)
        message(STATUS "cuSPARSE helper requires CUDA build (SD_CUDA=ON) — skipping")
        return()
    endif()

    # FindCUDAToolkit exposes CUDA::cusparse when the library is present.
    # find_package may have already run; re-running is idempotent.
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND AND TARGET CUDA::cusparse)
        add_compile_definitions(HAVE_CUSPARSE=1)
        set(HAVE_CUSPARSE TRUE PARENT_SCOPE)
        # CUSPARSE_LIBRARIES is provided for callers that link manually
        # (e.g. PartialLinking.cmake).  Modern targets use CUDA::cusparse directly.
        get_target_property(_cusparse_loc CUDA::cusparse IMPORTED_LOCATION)
        if(_cusparse_loc)
            set(CUSPARSE_LIBRARIES "${_cusparse_loc}" PARENT_SCOPE)
        else()
            set(CUSPARSE_LIBRARIES "cusparse" PARENT_SCOPE)
        endif()
        message(STATUS "✅ cuSPARSE found (CUDA::cusparse available) — HAVE_CUSPARSE=1")
    else()
        message(WARNING "cuSPARSE not found in CUDA Toolkit — sparse BLAS ops will be unavailable")
    endif()

    message(STATUS "✅ cuSPARSE setup complete (HAVE_CUSPARSE=${HAVE_CUSPARSE})")
endfunction()

# =============================================================================
# OpenVINO (Intel CPU graph backend — Snippets JIT + oneDNN BRGEMM)
# Builds ENTIRELY from source like Triton: downloads OpenVINO + all submodule
# dependencies (oneDNN, xbyak, pugixml, flatbuffers, ittapi, nlohmann_json,
# TBB), patches out Python, compiles C++ only with static linking.
# We manage every dependency ourselves for full control over optimizations
# and libc compatibility.
# =============================================================================
function(setup_openvino)
    set(HAVE_OPENVINO FALSE PARENT_SCOPE)

    # OpenVINO is triggered by SD_TRITON=ON (via -Dlibnd4j.triton=ON), not helpers.
    # It's an independent internal variable but part of the "triton" build umbrella.
    if(NOT SD_TRITON)
        message(STATUS "OpenVINO disabled (SD_TRITON is OFF, use -Dlibnd4j.triton=ON)")
        return()
    endif()

    if(SD_CUDA)
        message(STATUS "OpenVINO is CPU-only (skipping for CUDA build)")
        return()
    endif()

    # OpenVINO Intel CPU plugin requires x86/x86_64 (depends on xbyak JIT, x86 ISAs).
    # Disable for all non-x86 targets: Android, ARM64, Apple Silicon, cross-compilation.
    if(CMAKE_CROSSCOMPILING)
        message(STATUS "OpenVINO disabled (cross-compilation target: Intel CPU plugin is x86-only)")
        return()
    endif()

    if(SD_ANDROID_BUILD OR ANDROID)
        message(STATUS "OpenVINO disabled (Android target: Intel CPU plugin is x86-only)")
        return()
    endif()

    if(SD_ARM_BUILD OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64|armv8")
        message(STATUS "OpenVINO disabled (ARM target: Intel CPU plugin requires x86)")
        return()
    endif()

    if(APPLE)
        message(STATUS "OpenVINO disabled (Apple platform: Intel CPU plugin requires x86 Linux/Windows)")
        return()
    endif()

    if(WIN32 OR MINGW OR MSYS OR CYGWIN)
        message(STATUS "OpenVINO disabled (Windows/MSYS2: ExternalProject TBB/oneDNN build not supported)")
        return()
    endif()

    # ── Versions and pinned commits (from OpenVINO 2026.0.0 .gitmodules) ──
    set(OPENVINO_VERSION "2026.0.0")
    # Submodule pinned commits for 2026.0.0
    set(OV_ONEDNN_COMMIT   "c6b79c1207bd5f20b9395536dab1d71a47cfcb1d")
    set(OV_XBYAK_COMMIT    "0d67fd1530016b7c56f3cd74b3fca920f4c3e2b4")
    set(OV_PUGIXML_COMMIT  "ee86beb30e4973f5feffe3ce63bfa4fbadf72f38")
    set(OV_FLATBUF_COMMIT  "595bf0007ab1929570c7671f091313c8fc20644e")
    set(OV_ITTAPI_COMMIT   "ca45fef1a12cef3316e6ff362a4d36571270e392")
    set(OV_JSON_COMMIT     "9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03")
    set(OV_ZLIB_COMMIT     "51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf")
    # TBB: OpenVINO 2026.0 uses oneAPI TBB 2021.13.1
    set(OV_TBB_VERSION     "2021.13.1")

    set(OPENVINO_INSTALL_DIR "${CMAKE_BINARY_DIR}/openvino_install")
    set(OPENVINO_PREFIX "${CMAKE_BINARY_DIR}/openvino_external")
    set(OPENVINO_SOURCE_DIR "${OPENVINO_PREFIX}/src/openvino-${OPENVINO_VERSION}")
    set(OPENVINO_STAMP_DIR "${OPENVINO_PREFIX}/stamp")

    # ── Check if OpenVINO is already built from a previous run ──
    set(_OV_CONFIG_EXISTS FALSE)
    foreach(_cfg_path
            "${OPENVINO_INSTALL_DIR}/runtime/lib/cmake/OpenVINO/OpenVINOConfig.cmake"
            "${OPENVINO_INSTALL_DIR}/runtime/cmake/OpenVINOConfig.cmake"
            "${OPENVINO_INSTALL_DIR}/lib/cmake/OpenVINO/OpenVINOConfig.cmake")
        if(EXISTS "${_cfg_path}")
            set(_OV_CONFIG_EXISTS TRUE)
            break()
        endif()
    endforeach()

    if(_OV_CONFIG_EXISTS)
        message(STATUS "OpenVINO: reusing existing install at ${OPENVINO_INSTALL_DIR}")
        set(HAVE_OPENVINO ON CACHE BOOL "OpenVINO availability" FORCE)
        set(HAVE_OPENVINO ON PARENT_SCOPE)

        # Ensure TBB cmake config is present (may be missing from first install)
        set(_TBB_SRC_CMAKE "${OPENVINO_PREFIX}/tbb_install/lib64/cmake/TBB")
        set(_TBB_DST_CMAKE "${OPENVINO_INSTALL_DIR}/runtime/3rdparty/tbb/lib64/cmake/TBB")
        if(EXISTS "${_TBB_SRC_CMAKE}/TBBConfig.cmake" AND NOT EXISTS "${_TBB_DST_CMAKE}/TBBConfig.cmake")
            message(STATUS "  Copying TBB cmake config into OpenVINO install")
            file(MAKE_DIRECTORY "${_TBB_DST_CMAKE}")
            file(GLOB _tbb_cmake_files "${_TBB_SRC_CMAKE}/*.cmake")
            foreach(_f IN LISTS _tbb_cmake_files)
                get_filename_component(_fname "${_f}" NAME)
                file(COPY "${_f}" DESTINATION "${_TBB_DST_CMAKE}")
            endforeach()
        endif()

        if(NOT TARGET openvino_external)
            add_custom_target(openvino_external)
        endif()

        if(NOT TARGET openvino_interface)
            _openvino_create_interface_from_install("${OPENVINO_INSTALL_DIR}")
        endif()

        set(OPENVINO_LIB openvino_interface PARENT_SCOPE)
        sd_register_helper("openvino")
        message(STATUS "OpenVINO setup complete (HAVE_OPENVINO=ON, reused)")
        return()
    endif()

    if(TARGET openvino_external)
        message(STATUS "OpenVINO helper is enabled (target already exists)")
        set(HAVE_OPENVINO ON CACHE BOOL "OpenVINO availability" FORCE)
        set(HAVE_OPENVINO ON PARENT_SCOPE)
        set(OPENVINO_LIB openvino_interface PARENT_SCOPE)
        return()
    endif()

    # ── Dependency cache: restore if available ──
    if(SD_DEP_CACHE)
        sd_dep_cache_key("openvino" "${OPENVINO_VERSION}" "" _ov_cache_key)
        sd_dep_cache_check("openvino" "${_ov_cache_key}" _ov_hit _ov_cache_path)
        if(_ov_hit AND NOT _OV_CONFIG_EXISTS)
            sd_dep_cache_restore("openvino" "${_ov_cache_path}" "${OPENVINO_INSTALL_DIR}")
            foreach(_cfg_path
                    "${OPENVINO_INSTALL_DIR}/runtime/lib/cmake/OpenVINO/OpenVINOConfig.cmake"
                    "${OPENVINO_INSTALL_DIR}/runtime/cmake/OpenVINOConfig.cmake"
                    "${OPENVINO_INSTALL_DIR}/lib/cmake/OpenVINO/OpenVINOConfig.cmake")
                if(EXISTS "${_cfg_path}")
                    message(STATUS "OpenVINO restored from dependency cache")
                    setup_openvino()
                    # Propagate results from the recursive call to OUR caller's scope.
                    # The recursive call sets HAVE_OPENVINO in the cache (FORCE) and in
                    # our local scope (PARENT_SCOPE from its perspective). We need to
                    # forward it one more level to the scope that called US.
                    set(HAVE_OPENVINO ${HAVE_OPENVINO} PARENT_SCOPE)
                    if(DEFINED OPENVINO_LIB)
                        set(OPENVINO_LIB ${OPENVINO_LIB} PARENT_SCOPE)
                    endif()
                    return()
                endif()
            endforeach()
        endif()
    endif()

    # ── Build from source ──
    message(STATUS "")
    message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
    message(STATUS "║  Building OpenVINO ${OPENVINO_VERSION} from source (C++ only)             ║")
    message(STATUS "║  All dependencies managed by us — no system packages required      ║")
    message(STATUS "║  This is a one-time build (~15-25 min). ${DEP_PARALLEL_JOBS} parallel jobs             ║")
    message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
    message(STATUS "")

    set(HAVE_OPENVINO ON CACHE BOOL "OpenVINO availability" FORCE)
    set(HAVE_OPENVINO ON PARENT_SCOPE)
    # OpenVINO is driven by SD_TRITON, not helpers

    file(MAKE_DIRECTORY "${OPENVINO_STAMP_DIR}")
    file(MAKE_DIRECTORY "${OPENVINO_PREFIX}/src")
    file(MAKE_DIRECTORY "${OPENVINO_PREFIX}/build")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")

    # Clean stale stamp files
    if(EXISTS "${OPENVINO_STAMP_DIR}")
        file(GLOB _OV_STALE_STAMPS "${OPENVINO_STAMP_DIR}/*.txt")
        foreach(stamp ${_OV_STALE_STAMPS})
            file(REMOVE "${stamp}")
        endforeach()
    endif()

    set(OPENVINO_URL "https://github.com/openvinotoolkit/openvino/archive/refs/tags/${OPENVINO_VERSION}.tar.gz")
    set(OPENVINO_URL_HASH "SHA256=529ce766bcca30991c21d0e065886e175b5210d81d6f6b3d7cdaaa89fe22ea8a")

    # ── Submodule URLs (downloaded by patch_openvino.cmake into source tree) ──
    # We pass these as cmake args to the patch script so it can fetch them.
    set(OV_SUBMODULE_URLS
        "ONEDNN_URL=https://github.com/openvinotoolkit/oneDNN/archive/${OV_ONEDNN_COMMIT}.tar.gz"
        "XBYAK_URL=https://github.com/herumi/xbyak/archive/${OV_XBYAK_COMMIT}.tar.gz"
        "PUGIXML_URL=https://github.com/zeux/pugixml/archive/${OV_PUGIXML_COMMIT}.tar.gz"
        "FLATBUF_URL=https://github.com/google/flatbuffers/archive/${OV_FLATBUF_COMMIT}.tar.gz"
        "ITTAPI_URL=https://github.com/intel/ittapi/archive/${OV_ITTAPI_COMMIT}.tar.gz"
        "JSON_URL=https://github.com/nlohmann/json/archive/${OV_JSON_COMMIT}.tar.gz"
        "ZLIB_URL=https://github.com/madler/zlib/archive/${OV_ZLIB_COMMIT}.tar.gz"
    )

    # ── Build cmake args: C++ inference runtime ONLY ──
    set(OPENVINO_CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OPENVINO_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        # Static libs for embedding into libnd4j
        -DBUILD_SHARED_LIBS=OFF
        # ── Disable Python entirely ──
        -DENABLE_PYTHON=OFF
        -DENABLE_WHEEL=OFF
        -DENABLE_GIL_PYTHON_API=OFF
        # ── Disable tests, samples, docs ──
        -DENABLE_TESTS=OFF
        -DENABLE_SAMPLES=OFF
        -DENABLE_FUNCTIONAL_TESTS=OFF
        -DENABLE_DOCS=OFF
        # ── CPU plugin ONLY ──
        -DENABLE_INTEL_CPU=ON
        -DENABLE_INTEL_GPU=OFF
        -DENABLE_INTEL_NPU=OFF
        -DENABLE_INTEL_NPU_INTERNAL=OFF
        # ── Disable ALL model frontends (we build ov::Model programmatically) ──
        -DENABLE_OV_ONNX_FRONTEND=OFF
        -DENABLE_OV_TF_FRONTEND=OFF
        -DENABLE_OV_TF_LITE_FRONTEND=OFF
        -DENABLE_OV_PADDLE_FRONTEND=OFF
        -DENABLE_OV_PYTORCH_FRONTEND=OFF
        -DENABLE_OV_JAX_FRONTEND=OFF
        -DENABLE_OV_IR_FRONTEND=ON
        # ── Disable multi-device plugins ──
        -DENABLE_MULTI=OFF
        -DENABLE_AUTO=OFF
        -DENABLE_AUTO_BATCH=OFF
        -DENABLE_HETERO=OFF
        -DENABLE_TEMPLATE=OFF
        -DENABLE_PROXY=OFF
        # ── Disable JS bindings ──
        -DENABLE_JS=OFF
        # ── CPU performance: Snippets JIT (core reason for OpenVINO) ──
        -DENABLE_SNIPPETS_LIBXSMM_TPP=OFF
        -DENABLE_MLAS_FOR_CPU=OFF
        # ── Threading: TBB (bundled, not system) ──
        -DTHREADING=TBB
        -DENABLE_SYSTEM_TBB=OFF
        -DENABLE_TBBBIND_2_5=OFF
        # ── Use bundled deps (we fetched them) ──
        -DENABLE_SYSTEM_PUGIXML=OFF
        -DENABLE_SYSTEM_FLATBUFFERS=OFF
        -DENABLE_SYSTEM_PROTOBUF=OFF
        -DENABLE_SYSTEM_SNAPPY=OFF
        -DENABLE_SNAPPY_COMPRESSION=OFF
        # ── Binary size / debug ──
        -DENABLE_LTO=OFF
        -DENABLE_DEBUG_CAPS=OFF
        -DENABLE_CPU_DEBUG_CAPS=OFF
        -DENABLE_SNIPPETS_DEBUG_CAPS=OFF
        -DENABLE_PROFILING_FIRST_INFERENCE=OFF
        # ── Compilers ──
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )

    # Pass ccache
    if(CMAKE_C_COMPILER_LAUNCHER AND EXISTS "${CMAKE_C_COMPILER_LAUNCHER}")
        list(APPEND OPENVINO_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CMAKE_C_COMPILER_LAUNCHER}")
    elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
        list(APPEND OPENVINO_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
    endif()
    if(CMAKE_CXX_COMPILER_LAUNCHER AND EXISTS "${CMAKE_CXX_COMPILER_LAUNCHER}")
        list(APPEND OPENVINO_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CMAKE_CXX_COMPILER_LAUNCHER}")
    elseif(SD_PLAIN_CCACHE_PATH AND EXISTS "${SD_PLAIN_CCACHE_PATH}")
        list(APPEND OPENVINO_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${SD_PLAIN_CCACHE_PATH}")
    endif()

    # Smart ccache partition key
    set(_OV_SHAPE_KEY_RAW "${OPENVINO_VERSION}-cpu-static-Release")
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" OV_SHAPE_KEY "${_OV_SHAPE_KEY_RAW}")
    set(OPENVINO_BUILD_COMMAND
        ${CMAKE_COMMAND} -E env
        "SD_SMART_CCACHE_SEGMENT=openvino"
        "SD_SMART_CCACHE_SHAPE_KEY=${OV_SHAPE_KEY}"
        ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --parallel ${DEP_PARALLEL_JOBS}
    )
    message(STATUS "   Smart ccache segment=openvino shape=${OV_SHAPE_KEY}")

    # ── Main ExternalProject: download OpenVINO source + populate submodules ──
    # PATCH_COMMAND runs patch_openvino.cmake which downloads and extracts all
    # submodule dependencies into their expected paths in the source tree.
    ExternalProject_Add(openvino_external
        PREFIX            "${OPENVINO_PREFIX}"
        URL               "${OPENVINO_URL}"
        URL_HASH          "${OPENVINO_URL_HASH}"
        DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
        ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
        SOURCE_DIR        "${OPENVINO_SOURCE_DIR}"
        BINARY_DIR        "${OPENVINO_PREFIX}/build"
        STAMP_DIR         "${OPENVINO_STAMP_DIR}"
        PATCH_COMMAND     ${CMAKE_COMMAND}
                          -DSOURCE_DIR=${OPENVINO_SOURCE_DIR}
                          -DDOWNLOAD_DIR=${CMAKE_BINARY_DIR}/downloads
                          -DTBB_INSTALL_DIR=${OPENVINO_PREFIX}/tbb_install
                          -DTBB_VERSION=${OV_TBB_VERSION}
                          -DPARALLEL_JOBS=${DEP_PARALLEL_JOBS}
                          -DCCACHE_PATH=${SD_PLAIN_CCACHE_PATH}
                          -DONEDNN_URL=https://github.com/openvinotoolkit/oneDNN/archive/${OV_ONEDNN_COMMIT}.tar.gz
                          -DONEDNN_COMMIT=${OV_ONEDNN_COMMIT}
                          -DXBYAK_URL=https://github.com/herumi/xbyak/archive/${OV_XBYAK_COMMIT}.tar.gz
                          -DXBYAK_COMMIT=${OV_XBYAK_COMMIT}
                          -DPUGIXML_URL=https://github.com/zeux/pugixml/archive/${OV_PUGIXML_COMMIT}.tar.gz
                          -DPUGIXML_COMMIT=${OV_PUGIXML_COMMIT}
                          -DFLATBUF_URL=https://github.com/google/flatbuffers/archive/${OV_FLATBUF_COMMIT}.tar.gz
                          -DFLATBUF_COMMIT=${OV_FLATBUF_COMMIT}
                          -DITTAPI_URL=https://github.com/intel/ittapi/archive/${OV_ITTAPI_COMMIT}.tar.gz
                          -DITTAPI_COMMIT=${OV_ITTAPI_COMMIT}
                          -DJSON_URL=https://github.com/nlohmann/json/archive/${OV_JSON_COMMIT}.tar.gz
                          -DJSON_COMMIT=${OV_JSON_COMMIT}
                          -DZLIB_URL=https://github.com/madler/zlib/archive/${OV_ZLIB_COMMIT}.tar.gz
                          -DZLIB_COMMIT=${OV_ZLIB_COMMIT}
                          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_openvino.cmake"
        CMAKE_ARGS        ${OPENVINO_CMAKE_ARGS}
                          -DTBBROOT=${OPENVINO_PREFIX}/tbb_install
        BUILD_COMMAND     ${OPENVINO_BUILD_COMMAND}
        INSTALL_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config Release
        TIMEOUT           2700  # 45 minutes (includes submodule downloads)
        LOG_DOWNLOAD      TRUE
        LOG_CONFIGURE     TRUE
        LOG_BUILD         TRUE
        LOG_INSTALL       TRUE
    )

    # ── Copy TBB cmake config into OpenVINO install so find_package(OpenVINO) can find TBB ──
    # TBB installs cmake config under lib64/ on some distros and lib/ on others.
    # Use a script step that tries both paths, succeeding if either exists.
    set(_TBB_DST_CMAKE "${OPENVINO_INSTALL_DIR}/runtime/3rdparty/tbb/lib64/cmake/TBB")
    ExternalProject_Add_Step(openvino_external copy_tbb_cmake
        COMMENT "Finalizing the OpenVINO static link contract"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_TBB_DST_CMAKE}"
        COMMAND ${CMAKE_COMMAND}
            -DTBB_PREFIX=${OPENVINO_PREFIX}/tbb_install
            -DTBB_DST=${_TBB_DST_CMAKE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_tbb_cmake.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINSTALL_DIR=${OPENVINO_INSTALL_DIR}
            -DRESPONSE_FILE=${OPENVINO_INSTALL_DIR}/runtime/lib/intel64/openvino-static-link.rsp
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_openvino.cmake
        BYPRODUCTS
            "${OPENVINO_INSTALL_DIR}/runtime/lib/intel64/openvino-static-link.rsp"
        DEPENDEES install
    )

    # ── Create interface library for linking ──
    if(NOT TARGET openvino_interface)
        add_library(openvino_interface INTERFACE)
        add_dependencies(openvino_interface openvino_external)

        # Headers
        target_include_directories(openvino_interface SYSTEM INTERFACE
            "${OPENVINO_INSTALL_DIR}/runtime/include"
            "${OPENVINO_INSTALL_DIR}/include"
        )

        # Static OpenVINO installs place their archives below runtime/. The
        # generated response keeps all circular archive dependencies in one rescan
        # group and links the separately installed oneTBB shared libraries.
        set(_OV_LIB_DIR "${OPENVINO_INSTALL_DIR}/runtime/lib/intel64")

        if(WIN32)
            target_link_libraries(openvino_interface INTERFACE
                "${_OV_LIB_DIR}/openvino.lib"
                "${_OV_LIB_DIR}/openvino_intel_cpu_plugin.lib"
            )
        else()
            _openvino_link_static_response(
                openvino_interface "${OPENVINO_INSTALL_DIR}")
        endif()
    endif()

    set(OPENVINO_LIB openvino_interface PARENT_SCOPE)
    sd_register_helper("openvino")

    # ── Save to dependency cache ──
    if(SD_DEP_CACHE)
        sd_dep_cache_key("openvino" "${OPENVINO_VERSION}" "" _ov_cache_key)
        sd_dep_cache_store("openvino" "${_ov_cache_key}" "${OPENVINO_INSTALL_DIR}" "openvino_external")
    endif()

    message(STATUS "OpenVINO setup complete (HAVE_OPENVINO=ON, building from source)")
endfunction()

# Link every archive installed by a static OpenVINO build through a generated
# GNU ld response file. The archive set is not stable across OpenVINO releases,
# and the CPU plugin's PRIVATE dependencies are otherwise omitted when the
# ExternalProject is configured before its package export exists.
function(_openvino_link_static_response _target _install_dir)
    set(_ov_link_response
        "${_install_dir}/runtime/lib/intel64/openvino-static-link.rsp")
    set(_ov_core_archive
        "${_install_dir}/runtime/lib/intel64/libopenvino.a")

    # Cached/reused installs already exist at configure time. Fresh
    # ExternalProject builds generate the same response in copy_tbb_cmake after
    # installation, and declare it as a byproduct above.
    if(EXISTS "${_ov_core_archive}")
        execute_process(
            COMMAND ${CMAKE_COMMAND}
                -DINSTALL_DIR=${_install_dir}
                -DRESPONSE_FILE=${_ov_link_response}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_openvino.cmake"
            RESULT_VARIABLE _ov_link_response_result
        )
        if(NOT _ov_link_response_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to generate the OpenVINO static linker response")
        endif()
    endif()

    target_link_libraries(${_target} INTERFACE
        "-Wl,@${_ov_link_response}"
        -lpthread -ldl -lm -lrt)
    set_property(TARGET ${_target} APPEND PROPERTY
        INTERFACE_LINK_DEPENDS "${_ov_link_response}")
endfunction()

# Helper: create openvino_interface from an existing install directory.
# Used by both the "reuse existing" and "restored from cache" paths.
function(_openvino_create_interface_from_install _install_dir)
    add_library(openvino_interface INTERFACE)

    # Find cmake config dir
    set(_ov_cfg_found FALSE)
    foreach(_ov_cfg_dir
            "${_install_dir}/runtime/lib/cmake/OpenVINO"
            "${_install_dir}/runtime/cmake"
            "${_install_dir}/lib/cmake/OpenVINO")
        if(EXISTS "${_ov_cfg_dir}/OpenVINOConfig.cmake")
            set(OpenVINO_DIR "${_ov_cfg_dir}")
            set(_ov_cfg_found TRUE)
            break()
        endif()
    endforeach()

    if(_ov_cfg_found)
        # Load OpenVINO's package to validate the install and configure its
        # bundled TBB package. The static CPU extension itself is linked by the
        # complete response below because it is not part of openvino::runtime.
        find_package(OpenVINO REQUIRED CONFIG PATHS "${OpenVINO_DIR}" NO_DEFAULT_PATH)
    else()
        message(WARNING
            "OpenVINO cmake config not found; using the installed static archives")
    endif()

    _openvino_link_static_response(openvino_interface "${_install_dir}")

    # Add include dirs explicitly
    foreach(_ov_inc_dir
            "${_install_dir}/runtime/include"
            "${_install_dir}/include")
        if(EXISTS "${_ov_inc_dir}/openvino/openvino.hpp")
            target_include_directories(openvino_interface SYSTEM INTERFACE "${_ov_inc_dir}")
            break()
        endif()
    endforeach()
endfunction()

# =============================================================================
# VULKAN (standalone SD_VULKAN device backend)
# This is distinct from MLIR_ENABLE_VULKAN (MLIR Vulkan/SPIR-V dialect).
# Discovery and bootstrap are legal only for the Vulkan chip build; CPU and
# CUDA builds never probe, download, or link Vulkan through this function.
# =============================================================================
function(setup_vulkan)
    if(NOT SD_VULKAN)
        message(FATAL_ERROR
            "setup_vulkan() is valid only for an SD_VULKAN chip build")
    endif()

    if(NOT LIBND4J_ENABLE_VULKAN)
        message(STATUS "Vulkan compute backend disabled (LIBND4J_ENABLE_VULKAN=OFF)")
        set(HAVE_VULKAN FALSE PARENT_SCOPE)
        return()
    endif()

    # Search for vulkan/vulkan.h in standard locations and common SDK paths.
    # The bootstrap install dir is first so a previously downloaded copy is
    # found even if the cache variable was cleared.
    find_path(VULKAN_INCLUDE_DIR
        NAMES vulkan/vulkan.h
        HINTS
            "${CMAKE_BINARY_DIR}/vulkan_headers_install/include"
            "$ENV{VULKAN_SDK}/include"
            "$ENV{VULKAN_SDK}/../include"
            "/home/linuxbrew/.linuxbrew/include"
            "/usr/local/include"
            "/opt/homebrew/include"
        PATHS
            "/usr/include"
    )

    if(NOT VULKAN_INCLUDE_DIR)
        message(STATUS "Vulkan headers not found locally — bootstrapping from Khronos repository")

        # Download Vulkan headers from Khronos if not found
        set(VULKAN_HEADERS_VERSION "1.3.268")
        set(VULKAN_INSTALL_DIR "${CMAKE_BINARY_DIR}/vulkan_headers_install")

        # --- Dependency cache check ---
        if(SD_DEP_CACHE)
            sd_dep_cache_key("vulkan-headers" "${VULKAN_HEADERS_VERSION}" "" _vulkan_cache_key)
            sd_dep_cache_check("vulkan-headers" "${_vulkan_cache_key}" _vulkan_hit _vulkan_cache_path)
            if(_vulkan_hit)
                sd_dep_cache_restore("vulkan-headers" "${_vulkan_cache_path}" "${VULKAN_INSTALL_DIR}")
                message(STATUS "✅ Vulkan headers restored from cache")
            endif()
        endif()

        if(NOT EXISTS "${VULKAN_INSTALL_DIR}/include/vulkan/vulkan.h")
            # The headers are needed at configure time (find_path + the
            # include_directories call below), so download and extract them
            # here rather than via ExternalProject_Add, whose steps only run
            # at build time — too late for this function's own existence check.
            set(VULKAN_URL "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v${VULKAN_HEADERS_VERSION}.tar.gz")
            set(_vulkan_tarball "${CMAKE_BINARY_DIR}/downloads/vulkan-headers-v${VULKAN_HEADERS_VERSION}.tar.gz")
            message(STATUS "Downloading Vulkan ${VULKAN_HEADERS_VERSION} headers from ${VULKAN_URL}")
            file(DOWNLOAD "${VULKAN_URL}" "${_vulkan_tarball}"
                 STATUS _vulkan_dl_status
                 TIMEOUT 300)
            list(GET _vulkan_dl_status 0 _vulkan_dl_code)
            if(NOT _vulkan_dl_code EQUAL 0)
                list(GET _vulkan_dl_status 1 _vulkan_dl_msg)
                message(FATAL_ERROR "Failed to download Vulkan headers from ${VULKAN_URL}: ${_vulkan_dl_msg}")
            endif()

            set(_vulkan_extract_dir "${CMAKE_BINARY_DIR}/vulkan_headers_src")
            file(REMOVE_RECURSE "${_vulkan_extract_dir}")
            file(ARCHIVE_EXTRACT INPUT "${_vulkan_tarball}" DESTINATION "${_vulkan_extract_dir}")

            # The tarball extracts to Vulkan-Headers-<version>/
            file(GLOB _vulkan_src_root "${_vulkan_extract_dir}/Vulkan-Headers-*")
            list(LENGTH _vulkan_src_root _vulkan_src_root_count)
            if(NOT _vulkan_src_root_count EQUAL 1)
                message(FATAL_ERROR "Unexpected Vulkan headers archive layout under ${_vulkan_extract_dir}")
            endif()
            list(GET _vulkan_src_root 0 _vulkan_src_root)

            file(MAKE_DIRECTORY "${VULKAN_INSTALL_DIR}/include")
            file(COPY "${_vulkan_src_root}/include/vulkan" DESTINATION "${VULKAN_INSTALL_DIR}/include")
            # vulkan_core.h includes vk_video headers since 1.2.x
            if(EXISTS "${_vulkan_src_root}/include/vk_video")
                file(COPY "${_vulkan_src_root}/include/vk_video" DESTINATION "${VULKAN_INSTALL_DIR}/include")
            endif()
        endif()

        if(NOT EXISTS "${VULKAN_INSTALL_DIR}/include/vulkan/vulkan.h")
            message(FATAL_ERROR "Failed to bootstrap Vulkan headers into ${VULKAN_INSTALL_DIR}")
        endif()

        set(VULKAN_INCLUDE_DIR "${VULKAN_INSTALL_DIR}/include")

        # --- Cache store (configure-time; mirrors sd_dep_cache_store's
        #     install/ + .cache_complete marker layout) ---
        if(SD_DEP_CACHE AND DEFINED _vulkan_cache_key AND NOT _vulkan_hit)
            set(_vulkan_cache_dir "${SD_DEP_CACHE_DIR}/vulkan-headers/${_vulkan_cache_key}")
            set(_vulkan_cache_marker "${_vulkan_cache_dir}/.cache_complete")
            if(NOT EXISTS "${_vulkan_cache_marker}")
                file(MAKE_DIRECTORY "${_vulkan_cache_dir}/install")
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E copy_directory "${VULKAN_INSTALL_DIR}" "${_vulkan_cache_dir}/install"
                    RESULT_VARIABLE _vulkan_store_result
                )
                if(_vulkan_store_result EQUAL 0)
                    file(WRITE "${_vulkan_cache_marker}" "cached by libnd4j at configure time")
                    message(STATUS "DEP-CACHE [vulkan-headers] Cache stored successfully")
                else()
                    message(WARNING "DEP-CACHE [vulkan-headers] Failed to store cache (exit code ${_vulkan_store_result})")
                    file(REMOVE_RECURSE "${_vulkan_cache_dir}")
                endif()
            endif()
        endif()
    endif()

    # Link the platform Vulkan loader; the loader's normal ICD discovery chooses
    # AMD, NVIDIA, Intel, or another conforming implementation at runtime.
    # Cross-target searches must never admit a host loader.
    if(ANDROID)
        # The NDK toolchain resolves this in the target sysroot. Android Vulkan
        # starts at API 24, which verify_vulkan_chip_requirements() enforces.
        find_library(VULKAN_LIBRARY NAMES vulkan)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _vulkan_target_processor)
        set(_vulkan_linux_library_paths "")

        # Prefer the toolchain's canonical target multiarch tuple when available.
        if(CMAKE_LIBRARY_ARCHITECTURE)
            list(APPEND _vulkan_linux_library_paths
                "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
                "/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
        endif()

        # Keep architecture-specific paths disjoint: an ARM64 target must not
        # probe x86-64 loader directories, and vice versa.
        if(_vulkan_target_processor MATCHES "^(x86_64|amd64)$")
            list(APPEND _vulkan_linux_library_paths
                "/usr/lib/x86_64-linux-gnu"
                "/lib/x86_64-linux-gnu"
                "/usr/lib64"
                "/lib64"
                "/usr/lib"
                "/lib")
            set(_vulkan_expected_machine_regex "x86-64")
        elseif(_vulkan_target_processor MATCHES "^(aarch64|arm64)$")
            list(APPEND _vulkan_linux_library_paths
                "/usr/lib/aarch64-linux-gnu"
                "/lib/aarch64-linux-gnu"
                "/usr/lib64"
                "/lib64"
                "/usr/lib"
                "/lib")
            set(_vulkan_expected_machine_regex "(ARM aarch64|aarch64)")
        else()
            list(APPEND _vulkan_linux_library_paths
                "/usr/lib64"
                "/lib64"
                "/usr/lib"
                "/lib")
            set(_vulkan_expected_machine_regex "")
        endif()

        # A host Vulkan SDK is valid only for a native build. Cross builds must
        # resolve exclusively through the target sysroot/toolchain.
        if(NOT CMAKE_CROSSCOMPILING AND DEFINED ENV{VULKAN_SDK})
            list(PREPEND _vulkan_linux_library_paths "$ENV{VULKAN_SDK}/lib")
        endif()
        list(REMOVE_DUPLICATES _vulkan_linux_library_paths)

        if(CMAKE_CROSSCOMPILING)
            find_library(VULKAN_LIBRARY
                NAMES vulkan vulkan-1 vulkan.so.1 libvulkan.so.1
                PATHS ${_vulkan_linux_library_paths}
                NO_DEFAULT_PATH
                ONLY_CMAKE_FIND_ROOT_PATH)
        else()
            find_library(VULKAN_LIBRARY
                NAMES vulkan vulkan-1 vulkan.so.1 libvulkan.so.1
                PATHS ${_vulkan_linux_library_paths}
                NO_DEFAULT_PATH)
        endif()
    else()
        # Windows (MSYS2/mingw: vulkan-loader provides libvulkan-1.dll.a) and
        # macOS (LunarG SDK or MoltenVK) still link the vendor-neutral loader.
        # MSYS2's shell does not always export MSYSTEM_PREFIX to Maven. Derive
        # the prefix from the compiler CMake selected so the loader search stays
        # toolchain-relative instead of depending on a fixed host path.
        set(_vulkan_windows_library_paths "")
        if(DEFINED ENV{VULKAN_SDK})
            list(APPEND _vulkan_windows_library_paths
                "$ENV{VULKAN_SDK}/lib"
                "$ENV{VULKAN_SDK}/Lib")
        endif()
        if(DEFINED ENV{MSYSTEM_PREFIX})
            list(APPEND _vulkan_windows_library_paths "$ENV{MSYSTEM_PREFIX}/lib")
        endif()
        if(DEFINED ENV{MINGW_PREFIX})
            list(APPEND _vulkan_windows_library_paths "$ENV{MINGW_PREFIX}/lib")
        endif()
        if(DEFINED ENV{MSYS2_PREFIX})
            list(APPEND _vulkan_windows_library_paths "$ENV{MSYS2_PREFIX}/mingw64/lib")
        endif()
        set(_vulkan_compiler_path "")
        if(CMAKE_C_COMPILER)
            if(IS_ABSOLUTE "${CMAKE_C_COMPILER}")
                set(_vulkan_compiler_path "${CMAKE_C_COMPILER}")
            else()
                find_program(_vulkan_compiler_path NAMES "${CMAKE_C_COMPILER}")
            endif()
        endif()
        if(NOT _vulkan_compiler_path AND CMAKE_CXX_COMPILER)
            if(IS_ABSOLUTE "${CMAKE_CXX_COMPILER}")
                set(_vulkan_compiler_path "${CMAKE_CXX_COMPILER}")
            else()
                find_program(_vulkan_compiler_path NAMES "${CMAKE_CXX_COMPILER}")
            endif()
        endif()
        set(_vulkan_mingw_prefix "")
        if(_vulkan_compiler_path)
            get_filename_component(_vulkan_compiler_bin "${_vulkan_compiler_path}" DIRECTORY)
            get_filename_component(_vulkan_mingw_prefix "${_vulkan_compiler_bin}" DIRECTORY)
            list(APPEND _vulkan_windows_library_paths "${_vulkan_mingw_prefix}/lib")
        endif()
        list(REMOVE_DUPLICATES _vulkan_windows_library_paths)
        set(_vulkan_mingw_loader "")
        if(_vulkan_mingw_prefix)
            foreach(_vulkan_loader_name IN ITEMS libvulkan-1.dll.a vulkan-1.dll.a)
                if(EXISTS "${_vulkan_mingw_prefix}/lib/${_vulkan_loader_name}")
                    set(_vulkan_mingw_loader "${_vulkan_mingw_prefix}/lib/${_vulkan_loader_name}")
                    break()
                endif()
            endforeach()
        endif()
        find_library(VULKAN_LIBRARY
            NAMES vulkan vulkan-1 libvulkan-1
            HINTS
                ${_vulkan_windows_library_paths}
                "/home/linuxbrew/.linuxbrew/lib"
                "/usr/local/lib"
                "/opt/homebrew/lib"
            PATHS
                "/usr/lib64"
                "/usr/lib/x86_64-linux-gnu"
                "/usr/lib")
        if(NOT VULKAN_LIBRARY AND _vulkan_mingw_loader)
            set(VULKAN_LIBRARY "${_vulkan_mingw_loader}")
        endif()
    endif()

    if(NOT VULKAN_LIBRARY)
        message(STATUS "Vulkan loader library not found — Vulkan compute backend disabled")
        message(STATUS "  Install vulkan-loader (Fedora: dnf install vulkan-loader)")
        set(HAVE_VULKAN FALSE PARENT_SCOPE)
        return()
    endif()

    # Verify both ELF class and target machine. Bitness alone cannot distinguish
    # an x86-64 host loader from an AArch64 target loader.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        execute_process(
            COMMAND file -L "${VULKAN_LIBRARY}"
            OUTPUT_VARIABLE _vulkan_file_output
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(_vulkan_library_architecture_valid TRUE)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT _vulkan_file_output MATCHES "ELF 64-bit")
            set(_vulkan_library_architecture_valid FALSE)
        elseif(CMAKE_SIZEOF_VOID_P EQUAL 4 AND NOT _vulkan_file_output MATCHES "ELF 32-bit")
            set(_vulkan_library_architecture_valid FALSE)
        endif()
        if(_vulkan_expected_machine_regex
                AND NOT _vulkan_file_output MATCHES "${_vulkan_expected_machine_regex}")
            set(_vulkan_library_architecture_valid FALSE)
        endif()

        if(NOT _vulkan_library_architecture_valid)
            message(STATUS
                "Vulkan loader target mismatch for ${CMAKE_SYSTEM_PROCESSOR}: "
                "${VULKAN_LIBRARY} resolves as '${_vulkan_file_output}'")
            unset(VULKAN_LIBRARY CACHE)
            set(HAVE_VULKAN FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    message(STATUS "✅ Vulkan headers found: ${VULKAN_INCLUDE_DIR}")
    message(STATUS "✅ Vulkan loader found:  ${VULKAN_LIBRARY}")
    set(HAVE_VULKAN TRUE CACHE BOOL "Vulkan availability" FORCE)
    set(HAVE_VULKAN TRUE PARENT_SCOPE)
    set(VULKAN_INCLUDE_DIR "${VULKAN_INCLUDE_DIR}" CACHE PATH "Vulkan include directory" FORCE)
    set(VULKAN_LIBRARY "${VULKAN_LIBRARY}" CACHE FILEPATH "Vulkan loader library" FORCE)
    set(VULKAN_LIBRARY "${VULKAN_LIBRARY}" PARENT_SCOPE)
    add_compile_definitions(HAVE_VULKAN=1)
    include_directories(SYSTEM "${VULKAN_INCLUDE_DIR}")
    message(STATUS "Vulkan compute backend ENABLED (HAVE_VULKAN=1)")
endfunction()
