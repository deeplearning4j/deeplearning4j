# cmake/Dependencies.cmake
# Manages all third-party dependencies. Logic is encapsulated in functions.

include(ExternalProject)
include(ProcessorCount)

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

# Dependencies (LLVM, flatbuffers, etc.) use less memory per compile (~1GB vs 2-4GB for CUDA)
# Always auto-compute from available memory and cores — don't use SD_PARALLEL_COMPILE_JOBS
# which is tuned for CUDA compilation and would under-utilize the machine.
cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
math(EXPR MEM_BASED_JOBS "${AVAILABLE_MEMORY} / 1000")  # 1GB per job for deps

# Cap at processor count, minimum 1
if(MEM_BASED_JOBS GREATER NPROC)
    set(DEP_PARALLEL_JOBS ${NPROC})
elseif(MEM_BASED_JOBS LESS 1)
    set(DEP_PARALLEL_JOBS 1)
else()
    set(DEP_PARALLEL_JOBS ${MEM_BASED_JOBS})
endif()

message(STATUS "🔧 Dependency builds will use ${DEP_PARALLEL_JOBS} parallel jobs (${NPROC} cores, ${AVAILABLE_MEMORY}MB available)")

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
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${cache_path}" "${install_dir}"
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
    set(_store_script "${CMAKE_BINARY_DIR}/dep_cache_store_${dep_name}.cmake")
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
                "${OPENBLAS_PATH}/lib/libopenblas.a"
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
            message(STATUS "❌ OPENBLAS_PATH not set")
            return()
        endif()
    endif()

    # Handle Android path normalization at CMake level (in case shell script normalization didn't work)
    setup_android_arm_openblas()


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
            "${OPENBLAS_PATH}/libopenblas.so"
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
    set(FLATBUFFERS_URL "https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VERSION}.tar.gz")
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
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
                target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib")
            elseif(WIN32)
                target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/lib64/libdnnl.a")
            else()
                target_link_libraries(onednn_interface INTERFACE "${ONEDNN_INSTALL_DIR}/lib64/libdnnl.a")
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
    if(WIN32)
        set(ONEDNN_LIB_DIR "lib")
    else()
        set(ONEDNN_LIB_DIR "lib64")
    endif()
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

    # Pass Android cross-compilation args to OneDNN (same pattern as FlatBuffers)
    if(CMAKE_TOOLCHAIN_FILE)
        list(APPEND ONEDNN_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    endif()
    if(CMAKE_SYSTEM_NAME)
        list(APPEND ONEDNN_CMAKE_ARGS -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})
    endif()
    if(CMAKE_SYSTEM_VERSION)
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

    ExternalProject_Add(onednn_external
            PREFIX            "${ONEDNN_PREFIX}"
            URL               "${ONEDNN_URL}"
            URL_HASH          "${ONEDNN_URL_HASH}"
            DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
            SOURCE_DIR        "${ONEDNN_PREFIX}/src/oneDNN-${ONEDNN_VERSION}"
            BINARY_DIR        "${ONEDNN_PREFIX}/build"
            STAMP_DIR         "${ONEDNN_STAMP_DIR}"
            DOWNLOAD_NO_PROGRESS FALSE
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
                target_include_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/include")
                target_link_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/lib")
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
                INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/${ARMCOMPUTE_PKG_NAME} ${ARMCOMPUTE_INSTALL_DIR}
                BUILD_BYPRODUCTS "${ARMCOMPUTE_INSTALL_DIR}/include/arm_compute/core/CL/CLKernelLibrary.h"
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        add_library(armcompute_interface INTERFACE)
        target_include_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/include")
        target_link_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/lib")
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
        message(WARNING "MLIR/LLVM ${MLIR_VERSION}+ not found. MLIR support will be disabled.")
        message(WARNING "To enable MLIR support:")
        message(WARNING "  1. Install LLVM ${MLIR_VERSION}+ with MLIR enabled")
        message(WARNING "  2. Set LLVM_DIR or LLVM_ROOT to your LLVM installation")
        set(HAVE_MLIR FALSE PARENT_SCOPE)
        set(MLIR "" PARENT_SCOPE)
        return()
    endif()

    set(HAVE_MLIR TRUE PARENT_SCOPE)
    set(MLIR MLIR::MLIR PARENT_SCOPE)

    # Configure GPU support if requested and CUDA is enabled
    if(MLIR_ENABLE_GPU AND SD_CUDA)
        if(TARGET MLIR::GPU)
            message(STATUS "   MLIR GPU dialect: enabled")
        else()
            message(WARNING "   MLIR GPU libraries not found. GPU dialect will not be available.")
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
# METAL PERFORMANCE SHADERS (Optional, for macOS/iOS builds)
# =============================================================================
# =============================================================================
# ZLUDA Transpiler (Optional, for AMD/Intel GPU support via CUDA translation)
# Downloads ZLUDA and optionally MIOpen for AMD targets
# =============================================================================
function(setup_zluda_download)
    if(NOT SD_ZLUDA)
        message(STATUS "ZLUDA is disabled (SD_ZLUDA=${SD_ZLUDA})")
        set(HAVE_ZLUDA FALSE PARENT_SCOPE)
        return()
    endif()

    # Check if ZLUDA is already available via environment
    if(DEFINED ENV{ZLUDA_PATH} AND EXISTS "$ENV{ZLUDA_PATH}")
        message(STATUS "Using existing ZLUDA installation: $ENV{ZLUDA_PATH}")
        set(ZLUDA_ROOT "$ENV{ZLUDA_PATH}")
        set(HAVE_ZLUDA TRUE PARENT_SCOPE)
        set(ZLUDA_PATH "${ZLUDA_ROOT}" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "ZLUDA automatic download enabled")

    # ZLUDA release configuration
    # ZLUDA v3 supports both AMD (ROCm/HIP) and Intel (Level Zero) GPUs
    set(ZLUDA_VERSION "3")
    set(ZLUDA_INSTALL_DIR "${CMAKE_BINARY_DIR}/zluda_install")

    # Determine platform-specific download
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            set(ZLUDA_PLATFORM "linux-x86_64")
            set(ZLUDA_ARCHIVE_EXT "tar.gz")
        else()
            message(WARNING "ZLUDA: Unsupported processor ${CMAKE_SYSTEM_PROCESSOR} on Linux")
            set(HAVE_ZLUDA FALSE PARENT_SCOPE)
            return()
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
            set(ZLUDA_PLATFORM "windows-x86_64")
            set(ZLUDA_ARCHIVE_EXT "zip")
        else()
            message(WARNING "ZLUDA: Unsupported processor ${CMAKE_SYSTEM_PROCESSOR} on Windows")
            set(HAVE_ZLUDA FALSE PARENT_SCOPE)
            return()
        endif()
    else()
        message(WARNING "ZLUDA: Unsupported platform ${CMAKE_SYSTEM_NAME}")
        set(HAVE_ZLUDA FALSE PARENT_SCOPE)
        return()
    endif()

    # ZLUDA GitHub releases URL
    # Note: ZLUDA releases may vary in naming convention - adjust as needed
    set(ZLUDA_URL "https://github.com/vosen/ZLUDA/releases/download/v${ZLUDA_VERSION}/zluda-${ZLUDA_PLATFORM}.${ZLUDA_ARCHIVE_EXT}")

    message(STATUS "ZLUDA download URL: ${ZLUDA_URL}")

    # --- Dependency cache check ---
    if(SD_DEP_CACHE)
        sd_dep_cache_key("zluda" "${ZLUDA_VERSION}" "${ZLUDA_PLATFORM}" _zluda_cache_key)
        sd_dep_cache_check("zluda" "${_zluda_cache_key}" _zluda_hit _zluda_cache_path)
        if(_zluda_hit)
            sd_dep_cache_restore("zluda" "${_zluda_cache_path}" "${ZLUDA_INSTALL_DIR}")
            if(NOT TARGET zluda_external)
                add_custom_target(zluda_external)
            endif()
            add_library(zluda_interface INTERFACE)
            target_include_directories(zluda_interface INTERFACE "${ZLUDA_INSTALL_DIR}/include")
            target_link_directories(zluda_interface INTERFACE "${ZLUDA_INSTALL_DIR}/lib")
            add_dependencies(zluda_interface zluda_external)
            set(HAVE_ZLUDA TRUE PARENT_SCOPE)
            set(ZLUDA_PATH "${ZLUDA_INSTALL_DIR}" PARENT_SCOPE)
            set(ZLUDA zluda_interface PARENT_SCOPE)
            set(ENV{ZLUDA_PATH} "${ZLUDA_INSTALL_DIR}")
            message(STATUS "✅ ZLUDA setup complete (from cache)")
            return()
        endif()
    endif()

    # Download and extract ZLUDA
    ExternalProject_Add(zluda_external
            PREFIX            "${CMAKE_BINARY_DIR}/zluda_external"
            URL               "${ZLUDA_URL}"
            DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
            SOURCE_DIR        "${ZLUDA_INSTALL_DIR}"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND     ""
            INSTALL_COMMAND   ""
            BUILD_BYPRODUCTS  "${ZLUDA_INSTALL_DIR}/lib/libcuda.so"
            TIMEOUT           300
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            LOG_DOWNLOAD      OFF
            LOG_CONFIGURE     OFF
            LOG_BUILD         OFF
            LOG_INSTALL       OFF
    )

    # Create interface library for ZLUDA
    add_library(zluda_interface INTERFACE)
    target_include_directories(zluda_interface INTERFACE "${ZLUDA_INSTALL_DIR}/include")
    if(WIN32)
        target_link_directories(zluda_interface INTERFACE "${ZLUDA_INSTALL_DIR}/lib")
    else()
        target_link_directories(zluda_interface INTERFACE "${ZLUDA_INSTALL_DIR}/lib")
    endif()
    add_dependencies(zluda_interface zluda_external)

    set(HAVE_ZLUDA TRUE PARENT_SCOPE)
    set(ZLUDA_PATH "${ZLUDA_INSTALL_DIR}" PARENT_SCOPE)
    set(ZLUDA zluda_interface PARENT_SCOPE)

    # Set environment variable for runtime
    set(ENV{ZLUDA_PATH} "${ZLUDA_INSTALL_DIR}")

    # --- Cache store ---
    if(SD_DEP_CACHE AND DEFINED _zluda_cache_key)
        sd_dep_cache_store("zluda" "${_zluda_cache_key}" "${ZLUDA_INSTALL_DIR}" "zluda_external")
    endif()

    message(STATUS "ZLUDA setup complete")
    message(STATUS "   Install directory: ${ZLUDA_INSTALL_DIR}")
    message(STATUS "   Platform: ${ZLUDA_PLATFORM}")

    # Setup MIOpen for AMD targets
    if(SD_ZLUDA_TARGET STREQUAL "AMD" OR SD_ZLUDA_TARGET STREQUAL "amd")
        setup_miopen_download()
    endif()
endfunction()

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
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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

    message(STATUS "✅ MPS setup complete")
endfunction()

# =============================================================================
# TRITON GPU COMPILER (Optional)
# =============================================================================
function(setup_triton)
    # Triton is OFF by default.
    # Enable explicitly with -DSD_TRITON=ON (or SD_TRITON=ON env var).
    # NVRTC and PTX backends are also available on CUDA builds.
    set(_triton_requested OFF)
    # Legacy opt-in path via helper flag (kept for backward compatibility).
    if(HELPERS_triton)
        set(_triton_requested ON)
    endif()
    # Preferred explicit opt-in flags.
    if(DEFINED SD_TRITON AND SD_TRITON)
        set(_triton_requested ON)
    endif()
    if(DEFINED ENV{SD_TRITON} AND "$ENV{SD_TRITON}" STREQUAL "ON")
        set(_triton_requested ON)
    endif()
    # Explicit opt-out overrides
    if(DEFINED SD_TRITON AND NOT SD_TRITON)
        set(_triton_requested OFF)
    endif()
    if(DEFINED ENV{SD_TRITON} AND "$ENV{SD_TRITON}" STREQUAL "OFF")
        set(_triton_requested OFF)
    endif()
    if(NOT _triton_requested)
        message(STATUS "Triton disabled (use -DSD_TRITON=ON to enable)")
        set(HAVE_TRITON OFF CACHE BOOL "Triton availability" FORCE)
        set(TRITON "" PARENT_SCOPE)
        return()
    endif()
    # Triton JIT compilation requires LLVM headers that are incompatible with
    # MSVC's STL (constexpr string_view::substr in TypeName.h).  The Triton
    # runtime also needs a Linux-only PTX/NVRTC toolchain, so skip on Windows.
    if(WIN32)
        message(STATUS "Triton disabled on Windows (LLVM headers incompatible with MSVC)")
        set(HAVE_TRITON OFF CACHE BOOL "Triton availability" FORCE)
        set(TRITON "" PARENT_SCOPE)
        return()
    endif()
    # Triton supports both GPU (CUDA/HIP/Level-Zero) and CPU backends.
    # On CPU builds, triton-cpu provides LLVM-based CPU codegen.
    # SD_TRITON=ON is valid for all build types.
    set(_TRITON_CPU_ONLY FALSE)
    if(NOT SD_CUDA AND NOT SD_HIP AND NOT SD_LEVEL_ZERO)
        set(_TRITON_CPU_ONLY TRUE)
    endif()

    # Check if Triton and LLVM are already built from a previous run.
    # After CMake cache clean, targets don't persist but install dirs do.
    # Use different install dirs for CPU vs GPU triton to avoid conflicts.
    if(_TRITON_CPU_ONLY)
        set(TRITON_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_cpu_install")
        set(TRITON_LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_cpu_llvm_install")
    else()
        set(TRITON_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_install")
        set(TRITON_LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/triton_llvm_install")
    endif()

    set(_TRITON_LIB_EXISTS FALSE)
    if(EXISTS "${TRITON_INSTALL_DIR}/lib/libtriton.a" OR
       EXISTS "${TRITON_INSTALL_DIR}/lib/triton.lib")
        set(_TRITON_LIB_EXISTS TRUE)
    endif()
    if(_TRITON_LIB_EXISTS AND
       EXISTS "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
        message(STATUS "Triton helper: reusing existing install at ${TRITON_INSTALL_DIR}")
        set(HAVE_TRITON ON CACHE BOOL "Triton availability" FORCE)
        set(HAVE_TRITON ON PARENT_SCOPE)
        if(_TRITON_CPU_ONLY)
            set(HAVE_TRITON_CPU ON CACHE BOOL "Triton CPU backend" FORCE)
            set(HAVE_TRITON_CPU ON PARENT_SCOPE)
        endif()
        # Create a dummy triton_external target so MainBuildFlow's dependency block picks us up
        if(NOT TARGET triton_external)
            add_custom_target(triton_external)
        endif()
        if(NOT TARGET triton_interface)
            add_library(triton_interface INTERFACE)
            # Use SYSTEM to prevent Triton/LLVM headers from conflicting
            # with libnd4j's DataType enum names (BOOL, INT8, FLOAT32 etc.)
            target_include_directories(triton_interface SYSTEM INTERFACE
                "${TRITON_INSTALL_DIR}/include"
                "${TRITON_LLVM_INSTALL_DIR}/include"
            )
            if(WIN32)
                target_link_libraries(triton_interface INTERFACE "${TRITON_INSTALL_DIR}/lib/triton.lib")
            else()
                target_link_libraries(triton_interface INTERFACE "${TRITON_INSTALL_DIR}/lib/libtriton.a")
            endif()
            # Glob ALL MLIR and LLVM static libraries — avoids missing transitive deps
            if(WIN32)
                file(GLOB _TRITON_MLIR_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/MLIR*.lib")
                file(GLOB _TRITON_LLVM_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/LLVM*.lib")
                file(GLOB _TRITON_MLIR_CAPI_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/MLIRCAPI*.lib")
            else()
                file(GLOB _TRITON_MLIR_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/libMLIR*.a")
                file(GLOB _TRITON_LLVM_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/libLLVM*.a")
                file(GLOB _TRITON_MLIR_CAPI_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/libMLIRCAPI*.a")
            endif()
            list(SORT _TRITON_MLIR_LIBS)
            list(SORT _TRITON_LLVM_LIBS)
            if(WIN32)
                # MSVC linker handles circular deps automatically (no --start-group needed)
                target_link_libraries(triton_interface INTERFACE
                    ${_TRITON_MLIR_LIBS}
                    ${_TRITON_LLVM_LIBS}
                    ${_TRITON_MLIR_CAPI_LIBS}
                    zlib.lib zstd.lib
                )
                if(NOT _TRITON_CPU_ONLY)
                    # Link nvrtc and cuda driver for NVRTC JIT and PTX backends
                    target_link_libraries(triton_interface INTERFACE nvrtc.lib cuda.lib)
                endif()
            else()
                # Apple ld and Android lld resolve circular deps automatically (like MSVC).
                # --start-group/--end-group is GNU ld only.
                if(APPLE OR ANDROID)
                    target_link_libraries(triton_interface INTERFACE
                        ${_TRITON_MLIR_LIBS}
                        ${_TRITON_LLVM_LIBS}
                        ${_TRITON_MLIR_CAPI_LIBS}
                    )
                else()
                    target_link_libraries(triton_interface INTERFACE
                        -Wl,--start-group
                        ${_TRITON_MLIR_LIBS}
                        ${_TRITON_LLVM_LIBS}
                        ${_TRITON_MLIR_CAPI_LIBS}
                        -Wl,--end-group
                    )
                endif()
                # Platform-appropriate system libs
                target_link_libraries(triton_interface INTERFACE -lz -lm)
                if(NOT APPLE AND NOT ANDROID)
                    target_link_libraries(triton_interface INTERFACE -lzstd -lrt -ldl -lpthread)
                elseif(APPLE)
                    target_link_libraries(triton_interface INTERFACE -lzstd -ldl -lpthread)
                else()
                    # Android: no -lrt, -lpthread, or -lzstd
                    target_link_libraries(triton_interface INTERFACE -ldl)
                endif()
                if(NOT _TRITON_CPU_ONLY)
                    # Link nvrtc and cuda driver for NVRTC JIT and PTX backends
                    target_link_libraries(triton_interface INTERFACE -lnvrtc -lcuda)
                endif()
            endif()
            list(LENGTH _TRITON_MLIR_LIBS _mlir_count)
            list(LENGTH _TRITON_LLVM_LIBS _llvm_count)
            message(STATUS "Triton interface: ${_mlir_count} MLIR + ${_llvm_count} LLVM static libs (from existing install)")
        endif()
        set(TRITON triton_interface PARENT_SCOPE)
        return()
    endif()
    if(TARGET triton_external)
        message(STATUS "Triton helper is enabled (target already exists)")
        set(HAVE_TRITON ON CACHE BOOL "Triton availability" FORCE)
        set(TRITON triton_interface PARENT_SCOPE)
        return()
    endif()

    message(STATUS "Triton helper is enabled (CPU_ONLY=${_TRITON_CPU_ONLY})")
    set(HAVE_TRITON ON CACHE BOOL "Triton availability" FORCE)
    set(HAVE_TRITON ON PARENT_SCOPE)
    if(_TRITON_CPU_ONLY)
        set(HAVE_TRITON_CPU ON CACHE BOOL "Triton CPU backend" FORCE)
        set(HAVE_TRITON_CPU ON PARENT_SCOPE)
    endif()
    set(HELPERS_triton ON PARENT_SCOPE)

    # triton-cpu pinned commit (declared early so TRITON_VERSION can reference it)
    set(TRITON_CPU_COMMIT "c4ccb98970bfe0fa17548b5b32def8d0de2bdc53")

    if(_TRITON_CPU_ONLY)
        # triton-cpu: experimental CPU backend from triton-lang/triton-cpu
        # Uses a different LLVM pin and adds third_party/cpu backend
        set(TRITON_PREFIX "${CMAKE_BINARY_DIR}/triton_cpu_external")
        set(TRITON_VERSION "cpu-${TRITON_CPU_COMMIT}")  # Pinned commit, no tagged releases
    else()
        set(TRITON_PREFIX "${CMAKE_BINARY_DIR}/triton_external")
        set(TRITON_VERSION "3.6.0")
    endif()
    set(TRITON_STAMP_DIR "${TRITON_PREFIX}/stamp")

    # Ensure directories exist
    file(MAKE_DIRECTORY "${TRITON_STAMP_DIR}")
    file(MAKE_DIRECTORY "${TRITON_PREFIX}/src")
    file(MAKE_DIRECTORY "${TRITON_PREFIX}/build")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")

    # Clean stale stamp files
    if(EXISTS "${TRITON_STAMP_DIR}")
        file(GLOB STALE_STAMPS "${TRITON_STAMP_DIR}/*-lastrun.txt" "${TRITON_STAMP_DIR}/*.txt")
        foreach(stamp ${STALE_STAMPS})
            message(STATUS "Cleaning stale Triton stamp file: ${stamp}")
            file(REMOVE "${stamp}")
        endforeach()
    endif()

    if(_TRITON_CPU_ONLY)
        # triton-cpu pinned to a specific commit for reproducibility (commit set above)
        set(TRITON_URL "https://github.com/triton-lang/triton-cpu/archive/${TRITON_CPU_COMMIT}.tar.gz")
        set(TRITON_URL_HASH "SHA256=29bbed27580d8785605216fe628759165b195cfcf81ee7bc2a3ef681c4f161b3")
    elseif(WIN32)
        set(TRITON_URL "https://github.com/triton-lang/triton-windows/archive/refs/heads/release/3.6.x-windows.tar.gz")
    else()
        set(TRITON_URL "https://github.com/triton-lang/triton/archive/refs/tags/v${TRITON_VERSION}.tar.gz")
        set(TRITON_URL_HASH "SHA256=be270ed11ca5a8fbd9d7941c5bbe9a23a9f6e2ffd372c8398346928bee464774")
    endif()

    # Determine codegen backends based on TRITON_GPU_TARGET and build config.
    # This MUST be done before the LLVM build so LLVM_TARGETS_TO_BUILD can be set correctly.
    set(TRITON_CODEGEN_BACKENDS "")

    if(_TRITON_CPU_ONLY)
        # CPU-only build: use triton-cpu's CPU backend
        set(TRITON_CODEGEN_BACKENDS "cpu")
    elseif(TRITON_GPU_TARGET STREQUAL "AUTO")
        # NVIDIA backend: always for CUDA builds (pure or ZLUDA)
        if(SD_CUDA)
            list(APPEND TRITON_CODEGEN_BACKENDS "nvidia")
        endif()
        # AMD backend: only when ZLUDA targets AMD, or native HIP build
        if(SD_HIP OR (SD_ZLUDA AND ZLUDA_TARGET_BACKEND STREQUAL "AMD"))
            list(APPEND TRITON_CODEGEN_BACKENDS "amd")
        endif()
        # Intel backend: only when ZLUDA targets Intel, or native Level Zero build
        if(SD_LEVEL_ZERO OR (SD_ZLUDA AND ZLUDA_TARGET_BACKEND STREQUAL "INTEL"))
            list(APPEND TRITON_CODEGEN_BACKENDS "intel")
        endif()
        # Fallback
        if(NOT TRITON_CODEGEN_BACKENDS)
            list(APPEND TRITON_CODEGEN_BACKENDS "nvidia")
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
    message(STATUS "   Triton codegen backends: ${TRITON_CODEGEN_BACKENDS}")

    # Each Triton variant pins a specific LLVM commit for ABI compatibility.
    # We build LLVM/MLIR from source at that commit.
    if(_TRITON_CPU_ONLY)
        # triton-cpu pins a different LLVM commit than GPU triton
        set(TRITON_LLVM_COMMIT "20902f0b721ba6cf2fb134362d27144bd8584d53")
        set(TRITON_LLVM_URL_HASH "SHA256=1736af3127e73eab0f2a2f489275c9509d5b60f80c050c42be3a1f85843993e2")
        set(TRITON_LLVM_PREFIX "${CMAKE_BINARY_DIR}/triton_cpu_llvm")
        # CPU-only: just need host target for LLVM CPU codegen
        set(TRITON_LLVM_TARGETS "host")
        message(STATUS "   LLVM targets: host (CPU-only triton-cpu)")
    else()
        # GPU triton v3.6.0 pins f6ded0be
        set(TRITON_LLVM_COMMIT "f6ded0be897e2878612dd903f7e8bb85448269e5")
        set(TRITON_LLVM_URL_HASH "SHA256=f63c624aa63eda73508b9df2be2a6945ea4fddbee58615fbe1cd747b6884dd5e")
        set(TRITON_LLVM_PREFIX "${CMAKE_BINARY_DIR}/triton_llvm")
        # Determine LLVM targets based on which Triton codegen backends are needed.
        # Always include host and NVPTX (for CUDA). Only add AMDGPU if "amd" backend is selected.
        set(TRITON_LLVM_TARGETS "host$<SEMICOLON>NVPTX")
        if("amd" IN_LIST TRITON_CODEGEN_BACKENDS)
            set(TRITON_LLVM_TARGETS "${TRITON_LLVM_TARGETS}$<SEMICOLON>AMDGPU")
            message(STATUS "   LLVM targets: host, NVPTX, AMDGPU (AMD backend requested)")
        else()
            message(STATUS "   LLVM targets: host, NVPTX (no AMD backend)")
        endif()
    endif()

    # --- Dependency cache: restore Triton LLVM and Triton into build dirs ---
    # This is done BEFORE the existing "reuse existing install" check so that
    # the existing fast-path logic picks up restored files naturally.
    if(SD_DEP_CACHE)
        # Check Triton LLVM cache
        string(SUBSTRING "${TRITON_LLVM_COMMIT}" 0 8 TRITON_LLVM_COMMIT_SHORT)
        sd_dep_cache_key("triton_llvm" "${TRITON_LLVM_COMMIT_SHORT}" "TARGETS=${TRITON_LLVM_TARGETS}" _tllvm_cache_key)
        sd_dep_cache_check("triton_llvm" "${_tllvm_cache_key}" _tllvm_hit _tllvm_cache_path)
        if(_tllvm_hit AND NOT EXISTS "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
            sd_dep_cache_restore("triton_llvm" "${_tllvm_cache_path}" "${TRITON_LLVM_INSTALL_DIR}")
        endif()

        # Check Triton cache
        set(_triton_ver "${TRITON_VERSION}")
        string(REPLACE ";" "_" _triton_backends_str "${TRITON_CODEGEN_BACKENDS}")
        sd_dep_cache_key("triton" "${_triton_ver}" "BACKENDS=${_triton_backends_str}" _triton_cache_key)
        sd_dep_cache_check("triton" "${_triton_cache_key}" _triton_hit _triton_cache_path)
        if(_triton_hit)
            set(_need_triton_restore TRUE)
            if(EXISTS "${TRITON_INSTALL_DIR}/lib/libtriton.a" OR EXISTS "${TRITON_INSTALL_DIR}/lib/triton.lib")
                set(_need_triton_restore FALSE)
            endif()
            if(_need_triton_restore)
                sd_dep_cache_restore("triton" "${_triton_cache_path}" "${TRITON_INSTALL_DIR}")
            endif()
        endif()
    endif()

    # Build-time-only SmartCcache partition key for Triton LLVM external build.
    # This stays in CMake/source-tree scope; runtime extraction infra is separate.
    set(_TRITON_LLVM_SHAPE_KEY_RAW "${TRITON_LLVM_COMMIT}-${TRITON_LLVM_TARGETS}-Release")
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" TRITON_LLVM_SHAPE_KEY "${_TRITON_LLVM_SHAPE_KEY_RAW}")
    set(TRITON_LLVM_BUILD_COMMAND
            ${CMAKE_COMMAND} -E env
            "SD_SMART_CCACHE_SEGMENT=triton_llvm"
            "SD_SMART_CCACHE_SHAPE_KEY=${TRITON_LLVM_SHAPE_KEY}"
            ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --parallel ${DEP_PARALLEL_JOBS}
    )
    message(STATUS "   Triton LLVM smart ccache segment=triton_llvm shape=${TRITON_LLVM_SHAPE_KEY}")

    if(NOT EXISTS "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
        message(STATUS "   Building LLVM/MLIR from Triton-pinned commit ${TRITON_LLVM_COMMIT}...")
        message(STATUS "   This is a one-time build (~15-30 min). Install dir: ${TRITON_LLVM_INSTALL_DIR}")

        file(MAKE_DIRECTORY "${TRITON_LLVM_PREFIX}/stamp")

        # Build LLVM cmake args, including compiler launcher if available
        set(TRITON_LLVM_CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=${TRITON_LLVM_INSTALL_DIR}
                -DCMAKE_BUILD_TYPE=Release
                -DLLVM_ENABLE_PROJECTS=mlir
                -DLLVM_TARGETS_TO_BUILD=${TRITON_LLVM_TARGETS}
                -DLLVM_ENABLE_ASSERTIONS=ON
                -DLLVM_ENABLE_RTTI=ON
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

        # MSVC-specific flags for LLVM build
        if(MSVC)
            list(APPEND TRITON_LLVM_CMAKE_ARGS
                -DLLVM_BUILD_SHARED_LIBS=OFF
                "-DCMAKE_C_FLAGS=/utf-8 /D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING"
                "-DCMAKE_CXX_FLAGS=/utf-8 /D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING"
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

        set(TRITON_LLVM_URL "https://github.com/llvm/llvm-project/archive/${TRITON_LLVM_COMMIT}.tar.gz")

        ExternalProject_Add(triton_llvm_external
                PREFIX            "${TRITON_LLVM_PREFIX}"
                URL               "${TRITON_LLVM_URL}"
                URL_HASH          "${TRITON_LLVM_URL_HASH}"
                DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                SOURCE_SUBDIR     llvm
                BINARY_DIR        "${TRITON_LLVM_PREFIX}/build"
                STAMP_DIR         "${TRITON_LLVM_PREFIX}/stamp"
                CMAKE_ARGS        ${TRITON_LLVM_CMAKE_ARGS}
                BUILD_COMMAND     ${TRITON_LLVM_BUILD_COMMAND}
                INSTALL_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config Release
                BUILD_BYPRODUCTS
                    "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake"
                    "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake"
                TIMEOUT           7200
                LOG_DOWNLOAD      OFF
                LOG_CONFIGURE     OFF
                LOG_BUILD         OFF
                LOG_INSTALL       OFF
        )

        # Patch MLIRConfig.cmake after install: replace bare "mlir-tblgen" with absolute path.
        # The installed MLIRConfig.cmake sets MLIR_TABLEGEN_EXE to just "mlir-tblgen" (bare name).
        # When LLVM is consumed as an external project, there is no imported target for mlir-tblgen,
        # so cmake's tablegen() function creates broken file-level dependencies.
        set(_PATCH_MLIR_SCRIPT "${CMAKE_BINARY_DIR}/patch_mlir_config.cmake")
        file(WRITE "${_PATCH_MLIR_SCRIPT}" "
            set(F \"${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake\")
            file(READ \"\${F}\" C)
            string(REPLACE
                \"set(MLIR_TABLEGEN_EXE \\\"mlir-tblgen\\\")\"
                \"set(MLIR_TABLEGEN_EXE \\\"${TRITON_LLVM_INSTALL_DIR}/bin/mlir-tblgen\\\")\"
                C \"\${C}\")
            string(REPLACE
                \"set(MLIR_PDLL_TABLEGEN_EXE \\\"mlir-pdll\\\")\"
                \"set(MLIR_PDLL_TABLEGEN_EXE \\\"${TRITON_LLVM_INSTALL_DIR}/bin/mlir-pdll\\\")\"
                C \"\${C}\")
            file(WRITE \"\${F}\" \"\${C}\")
            message(STATUS \"Patched MLIRConfig.cmake with absolute tablegen paths\")
        ")
        ExternalProject_Add_Step(triton_llvm_external patch_mlir_config
            COMMAND ${CMAKE_COMMAND} -P "${_PATCH_MLIR_SCRIPT}"
            DEPENDEES install
            COMMENT "Patching MLIRConfig.cmake with absolute tablegen executable paths"
        )

        # Patch Matchers.h for NVCC compatibility: NVCC cannot handle lambda-to-function-pointer
        # conversion inside aggregate initialization. Guard the affected functions with __CUDACC__.
        set(_PATCH_MATCHERS_SCRIPT "${CMAKE_BINARY_DIR}/patch_matchers.cmake")
        file(WRITE "${_PATCH_MATCHERS_SCRIPT}" "
            set(F \"${TRITON_LLVM_INSTALL_DIR}/include/mlir/IR/Matchers.h\")
            if(EXISTS \"\${F}\")
                file(READ \"\${F}\" C)
                string(FIND \"\${C}\" \"__CUDACC__\" _HAS_GUARD)
                if(_HAS_GUARD EQUAL -1)
                    string(REPLACE
                        \"/// Matches a constant scalar / vector splat / tensor splat float (both positive\"
                        \"#ifndef __CUDACC__\\n/// Matches a constant scalar / vector splat / tensor splat float (both positive\"
                        C \"\${C}\")
                    string(REPLACE
                        \"/// Matches the given OpClass.\"
                        \"#endif // __CUDACC__\\n\\n/// Matches the given OpClass.\"
                        C \"\${C}\")
                    file(WRITE \"\${F}\" \"\${C}\")
                    message(STATUS \"Patched Matchers.h with __CUDACC__ guard for NVCC compatibility\")
                else()
                    message(STATUS \"Matchers.h already has __CUDACC__ guard\")
                endif()
            endif()
        ")
        ExternalProject_Add_Step(triton_llvm_external patch_matchers
            COMMAND ${CMAKE_COMMAND} -P "${_PATCH_MATCHERS_SCRIPT}"
            DEPENDEES patch_mlir_config
            COMMENT "Patching Matchers.h for NVCC compatibility"
        )

        # --- Cache store for Triton LLVM ---
        if(SD_DEP_CACHE AND DEFINED _tllvm_cache_key)
            sd_dep_cache_store("triton_llvm" "${_tllvm_cache_key}" "${TRITON_LLVM_INSTALL_DIR}" "triton_llvm_external")
        endif()
    else()
        message(STATUS "   Reusing existing LLVM/MLIR at ${TRITON_LLVM_INSTALL_DIR}")
        # Create a dummy target so triton_external can depend on it
        add_custom_target(triton_llvm_external)

        # Patch MLIRConfig.cmake if it has bare tablegen names (idempotent)
        set(_MLIR_CONFIG_FILE "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
        if(EXISTS "${_MLIR_CONFIG_FILE}")
            file(READ "${_MLIR_CONFIG_FILE}" _MLIR_CFG)
            string(FIND "${_MLIR_CFG}" "set(MLIR_TABLEGEN_EXE \"mlir-tblgen\")" _NEEDS_PATCH)
            if(NOT _NEEDS_PATCH EQUAL -1)
                string(REPLACE
                    "set(MLIR_TABLEGEN_EXE \"mlir-tblgen\")"
                    "set(MLIR_TABLEGEN_EXE \"${TRITON_LLVM_INSTALL_DIR}/bin/mlir-tblgen\")"
                    _MLIR_CFG "${_MLIR_CFG}")
                string(REPLACE
                    "set(MLIR_PDLL_TABLEGEN_EXE \"mlir-pdll\")"
                    "set(MLIR_PDLL_TABLEGEN_EXE \"${TRITON_LLVM_INSTALL_DIR}/bin/mlir-pdll\")"
                    _MLIR_CFG "${_MLIR_CFG}")
                file(WRITE "${_MLIR_CONFIG_FILE}" "${_MLIR_CFG}")
                message(STATUS "   Patched MLIRConfig.cmake with absolute tablegen paths")
            endif()
        endif()

        # Patch Matchers.h for NVCC compatibility (idempotent)
        set(_MATCHERS_FILE "${TRITON_LLVM_INSTALL_DIR}/include/mlir/IR/Matchers.h")
        if(EXISTS "${_MATCHERS_FILE}")
            file(READ "${_MATCHERS_FILE}" _MATCHERS_CONTENT)
            string(FIND "${_MATCHERS_CONTENT}" "__CUDACC__" _HAS_GUARD)
            if(_HAS_GUARD EQUAL -1)
                string(REPLACE
                    "/// Matches a constant scalar / vector splat / tensor splat float (both positive"
                    "#ifndef __CUDACC__\n/// Matches a constant scalar / vector splat / tensor splat float (both positive"
                    _MATCHERS_CONTENT "${_MATCHERS_CONTENT}")
                string(REPLACE
                    "/// Matches the given OpClass."
                    "#endif // __CUDACC__\n\n/// Matches the given OpClass."
                    _MATCHERS_CONTENT "${_MATCHERS_CONTENT}")
                file(WRITE "${_MATCHERS_FILE}" "${_MATCHERS_CONTENT}")
                message(STATUS "   Patched Matchers.h with __CUDACC__ guard for NVCC compatibility")
            endif()
        endif()
    endif()

    set(TRITON_MLIR_DIR "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/mlir")
    set(TRITON_LLVM_DIR "${TRITON_LLVM_INSTALL_DIR}/lib/cmake/llvm")

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
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )

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
    set(_TRITON_SHAPE_KEY_RAW "${TRITON_VERSION}-${_TRITON_BACKENDS_KEY}-${CMAKE_BUILD_TYPE}")
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
    if(_TRITON_CPU_ONLY)
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
    if(_TRITON_CPU_ONLY)
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
    if(_TRITON_CPU_ONLY)
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

    ExternalProject_Add(triton_external
            PREFIX            "${TRITON_PREFIX}"
            URL               "${TRITON_URL}"
            URL_HASH          "${_TRITON_EP_URL_HASH}"
            DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
            SOURCE_DIR        "${TRITON_SOURCE_DIR}"
            BINARY_DIR        "${TRITON_PREFIX}/build"
            STAMP_DIR         "${TRITON_STAMP_DIR}"
            DOWNLOAD_NO_PROGRESS FALSE
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            PATCH_COMMAND     ${_TRITON_EP_PATCH_CMD}
            CMAKE_ARGS        ${TRITON_CMAKE_ARGS}
            BUILD_COMMAND     ${TRITON_BUILD_COMMAND}
            INSTALL_COMMAND   ${CMAKE_COMMAND} -DBINARY_DIR=<BINARY_DIR> -DSOURCE_DIR=<SOURCE_DIR> -DINSTALL_DIR=${TRITON_INSTALL_DIR} -P "${TRITON_INSTALL_SCRIPT}"
            BUILD_BYPRODUCTS
                "${TRITON_INSTALL_DIR}/include/triton/Compiler/Compiler.h"
                "${_TRITON_LIB_BYPRODUCT}"
            TIMEOUT           1800
            LOG_DOWNLOAD      OFF
            LOG_CONFIGURE     OFF
            LOG_BUILD         OFF
            LOG_INSTALL       OFF
            DEPENDS           triton_llvm_external
    )

    add_library(triton_interface INTERFACE)
    # Use SYSTEM to prevent Triton/LLVM headers from conflicting
    # with libnd4j's DataType enum names (BOOL, INT8, FLOAT32 etc.)
    target_include_directories(triton_interface SYSTEM INTERFACE
        "${TRITON_INSTALL_DIR}/include"
        "${TRITON_LLVM_INSTALL_DIR}/include"
    )

    if(WIN32)
        target_link_libraries(triton_interface INTERFACE "${TRITON_INSTALL_DIR}/lib/triton.lib")
    else()
        target_link_libraries(triton_interface INTERFACE "${TRITON_INSTALL_DIR}/lib/libtriton.a")
    endif()

    # Link MLIR/LLVM static libraries from our built LLVM.
    # On first build, LLVM hasn't been built yet at configure time so we use -l flags
    # (resolved at link time). On subsequent builds, the early-return path above handles it.
    target_link_directories(triton_interface INTERFACE "${TRITON_LLVM_INSTALL_DIR}/lib")

    # Try glob first (works when LLVM already built from previous run)
    if(WIN32)
        file(GLOB _TRITON_MLIR_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/MLIR*.lib")
        file(GLOB _TRITON_LLVM_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/LLVM*.lib")
    else()
        file(GLOB _TRITON_MLIR_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/libMLIR*.a")
        file(GLOB _TRITON_LLVM_LIBS "${TRITON_LLVM_INSTALL_DIR}/lib/libLLVM*.a")
    endif()
    list(LENGTH _TRITON_MLIR_LIBS _mlir_count)
    list(LENGTH _TRITON_LLVM_LIBS _llvm_count)

    if(_mlir_count GREATER 0 AND _llvm_count GREATER 0)
        # Libraries exist — use full paths (most reliable)
        list(SORT _TRITON_MLIR_LIBS)
        list(SORT _TRITON_LLVM_LIBS)
        if(WIN32)
            target_link_libraries(triton_interface INTERFACE
                ${_TRITON_MLIR_LIBS}
                ${_TRITON_LLVM_LIBS}
                zlib.lib zstd.lib
            )
            if(NOT _TRITON_CPU_ONLY)
                target_link_libraries(triton_interface INTERFACE nvrtc.lib cuda.lib)
            endif()
        else()
            if(APPLE OR ANDROID)
                target_link_libraries(triton_interface INTERFACE
                    ${_TRITON_MLIR_LIBS}
                    ${_TRITON_LLVM_LIBS}
                )
            else()
                target_link_libraries(triton_interface INTERFACE
                    -Wl,--start-group
                    ${_TRITON_MLIR_LIBS}
                    ${_TRITON_LLVM_LIBS}
                    -Wl,--end-group
                )
            endif()
            target_link_libraries(triton_interface INTERFACE -lz -lm)
            if(NOT APPLE AND NOT ANDROID)
                target_link_libraries(triton_interface INTERFACE -lzstd -lrt -ldl -lpthread)
            elseif(APPLE)
                target_link_libraries(triton_interface INTERFACE -lzstd -ldl -lpthread)
            else()
                target_link_libraries(triton_interface INTERFACE -ldl)
            endif()
            if(NOT _TRITON_CPU_ONLY)
                target_link_libraries(triton_interface INTERFACE -lnvrtc -lcuda)
            endif()
        endif()
        message(STATUS "Triton linking ${_mlir_count} MLIR + ${_llvm_count} LLVM static libraries (globbed)")
    else()
        # Fresh build — LLVM not built yet, use -l flags resolved at link time.
        # COMPLETE list of all MLIR + LLVM libraries from Triton's LLVM fork.
        # --start-group/--end-group handles circular dependencies.
        # Unused libraries won't increase binary size (linker only pulls needed objects).
        if(WIN32)
            # On Windows (MSVC), deferred linking uses .lib names directly.
            # MSVC linker resolves circular deps automatically.
            message(STATUS "Triton: LLVM not yet built, using deferred linking (Windows)")
            # Will be resolved at link time via target_link_directories above.
            # The glob path above will handle this on rebuild.
        else()
            message(STATUS "Triton: LLVM not yet built, using -l flags for deferred linking")
            # Collect all MLIR/LLVM -l flags into a list, then wrap with
            # --start-group/--end-group only on platforms that need it (GNU ld).
            # Apple ld and Android lld resolve circular deps automatically.
            set(_TRITON_DEFERRED_LIBS
                # === ALL MLIR libraries (Triton 3.6.0 LLVM pin: f6ded0be) ===
                # AUTO-GENERATED from triton_llvm_install/lib/libMLIR*.a (357 libraries)
                -lMLIRAffineAnalysis -lMLIRAffineDialect -lMLIRAffineToStandard -lMLIRAffineTransformOps
                -lMLIRAffineTransforms -lMLIRAffineUtils -lMLIRAMDGPUDialect -lMLIRAMDGPUToROCDL
                -lMLIRAMDGPUTransforms -lMLIRAMDGPUUtils -lMLIRAMXDialect -lMLIRAMXTransforms -lMLIRAnalysis
                -lMLIRArithAttrToLLVMConversion -lMLIRArithDialect -lMLIRArithToAMDGPU -lMLIRArithToArmSME
                -lMLIRArithToEmitC -lMLIRArithToLLVM -lMLIRArithToSPIRV -lMLIRArithTransforms -lMLIRArithUtils
                -lMLIRArithValueBoundsOpInterfaceImpl -lMLIRArmNeon2dToIntr -lMLIRArmNeonDialect
                -lMLIRArmNeonToLLVMIRTranslation -lMLIRArmNeonTransforms -lMLIRArmNeonVectorTransformOps
                -lMLIRArmSMEDialect -lMLIRArmSMEToLLVM -lMLIRArmSMEToLLVMIRTranslation -lMLIRArmSMEToSCF
                -lMLIRArmSMETransforms -lMLIRArmSVEDialect -lMLIRArmSVEToLLVMIRTranslation -lMLIRArmSVETransforms
                -lMLIRArmSVEVectorTransformOps -lMLIRAsmParser -lMLIRAsyncDialect -lMLIRAsyncToLLVM
                -lMLIRAsyncTransforms -lMLIRBufferizationDialect -lMLIRBufferizationPipelines
                -lMLIRBufferizationToMemRef -lMLIRBufferizationTransformOps -lMLIRBufferizationTransforms
                -lMLIRBuiltinToLLVMIRTranslation -lMLIRBytecodeOpInterface -lMLIRBytecodeReader
                -lMLIRBytecodeWriter -lMLIRCallInterfaces -lMLIRCAPIAMDGPU -lMLIRCAPIArith -lMLIRCAPIAsync
                -lMLIRCAPIControlFlow -lMLIRCAPIConversion -lMLIRCAPIDebug -lMLIRCAPIEmitC
                -lMLIRCAPIExecutionEngine -lMLIRCAPIExportSMTLIB -lMLIRCAPIFunc -lMLIRCAPIGPU -lMLIRCAPIIndex
                -lMLIRCAPIInterfaces -lMLIRCAPIIR -lMLIRCAPIIRDL -lMLIRCAPILinalg -lMLIRCAPILLVM -lMLIRCAPIMath
                -lMLIRCAPIMemRef -lMLIRCAPIMLProgram -lMLIRCAPINVGPU -lMLIRCAPINVVM -lMLIRCAPIOpenMP -lMLIRCAPIPDL
                -lMLIRCAPIQuant -lMLIRCAPIRegisterEverything -lMLIRCAPIROCDL -lMLIRCAPISCF -lMLIRCAPIShape
                -lMLIRCAPISMT -lMLIRCAPISparseTensor -lMLIRCAPISPIRV -lMLIRCAPITarget -lMLIRCAPITensor
                -lMLIRCAPITransformDialect -lMLIRCAPITransformDialectTransforms -lMLIRCAPITransforms
                -lMLIRCAPIVector -lMLIRCastInterfaces -lMLIRComplexDialect -lMLIRComplexDivisionConversion
                -lMLIRComplexToLibm -lMLIRComplexToLLVM -lMLIRComplexToROCDLLibraryCalls -lMLIRComplexToSPIRV
                -lMLIRComplexToStandard -lMLIRControlFlowDialect -lMLIRControlFlowInterfaces
                -lMLIRControlFlowToLLVM -lMLIRControlFlowToSCF -lMLIRControlFlowToSPIRV -lMLIRControlFlowTransforms
                -lMLIRConvertToEmitC -lMLIRConvertToLLVMInterface -lMLIRConvertToLLVMPass
                -lMLIRDataLayoutInterfaces -lMLIRDebug -lMLIRDerivedAttributeOpInterface
                -lMLIRDestinationStyleOpInterface -lMLIRDialect -lMLIRDialectUtils -lMLIRDLTIDialect
                -lMLIRDLTITransformOps -lMLIREmitCDialect -lMLIREmitCTransforms -lMLIRExecutionEngine
                -lMLIRExecutionEngineUtils -lMLIRExportSMTLIB -lMLIRFromLLVMIRTranslationRegistration
                -lMLIRFuncAllExtensions -lMLIRFuncDialect -lMLIRFuncInlinerExtension -lMLIRFuncShardingExtensions
                -lMLIRFunctionInterfaces -lMLIRFuncToEmitC -lMLIRFuncToLLVM -lMLIRFuncToSPIRV
                -lMLIRFuncTransformOps -lMLIRFuncTransforms -lMLIRFuncUtils -lMLIRGPUDialect -lMLIRGPUPipelines
                -lMLIRGPUToGPURuntimeTransforms -lMLIRGPUToLLVMIRTranslation -lMLIRGPUToLLVMSPV
                -lMLIRGPUToNVVMTransforms -lMLIRGPUToROCDLTransforms -lMLIRGPUToSPIRV -lMLIRGPUTransformOps
                -lMLIRGPUTransforms -lMLIRGPUUtils -lMLIRIndexDialect -lMLIRIndexingMapOpInterface
                -lMLIRIndexToLLVM -lMLIRIndexToSPIRV -lMLIRInferIntRangeCommon -lMLIRInferIntRangeInterface
                -lMLIRInferTypeOpInterface -lMLIRIR -lMLIRIRDL -lMLIRJitRunner -lMLIRLinalgDialect
                -lMLIRLinalgToStandard -lMLIRLinalgTransformOps -lMLIRLinalgTransforms -lMLIRLinalgUtils
                -lMLIRLLVMCommonConversion -lMLIRLLVMDialect -lMLIRLLVMIRToLLVMTranslation
                -lMLIRLLVMIRToNVVMTranslation -lMLIRLLVMIRTransforms -lMLIRLLVMToLLVMIRTranslation
                -lMLIRLoopLikeInterface -lMLIRLspServerLib -lMLIRLspServerSupportLib -lMLIRMaskableOpInterface
                -lMLIRMaskingOpInterface -lMLIRMathDialect -lMLIRMathToEmitC -lMLIRMathToFuncs -lMLIRMathToLibm
                -lMLIRMathToLLVM -lMLIRMathToROCDL -lMLIRMathToSPIRV -lMLIRMathTransforms -lMLIRMemOpInterfaces
                -lMLIRMemorySlotInterfaces -lMLIRMemRefDialect -lMLIRMemRefToEmitC -lMLIRMemRefToLLVM
                -lMLIRMemRefToSPIRV -lMLIRMemRefTransformOps -lMLIRMemRefTransforms -lMLIRMemRefUtils
                -lMLIRMlirOptMain -lMLIRMLProgramDialect -lMLIRMLProgramTransforms -lMLIRMPIDialect -lMLIRMPIToLLVM
                -lMLIRNVGPUDialect -lMLIRNVGPUToNVVM -lMLIRNVGPUTransformOps -lMLIRNVGPUTransforms -lMLIRNVGPUUtils
                -lMLIRNVVMDialect -lMLIRNVVMTarget -lMLIRNVVMToLLVM -lMLIRNVVMToLLVMIRTranslation -lMLIRObservers
                -lMLIROpenACCDialect -lMLIROpenACCMPCommon -lMLIROpenACCToLLVMIRTranslation -lMLIROpenACCToSCF
                -lMLIROpenACCTransforms -lMLIROpenMPDialect -lMLIROpenMPToLLVM -lMLIROpenMPToLLVMIRTranslation
                -lMLIROptLib -lMLIRParallelCombiningOpInterface -lMLIRParser -lMLIRPass -lMLIRPDLDialect
                -lMLIRPDLInterpDialect -lMLIRPDLLAST -lMLIRPDLLCodeGen -lMLIRPDLLODS -lMLIRPDLToPDLInterp
                -lMLIRPluginsLib -lMLIRPresburger -lMLIRPtrDialect -lMLIRPtrMemorySpaceInterfaces -lMLIRPtrToLLVM
                -lMLIRPtrToLLVMIRTranslation -lMLIRQuantDialect -lMLIRQuantTransforms -lMLIRQuantUtils -lMLIRQuery
                -lMLIRQueryLib -lMLIRQueryMatcher -lMLIRReconcileUnrealizedCasts -lMLIRReduce -lMLIRReduceLib
                -lMLIRRegisterAllDialects -lMLIRRegisterAllExtensions -lMLIRRegisterAllPasses -lMLIRRemarkStreamer
                -lMLIRRewrite -lMLIRRewritePDL -lMLIRROCDLDialect -lMLIRROCDLTarget -lMLIRROCDLToLLVMIRTranslation
                -lMLIRRuntimeVerifiableOpInterface -lMLIRSCFDialect -lMLIRSCFToControlFlow -lMLIRSCFToEmitC
                -lMLIRSCFToGPU -lMLIRSCFToOpenMP -lMLIRSCFToSPIRV -lMLIRSCFTransformOps -lMLIRSCFTransforms
                -lMLIRSCFUtils -lMLIRShapeDialect -lMLIRShapedOpInterfaces -lMLIRShapeOpsTransforms
                -lMLIRShapeToStandard -lMLIRShardDialect -lMLIRShardingInterface -lMLIRShardToMPI
                -lMLIRShardTransforms -lMLIRSideEffectInterfaces -lMLIRSMT -lMLIRSparseTensorDialect
                -lMLIRSparseTensorPipelines -lMLIRSparseTensorRuntime -lMLIRSparseTensorTransformOps
                -lMLIRSparseTensorTransforms -lMLIRSparseTensorUtils -lMLIRSPIRVAttrToLLVMConversion
                -lMLIRSPIRVBinaryUtils -lMLIRSPIRVConversion -lMLIRSPIRVDeserialization -lMLIRSPIRVDialect
                -lMLIRSPIRVImageInterfaces -lMLIRSPIRVModuleCombiner -lMLIRSPIRVSerialization -lMLIRSPIRVTarget
                -lMLIRSPIRVToLLVM -lMLIRSPIRVToLLVMIRTranslation -lMLIRSPIRVTransforms
                -lMLIRSPIRVTranslateRegistration -lMLIRSPIRVUtils -lMLIRSubsetOpInterface -lMLIRSupport
                -lMLIRTableGen -lMLIRTargetCpp -lMLIRTargetIRDLToCpp -lMLIRTargetLLVM -lMLIRTargetLLVMIRExport
                -lMLIRTargetLLVMIRImport -lMLIRTargetLLVMIRTransforms -lMLIRTargetWasmImport -lMLIRTblgenLib
                -lMLIRTensorAllExtensions -lMLIRTensorDialect -lMLIRTensorInferTypeOpInterfaceImpl
                -lMLIRTensorShardingExtensions -lMLIRTensorTilingInterfaceImpl -lMLIRTensorToLinalg
                -lMLIRTensorToSPIRV -lMLIRTensorTransformOps -lMLIRTensorTransforms -lMLIRTensorUtils
                -lMLIRTilingInterface -lMLIRToLLVMIRTranslationRegistration -lMLIRTosaDialect
                -lMLIRTosaShardingInterfaceImpl -lMLIRTosaToArith -lMLIRTosaToLinalg -lMLIRTosaToMLProgram
                -lMLIRTosaToSCF -lMLIRTosaToTensor -lMLIRTosaTransforms -lMLIRTransformDebugExtension
                -lMLIRTransformDialect -lMLIRTransformDialectInterfaces -lMLIRTransformDialectIRDLExtension
                -lMLIRTransformDialectTransforms -lMLIRTransformDialectUtils -lMLIRTransformLoopExtension
                -lMLIRTransformPDLExtension -lMLIRTransforms -lMLIRTransformSMTExtension
                -lMLIRTransformTuneExtension -lMLIRTransformUtils -lMLIRTranslateLib -lMLIRUBDialect -lMLIRUBToLLVM
                -lMLIRUBToSPIRV -lMLIRValueBoundsOpInterface -lMLIRVCIXDialect -lMLIRVCIXToLLVMIRTranslation
                -lMLIRVectorDialect -lMLIRVectorInterfaces -lMLIRVectorToAMX -lMLIRVectorToArmSME -lMLIRVectorToGPU
                -lMLIRVectorToLLVM -lMLIRVectorToLLVMPass -lMLIRVectorToSCF -lMLIRVectorToSPIRV -lMLIRVectorToXeGPU
                -lMLIRVectorTransformOps -lMLIRVectorTransforms -lMLIRVectorUtils -lMLIRViewLikeInterface
                -lMLIRWasmSSADialect -lMLIRX86VectorDialect -lMLIRX86VectorTransforms -lMLIRXeGPUDialect
                -lMLIRXeGPUToXeVM -lMLIRXeGPUTransforms -lMLIRXeGPUUtils -lMLIRXeVMDialect -lMLIRXeVMTarget
                -lMLIRXeVMToLLVM -lMLIRXeVMToLLVMIRTranslation
                # === ALL LLVM libraries (Triton 3.6.0 LLVM pin: f6ded0be) ===
                # AUTO-GENERATED from triton_llvm_install/lib/libLLVM*.a (110 libraries)
                -lLLVMAggressiveInstCombine -lLLVMAnalysis -lLLVMAsmParser
                -lLLVMAsmPrinter -lLLVMBinaryFormat -lLLVMBitReader
                -lLLVMBitstreamReader -lLLVMBitWriter -lLLVMCAS
                -lLLVMCFGuard -lLLVMCFIVerify -lLLVMCGData -lLLVMCodeGen
                -lLLVMCodeGenTypes -lLLVMCore -lLLVMCoroutines
                -lLLVMCoverage -lLLVMDebugInfoBTF -lLLVMDebugInfoCodeView
                -lLLVMDebuginfod -lLLVMDebugInfoDWARF -lLLVMDebugInfoDWARFLowLevel
                -lLLVMDebugInfoGSYM -lLLVMDebugInfoLogicalView -lLLVMDebugInfoMSF
                -lLLVMDebugInfoPDB -lLLVMDemangle -lLLVMDiff -lLLVMDlltoolDriver
                -lLLVMDWARFCFIChecker -lLLVMDWARFLinker -lLLVMDWARFLinkerClassic
                -lLLVMDWARFLinkerParallel -lLLVMDWP -lLLVMExecutionEngine
                -lLLVMExegesis -lLLVMExegesisX86 -lLLVMExtensions -lLLVMFileCheck
                -lLLVMFrontendAtomic -lLLVMFrontendDirective -lLLVMFrontendDriver
                -lLLVMFrontendHLSL -lLLVMFrontendOffloading -lLLVMFrontendOpenACC
                -lLLVMFrontendOpenMP -lLLVMFuzzerCLI -lLLVMFuzzMutate -lLLVMGlobalISel
                -lLLVMHipStdPar -lLLVMInstCombine -lLLVMInstrumentation
                -lLLVMInterfaceStub -lLLVMInterpreter -lLLVMipo
                -lLLVMIRPrinter -lLLVMIRReader -lLLVMJITLink
                -lLLVMLibDriver -lLLVMLineEditor -lLLVMLinker
                -lLLVMLTO -lLLVMMC -lLLVMMCA
                -lLLVMMCDisassembler -lLLVMMCJIT -lLLVMMCParser
                -lLLVMMIRParser -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc
                -lLLVMNVPTXInfo -lLLVMObjCARCOpts -lLLVMObjCopy
                -lLLVMObject -lLLVMObjectYAML -lLLVMOptDriver
                -lLLVMOption -lLLVMOrcDebugging -lLLVMOrcJIT
                -lLLVMOrcShared -lLLVMOrcTargetProcess -lLLVMPasses
                -lLLVMProfileData -lLLVMRemarks -lLLVMRuntimeDyld
                -lLLVMSandboxIR -lLLVMScalarOpts -lLLVMSelectionDAG
                -lLLVMSupport -lLLVMSupportLSP -lLLVMSymbolize -lLLVMTableGen
                -lLLVMTableGenBasic -lLLVMTableGenCommon -lLLVMTarget
                -lLLVMTargetParser -lLLVMTelemetry -lLLVMTextAPI -lLLVMTextAPIBinaryReader
                -lLLVMTransformUtils -lLLVMVectorize -lLLVMWindowsDriver
                -lLLVMWindowsManifest -lLLVMX86AsmParser -lLLVMX86CodeGen
                -lLLVMX86Desc -lLLVMX86Disassembler -lLLVMX86Info
                -lLLVMX86TargetMCA -lLLVMXRay
            )
            if(APPLE OR ANDROID)
                target_link_libraries(triton_interface INTERFACE ${_TRITON_DEFERRED_LIBS})
            else()
                target_link_libraries(triton_interface INTERFACE
                    -Wl,--start-group ${_TRITON_DEFERRED_LIBS} -Wl,--end-group)
            endif()
            target_link_libraries(triton_interface INTERFACE -lz -lm)
            if(NOT APPLE AND NOT ANDROID)
                target_link_libraries(triton_interface INTERFACE -lzstd -lrt -ldl -lpthread)
            elseif(APPLE)
                target_link_libraries(triton_interface INTERFACE -lzstd -ldl -lpthread)
            else()
                target_link_libraries(triton_interface INTERFACE -ldl)
            endif()
        endif()
    endif()

    # Link nvrtc and cuda driver for NVRTC JIT and PTX backends (GPU only)
    if(NOT _TRITON_CPU_ONLY)
        if(WIN32)
            target_link_libraries(triton_interface INTERFACE nvrtc.lib cuda.lib)
        else()
            target_link_libraries(triton_interface INTERFACE -lnvrtc -lcuda)
        endif()
    endif()

    add_dependencies(triton_interface triton_external)
    set(TRITON triton_interface PARENT_SCOPE)

    # --- Cache store for Triton ---
    if(SD_DEP_CACHE AND DEFINED _triton_cache_key)
        sd_dep_cache_store("triton" "${_triton_cache_key}" "${TRITON_INSTALL_DIR}" "triton_external")
    endif()

    message(STATUS "Triton ${TRITON_VERSION} setup complete (target: ${TRITON_GPU_TARGET})")
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
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
        COMMENT "Copying TBB cmake config into OpenVINO install"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_TBB_DST_CMAKE}"
        COMMAND ${CMAKE_COMMAND}
            -DTBB_PREFIX=${OPENVINO_PREFIX}/tbb_install
            -DTBB_DST=${_TBB_DST_CMAKE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_tbb_cmake.cmake
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

        # Static OpenVINO install produces libs under runtime/lib/intel64/ (Linux x86_64).
        # We link with --whole-archive for the core runtime to ensure plugin registration
        # symbols are not stripped. All static deps (oneDNN, pugixml, TBB) are bundled.
        set(_OV_LIB_DIR "${OPENVINO_INSTALL_DIR}/runtime/lib/intel64")
        set(_OV_LIB_DIR_ALT "${OPENVINO_INSTALL_DIR}/runtime/lib")
        set(_OV_LIB_3P "${OPENVINO_INSTALL_DIR}/runtime/3rdparty/lib")

        if(WIN32)
            target_link_libraries(openvino_interface INTERFACE
                "${_OV_LIB_DIR}/openvino.lib"
                "${_OV_LIB_DIR}/openvino_intel_cpu_plugin.lib"
            )
        else()
            # Use a post-build step (install_openvino.cmake) to glob all static
            # libs and create a proper link line. For now, link the known main libs.
            # The install produces: libopenvino.a, libopenvino_intel_cpu_plugin.a,
            # libopenvino_ir_frontend.a, libopenvino_reference.a, plus 3rdparty
            # deps (libdnnl.a, libpugixml.a, libtbb.a, etc.)
            set(_OV_STATIC_LIBS
                "${_OV_LIB_DIR}/libopenvino.a"
                "${_OV_LIB_DIR}/libopenvino_intel_cpu_plugin.a"
                "${_OV_LIB_DIR}/libopenvino_ir_frontend.a"
                "${_OV_LIB_DIR}/libopenvino_reference.a"
                "${_OV_LIB_DIR}/libopenvino_shape_inference.a"
            )
            if(APPLE)
                # Apple ld resolves circular deps automatically
                target_link_libraries(openvino_interface INTERFACE
                    ${_OV_STATIC_LIBS} -ldl -lm -lpthread)
            else()
                target_link_libraries(openvino_interface INTERFACE
                    -Wl,--start-group ${_OV_STATIC_LIBS} -Wl,--end-group
                    -lpthread -ldl -lm -lrt)
            endif()
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
        # Use OpenVINO's own cmake config — handles all transitive deps
        find_package(OpenVINO REQUIRED CONFIG PATHS "${OpenVINO_DIR}" NO_DEFAULT_PATH)
        target_link_libraries(openvino_interface INTERFACE openvino::runtime)
    else()
        # Fallback: manual linking (shouldn't happen with proper install)
        message(WARNING "OpenVINO cmake config not found, using manual linking")
        set(_OV_LIB_DIR "${_install_dir}/runtime/lib/intel64")
        if(WIN32)
            target_link_libraries(openvino_interface INTERFACE
                "${_OV_LIB_DIR}/openvino.lib")
        else()
            # Glob all .a files
            file(GLOB _OV_ALL_LIBS "${_OV_LIB_DIR}/*.a")
            if(_OV_ALL_LIBS)
                if(APPLE)
                    target_link_libraries(openvino_interface INTERFACE
                        ${_OV_ALL_LIBS} -ldl -lm -lpthread)
                else()
                    target_link_libraries(openvino_interface INTERFACE
                        -Wl,--start-group ${_OV_ALL_LIBS} -Wl,--end-group
                        -lpthread -ldl -lm -lrt)
                endif()
            endif()
        endif()
    endif()

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

