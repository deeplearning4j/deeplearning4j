# patch_openvino.cmake — Download and populate ALL OpenVINO dependencies
#
# Called as PATCH_COMMAND from ExternalProject_Add. All variables are passed
# via -D flags from the parent cmake.
#
# Required variables:
#   SOURCE_DIR   — OpenVINO source tree root
#   DOWNLOAD_DIR — Shared download cache directory
#   TBB_INSTALL_DIR — Where to build/install TBB (passed back to OV cmake as TBBROOT)
#   *_URL / *_COMMIT — Download URL + commit for each submodule tarball
#
# This script:
# 1. Downloads each submodule tarball (if not cached)
# 2. Extracts into the correct path in the OpenVINO source tree
# 3. Downloads and builds TBB from source (so OpenVINO never hits its wget path)
# 4. Patches OpenVINO's cmake to use our TBB

if(NOT SOURCE_DIR OR NOT EXISTS "${SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "patch_openvino: SOURCE_DIR not set or invalid: '${SOURCE_DIR}'")
endif()

message(STATUS "patch_openvino: populating dependencies for ${SOURCE_DIR}")

# ── Helper function: download and extract a dependency ──
# Args: name url dest_path [sha256_hash]
function(_ov_populate_dep name url dest_path)
    set(_full_dest "${SOURCE_DIR}/${dest_path}")
    set(_expected_hash "${ARGN}")  # Optional 4th argument: SHA256 hash

    # Skip if already populated (idempotent)
    if(EXISTS "${_full_dest}/CMakeLists.txt" OR EXISTS "${_full_dest}/include")
        message(STATUS "  [skip] ${name}: already populated")
        return()
    endif()

    get_filename_component(_tarball_name "${url}" NAME)
    set(_tarball "${DOWNLOAD_DIR}/${_tarball_name}")

    if(NOT EXISTS "${_tarball}")
        message(STATUS "  [download] ${name}: ${_tarball_name}")
        set(_dl_args "${url}" "${_tarball}" STATUS _dl_status TIMEOUT 300 SHOW_PROGRESS TLS_VERIFY OFF)
        if(_expected_hash)
            list(APPEND _dl_args EXPECTED_HASH "SHA256=${_expected_hash}")
        endif()
        file(DOWNLOAD ${_dl_args})
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            list(GET _dl_status 1 _dl_msg)
            message(FATAL_ERROR "  ${name}: download FAILED: ${_dl_msg}")
        endif()
    else()
        # Verify cached tarball integrity if hash provided
        if(_expected_hash)
            file(SHA256 "${_tarball}" _actual_hash)
            if(NOT _actual_hash STREQUAL _expected_hash)
                message(STATUS "  [redownload] ${name}: hash mismatch, redownloading")
                file(REMOVE "${_tarball}")
                file(DOWNLOAD "${url}" "${_tarball}"
                     STATUS _dl_status TIMEOUT 300 SHOW_PROGRESS
                     TLS_VERIFY OFF EXPECTED_HASH "SHA256=${_expected_hash}")
                list(GET _dl_status 0 _dl_code)
                if(NOT _dl_code EQUAL 0)
                    list(GET _dl_status 1 _dl_msg)
                    message(FATAL_ERROR "  ${name}: redownload FAILED: ${_dl_msg}")
                endif()
            endif()
        endif()
        message(STATUS "  [cached] ${name}: ${_tarball_name}")
    endif()

    set(_extract_tmp "${SOURCE_DIR}/_extract_tmp_${name}")
    file(MAKE_DIRECTORY "${_extract_tmp}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${_tarball}"
        WORKING_DIRECTORY "${_extract_tmp}"
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "  ${name}: extraction FAILED")
    endif()

    # GitHub tarballs extract to <repo>-<hash>/ — find it
    file(GLOB _extracted_dirs "${_extract_tmp}/*")
    list(LENGTH _extracted_dirs _n)
    if(_n EQUAL 0)
        message(FATAL_ERROR "  ${name}: empty extraction")
    endif()
    list(GET _extracted_dirs 0 _extracted_dir)

    get_filename_component(_dest_parent "${_full_dest}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dest_parent}")
    if(EXISTS "${_full_dest}")
        file(REMOVE_RECURSE "${_full_dest}")
    endif()
    file(RENAME "${_extracted_dir}" "${_full_dest}")
    file(REMOVE_RECURSE "${_extract_tmp}")
    message(STATUS "  [done] ${name} -> ${dest_path}")
endfunction()

# ══════════════════════════════════════════════════════════════════════════
# 1. Populate submodule sources into OpenVINO source tree
# ══════════════════════════════════════════════════════════════════════════

_ov_populate_dep("oneDNN"        "${ONEDNN_URL}"   "src/plugins/intel_cpu/thirdparty/onednn"
    "c6825a7aad5bc83686da759aeb89cf979bd820ba4795b55241fde0d6b093f4d6")
_ov_populate_dep("xbyak"         "${XBYAK_URL}"    "thirdparty/xbyak"
    "18aa05ac8e4bd5f5cb52c5481b86ba02cac395ad8cdbec6fe5acac94ad546dc8")
_ov_populate_dep("pugixml"       "${PUGIXML_URL}"  "thirdparty/pugixml"
    "51c102d4187fac99daa38af281b0772c5e6c586f65004cdc63f8f2e011a21492")
_ov_populate_dep("flatbuffers"   "${FLATBUF_URL}"  "thirdparty/flatbuffers/flatbuffers"
    "987300083ec1f1b095d5596ef8fb657ba46c45d786bc866a5e9029d7590a5e48")
_ov_populate_dep("ittapi"        "${ITTAPI_URL}"   "thirdparty/ittapi/ittapi"
    "62a0802b8680c2dbd6947bf7f020290ea9a7f0b523a846eb189759db956cd44a")
_ov_populate_dep("nlohmann_json" "${JSON_URL}"     "thirdparty/json/nlohmann_json"
    "0dbc5e40a01ff142e7e68c03e85247a4dcede2f592d12d3677dee3664d17975a")
_ov_populate_dep("zlib"          "${ZLIB_URL}"     "thirdparty/zlib/zlib"
    "d9e270d46252734aa49770fbc544125391617956266f220bd63216c834f3a522")

# ══════════════════════════════════════════════════════════════════════════
# 2. Build TBB from source so OpenVINO never tries its wget-based download
# ══════════════════════════════════════════════════════════════════════════

if(NOT TBB_INSTALL_DIR)
    set(TBB_INSTALL_DIR "${SOURCE_DIR}/_tbb_install")
endif()

if(EXISTS "${TBB_INSTALL_DIR}/include/tbb/tbb.h" OR
   EXISTS "${TBB_INSTALL_DIR}/include/oneapi/tbb.h" OR
   EXISTS "${TBB_INSTALL_DIR}/include/tbb/parallel_for.h")
    message(STATUS "  [skip] TBB: already built at ${TBB_INSTALL_DIR}")
else()
    set(_TBB_VERSION "${TBB_VERSION}")
    if(NOT _TBB_VERSION)
        set(_TBB_VERSION "2021.13.1")
    endif()
    set(_TBB_URL "https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v${_TBB_VERSION}.tar.gz")
    set(_TBB_SHA256 "414db3c84c96c05353eb02d4699599fd156d76458283e528805754e45219314b")
    get_filename_component(_tbb_tarball_name "${_TBB_URL}" NAME)
    set(_tbb_tarball "${DOWNLOAD_DIR}/${_tbb_tarball_name}")

    if(NOT EXISTS "${_tbb_tarball}")
        message(STATUS "  [download] TBB ${_TBB_VERSION}")
        file(DOWNLOAD "${_TBB_URL}" "${_tbb_tarball}"
             STATUS _dl_status TIMEOUT 300 SHOW_PROGRESS
             TLS_VERIFY OFF
             EXPECTED_HASH SHA256=${_TBB_SHA256})
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            list(GET _dl_status 1 _dl_msg)
            message(FATAL_ERROR "  TBB: download FAILED: ${_dl_msg}")
        endif()
    else()
        # Verify cached tarball integrity
        file(SHA256 "${_tbb_tarball}" _tbb_actual_hash)
        if(NOT _tbb_actual_hash STREQUAL _TBB_SHA256)
            message(STATUS "  [redownload] TBB: cached tarball hash mismatch, redownloading")
            file(REMOVE "${_tbb_tarball}")
            file(DOWNLOAD "${_TBB_URL}" "${_tbb_tarball}"
                 STATUS _dl_status TIMEOUT 300 SHOW_PROGRESS
                 TLS_VERIFY OFF
                 EXPECTED_HASH SHA256=${_TBB_SHA256})
        endif()
        message(STATUS "  [cached] TBB ${_TBB_VERSION}")
    endif()

    # Extract
    set(_tbb_extract "${SOURCE_DIR}/_tbb_build_src")
    if(EXISTS "${_tbb_extract}")
        file(REMOVE_RECURSE "${_tbb_extract}")
    endif()
    file(MAKE_DIRECTORY "${_tbb_extract}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${_tbb_tarball}"
        WORKING_DIRECTORY "${_tbb_extract}"
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "  TBB: extraction FAILED")
    endif()

    file(GLOB _tbb_src_dirs "${_tbb_extract}/*")
    list(GET _tbb_src_dirs 0 _tbb_src)

    # Patch TBB to disable LTO — GCC 12.x has ICE bugs with -flto on TBB code.
    # TBB's cmake/compilers/GNU.cmake sets TBB_IPO_COMPILE_FLAGS to -flto which
    # overrides any CMAKE_*_FLAGS we pass. Patch it to empty.
    set(_tbb_gnu_cmake "${_tbb_src}/cmake/compilers/GNU.cmake")
    if(EXISTS "${_tbb_gnu_cmake}")
        file(READ "${_tbb_gnu_cmake}" _tbb_gnu_content)
        string(REPLACE
            "set(TBB_IPO_COMPILE_FLAGS $<$<NOT:$<CONFIG:Debug>>:-flto>)"
            "set(TBB_IPO_COMPILE_FLAGS \"\")"
            _tbb_gnu_content "${_tbb_gnu_content}")
        string(REPLACE
            "set(TBB_IPO_LINK_FLAGS $<$<NOT:$<CONFIG:Debug>>:-flto>)"
            "set(TBB_IPO_LINK_FLAGS \"\")"
            _tbb_gnu_content "${_tbb_gnu_content}")
        file(WRITE "${_tbb_gnu_cmake}" "${_tbb_gnu_content}")
        message(STATUS "  [patch] Disabled LTO in TBB cmake/compilers/GNU.cmake")
    endif()

    # Build TBB
    set(_tbb_build "${SOURCE_DIR}/_tbb_build")
    file(MAKE_DIRECTORY "${_tbb_build}")

    # Build ccache args if available
    set(_tbb_ccache_args "")
    if(CCACHE_PATH AND EXISTS "${CCACHE_PATH}")
        list(APPEND _tbb_ccache_args "-DCMAKE_C_COMPILER_LAUNCHER:FILEPATH=${CCACHE_PATH}")
        list(APPEND _tbb_ccache_args "-DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH=${CCACHE_PATH}")
        message(STATUS "  [ccache] TBB using ccache: ${CCACHE_PATH}")
    endif()

    message(STATUS "  [build] TBB ${_TBB_VERSION} (cmake configure)...")
    execute_process(
        COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${TBB_INSTALL_DIR}
            -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
            "-DCMAKE_C_FLAGS=-fno-lto -fno-fat-lto-objects"
            "-DCMAKE_CXX_FLAGS=-fno-lto -fno-fat-lto-objects"
            "-DCMAKE_EXE_LINKER_FLAGS=-fno-lto"
            "-DCMAKE_SHARED_LINKER_FLAGS=-fno-lto"
            ${_tbb_ccache_args}
            -DTBB_TEST=OFF
            -DTBB_EXAMPLES=OFF
            -DTBB_BENCH=OFF
            -DTBB_STRICT=OFF
            -DTBB_ENABLE_IPO=OFF
            -DBUILD_SHARED_LIBS=ON
            "${_tbb_src}"
        WORKING_DIRECTORY "${_tbb_build}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "  TBB cmake configure FAILED:\n${_err}")
    endif()

    # Use PARALLEL_JOBS from parent build (DEP_PARALLEL_JOBS), fall back to ProcessorCount
    if(PARALLEL_JOBS AND PARALLEL_JOBS GREATER 0)
        set(_nproc ${PARALLEL_JOBS})
    else()
        include(ProcessorCount)
        ProcessorCount(_nproc)
        if(_nproc EQUAL 0)
            set(_nproc 4)
        endif()
    endif()

    message(STATUS "  [build] TBB ${_TBB_VERSION} (compiling with ${_nproc} jobs)...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . --config Release --parallel ${_nproc}
        WORKING_DIRECTORY "${_tbb_build}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "  TBB build FAILED:\n${_err}")
    endif()

    message(STATUS "  [build] TBB ${_TBB_VERSION} (installing)...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --install . --config Release
        WORKING_DIRECTORY "${_tbb_build}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "  TBB install FAILED:\n${_err}")
    endif()

    # Cleanup build artifacts (keep install)
    file(REMOVE_RECURSE "${_tbb_extract}")
    file(REMOVE_RECURSE "${_tbb_build}")

    message(STATUS "  [done] TBB ${_TBB_VERSION} installed at ${TBB_INSTALL_DIR}")
endif()

# ══════════════════════════════════════════════════════════════════════════
# 3. Patch OpenVINO cmake to use our TBB instead of downloading
# ══════════════════════════════════════════════════════════════════════════
# OpenVINO checks TBBROOT first. If set, it skips ov_download_tbb() entirely.
# We write a small cmake snippet that sets TBBROOT before dependencies.cmake runs.
# The cleanest way: prepend a set(TBBROOT ...) to cmake/dependencies.cmake.

set(_deps_cmake "${SOURCE_DIR}/cmake/dependencies.cmake")
if(EXISTS "${_deps_cmake}")
    file(READ "${_deps_cmake}" _deps_content)
    # Only patch once (idempotent)
    string(FIND "${_deps_content}" "# PATCHED_BY_LIBND4J" _patch_pos)
    if(_patch_pos EQUAL -1)
        set(_tbb_patch "# PATCHED_BY_LIBND4J: use locally-built TBB\nset(TBBROOT \"${TBB_INSTALL_DIR}\" CACHE PATH \"\" FORCE)\nset(ENV{TBBROOT} \"${TBB_INSTALL_DIR}\")\n\n")
        string(PREPEND _deps_content "${_tbb_patch}")
        file(WRITE "${_deps_cmake}" "${_deps_content}")
        message(STATUS "  [patch] Set TBBROOT=${TBB_INSTALL_DIR} in cmake/dependencies.cmake")
    else()
        message(STATUS "  [skip] cmake/dependencies.cmake already patched")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════
# 4. Patch src/bindings/CMakeLists.txt to skip Python when ENABLE_PYTHON=OFF
# ══════════════════════════════════════════════════════════════════════════
# OpenVINO 2026.0.0 unconditionally does add_subdirectory(python) in
# src/bindings/CMakeLists.txt, which triggers ov_find_python3() and a
# Python >= 3.13 GIL check even when ENABLE_PYTHON=OFF.
# We patch the file to guard the python subdirectory inclusion.

set(_bindings_cmake "${SOURCE_DIR}/src/bindings/CMakeLists.txt")
if(EXISTS "${_bindings_cmake}")
    file(READ "${_bindings_cmake}" _bindings_content)
    string(FIND "${_bindings_content}" "# PATCHED_BY_LIBND4J_PYTHON" _patch_pos)
    if(_patch_pos EQUAL -1)
        # Replace unconditional add_subdirectory(python) with guarded version
        string(REPLACE
            "add_subdirectory(python)"
            "# PATCHED_BY_LIBND4J_PYTHON: guard python bindings\nif(ENABLE_PYTHON)\n    add_subdirectory(python)\nendif()"
            _bindings_content "${_bindings_content}")
        file(WRITE "${_bindings_cmake}" "${_bindings_content}")
        message(STATUS "  [patch] Guarded add_subdirectory(python) in src/bindings/CMakeLists.txt")
    else()
        message(STATUS "  [skip] src/bindings/CMakeLists.txt already patched")
    endif()
else()
    message(STATUS "  [skip] src/bindings/CMakeLists.txt not found (OK for older OpenVINO)")
endif()

message(STATUS "patch_openvino: all dependencies ready")
