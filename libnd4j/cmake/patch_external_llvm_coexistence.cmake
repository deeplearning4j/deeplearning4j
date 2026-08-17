# patch_external_llvm_coexistence.cmake
#
# Idempotently applies project-required patches to downloaded LLVM or Triton
# sources: normal coexistence of versioned LLVM/MLIR shared libraries and
# correctness fixes needed by the pinned MLIR conversion pipeline.
#
# Required:
#   SOURCE_DIR           extracted external-project source root
#   SD_EXTERNAL_PROJECT  LLVM or TRITON
# Optional:
#   SD_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP
#                        ON for the pinned GPU/Vulkan LLVM package

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED SOURCE_DIR OR NOT IS_DIRECTORY "${SOURCE_DIR}")
    message(FATAL_ERROR
        "patch_external_llvm_coexistence: SOURCE_DIR is missing or invalid: '${SOURCE_DIR}'")
endif()
if(NOT DEFINED SD_EXTERNAL_PROJECT)
    message(FATAL_ERROR
        "patch_external_llvm_coexistence: SD_EXTERNAL_PROJECT must be LLVM or TRITON")
endif()

string(TOUPPER "${SD_EXTERNAL_PROJECT}" _sd_external_project)
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_target_file
        "${SOURCE_DIR}/llvm/cmake/modules/HandleLLVMOptions.cmake")
    # HandleLLVMOptions is included before LLVM/MLIR targets are created.
    set(_sd_anchor "include(LLVMProcessSources)\n")
elseif(_sd_external_project STREQUAL "TRITON")
    set(_sd_target_file "${SOURCE_DIR}/CMakeLists.txt")
    # Both pinned Triton variants use this project declaration before targets.
    set(_sd_anchor "project(triton CXX C)\n")
else()
    message(FATAL_ERROR
        "patch_external_llvm_coexistence: unsupported SD_EXTERNAL_PROJECT='${SD_EXTERNAL_PROJECT}'")
endif()

if(NOT EXISTS "${_sd_target_file}")
    message(FATAL_ERROR
        "patch_external_llvm_coexistence: expected source file does not exist: ${_sd_target_file}")
endif()

file(READ "${_sd_target_file}" _sd_source)

# Older pinned LLVM snapshots perform the GNU-unique probe unconditionally,
# even when the external project is being configured with the Android Clang
# target compiler.  That compiler intentionally does not accept the host-only
# flag; only GNU host builds need it.  Normalize the legacy block before adding
# the idempotent coexistence marker.  Newer snapshots already contain this
# compiler-aware form, so the replacement is a no-op for them.
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_legacy_unique_block [=[
if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fno-gnu-unique" SD_CXX_SUPPORTS_FNO_GNU_UNIQUE)
  if(NOT SD_CXX_SUPPORTS_FNO_GNU_UNIQUE)
    message(FATAL_ERROR
      "Bundled LLVM/Triton requires C++ compiler support for -fno-gnu-unique "
      "so multiple LLVM/MLIR shared-library versions can coexist")
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-gnu-unique>")
endif()
]=])
    set(_sd_compiler_aware_unique_block [=[
if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fno-gnu-unique" SD_CXX_SUPPORTS_FNO_GNU_UNIQUE)
    if(NOT SD_CXX_SUPPORTS_FNO_GNU_UNIQUE)
      message(FATAL_ERROR
        "Bundled LLVM/Triton requires C++ compiler support for -fno-gnu-unique "
        "so multiple LLVM/MLIR shared-library versions can coexist")
    endif()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-gnu-unique>")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
    message(STATUS
      "Clang emits ordinary weak template statics; -fno-gnu-unique is not required")
  else()
    message(FATAL_ERROR
      "Unsupported C++ compiler for LLVM/MLIR coexistence: ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()
]=])
    string(REPLACE
        "${_sd_legacy_unique_block}"
        "${_sd_compiler_aware_unique_block}"
        _sd_source
        "${_sd_source}")
endif()

set(_sd_unique_marker "# SD_EXTERNAL_LLVM_COEXISTENCE_V1")
string(FIND "${_sd_source}" "${_sd_unique_marker}" _sd_unique_marker_pos)

if(_sd_unique_marker_pos EQUAL -1)
    string(FIND "${_sd_source}" "${_sd_anchor}" _sd_anchor_pos)
    if(_sd_anchor_pos EQUAL -1)
        message(FATAL_ERROR
            "patch_external_llvm_coexistence: expected ${_sd_external_project} anchor was not found in ${_sd_target_file}")
    endif()

    set(_sd_unique_patch [=[
# SD_EXTERNAL_LLVM_COEXISTENCE_V1
# STB_GNU_UNIQUE makes C++ template statics process-global and bypasses normal
# ELF symbol-version selection. LLVM/MLIR versions with distinct SONAMEs must
# retain their own registries, so emit ordinary weak/versioned definitions.
if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fno-gnu-unique" SD_CXX_SUPPORTS_FNO_GNU_UNIQUE)
    if(NOT SD_CXX_SUPPORTS_FNO_GNU_UNIQUE)
      message(FATAL_ERROR
        "Bundled LLVM/Triton requires C++ compiler support for -fno-gnu-unique "
        "so multiple LLVM/MLIR shared-library versions can coexist")
    endif()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-gnu-unique>")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
    message(STATUS
      "Clang emits ordinary weak template statics; -fno-gnu-unique is not required")
  else()
    message(FATAL_ERROR
      "Unsupported C++ compiler for LLVM/MLIR coexistence: ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()
]=])

    string(REPLACE
        "${_sd_anchor}"
        "${_sd_anchor}\n${_sd_unique_patch}\n"
        _sd_patched_source
        "${_sd_source}")
    if(_sd_patched_source STREQUAL _sd_source)
        message(FATAL_ERROR
            "patch_external_llvm_coexistence: failed to modify ${_sd_target_file}")
    endif()
    file(WRITE "${_sd_target_file}" "${_sd_patched_source}")
else()
    message(STATUS
        "External ${_sd_external_project} GNU-unique patch already applied: ${_sd_target_file}")
endif()

# Android's cross toolchain can cause LLVM's dependent shared-library options
# to remain disabled even when the external-project cache receives ON values.
# The Vulkan/SDX consumer requires the monolithic LLVM/MLIR DSOs and the
# MLIRExecutionEngineShared target, so force the shared-runtime contract in the
# downloaded LLVM project before its target directories are configured.
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_llvm_options_file "${SOURCE_DIR}/llvm/CMakeLists.txt")
    if(EXISTS "${_sd_llvm_options_file}")
        file(READ "${_sd_llvm_options_file}" _sd_llvm_options_source)
        set(_sd_android_shared_marker "# SD_ANDROID_LLVM_SHARED_RUNTIME_V1")
        string(FIND "${_sd_llvm_options_source}" "${_sd_android_shared_marker}"
            _sd_android_shared_marker_pos)
        if(_sd_android_shared_marker_pos EQUAL -1)
            set(_sd_llvm_options_anchor [=[cmake_dependent_option(LLVM_BUILD_LLVM_DYLIB "Build libllvm dynamic library" ${LLVM_BUILD_LLVM_DYLIB_default}
                       "CAN_BUILD_LLVM_DYLIB" OFF)
]=])
            set(_sd_android_shared_patch [=[
# SD_ANDROID_LLVM_SHARED_RUNTIME_V1
if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(CAN_BUILD_LLVM_DYLIB ON CACHE BOOL "" FORCE)
    set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "Build libllvm dynamic library" FORCE)
    set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "Link tools against the libllvm dynamic library" FORCE)
    set(MLIR_BUILD_MLIR_DYLIB ON CACHE BOOL "Build MLIR dynamic library" FORCE)
    set(MLIR_LINK_MLIR_DYLIB ON CACHE BOOL "Link tools against the MLIR dynamic library" FORCE)
endif()
]=])
            string(REPLACE
                "${_sd_llvm_options_anchor}"
                "${_sd_llvm_options_anchor}
${_sd_android_shared_patch}
"
                _sd_llvm_options_patched
                "${_sd_llvm_options_source}")
            if(_sd_llvm_options_patched STREQUAL _sd_llvm_options_source)
                message(FATAL_ERROR
                    "patch_external_llvm_coexistence: failed to enable Android LLVM shared runtime")
            endif()
            file(WRITE "${_sd_llvm_options_file}" "${_sd_llvm_options_patched}")
        endif()
    endif()
endif()

# The project-managed runtime contract requires MLIRExecutionEngineShared on
# Windows as well as ELF hosts. Upstream currently excludes that shared target
# from WIN32/MINGW builds; the existing MinGW export and ArmSME patches make the
# producer-side Windows build viable, so remove only that platform exclusion.
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_mlir_execution_engine_file
        "${SOURCE_DIR}/mlir/lib/ExecutionEngine/CMakeLists.txt")
    if(EXISTS "${_sd_mlir_execution_engine_file}")
        file(READ "${_sd_mlir_execution_engine_file}"
            _sd_mlir_execution_engine_source)
        set(_sd_mlir_execution_engine_marker
            "# SD_WINDOWS_MLIR_EXECUTION_ENGINE_SHARED_V1")
        string(FIND "${_sd_mlir_execution_engine_source}"
            "${_sd_mlir_execution_engine_marker}"
            _sd_mlir_execution_engine_marker_pos)
        if(_sd_mlir_execution_engine_marker_pos EQUAL -1)
            set(_sd_mlir_execution_engine_anchor
                "if(LLVM_BUILD_LLVM_DYLIB AND NOT (WIN32 OR MINGW OR CYGWIN))")
            set(_sd_mlir_execution_engine_patch [=[
# SD_WINDOWS_MLIR_EXECUTION_ENGINE_SHARED_V1
if(LLVM_BUILD_LLVM_DYLIB)
]=])
            string(REPLACE
                "${_sd_mlir_execution_engine_anchor}"
                "${_sd_mlir_execution_engine_patch}"
                _sd_mlir_execution_engine_patched
                "${_sd_mlir_execution_engine_source}")
            if(_sd_mlir_execution_engine_patched STREQUAL
               _sd_mlir_execution_engine_source)
                message(FATAL_ERROR
                    "patch_external_llvm_coexistence: failed to enable "
                    "MLIRExecutionEngineShared on Windows")
            endif()
            file(WRITE "${_sd_mlir_execution_engine_file}"
                "${_sd_mlir_execution_engine_patched}")
        endif()
    endif()
endif()

# Upstream MLIR deliberately resets this option from the native host
# architecture, which disables the execution-engine shared runtime for an
# Android cross build even when the external-project cache receives ON.
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_mlir_options_file "${SOURCE_DIR}/mlir/CMakeLists.txt")
    if(EXISTS "${_sd_mlir_options_file}")
        file(READ "${_sd_mlir_options_file}" _sd_mlir_options_source)
        set(_sd_android_mlir_engine_marker "# SD_ANDROID_MLIR_EXECUTION_ENGINE_V2")
        string(FIND "${_sd_mlir_options_source}" "${_sd_android_mlir_engine_marker}"
            _sd_android_mlir_engine_marker_pos)
        if(_sd_android_mlir_engine_marker_pos EQUAL -1)
            # The two pinned LLVM snapshots use different complete option
            # blocks. Patch after the whole conditional, never after an individual
            # assignment that a following else() branch can overwrite.
            set(_sd_mlir_direct_options_anchor [=[if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE 0)
endif()]=])
            set(_sd_mlir_legacy_options_anchor [=[if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE_default 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE_default 0)
endif()
option(MLIR_ENABLE_EXECUTION_ENGINE
       "Enable building the MLIR Execution Engine."
       ${MLIR_ENABLE_EXECUTION_ENGINE_default})]=])
            set(_sd_mlir_options_anchor "")
            string(FIND "${_sd_mlir_options_source}"
                "${_sd_mlir_direct_options_anchor}"
                _sd_mlir_direct_options_anchor_pos)
            if(NOT _sd_mlir_direct_options_anchor_pos EQUAL -1)
                set(_sd_mlir_options_anchor "${_sd_mlir_direct_options_anchor}")
            else()
                string(FIND "${_sd_mlir_options_source}"
                    "${_sd_mlir_legacy_options_anchor}"
                    _sd_mlir_legacy_options_anchor_pos)
                if(NOT _sd_mlir_legacy_options_anchor_pos EQUAL -1)
                    set(_sd_mlir_options_anchor "${_sd_mlir_legacy_options_anchor}")
                else()
                    message(FATAL_ERROR
                        "patch_external_llvm_coexistence: unsupported MLIR execution-engine option layout")
                endif()
            endif()
            set(_sd_android_mlir_engine_patch [=[
# SD_ANDROID_MLIR_EXECUTION_ENGINE_V2
if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1 CACHE BOOL "Enable MLIR execution engine" FORCE)
endif()
]=])
            string(REPLACE
                "${_sd_mlir_options_anchor}"
                "${_sd_mlir_options_anchor}
${_sd_android_mlir_engine_patch}"
                _sd_mlir_options_patched
                "${_sd_mlir_options_source}")
            if(_sd_mlir_options_patched STREQUAL _sd_mlir_options_source)
                message(FATAL_ERROR
                    "patch_external_llvm_coexistence: failed to enable Android MLIR execution engine")
            endif()
            file(WRITE "${_sd_mlir_options_file}" "${_sd_mlir_options_patched}")
        endif()
    endif()
endif()

# The Android NDK libc headers do not expose aligned_alloc even though the
# pinned MLIR execution-engine runtime uses it for its generic aligned allocator.
# Use POSIX memalign on Android; this preserves the allocator contract without
# changing host or Apple builds.
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_crunner_utils_file
        "${SOURCE_DIR}/mlir/lib/ExecutionEngine/CRunnerUtils.cpp")
    if(EXISTS "${_sd_crunner_utils_file}")
        file(READ "${_sd_crunner_utils_file}" _sd_crunner_utils_source)
        set(_sd_android_aligned_alloc_marker
            "// SD_ANDROID_CRUNNERUTILS_ALIGNED_ALLOC_V1")
        string(FIND "${_sd_crunner_utils_source}"
            "${_sd_android_aligned_alloc_marker}"
            _sd_android_aligned_alloc_marker_pos)
        if(_sd_android_aligned_alloc_marker_pos EQUAL -1)
            set(_sd_android_aligned_alloc_anchor [=[#elif defined(__APPLE__)
  // aligned_alloc was added in MacOS 10.15. Fall back to posix_memalign to also
  // support older versions.
]=])
            set(_sd_android_aligned_alloc_patch [=[#elif defined(__APPLE__) || defined(__ANDROID__)
  // SD_ANDROID_CRUNNERUTILS_ALIGNED_ALLOC_V1
  // aligned_alloc is not exposed by the Android NDK libc headers; use the POSIX
  // allocator with the same alignment contract.
]=])
            string(REPLACE
                "${_sd_android_aligned_alloc_anchor}"
                "${_sd_android_aligned_alloc_patch}"
                _sd_crunner_utils_patched
                "${_sd_crunner_utils_source}")
            if(_sd_crunner_utils_patched STREQUAL _sd_crunner_utils_source)
                message(FATAL_ERROR
                    "patch_external_llvm_coexistence: failed to patch Android "
                    "aligned allocation in ${_sd_crunner_utils_file}")
            endif()
            file(WRITE "${_sd_crunner_utils_file}"
                "${_sd_crunner_utils_patched}")
        endif()
    endif()
endif()

# MinGW's object-library export define is not propagated to this upstream MLIR
# source, so ArmSMEStubs.cpp sees __declspec(dllimport) while it is compiling
# the stub definitions themselves.  Vulkan does not consume the Arm SME runtime,
# but the shared MLIR execution-engine build still compiles this source on
# Windows.  Normalize the producer-side export annotation without changing
# native ELF or MSVC behavior.
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_arm_sme_stubs_file
        "${SOURCE_DIR}/mlir/lib/ExecutionEngine/ArmSMEStubs.cpp")
    if(EXISTS "${_sd_arm_sme_stubs_file}")
        file(READ "${_sd_arm_sme_stubs_file}" _sd_arm_sme_stubs_source)
        set(_sd_mingw_arm_sme_marker "// SD_MINGW_ARMSME_EXPORT_V1")
        string(FIND "${_sd_arm_sme_stubs_source}"
            "${_sd_mingw_arm_sme_marker}"
            _sd_mingw_arm_sme_marker_pos)
        if(_sd_mingw_arm_sme_marker_pos EQUAL -1)
            set(_sd_mingw_arm_sme_anchor
                "#endif // (defined(_WIN32) || defined(__CYGWIN__))\n")
            set(_sd_mingw_arm_sme_patch [=[
#if defined(_WIN32) && defined(__GNUC__)
// SD_MINGW_ARMSME_EXPORT_V1
#undef MLIR_ARMSMEABISTUBS_EXPORTED
#define MLIR_ARMSMEABISTUBS_EXPORTED __declspec(dllexport)
#endif
]=])
            string(REPLACE
                "${_sd_mingw_arm_sme_anchor}"
                "${_sd_mingw_arm_sme_anchor}${_sd_mingw_arm_sme_patch}\n"
                _sd_arm_sme_stubs_patched
                "${_sd_arm_sme_stubs_source}")
            if(_sd_arm_sme_stubs_patched STREQUAL _sd_arm_sme_stubs_source)
                message(FATAL_ERROR
                    "patch_external_llvm_coexistence: failed to patch MinGW "
                    "ArmSME export annotations in ${_sd_arm_sme_stubs_file}")
            endif()
            file(WRITE "${_sd_arm_sme_stubs_file}"
                "${_sd_arm_sme_stubs_patched}")
        endif()
    endif()
endif()

# LLVM's MinGW shared-library recipe explicitly enables --export-all-symbols.
# That defeats the hidden-visibility settings above and can exceed PE/COFF's
# 65,535-export ordinal limit when LLVM is linked from its component archives.
# Keep the target's intentional exports while preventing archive-wide auto-export.
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_mingw_llvm_shlib_file
        "${SOURCE_DIR}/llvm/tools/llvm-shlib/CMakeLists.txt")
    if(EXISTS "${_sd_mingw_llvm_shlib_file}")
        file(READ "${_sd_mingw_llvm_shlib_file}" _sd_mingw_llvm_shlib_source)
        set(_sd_mingw_llvm_export_marker "# SD_MINGW_LLVM_EXPORT_LIMIT_V1")
        string(FIND "${_sd_mingw_llvm_shlib_source}"
            "${_sd_mingw_llvm_export_marker}" _sd_mingw_llvm_export_marker_pos)
        if(_sd_mingw_llvm_export_marker_pos EQUAL -1)
            set(_sd_mingw_llvm_export_flag
                "target_link_options(LLVM PRIVATE LINKER:--export-all-symbols)")
            string(FIND "${_sd_mingw_llvm_shlib_source}"
                "${_sd_mingw_llvm_export_flag}" _sd_mingw_llvm_export_flag_pos)
            if(NOT _sd_mingw_llvm_export_flag_pos EQUAL -1)
                set(_sd_mingw_llvm_export_patch [=[
# SD_MINGW_LLVM_EXPORT_LIMIT_V1
  target_link_options(LLVM PRIVATE LINKER:--exclude-all-symbols)
]=])
                string(REPLACE
                    "${_sd_mingw_llvm_export_flag}"
                    "${_sd_mingw_llvm_export_patch}"
                    _sd_mingw_llvm_shlib_patched
                    "${_sd_mingw_llvm_shlib_source}")
                if(_sd_mingw_llvm_shlib_patched STREQUAL _sd_mingw_llvm_shlib_source)
                    message(FATAL_ERROR
                        "patch_external_llvm_coexistence: failed to patch MinGW LLVM export limit in ${_sd_mingw_llvm_shlib_file}")
                endif()
                file(WRITE "${_sd_mingw_llvm_shlib_file}"
                    "${_sd_mingw_llvm_shlib_patched}")
            endif()
        endif()
    endif()
endif()

# The pinned MLIR SCF-to-SPIR-V conversion represents scf.for results with
# Function-scope variables. Upstream initializes those variables only from
# scf.yield in the loop body, so a zero-trip loop reads undefined data instead
# of returning its init args. Apply the general SCF conversion fix to downloaded
# LLVM sources; no Vulkan op or machine-specific behavior belongs here.
if(_sd_external_project STREQUAL "LLVM" AND
   SD_LLVM_PATCH_SCF_TO_SPIRV_ZERO_TRIP)
    set(_sd_scf_to_spirv_file
        "${SOURCE_DIR}/mlir/lib/Conversion/SCFToSPIRV/SCFToSPIRV.cpp")
    if(NOT EXISTS "${_sd_scf_to_spirv_file}")
        message(FATAL_ERROR
            "patch_external_llvm_coexistence: expected MLIR source file does not exist: ${_sd_scf_to_spirv_file}")
    endif()

    file(READ "${_sd_scf_to_spirv_file}" _sd_scf_source)
    set(_sd_scf_zero_trip_marker "// SD_SCF_TO_SPIRV_ZERO_TRIP_INIT_V3")
    set(_sd_scf_zero_trip_v2_marker "// SD_SCF_TO_SPIRV_ZERO_TRIP_INIT_V2")
    set(_sd_scf_zero_trip_legacy_marker "// SD_SCF_TO_SPIRV_ZERO_TRIP_INIT_V1")
    string(FIND "${_sd_scf_source}" "${_sd_scf_zero_trip_marker}"
        _sd_scf_zero_trip_marker_pos)
    if(_sd_scf_zero_trip_marker_pos EQUAL -1)
        set(_sd_scf_anchor [=[    replaceSCFOutputValue(forOp, loopOp, rewriter, scfToSPIRVContext,
                          initTypes);]=])
        set(_sd_scf_zero_trip_block [=[
    // SD_SCF_TO_SPIRV_ZERO_TRIP_INIT_V3
    // A zero-trip scf.for must produce its init args. The SPIR-V serializer
    // deliberately omits spirv.mlir.loop's structural entry block, so initialize
    // the result variables in the enclosing block immediately before the loop.
    // scf.yield stores overwrite them after each real iteration.
    auto &resultAllocas = scfToSPIRVContext->outputVars[loopOp];
    rewriter.setInsertionPoint(loopOp);
    for (const auto &it : llvm::enumerate(adaptor.getInitArgs()))
      spirv::StoreOp::create(rewriter, loc, resultAllocas[it.index()],
                             it.value());]=])
        set(_sd_scf_zero_trip_v2_block [=[
    // SD_SCF_TO_SPIRV_ZERO_TRIP_INIT_V2
    // A zero-trip scf.for must produce its init args. The structured loop entry
    // block always executes, even when the header immediately branches to merge.
    // Seed result variables there so later SPIR-V canonicalization retains the
    // initialization; scf.yield stores overwrite it after each real iteration.
    auto &resultAllocas = scfToSPIRVContext->outputVars[loopOp];
    Block &entryBlock = loopOp.getBody().front();
    rewriter.setInsertionPoint(entryBlock.getTerminator());
    for (const auto &it : llvm::enumerate(adaptor.getInitArgs()))
      spirv::StoreOp::create(rewriter, loc, resultAllocas[it.index()],
                             it.value());]=])
        string(FIND "${_sd_scf_source}" "${_sd_scf_zero_trip_v2_marker}"
            _sd_scf_zero_trip_v2_marker_pos)
        string(FIND "${_sd_scf_source}" "${_sd_scf_zero_trip_legacy_marker}"
            _sd_scf_zero_trip_legacy_marker_pos)
        if(NOT _sd_scf_zero_trip_v2_marker_pos EQUAL -1)
            string(REPLACE
                "${_sd_scf_zero_trip_v2_block}"
                "${_sd_scf_zero_trip_block}"
                _sd_scf_patched_source
                "${_sd_scf_source}")
        elseif(NOT _sd_scf_zero_trip_legacy_marker_pos EQUAL -1)
            set(_sd_scf_legacy_block [=[
    // SD_SCF_TO_SPIRV_ZERO_TRIP_INIT_V1
    // A zero-trip scf.for must produce its init args. Seed the result variables
    // before the loop; scf.yield stores overwrite them after each iteration.
    auto &resultAllocas = scfToSPIRVContext->outputVars[loopOp];
    rewriter.setInsertionPoint(loopOp);
    for (const auto &it : llvm::enumerate(adaptor.getInitArgs()))
      spirv::StoreOp::create(rewriter, loc, resultAllocas[it.index()],
                             it.value());]=])
            string(REPLACE
                "${_sd_scf_legacy_block}"
                "${_sd_scf_zero_trip_block}"
                _sd_scf_patched_source
                "${_sd_scf_source}")
        else()
            string(FIND "${_sd_scf_source}" "${_sd_scf_anchor}" _sd_scf_anchor_pos)
            if(_sd_scf_anchor_pos EQUAL -1)
                message(FATAL_ERROR
                    "patch_external_llvm_coexistence: expected SCF-to-SPIR-V for-loop anchor was not found in ${_sd_scf_to_spirv_file}")
            endif()
            string(REPLACE
                "${_sd_scf_anchor}"
                "${_sd_scf_anchor}\n${_sd_scf_zero_trip_block}"
                _sd_scf_patched_source
                "${_sd_scf_source}")
        endif()
        if(_sd_scf_patched_source STREQUAL _sd_scf_source)
            message(FATAL_ERROR
                "patch_external_llvm_coexistence: failed to patch zero-trip SCF loop results in ${_sd_scf_to_spirv_file}")
        endif()
        file(WRITE "${_sd_scf_to_spirv_file}" "${_sd_scf_patched_source}")
    else()
        message(STATUS
            "External LLVM zero-trip SCF-to-SPIR-V patch already applied: ${_sd_scf_to_spirv_file}")
    endif()
endif()

# Symbol hiding is not a coexistence mechanism. Each versioned LLVM/MLIR DSO
# must remain a normal dynamically linked library with its public symbols
# available to the process. Remove upstream link flags that turn dependency
# archives into private implementation details.
file(GLOB_RECURSE _sd_cmake_files LIST_DIRECTORIES FALSE
    "${SOURCE_DIR}/CMakeLists.txt"
    "${SOURCE_DIR}/*.cmake")
set(_sd_symbol_hiding_files 0)
foreach(_sd_cmake_file IN LISTS _sd_cmake_files)
    file(READ "${_sd_cmake_file}" _sd_cmake_source)
    set(_sd_cleaned_source "${_sd_cmake_source}")
    string(REPLACE
        "$<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>"
        ""
        _sd_cleaned_source
        "${_sd_cleaned_source}")
    string(REPLACE
        "-Wl,--exclude-libs,ALL"
        ""
        _sd_cleaned_source
        "${_sd_cleaned_source}")
    string(REPLACE
        "LINKER:--exclude-libs,ALL"
        ""
        _sd_cleaned_source
        "${_sd_cleaned_source}")

    string(FIND "${_sd_cleaned_source}" "--exclude-libs" _sd_remaining_symbol_hiding)
    if(NOT _sd_remaining_symbol_hiding EQUAL -1)
        message(FATAL_ERROR
            "patch_external_llvm_coexistence: unsupported symbol-hiding flag remains in ${_sd_cmake_file}")
    endif()

    if(NOT _sd_cleaned_source STREQUAL _sd_cmake_source)
        file(WRITE "${_sd_cmake_file}" "${_sd_cleaned_source}")
        math(EXPR _sd_symbol_hiding_files "${_sd_symbol_hiding_files} + 1")
    endif()
endforeach()

message(STATUS
    "Patched downloaded ${_sd_external_project} for normal dynamic multi-LLVM coexistence; "
    "removed symbol-hiding flags from ${_sd_symbol_hiding_files} CMake file(s)")
