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

# Upstream MLIR deliberately resets this option from the native host
# architecture, which disables the execution-engine shared runtime for an
# Android cross build even when the external-project cache receives ON.
if(_sd_external_project STREQUAL "LLVM")
    set(_sd_mlir_options_file "${SOURCE_DIR}/mlir/CMakeLists.txt")
    if(EXISTS "${_sd_mlir_options_file}")
        file(READ "${_sd_mlir_options_file}" _sd_mlir_options_source)
        set(_sd_android_mlir_engine_marker "# SD_ANDROID_MLIR_EXECUTION_ENGINE_V1")
        string(FIND "${_sd_mlir_options_source}" "${_sd_android_mlir_engine_marker}"
            _sd_android_mlir_engine_marker_pos)
        if(_sd_android_mlir_engine_marker_pos EQUAL -1)
            set(_sd_mlir_options_anchor [=[if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE 0)
endif()]=])
            set(_sd_android_mlir_engine_patch [=[
# SD_ANDROID_MLIR_EXECUTION_ENGINE_V1
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
