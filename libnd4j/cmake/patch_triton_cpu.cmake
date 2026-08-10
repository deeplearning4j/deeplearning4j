# patch_triton_cpu.cmake — Strip Python dependency, populate submodules,
#                          and remove ALL GPU code for CPU-only build.
#
# Called as PATCH_COMMAND from ExternalProject_Add.
# Required variables (passed via -D flags):
#   SOURCE_DIR        — triton-cpu source tree root
#   LLVM_INSTALL_DIR  — Where LLVM/MLIR is installed (for include paths)
#   DOWNLOAD_DIR      — Shared download cache directory

if(NOT SOURCE_DIR OR NOT EXISTS "${SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "patch_triton_cpu: SOURCE_DIR not set or invalid: '${SOURCE_DIR}'")
endif()
if(NOT DOWNLOAD_DIR)
    get_filename_component(DOWNLOAD_DIR "${SOURCE_DIR}/../../downloads" ABSOLUTE)
endif()

message(STATUS "patch_triton_cpu: patching ${SOURCE_DIR}")

# Cross builds must execute host generators, not Android target binaries.
# Apply these idempotent fixes before the legacy patch marker check so an
# already-populated ExternalProject source tree can pick up build-recipe fixes.
set(_TRITON_ROOT_CMAKE "${SOURCE_DIR}/CMakeLists.txt")
file(READ "${_TRITON_ROOT_CMAKE}" _cross_content)
string(FIND "${_cross_content}" "PATCHED_BY_LIBND4J_HOST_TABLEGEN" _host_tablegen_patch_pos)
if(_host_tablegen_patch_pos EQUAL -1)
    set(_mlir_find_anchor "find_package(MLIR REQUIRED CONFIG PATHS \${MLIR_DIR})")
    set(_mlir_find_replacement [=[find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
# PATCHED_BY_LIBND4J_HOST_TABLEGEN: generated sources are build-host work.
if(CMAKE_CROSSCOMPILING)
  # ExternalProject forwards LLVM_NATIVE_TOOL_DIR more reliably than custom
  # cache variables. Derive the explicit host generator paths from it before
  # validating them, then fall back to PATH for preinstalled host tools.
  if((NOT LLVM_HOST_TABLEGEN OR NOT EXISTS "${LLVM_HOST_TABLEGEN}") AND LLVM_NATIVE_TOOL_DIR)
    set(LLVM_HOST_TABLEGEN "${LLVM_NATIVE_TOOL_DIR}/llvm-tblgen")
  endif()
  if((NOT MLIR_HOST_TABLEGEN OR NOT EXISTS "${MLIR_HOST_TABLEGEN}") AND LLVM_NATIVE_TOOL_DIR)
    set(MLIR_HOST_TABLEGEN "${LLVM_NATIVE_TOOL_DIR}/mlir-tblgen")
  endif()
  if(NOT EXISTS "${LLVM_HOST_TABLEGEN}")
    find_program(_LIBND4J_LLVM_HOST_TABLEGEN NAMES llvm-tblgen
      HINTS "${LLVM_NATIVE_TOOL_DIR}" NO_DEFAULT_PATH)
    if(_LIBND4J_LLVM_HOST_TABLEGEN)
      set(LLVM_HOST_TABLEGEN "${_LIBND4J_LLVM_HOST_TABLEGEN}")
    endif()
  endif()
  if(NOT EXISTS "${MLIR_HOST_TABLEGEN}")
    find_program(_LIBND4J_MLIR_HOST_TABLEGEN NAMES mlir-tblgen
      HINTS "${LLVM_NATIVE_TOOL_DIR}" NO_DEFAULT_PATH)
    if(_LIBND4J_MLIR_HOST_TABLEGEN)
      set(MLIR_HOST_TABLEGEN "${_LIBND4J_MLIR_HOST_TABLEGEN}")
    endif()
  endif()
  if(NOT EXISTS "${LLVM_HOST_TABLEGEN}" OR NOT EXISTS "${MLIR_HOST_TABLEGEN}")
    message(FATAL_ERROR
      "Cross-building triton-cpu requires LLVM_HOST_TABLEGEN and "
      "MLIR_HOST_TABLEGEN to name executable host tools")
  endif()
  set(LLVM_TABLEGEN_EXE "${LLVM_HOST_TABLEGEN}")
  set(MLIR_TABLEGEN_EXE "${MLIR_HOST_TABLEGEN}")
endif()
]=])
    string(FIND "${_cross_content}" "${_mlir_find_anchor}" _mlir_find_pos)
    if(_mlir_find_pos EQUAL -1)
        message(FATAL_ERROR
            "patch_triton_cpu: could not find MLIR package anchor for host TableGen override")
    endif()
    string(REPLACE "${_mlir_find_anchor}" "${_mlir_find_replacement}"
        _cross_content "${_cross_content}")
endif()

# Android's libc++ has std::filesystem in the standard library and provides no
# separate libstdc++fs compatibility archive. ExternalProject's nested Triton
# configure does not always export the ANDROID variable, so check the platform
# name as well.
string(REPLACE
    "if (NOT WIN32 AND NOT APPLE AND NOT BSD AND NOT ANDROID)"
    "if (NOT WIN32 AND NOT APPLE AND NOT BSD AND NOT (ANDROID OR CMAKE_SYSTEM_NAME STREQUAL \"Android\"))"
    _cross_content "${_cross_content}")
string(REPLACE
    "if (NOT WIN32 AND NOT APPLE AND NOT BSD)"
    "if (NOT WIN32 AND NOT APPLE AND NOT BSD AND NOT (ANDROID OR CMAKE_SYSTEM_NAME STREQUAL \"Android\"))"
    _cross_content "${_cross_content}")
file(WRITE "${_TRITON_ROOT_CMAKE}" "${_cross_content}")

# ══════════════════════════════════════════════════════════════════════════
# 0. Populate SLEEF submodule (vectorized math library, required by cpu backend)
# ══════════════════════════════════════════════════════════════════════════
set(_SLEEF_DEST "${SOURCE_DIR}/third_party/sleef")
if(EXISTS "${_SLEEF_DEST}/CMakeLists.txt")
    message(STATUS "  [skip] SLEEF: already populated")
else()
    set(_SLEEF_VERSION "3.8")
    set(_SLEEF_URL "https://github.com/shibatch/sleef/archive/refs/tags/${_SLEEF_VERSION}.tar.gz")
    set(_SLEEF_TARBALL "${DOWNLOAD_DIR}/sleef-${_SLEEF_VERSION}.tar.gz")
    set(_SLEEF_EXPECTED_HASH "a12ccd50f57083c530e1c76f10d52865defbd19fc9e2c85b483493065709874a")

    if(EXISTS "${_SLEEF_TARBALL}")
        file(SHA256 "${_SLEEF_TARBALL}" _sleef_actual_hash)
        if(NOT "${_sleef_actual_hash}" STREQUAL "${_SLEEF_EXPECTED_HASH}")
            message(STATUS "  [hash mismatch] SLEEF: re-downloading")
            file(REMOVE "${_SLEEF_TARBALL}")
        else()
            message(STATUS "  [cached] SLEEF ${_SLEEF_VERSION} (hash OK)")
        endif()
    endif()

    if(NOT EXISTS "${_SLEEF_TARBALL}")
        message(STATUS "  [download] SLEEF ${_SLEEF_VERSION}")
        file(DOWNLOAD "${_SLEEF_URL}" "${_SLEEF_TARBALL}"
             STATUS _dl_status TIMEOUT 300 SHOW_PROGRESS TLS_VERIFY OFF
             EXPECTED_HASH SHA256=${_SLEEF_EXPECTED_HASH})
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            list(GET _dl_status 1 _dl_msg)
            message(FATAL_ERROR "  SLEEF: download FAILED: ${_dl_msg}")
        endif()
    endif()

    set(_SLEEF_TMP "${SOURCE_DIR}/_sleef_extract")
    file(MAKE_DIRECTORY "${_SLEEF_TMP}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${_SLEEF_TARBALL}"
        WORKING_DIRECTORY "${_SLEEF_TMP}"
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "  SLEEF: extraction FAILED")
    endif()

    file(GLOB _sleef_dirs "${_SLEEF_TMP}/*")
    list(GET _sleef_dirs 0 _sleef_extracted)
    if(EXISTS "${_SLEEF_DEST}")
        file(REMOVE_RECURSE "${_SLEEF_DEST}")
    endif()
    file(RENAME "${_sleef_extracted}" "${_SLEEF_DEST}")
    file(REMOVE_RECURSE "${_SLEEF_TMP}")
    message(STATUS "  [done] SLEEF ${_SLEEF_VERSION} -> third_party/sleef")
endif()

# ══════════════════════════════════════════════════════════════════════════
# 1. Pre-generate the triton-third-party-vars.cmake file
# ══════════════════════════════════════════════════════════════════════════
set(_VARS_FILE "${SOURCE_DIR}/triton-third-party-vars-pregenerated.cmake")

set(_JSON_DIR "${SOURCE_DIR}/third_party/json/include")
if(NOT EXISTS "${_JSON_DIR}")
    set(_JSON_DIR "${SOURCE_DIR}/third_party/nlohmann_json/include")
endif()

set(_VARS_CONTENT "")
string(APPEND _VARS_CONTENT "# Pre-generated by patch_triton_cpu.cmake (no Python needed)\n")
string(APPEND _VARS_CONTENT "if(NOT DEFINED LLVM_INCLUDE_DIRS OR \"\${LLVM_INCLUDE_DIRS}\" STREQUAL \"\")\n")
string(APPEND _VARS_CONTENT "  set(LLVM_INCLUDE_DIRS \"${LLVM_INSTALL_DIR}/include\")\n")
string(APPEND _VARS_CONTENT "endif()\n")
string(APPEND _VARS_CONTENT "if(NOT DEFINED LLVM_LIBRARY_DIR OR \"\${LLVM_LIBRARY_DIR}\" STREQUAL \"\")\n")
string(APPEND _VARS_CONTENT "  set(LLVM_LIBRARY_DIR \"${LLVM_INSTALL_DIR}/lib\")\n")
string(APPEND _VARS_CONTENT "endif()\n")
string(APPEND _VARS_CONTENT "if(NOT DEFINED JSON_INCLUDE_DIR OR \"\${JSON_INCLUDE_DIR}\" STREQUAL \"\")\n")
string(APPEND _VARS_CONTENT "  set(JSON_INCLUDE_DIR \"${_JSON_DIR}\")\n")
string(APPEND _VARS_CONTENT "endif()\n")
file(WRITE "${_VARS_FILE}" "${_VARS_CONTENT}")
message(STATUS "  [done] Pre-generated ${_VARS_FILE}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Patch CMakeLists.txt — strip Python, GPU, tests, examples, proton, bin
# ══════════════════════════════════════════════════════════════════════════
set(_CMAKELISTS "${SOURCE_DIR}/CMakeLists.txt")
file(STRINGS "${_CMAKELISTS}" _lines)

file(READ "${_CMAKELISTS}" _raw_content)
string(FIND "${_raw_content}" "PATCHED_BY_LIBND4J_NO_PYTHON" _patch_pos)
if(NOT _patch_pos EQUAL -1)
    message(STATUS "  [skip] CMakeLists.txt already patched")
    message(STATUS "patch_triton_cpu: done")
    return()
endif()

set(_output "")
set(_in_python_block FALSE)
set(_python_block_done FALSE)
set(_in_add_triton_object FALSE)
foreach(_line IN LISTS _lines)
    # --- Strip Python block ---
    if(NOT _python_block_done)
        if(_line MATCHES "^find_package\\(Python3 REQUIRED COMPONENTS Interpreter\\)")
            set(_in_python_block TRUE)
            string(APPEND _output "# PATCHED_BY_LIBND4J_NO_PYTHON: use pre-generated vars instead of Python\n")
            string(APPEND _output "set(TRITON_THIRD_PARTY_CMAKE_VARS_FILE \"${_VARS_FILE}\")\n")
            string(APPEND _output "include(\"\${TRITON_THIRD_PARTY_CMAKE_VARS_FILE}\")\n")
            continue()
        endif()
        if(_in_python_block)
            if(_line MATCHES "^include\\(.*TRITON_THIRD_PARTY_CMAKE_VARS_FILE")
                set(_in_python_block FALSE)
                set(_python_block_done TRUE)
            endif()
            continue()
        endif()
    endif()

    # --- Disable ccache (we pass our own) ---
    if(_line MATCHES "option\\(TRITON_BUILD_WITH_CCACHE.*ON\\)")
        string(REPLACE "ON)" "OFF)" _line "${_line}")
    endif()

    # --- Strip everything GPU/non-CPU ---
    if(_line MATCHES "^add_subdirectory\\(examples\\)")
        string(APPEND _output "# PATCHED: stripped (GPU targets)\n")
        continue()
    endif()
    if(_line MATCHES "^add_subdirectory\\(test\\)")
        string(APPEND _output "# PATCHED: stripped (needs Python lit)\n")
        continue()
    endif()
    if(_line MATCHES "add_subdirectory\\(third_party/proton")
        string(APPEND _output "# PATCHED: stripped (GPU profiler)\n")
        continue()
    endif()
    if(_line MATCHES "^add_subdirectory\\(bin\\)")
        string(APPEND _output "# PATCHED: stripped (links GPU test targets)\n")
        continue()
    endif()
    if(_line MATCHES "list\\(APPEND TRITON_PLUGIN_NAMES.*proton")
        string(APPEND _output "# PATCHED: stripped proton plugin\n")
        continue()
    endif()
    # Strip lib/ entirely — we replace it with a CPU-only version below
    if(_line MATCHES "^add_subdirectory\\(lib\\)")
        string(APPEND _output "# PATCHED: original lib/ stripped, CPU-only subset below\n")
        string(APPEND _output "add_subdirectory(lib/Dialect/Triton)\n")
        string(APPEND _output "add_subdirectory(lib/Dialect/TritonCPU)\n")
        string(APPEND _output "add_subdirectory(lib/Tools)\n")
        continue()
    endif()

    # --- Patch add_triton_object: inject tablegen deps ---
    if(_line MATCHES "^function\\(add_triton_object name\\)")
        string(APPEND _output "${_line}\n")
        set(_in_add_triton_object TRUE)
        continue()
    endif()
    if(_in_add_triton_object)
        string(APPEND _output "${_line}\n")
        if(_line MATCHES "add_library\\(\\$\\{name\\}")
            string(APPEND _output "  # PATCHED: Force tablegen ordering (includes all TritonGPU sub-targets)\n")
            string(APPEND _output "  foreach(_tgt TritonTableGen TritonGPUTableGen TritonGPUCGAAttrIncGen TritonGPUAttrDefsIncGen TritonGPUOpsEnumsIncGen TritonGPUTypeInterfacesIncGen TritonGPUOpInterfacesIncGen TritonCPUTableGen TritonInstrumentTableGen GluonTableGen)\n")
            string(APPEND _output "    if(TARGET \${_tgt})\n")
            string(APPEND _output "      add_dependencies(\${name} \${_tgt})\n")
            string(APPEND _output "    endif()\n")
            string(APPEND _output "  endforeach()\n")
        endif()
        if(_line MATCHES "^endfunction")
            set(_in_add_triton_object FALSE)
        endif()
        continue()
    endif()

    string(APPEND _output "${_line}\n")
endforeach()

file(WRITE "${_CMAKELISTS}" "${_output}")
message(STATUS "  [patch] Stripped Python + GPU from CMakeLists.txt")

# ══════════════════════════════════════════════════════════════════════════
# 3. Strip GPU from include/ tablegen — only generate headers CPU needs
# ══════════════════════════════════════════════════════════════════════════
# The CPU backend includes triton/Dialect/TritonGPU/IR/Dialect.h (header only,
# not linked). That header needs TritonGPU tablegen. But TritonNvidiaGPU,
# TritonInstrument, Gluon are NOT needed by CPU backend at all.

set(_INCL_DIALECT_CMAKE "${SOURCE_DIR}/include/triton/Dialect/CMakeLists.txt")
if(EXISTS "${_INCL_DIALECT_CMAKE}")
    file(WRITE "${_INCL_DIALECT_CMAKE}"
        "add_subdirectory(Triton)\n"
        "add_subdirectory(TritonCPU)\n"
        "add_subdirectory(TritonGPU)\n"
        "# PATCHED: TritonNvidiaGPU, TritonInstrument, Gluon tablegen stripped\n")
    message(STATUS "  [patch] Stripped GPU tablegen from include/triton/Dialect/")
endif()

# Strip TritonGPU Transforms tablegen (not needed, only keep IR tablegen)
set(_INCL_GPU_CMAKE "${SOURCE_DIR}/include/triton/Dialect/TritonGPU/CMakeLists.txt")
if(EXISTS "${_INCL_GPU_CMAKE}")
    file(WRITE "${_INCL_GPU_CMAKE}"
        "add_subdirectory(IR)\n"
        "# PATCHED: Transforms tablegen stripped\n")
    message(STATUS "  [patch] Stripped TritonGPU/Transforms tablegen")
endif()

# ══════════════════════════════════════════════════════════════════════════
# 4. Stub out TritonNvidiaGPU/IR/Dialect.h — empty header
# ══════════════════════════════════════════════════════════════════════════
# TritonGPU/IR source files #include this header. Since we're not compiling
# TritonGPU/IR .cpp files (only tablegen for headers), we just need this to
# not cause errors if anyone transitively includes it.
# The CPU backend only includes TritonGPU headers, never NvidiaGPU directly.

set(_NVGPU_DIALECT_H "${SOURCE_DIR}/include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h")
if(EXISTS "${_NVGPU_DIALECT_H}")
    file(WRITE "${_NVGPU_DIALECT_H}"
"// PATCHED: Stub header for CPU-only build — no NVIDIA GPU dialect
#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_

#include \"mlir/IR/BuiltinAttributes.h\"
#include \"mlir/IR/BuiltinTypes.h\"
#include \"mlir/IR/Dialect.h\"

// Stub: no NvidiaGPU types available in CPU-only build
namespace mlir {
namespace triton {
namespace nvidia_gpu {

// Empty namespace — types referenced by TritonGPU IR headers will cause
// compile errors only if actually USED (not just included).
// The CPU backend never uses NvidiaGPU types.

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_
")
    message(STATUS "  [patch] Stubbed TritonNvidiaGPU/IR/Dialect.h")
endif()

# Also stub the Transforms headers that might be transitively included
set(_NVGPU_TRANSFORMS_DIR "${SOURCE_DIR}/include/triton/Dialect/TritonNvidiaGPU/Transforms")
if(EXISTS "${_NVGPU_TRANSFORMS_DIR}")
    file(GLOB _nvgpu_transform_headers "${_NVGPU_TRANSFORMS_DIR}/*.h")
    foreach(_h IN LISTS _nvgpu_transform_headers)
        get_filename_component(_hname "${_h}" NAME)
        file(WRITE "${_h}"
"// PATCHED: Stub for CPU-only build
#ifndef TRITON_NVGPU_TRANSFORMS_STUB_${_hname}_
#define TRITON_NVGPU_TRANSFORMS_STUB_${_hname}_
#endif
")
    endforeach()
    message(STATUS "  [patch] Stubbed NvidiaGPU/Transforms headers")
endif()

# ══════════════════════════════════════════════════════════════════════════
# 5. Strip GPU includes from Conversion utility headers the CPU backend uses
# ══════════════════════════════════════════════════════════════════════════
# CPU backend includes triton/Conversion/TritonGPUToLLVM/Utility.h — keep
# the header but strip any NvidiaGPU includes from it.

set(_GPU_CONV_UTILITY "${SOURCE_DIR}/include/triton/Conversion/TritonGPUToLLVM/Utility.h")
if(EXISTS "${_GPU_CONV_UTILITY}")
    file(READ "${_GPU_CONV_UTILITY}" _u_content)
    string(REGEX REPLACE "#include [\"<]triton/Dialect/TritonNvidiaGPU/[^\"]*[>\"]" "// PATCHED: NvidiaGPU stripped" _u_content "${_u_content}")
    file(WRITE "${_GPU_CONV_UTILITY}" "${_u_content}")
endif()

set(_GPU_CONV_PATTERN "${SOURCE_DIR}/include/triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h")
if(EXISTS "${_GPU_CONV_PATTERN}")
    file(READ "${_GPU_CONV_PATTERN}" _p_content)
    string(REGEX REPLACE "#include [\"<]triton/Dialect/TritonNvidiaGPU/[^\"]*[>\"]" "// PATCHED: NvidiaGPU stripped" _p_content "${_p_content}")
    file(WRITE "${_GPU_CONV_PATTERN}" "${_p_content}")
endif()

# ══════════════════════════════════════════════════════════════════════════
# 6. Strip Conversion include tablegen (not compiled, but cmake may process)
# ══════════════════════════════════════════════════════════════════════════
set(_CONV_INCL_CMAKE "${SOURCE_DIR}/include/triton/Conversion/CMakeLists.txt")
if(EXISTS "${_CONV_INCL_CMAKE}")
    file(WRITE "${_CONV_INCL_CMAKE}"
        "# PATCHED: GPU conversion tablegen stripped for CPU-only build\n")
    message(STATUS "  [patch] Stripped Conversion tablegen")
endif()

# Also strip Target include tablegen
set(_TARGET_INCL_CMAKE "${SOURCE_DIR}/include/triton/Target/CMakeLists.txt")
if(EXISTS "${_TARGET_INCL_CMAKE}")
    file(WRITE "${_TARGET_INCL_CMAKE}"
        "# PATCHED: Target tablegen stripped for CPU-only build\n")
    message(STATUS "  [patch] Stripped Target tablegen")
endif()

# ══════════════════════════════════════════════════════════════════════════
# 7. Fix _Float16 on ARM64 — GCC uses __fp16, not _Float16
# ══════════════════════════════════════════════════════════════════════════
set(_CPU_RUNTIME "${SOURCE_DIR}/third_party/cpu/runtime/cpu_runtime.cpp")
if(EXISTS "${_CPU_RUNTIME}")
    file(READ "${_CPU_RUNTIME}" _rt_content)
    string(FIND "${_rt_content}" "_Float16" _float16_pos)
    if(NOT _float16_pos EQUAL -1)
        string(FIND "${_rt_content}" "PATCHED_FLOAT16" _patched_pos)
        if(_patched_pos EQUAL -1)
            # Replace _Float16 with a portable typedef
            string(REPLACE "_Float16" "fp16_t" _rt_content "${_rt_content}")
            set(_float16_preamble [=[// PATCHED_FLOAT16: _Float16 is not available on all compilers (e.g. GCC on ARM64).
// Use a portable typedef.
#if defined(__clang__) || (defined(__GNUC__) && defined(__x86_64__))
  typedef _Float16 fp16_t;
#elif defined(__GNUC__) && defined(__aarch64__)
  typedef __fp16 fp16_t;
#else
  #include <cstdint>
  typedef uint16_t fp16_t;  // fallback: raw bits
#endif
]=])
            string(PREPEND _rt_content "${_float16_preamble}")
            file(WRITE "${_CPU_RUNTIME}" "${_rt_content}")
            message(STATUS "  [patch] Fixed _Float16 -> fp16_t in cpu_runtime.cpp")
        endif()
    endif()
endif()

# Triton instantiates LLVM/MLIR C++ template statics in libtriton.a, so it must
# use the same coexistence policy as the downloaded LLVM/MLIR DSOs.
set(SD_EXTERNAL_PROJECT TRITON)
include("${CMAKE_CURRENT_LIST_DIR}/patch_external_llvm_coexistence.cmake")

message(STATUS "patch_triton_cpu: done (all patches applied)")
