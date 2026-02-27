# patch_triton.cmake — Cross-platform Triton source patching
#
# Usage: cmake -P patch_triton.cmake -DSOURCE_DIR=<triton_source> [-DREMOVE_AMD=ON]
#
# This replaces the bash-only patch_triton_no_amd.sh with a portable CMake script.
# It handles:
#   1. AMD dialect removal (when REMOVE_AMD is set)
#   2. NegFOp/TanhOp op pattern additions (always)

cmake_minimum_required(VERSION 3.18)

# Accept arguments via -D defines (passed before -P) or via -- positional args
if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR must be defined (-DSOURCE_DIR=<path>)")
endif()

# Helper: read file, apply replacements, write back
macro(patch_file_replace FILE_PATH OLD_TEXT NEW_TEXT)
    if(EXISTS "${FILE_PATH}")
        file(READ "${FILE_PATH}" _content)
        string(REPLACE "${OLD_TEXT}" "${NEW_TEXT}" _content "${_content}")
        file(WRITE "${FILE_PATH}" "${_content}")
    endif()
endmacro()

# Helper: remove lines matching a pattern from a file
macro(patch_file_remove_lines FILE_PATH PATTERN)
    if(EXISTS "${FILE_PATH}")
        file(READ "${FILE_PATH}" _content)
        # Convert to list of lines, filter, convert back
        string(REGEX REPLACE "\n" ";" _lines "${_content}")
        set(_filtered "")
        foreach(_line IN LISTS _lines)
            string(FIND "${_line}" "${PATTERN}" _pos)
            if(_pos EQUAL -1)
                string(APPEND _filtered "${_line}\n")
            endif()
        endforeach()
        file(WRITE "${FILE_PATH}" "${_filtered}")
    endif()
endmacro()

# === Part 1: AMD dialect removal (when REMOVE_AMD is set) ===
if(REMOVE_AMD)
    set(_REG_H "${SOURCE_DIR}/bin/RegisterTritonDialects.h")
    set(_BIN_CMAKE "${SOURCE_DIR}/bin/CMakeLists.txt")

    if(EXISTS "${_REG_H}")
        file(READ "${_REG_H}" _reg_content)
        # Remove lines containing AMD-specific patterns
        set(_amd_patterns
            "amd/include/"
            "TritonAMDGPUToLLVM"
            "TritonAMDGPUTransforms"
            "TritonAMDGPUDialect"
            "ProtonAMDGPUToLLVM"
            "ROCDLDialect"
            "ROCDL::"
            "registerConvertTritonAMDGPUToLLVM"
            "registerConvertBuiltinFuncToLLVM"
            "registerDecomposeUnsupportedAMDConversions"
            "registerOptimizeAMDLDSUsage"
            "registerTritonAMDGPU"
            "registerAllocateAMDGPU"
            "registerAMDTest"
            "registerTestAMDGPU"
            "registerTestTritonAMDGPU"
            "registerTritonAMDFold"
            "registerConvertProtonAMDGPU"
            "AddSchedBarriers"
            "amdgpu::"
        )
        foreach(_pat IN LISTS _amd_patterns)
            # Remove each line containing the pattern
            string(REGEX REPLACE "[^\n]*${_pat}[^\n]*\n?" "" _reg_content "${_reg_content}")
        endforeach()
        file(WRITE "${_REG_H}" "${_reg_content}")
        message(STATUS "Patched ${_REG_H}: removed AMD dialect references")
    endif()

    if(EXISTS "${_BIN_CMAKE}")
        file(READ "${_BIN_CMAKE}" _bin_content)
        # Remove AMD-related library references from bin CMakeLists
        set(_bin_amd_patterns
            "MLIRGPUToROCDLTransforms"
            "TritonAMDGPUTestAnalysis"
            "ProtonAMDGPUToLLVM"
        )
        foreach(_pat IN LISTS _bin_amd_patterns)
            string(REGEX REPLACE "[^\n]*${_pat}[^\n]*\n?" "" _bin_content "${_bin_content}")
        endforeach()
        file(WRITE "${_BIN_CMAKE}" "${_bin_content}")
        message(STATUS "Patched ${_BIN_CMAKE}: removed AMD library references")
    endif()
endif()

# === Part 2: NegFOp and TanhOp patches (always applied) ===

# Patch TritonToTritonGPUPass.cpp — add NegFOp and TanhOp as legal op patterns
set(_TTGPU_PASS "${SOURCE_DIR}/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp")
if(EXISTS "${_TTGPU_PASS}")
    file(READ "${_TTGPU_PASS}" _ttgpu_content)

    # Add NegFOp after ShRSIOp (upstream comments it out or omits it)
    string(REPLACE
        "GenericOpPattern<arith::ShRSIOp>, // NegFOp"
        "GenericOpPattern<arith::ShRSIOp>, GenericOpPattern<arith::NegFOp>,"
        _ttgpu_content "${_ttgpu_content}")

    # Add TanhOp after FmaOp (upstream omits it)
    string(REPLACE
        "GenericOpPattern<math::FmaOp>>"
        "GenericOpPattern<math::FmaOp>, GenericOpPattern<math::TanhOp>>"
        _ttgpu_content "${_ttgpu_content}")

    file(WRITE "${_TTGPU_PASS}" "${_ttgpu_content}")
    message(STATUS "Patched ${_TTGPU_PASS}: added NegFOp and TanhOp patterns")
endif()

# Patch ElementwiseOpToLLVM.cpp — add NegFOp and TanhOp LLVM lowering patterns
set(_ELEM_LLVM "${SOURCE_DIR}/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp")
if(EXISTS "${_ELEM_LLVM}")
    file(READ "${_ELEM_LLVM}" _elem_content)

    # Add NegFOp lowering after UIToFPOp
    string(REPLACE
        "POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)"
        "POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)\n  POPULATE_UNARY_OP(arith::NegFOp, LLVM::FNegOp)"
        _elem_content "${_elem_content}")

    # Add TanhOp lowering after ExpOp
    string(REPLACE
        "POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)"
        "POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)\n  POPULATE_UNARY_OP(math::TanhOp, math::TanhOp)"
        _elem_content "${_elem_content}")

    file(WRITE "${_ELEM_LLVM}" "${_elem_content}")
    message(STATUS "Patched ${_ELEM_LLVM}: added NegFOp and TanhOp LLVM lowering")
endif()

# === Part 3: Fix Proton AMD build when AMD backend not selected (Triton 3.6.0) ===
# The Proton profiling dialect always builds ProtonAMDGPUToLLVM even when only NVIDIA
# codegen backend is selected, causing build failure on missing AMD .h.inc files.
# We need to disable it in both lib/ and include/ CMakeLists.
if(REMOVE_AMD)
    # Disable ProtonAMDGPUToLLVM library build
    set(_PROTON_GPU_CMAKE "${SOURCE_DIR}/third_party/proton/Dialect/lib/ProtonGPUToLLVM/CMakeLists.txt")
    if(EXISTS "${_PROTON_GPU_CMAKE}")
        file(READ "${_PROTON_GPU_CMAKE}" _proton_gpu_content)
        string(REPLACE
            "add_subdirectory(ProtonAMDGPUToLLVM)"
            "# add_subdirectory(ProtonAMDGPUToLLVM)  # Removed: AMD backend not selected"
            _proton_gpu_content "${_proton_gpu_content}")
        file(WRITE "${_PROTON_GPU_CMAKE}" "${_proton_gpu_content}")
        message(STATUS "Patched ${_PROTON_GPU_CMAKE}: disabled ProtonAMDGPUToLLVM lib")
    endif()

    # Disable ProtonAMDGPUToLLVM include/tablegen
    set(_PROTON_INC_CMAKE "${SOURCE_DIR}/third_party/proton/Dialect/include/Conversion/ProtonGPUToLLVM/CMakeLists.txt")
    if(EXISTS "${_PROTON_INC_CMAKE}")
        file(READ "${_PROTON_INC_CMAKE}" _proton_inc_content)
        string(REPLACE
            "add_subdirectory(ProtonAMDGPUToLLVM)"
            "# add_subdirectory(ProtonAMDGPUToLLVM)  # Removed: AMD backend not selected"
            _proton_inc_content "${_proton_inc_content}")
        file(WRITE "${_PROTON_INC_CMAKE}" "${_proton_inc_content}")
        message(STATUS "Patched ${_PROTON_INC_CMAKE}: disabled ProtonAMDGPUToLLVM include")
    endif()
endif()

# === Part 4: Fix missing build dependencies (Triton 3.6.0 upstream bug) ===

# TritonIR/Traits.cpp includes TritonGPU/IR/Attributes.h -> CTAEncodingAttr.h
# which needs CTAEncodingAttr.h.inc from TritonGPUCTAAttrIncGen.
# But TritonIR's CMakeLists.txt doesn't declare this dependency, causing
# parallel builds to fail when TritonIR compiles before the tablegen runs.
set(_TRITON_IR_CMAKE "${SOURCE_DIR}/lib/Dialect/Triton/IR/CMakeLists.txt")
if(EXISTS "${_TRITON_IR_CMAKE}")
    file(READ "${_TRITON_IR_CMAKE}" _triton_ir_content)
    # Check if the fix is already applied
    string(FIND "${_triton_ir_content}" "TritonGPUCTAAttrIncGen" _already_patched)
    if(_already_patched EQUAL -1)
        # Add TritonGPU tablegen dependencies to TritonIR
        string(REPLACE
            "  DEPENDS\n  TritonTableGen\n  TritonCanonicalizeIncGen"
            "  DEPENDS\n  TritonTableGen\n  TritonCanonicalizeIncGen\n  TritonGPUCTAAttrIncGen\n  TritonGPUAttrDefsIncGen\n  TritonGPUTypeInterfacesIncGen\n  TritonGPUOpInterfacesIncGen"
            _triton_ir_content "${_triton_ir_content}")
        file(WRITE "${_TRITON_IR_CMAKE}" "${_triton_ir_content}")
        message(STATUS "Patched ${_TRITON_IR_CMAKE}: added TritonGPU tablegen dependencies to TritonIR")
    else()
        message(STATUS "${_TRITON_IR_CMAKE}: TritonGPU dependency fix already applied")
    endif()
endif()

# === Part 5: Fix TritonGPUIR missing dependency on TritonNvidiaGPU tablegen ===

# TritonGPUIR/Dialect.cpp includes TritonNvidiaGPU/IR/Dialect.h which needs
# Dialect.h.inc generated by TritonNvidiaGPU tablegen targets. But TritonGPUIR
# doesn't declare this dependency, causing parallel builds to fail.
# Use file(APPEND) with add_dependencies() instead of string(REPLACE) on
# DEPENDS blocks — this is more robust against formatting differences.
set(_TRITON_GPU_IR_CMAKE "${SOURCE_DIR}/lib/Dialect/TritonGPU/IR/CMakeLists.txt")
if(EXISTS "${_TRITON_GPU_IR_CMAKE}")
    file(READ "${_TRITON_GPU_IR_CMAKE}" _triton_gpu_ir_content)
    string(FIND "${_triton_gpu_ir_content}" "TritonNvidiaGPUTableGen" _already_patched)
    if(_already_patched EQUAL -1)
        # Append add_dependencies() call — works regardless of CMakeLists formatting
        file(APPEND "${_TRITON_GPU_IR_CMAKE}" "\n\n# [nd4j patch] Fix missing TritonNvidiaGPU tablegen dependency\nif(TARGET TritonNvidiaGPUTableGen)\n  add_dependencies(TritonGPUIR TritonNvidiaGPUTableGen)\nendif()\nif(TARGET TritonNvidiaGPUAttrDefsIncGen)\n  add_dependencies(TritonGPUIR TritonNvidiaGPUAttrDefsIncGen)\nendif()\nif(TARGET TritonNvidiaGPUOpInterfacesIncGen)\n  add_dependencies(TritonGPUIR TritonNvidiaGPUOpInterfacesIncGen)\nendif()\n")
        message(STATUS "Patched ${_TRITON_GPU_IR_CMAKE}: appended TritonNvidiaGPU tablegen dependencies (add_dependencies)")
    else()
        message(STATUS "${_TRITON_GPU_IR_CMAKE}: TritonNvidiaGPU dependency fix already applied")
    endif()
else()
    message(WARNING "TritonGPU/IR/CMakeLists.txt NOT FOUND at ${_TRITON_GPU_IR_CMAKE}")
endif()

message(STATUS "Triton patching complete (SOURCE_DIR=${SOURCE_DIR})")
