# cmake/FindMLIR.cmake
# Find MLIR and LLVM installations for JIT compilation support
#
# This module finds LLVM/MLIR and sets up the necessary variables and targets.
#
# Required: LLVM 18+ with MLIR support
#
# Variables defined:
#   MLIR_FOUND          - True if MLIR was found
#   MLIR_INCLUDE_DIRS   - Include directories for MLIR
#   MLIR_LIBRARY_DIRS   - Library directories for MLIR
#   MLIR_DEFINITIONS    - Compiler definitions for MLIR
#   MLIR_LIBRARIES      - List of MLIR libraries to link
#   LLVM_VERSION        - LLVM version string
#
# Imported targets:
#   MLIR::MLIR          - Interface library for all MLIR dependencies

include(CMakeFindDependencyMacro)

# Minimum required LLVM version
if(NOT DEFINED MLIR_VERSION)
    set(MLIR_VERSION "18")
endif()

# Search paths for LLVM/MLIR
set(LLVM_SEARCH_PATHS
    ${LLVM_DIR}
    ${LLVM_ROOT}
    $ENV{LLVM_DIR}
    $ENV{LLVM_ROOT}
    /usr/lib/llvm-${MLIR_VERSION}
    /usr/lib/llvm-18
    /usr/lib/llvm-19
    /usr/local/opt/llvm@${MLIR_VERSION}
    /usr/local/opt/llvm
    /opt/llvm
    /opt/homebrew/opt/llvm
)

# Find LLVM first
find_package(LLVM ${MLIR_VERSION} CONFIG
    PATHS ${LLVM_SEARCH_PATHS}
    PATH_SUFFIXES lib/cmake/llvm share/llvm/cmake
)

if(NOT LLVM_FOUND)
    message(STATUS "LLVM ${MLIR_VERSION}+ not found in standard paths.")
    message(STATUS "Searched paths: ${LLVM_SEARCH_PATHS}")
    message(STATUS "Set LLVM_DIR or LLVM_ROOT to the LLVM installation directory.")
    set(MLIR_FOUND FALSE)
    return()
endif()

message(STATUS "Found LLVM ${LLVM_VERSION} at ${LLVM_DIR}")

# Check LLVM version meets minimum requirement
if(LLVM_VERSION VERSION_LESS "${MLIR_VERSION}.0.0")
    message(WARNING "LLVM version ${LLVM_VERSION} is below minimum required ${MLIR_VERSION}")
    set(MLIR_FOUND FALSE)
    return()
endif()

# Find MLIR
find_package(MLIR CONFIG
    PATHS ${LLVM_DIR}/../mlir ${LLVM_DIR} ${MLIR_DIR}
    PATH_SUFFIXES lib/cmake/mlir share/mlir/cmake
)

if(NOT MLIR_FOUND)
    message(STATUS "MLIR not found. LLVM may have been built without MLIR support.")
    message(STATUS "Rebuild LLVM with -DLLVM_ENABLE_PROJECTS='mlir' to enable MLIR.")
    set(MLIR_FOUND FALSE)
    return()
endif()

message(STATUS "Found MLIR at ${MLIR_DIR}")

# Include LLVM and MLIR CMake modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Set include directories
set(MLIR_INCLUDE_DIRS
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

# Set library directories
set(MLIR_LIBRARY_DIRS
    ${LLVM_LIBRARY_DIRS}
)

# Set definitions
set(MLIR_DEFINITIONS ${LLVM_DEFINITIONS})

# Core MLIR libraries for dialect development and JIT
set(MLIR_CORE_LIBS
    # Core MLIR infrastructure
    MLIRSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils

    # Dialects needed for lowering
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRLinalgDialect
    MLIRVectorDialect
    MLIRAffineDialect
    MLIRMathDialect

    # Bufferization
    MLIRBufferizationDialect
    MLIRBufferizationTransforms

    # LLVM dialect and lowering
    MLIRLLVMDialect
    MLIRLLVMCommonConversion

    # Conversion passes
    MLIRLinalgToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRMathToLLVM
    MLIRControlFlowToLLVM
    MLIRSCFToControlFlow
    MLIRAffineToStandard
    MLIRVectorToLLVM

    # Execution engine for JIT
    MLIRExecutionEngine
    MLIRJitRunner

    # Analysis and optimization
    MLIRAnalysis
    MLIRSideEffectInterfaces
    MLIRTranslateLib
)

# GPU-specific libraries (conditional)
set(MLIR_GPU_LIBS
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRGPUToNVVMTransforms
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRSPIRVDialect
)

# SPIR-V / Vulkan libraries for ARM mobile GPU targets
set(MLIR_VULKAN_LIBS
    MLIRSPIRVDialect
    MLIRSPIRVTransforms
    MLIRSPIRVConversion
    MLIRSPIRVSerialization
    MLIRSPIRVDeserialization
    MLIRGPUToSPIRV
    MLIRArithToSPIRV
    MLIRFuncToSPIRV
    MLIRMemRefToSPIRV
    MLIRSCFToSPIRV
    MLIRMathToSPIRV
    MLIRGPUToVulkanTransforms
)

# LLVM libraries needed for JIT execution
set(LLVM_JIT_LIBS
    LLVMCore
    LLVMSupport
    LLVMTarget
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Desc
    LLVMX86Info
    LLVMOrcJIT
    LLVMExecutionEngine
    LLVMRuntimeDyld
    LLVMObject
    LLVMMCParser
    LLVMMC
    LLVMBitReader
    LLVMBitWriter
    LLVMAnalysis
    LLVMTransformUtils
    LLVMInstCombine
    LLVMScalarOpts
    LLVMipo
    LLVMPasses
)

# ARM64 / AArch64 libraries for AOT cross-compilation and native ARM JIT
set(LLVM_AARCH64_LIBS
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Desc
    LLVMAArch64Info
    LLVMAArch64Utils
)

# Create interface library for easy consumption
if(NOT TARGET MLIR::MLIR)
    add_library(MLIR::MLIR INTERFACE IMPORTED)

    # Link against MLIR core libs
    set(_mlir_libs_to_link)
    foreach(_lib ${MLIR_CORE_LIBS})
        if(TARGET ${_lib})
            list(APPEND _mlir_libs_to_link ${_lib})
        else()
            message(STATUS "MLIR library not found: ${_lib} (may be optional)")
        endif()
    endforeach()

    # Optionally add GPU libs
    if(MLIR_ENABLE_GPU)
        foreach(_lib ${MLIR_GPU_LIBS})
            if(TARGET ${_lib})
                list(APPEND _mlir_libs_to_link ${_lib})
            else()
                message(STATUS "MLIR GPU library not found: ${_lib}")
            endif()
        endforeach()
    endif()

    # Optionally add Vulkan/SPIR-V libs for ARM mobile GPU
    if(MLIR_ENABLE_VULKAN)
        foreach(_lib ${MLIR_VULKAN_LIBS})
            if(TARGET ${_lib})
                list(APPEND _mlir_libs_to_link ${_lib})
            else()
                message(STATUS "MLIR Vulkan library not found: ${_lib} (optional)")
            endif()
        endforeach()
    endif()

    # Link LLVM JIT libs
    foreach(_lib ${LLVM_JIT_LIBS})
        if(TARGET ${_lib})
            list(APPEND _mlir_libs_to_link ${_lib})
        endif()
    endforeach()

    # Link AArch64 backend libs (for AOT cross-compilation or native ARM JIT)
    if(MLIR_ENABLE_AARCH64 OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        foreach(_lib ${LLVM_AARCH64_LIBS})
            if(TARGET ${_lib})
                list(APPEND _mlir_libs_to_link ${_lib})
            else()
                message(STATUS "LLVM AArch64 library not found: ${_lib} (optional)")
            endif()
        endforeach()
    endif()

    set_target_properties(MLIR::MLIR PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MLIR_INCLUDE_DIRS}"
        INTERFACE_COMPILE_DEFINITIONS "${MLIR_DEFINITIONS}"
        INTERFACE_LINK_LIBRARIES "${_mlir_libs_to_link}"
    )

    set(MLIR_LIBRARIES ${_mlir_libs_to_link})
endif()

# Create GPU-specific interface library
if(MLIR_ENABLE_GPU AND NOT TARGET MLIR::GPU)
    add_library(MLIR::GPU INTERFACE IMPORTED)

    set(_mlir_gpu_libs_to_link)
    foreach(_lib ${MLIR_GPU_LIBS})
        if(TARGET ${_lib})
            list(APPEND _mlir_gpu_libs_to_link ${_lib})
        endif()
    endforeach()

    set_target_properties(MLIR::GPU PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_mlir_gpu_libs_to_link}"
    )
endif()

# Function to run TableGen for dialect definitions
function(mlir_tablegen_sd output_dir)
    set(LLVM_TARGET_DEFINITIONS ${ARGN})
    tablegen(MLIR SDOps.h.inc -gen-op-decls)
    tablegen(MLIR SDOps.cpp.inc -gen-op-defs)
    tablegen(MLIR SDDialect.h.inc -gen-dialect-decls)
    tablegen(MLIR SDDialect.cpp.inc -gen-dialect-defs)
    tablegen(MLIR SDTypes.h.inc -gen-typedef-decls)
    tablegen(MLIR SDTypes.cpp.inc -gen-typedef-defs)
    tablegen(MLIR SDAttrDefs.h.inc -gen-attrdef-decls)
    tablegen(MLIR SDAttrDefs.cpp.inc -gen-attrdef-defs)
    add_public_tablegen_target(SDDialectIncGen)
endfunction()

# Print configuration summary
message(STATUS "")
message(STATUS "=== MLIR Configuration ===")
message(STATUS "LLVM Version: ${LLVM_VERSION}")
message(STATUS "LLVM Include Dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR Include Dirs: ${MLIR_INCLUDE_DIRS}")
message(STATUS "MLIR GPU Support: ${MLIR_ENABLE_GPU}")
message(STATUS "MLIR Vulkan/SPIR-V: ${MLIR_ENABLE_VULKAN}")
message(STATUS "MLIR AArch64 AOT: ${MLIR_ENABLE_AARCH64}")
message(STATUS "MLIR AOT Target: ${MLIR_AOT_TARGET}")
message(STATUS "MLIR Libraries: ${MLIR_LIBRARIES}")
message(STATUS "==========================")
message(STATUS "")

set(MLIR_FOUND TRUE)
