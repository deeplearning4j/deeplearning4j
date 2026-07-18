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

# Consume LLVM/MLIR through their upstream monolithic shared-library targets.
# When setup_triton publishes a managed package root it is the sole valid
# compiler-runtime package. Ambient LLVM_DIR, CMAKE_PREFIX_PATH, package-manager,
# and system packages must not change either headers or linked DSOs.
set(LLVM_LINK_LLVM_DYLIB ON)
set(MLIR_LINK_MLIR_DYLIB ON)

# setup_triton publishes the exact patched, revision-marked shared LLVM/MLIR
# package selected for its explicit compiler consumer. FindMLIR consumes that
# identity directly; it must not independently infer CPU vs GPU from backend
# flags because Vulkan is a SPIR-V consumer of the GPU-pinned compiler package.

if(NOT DEFINED SD_TRITON_MANAGED_LLVM_ROOT OR
   "${SD_TRITON_MANAGED_LLVM_ROOT}" STREQUAL "")
    message(FATAL_ERROR
        "MLIR support requires the project-managed LLVM/MLIR package produced by "
        "the compiler dependency bootstrap. Ambient LLVM_DIR, MLIR_DIR, "
        "CMAKE_PREFIX_PATH, system, and package-manager LLVM installations are "
        "intentionally ignored.")
endif()

if(NOT SD_TRITON_CONSUMER_KIND STREQUAL "CPU_COMPILER" AND
   NOT SD_TRITON_CONSUMER_KIND STREQUAL "GPU_EMITTER" AND
   NOT SD_TRITON_CONSUMER_KIND STREQUAL "VULKAN_SPIRV")
    message(FATAL_ERROR
        "Unknown SD_TRITON_CONSUMER_KIND='${SD_TRITON_CONSUMER_KIND}' for "
        "managed LLVM/MLIR root ${SD_TRITON_MANAGED_LLVM_ROOT}.")
endif()
set(_sd_managed_llvm_root "${SD_TRITON_MANAGED_LLVM_ROOT}")

if(WIN32)
    set(_sd_managed_llvm_dso_dir "${_sd_managed_llvm_root}/bin")
else()
    set(_sd_managed_llvm_dso_dir "${_sd_managed_llvm_root}/lib")
endif()

foreach(_sd_managed_llvm_file IN ITEMS
        "${_sd_managed_llvm_root}/lib/cmake/llvm/LLVMConfig.cmake"
        "${_sd_managed_llvm_root}/lib/cmake/mlir/MLIRConfig.cmake"
        "${_sd_managed_llvm_dso_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}LLVM${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_sd_managed_llvm_dso_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}MLIR${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_sd_managed_llvm_dso_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}MLIRExecutionEngineShared${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_sd_managed_llvm_file}")
        message(FATAL_ERROR
            "The project-managed LLVM/MLIR package is incomplete: "
            "'${_sd_managed_llvm_file}' is missing. Run the compiler dependency "
            "bootstrap; ambient or system LLVM packages are never substituted.")
    endif()
endforeach()

set(LLVM_ROOT "${_sd_managed_llvm_root}" CACHE PATH
    "Project-managed LLVM/MLIR package root" FORCE)
set(LLVM_DIR "${_sd_managed_llvm_root}/lib/cmake/llvm" CACHE PATH
    "Project-managed LLVM package" FORCE)
set(MLIR_DIR "${_sd_managed_llvm_root}/lib/cmake/mlir" CACHE PATH
    "Project-managed MLIR package" FORCE)
message(STATUS
    "Using only the project-managed patched LLVM/MLIR install at ${_sd_managed_llvm_root}")

find_package(LLVM CONFIG REQUIRED PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)
if(LLVM_VERSION VERSION_LESS "${MLIR_VERSION}.0.0")
    message(FATAL_ERROR
        "MLIR support requires LLVM ${MLIR_VERSION}+; found ${LLVM_VERSION} at ${LLVM_DIR}.")
endif()
message(STATUS "Found LLVM ${LLVM_VERSION} at ${LLVM_DIR}")

find_package(MLIR CONFIG REQUIRED PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)
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

# Require the upstream shared-library boundary used by the Triton/CUDA path.
# The MLIR JIT keeps ExecutionEngine in its own shared DSO, so all three exported
# shared targets are part of the runtime contract. Component archives must never
# be embedded into a backend.
function(_sd_require_shared_imported_target target_name out_location)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR
            "The selected LLVM/MLIR package does not export shared target '${target_name}'.")
    endif()

    get_target_property(_shared_type ${target_name} TYPE)
    set(_candidate_configs "")
    if(CMAKE_BUILD_TYPE)
        string(TOUPPER "${CMAKE_BUILD_TYPE}" _active_config)
        list(APPEND _candidate_configs "${_active_config}")
    endif()
    get_target_property(_imported_configs ${target_name} IMPORTED_CONFIGURATIONS)
    if(_imported_configs)
        list(APPEND _candidate_configs ${_imported_configs})
    endif()
    list(APPEND _candidate_configs RELEASE RELWITHDEBINFO MINSIZEREL DEBUG)
    list(REMOVE_DUPLICATES _candidate_configs)

    set(_shared_location "")
    foreach(_config IN LISTS _candidate_configs)
        string(TOUPPER "${_config}" _config_upper)
        get_target_property(_config_location ${target_name} IMPORTED_LOCATION_${_config_upper})
        if(_config_location AND EXISTS "${_config_location}")
            set(_shared_location "${_config_location}")
            break()
        endif()
    endforeach()
    if(NOT _shared_location)
        get_target_property(_generic_location ${target_name} IMPORTED_LOCATION)
        if(_generic_location AND EXISTS "${_generic_location}")
            set(_shared_location "${_generic_location}")
        endif()
    endif()

    if(NOT _shared_type STREQUAL "SHARED_LIBRARY" OR NOT _shared_location)
        message(FATAL_ERROR
            "MLIR requires an installed shared ${target_name} target; "
            "got type='${_shared_type}', location='${_shared_location}'.")
    endif()
    message(STATUS "MLIR shared ${target_name}: ${_shared_location}")
    set(${out_location} "${_shared_location}" PARENT_SCOPE)
endfunction()

_sd_require_shared_imported_target(MLIR MLIR_SHARED_LIBRARY)
_sd_require_shared_imported_target(MLIRExecutionEngineShared MLIR_EXECUTION_ENGINE_SHARED_LIBRARY)
_sd_require_shared_imported_target(LLVM LLVM_SHARED_LIBRARY)

if(_sd_managed_llvm_root)
    get_filename_component(_sd_managed_llvm_root_real
        "${_sd_managed_llvm_root}" REALPATH)
    foreach(_sd_managed_mlir_path
            "${MLIR_SHARED_LIBRARY}"
            "${MLIR_EXECUTION_ENGINE_SHARED_LIBRARY}"
            "${LLVM_SHARED_LIBRARY}"
            ${LLVM_INCLUDE_DIRS}
            ${MLIR_INCLUDE_DIRS})
        if(NOT EXISTS "${_sd_managed_mlir_path}")
            continue()
        endif()
        get_filename_component(_sd_managed_mlir_path_real
            "${_sd_managed_mlir_path}" REALPATH)
        string(FIND "${_sd_managed_mlir_path_real}/"
            "${_sd_managed_llvm_root_real}/" _sd_managed_path_prefix)
        if(NOT _sd_managed_path_prefix EQUAL 0)
            message(FATAL_ERROR
                "LLVM/MLIR path '${_sd_managed_mlir_path_real}' is outside the "
                "project-managed package '${_sd_managed_llvm_root_real}'. "
                "System and Linuxbrew compiler packages are not valid here.")
        endif()
    endforeach()
endif()

# Optional LLVM/MLIR capabilities must come from the selected package, not
# from a different installation that happens to appear later on the compiler
# search path. Source-level __has_include cannot enforce that package boundary.
set(_sd_mlir_capability_definitions "")
macro(_sd_mlir_header_capability definition relative_header)
    set(${definition} FALSE)
    foreach(_mlir_include_dir IN LISTS MLIR_INCLUDE_DIRS)
        if(EXISTS "${_mlir_include_dir}/${relative_header}")
            set(${definition} TRUE)
            list(APPEND _sd_mlir_capability_definitions "${definition}=1")
            break()
        endif()
    endforeach()
endmacro()

_sd_mlir_header_capability(SD_MLIR_HAS_AFFINE_DIALECT
    "mlir/Dialect/Affine/IR/AffineOps.h")
_sd_mlir_header_capability(SD_MLIR_HAS_AFFINE_PASSES
    "mlir/Dialect/Affine/Passes.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARMNEON_DIALECT
    "mlir/Dialect/ArmNeon/ArmNeonDialect.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARMSVE_DIALECT
    "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARMSME_DIALECT
    "mlir/Dialect/ArmSME/IR/ArmSME.h")
_sd_mlir_header_capability(SD_MLIR_HAS_MATH_DIALECT
    "mlir/Dialect/Math/IR/Math.h")
_sd_mlir_header_capability(SD_MLIR_HAS_GPU_DIALECT
    "mlir/Dialect/GPU/IR/GPUDialect.h")
_sd_mlir_header_capability(SD_MLIR_HAS_GPU_TRANSFORM_PASSES
    "mlir/Dialect/GPU/Transforms/Passes.h")
_sd_mlir_header_capability(SD_MLIR_HAS_X86VECTOR_DIALECT
    "mlir/Dialect/X86Vector/X86VectorDialect.h")
_sd_mlir_header_capability(SD_MLIR_HAS_AMX_DIALECT
    "mlir/Dialect/AMX/AMXDialect.h")
_sd_mlir_header_capability(SD_MLIR_HAS_AMX_TRANSFORMS
    "mlir/Dialect/AMX/Transforms.h")
_sd_mlir_header_capability(SD_MLIR_HAS_AFFINE_TO_STANDARD
    "mlir/Conversion/AffineToStandard/AffineToStandard.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARMNEON2D_TO_INTR_PASS
    "mlir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARMSME_TO_LLVM_PASS
    "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h")
_sd_mlir_header_capability(SD_MLIR_HAS_LINALG_TO_LLVM_PASS
    "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h")
_sd_mlir_header_capability(SD_MLIR_HAS_LINALG_TO_STANDARD_PASS
    "mlir/Conversion/LinalgToStandard/LinalgToStandard.h")
_sd_mlir_header_capability(SD_MLIR_HAS_MATH_TO_LLVM_PASS
    "mlir/Conversion/MathToLLVM/MathToLLVM.h")
_sd_mlir_header_capability(SD_MLIR_HAS_VECTOR_TO_LLVM_PASS
    "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h")
_sd_mlir_header_capability(SD_MLIR_HAS_VECTOR_TO_ARMSME_PASS
    "mlir/Conversion/VectorToArmSME/VectorToArmSME.h")
_sd_mlir_header_capability(SD_MLIR_HAS_X86VECTOR_TRANSLATION
    "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARMNEON_TRANSLATION
    "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARMSVE_TRANSLATION
    "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARMSME_TRANSLATION
    "mlir/Target/LLVMIR/Dialect/ArmSME/ArmSMEToLLVMIRTranslation.h")
_sd_mlir_header_capability(SD_MLIR_HAS_AMX_TRANSLATION
    "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h")
_sd_mlir_header_capability(SD_MLIR_HAS_SPIRV_DIALECT
    "mlir/Dialect/SPIRV/IR/SPIRVDialect.h")
_sd_mlir_header_capability(SD_MLIR_HAS_SPIRV_OPS
    "mlir/Dialect/SPIRV/IR/SPIRVOps.h")
_sd_mlir_header_capability(SD_MLIR_HAS_SPIRV_PASSES
    "mlir/Dialect/SPIRV/Transforms/Passes.h")
_sd_mlir_header_capability(SD_MLIR_HAS_SPIRV_TARGET_ABI
    "mlir/Dialect/SPIRV/IR/TargetAndABI.h")
_sd_mlir_header_capability(SD_MLIR_HAS_GPU_TO_SPIRV_PASS_HEADER
    "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h")
_sd_mlir_header_capability(SD_MLIR_HAS_GPU_TO_SPIRV_LEGACY_HEADER
    "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h")
_sd_mlir_header_capability(SD_MLIR_HAS_ARITH_TO_SPIRV_HEADER
    "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h")
_sd_mlir_header_capability(SD_MLIR_HAS_CONVERSION_PASSES
    "mlir/Conversion/Passes.h")
_sd_mlir_header_capability(SD_MLIR_HAS_FUNC_TO_SPIRV_PASS_HEADER
    "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h")
_sd_mlir_header_capability(SD_MLIR_HAS_FUNC_TO_SPIRV_LEGACY_HEADER
    "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h")
_sd_mlir_header_capability(SD_MLIR_HAS_MEMREF_TO_SPIRV_PASS_HEADER
    "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h")
_sd_mlir_header_capability(SD_MLIR_HAS_MEMREF_TO_SPIRV_LEGACY_HEADER
    "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h")
_sd_mlir_header_capability(SD_MLIR_HAS_SCF_TO_SPIRV_PASS_HEADER
    "mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h")
_sd_mlir_header_capability(SD_MLIR_HAS_SCF_TO_SPIRV_LEGACY_HEADER
    "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h")
_sd_mlir_header_capability(SD_MLIR_HAS_MATH_TO_SPIRV_PASS_HEADER
    "mlir/Conversion/MathToSPIRV/MathToSPIRVPass.h")
_sd_mlir_header_capability(SD_MLIR_HAS_MATH_TO_SPIRV_LEGACY_HEADER
    "mlir/Conversion/MathToSPIRV/MathToSPIRV.h")
_sd_mlir_header_capability(SD_MLIR_HAS_SPIRV_SERIALIZATION
    "mlir/Target/SPIRV/Serialization.h")
_sd_mlir_header_capability(SD_MLIR_HAS_GPU_TO_VULKAN
    "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h")
_sd_mlir_header_capability(SD_MLIR_HAS_TARGET_MACHINE
    "llvm/Target/TargetMachine.h")
_sd_mlir_header_capability(SD_MLIR_HAS_TARGET_REGISTRY
    "llvm/MC/TargetRegistry.h")
_sd_mlir_header_capability(SD_MLIR_HAS_LLVMIR_EXPORT
    "mlir/Target/LLVMIR/Export.h")
_sd_mlir_header_capability(SD_MLIR_HAS_LEGACY_PM
    "llvm/IR/LegacyPassManager.h")
_sd_mlir_header_capability(SD_MLIR_HAS_LLVM_FILESYSTEM
    "llvm/Support/FileSystem.h")
_sd_mlir_header_capability(SD_MLIR_HAS_HOST_TARGETPARSER
    "llvm/TargetParser/Host.h")
_sd_mlir_header_capability(SD_MLIR_HAS_HOST_SUPPORT
    "llvm/Support/Host.h")

# Vulkan is a real backend contract, not an optional collection of whatever
# SPIR-V headers happen to be installed.  Reject an incomplete MLIR package at
# configure time so every OS/architecture classifier has the same lowering and
# serialization guarantees, independent of the installed Vulkan vendor ICD.
if(MLIR_ENABLE_VULKAN)
    set(_sd_missing_spirv_capabilities "")
    foreach(_sd_required_capability IN ITEMS
            SD_MLIR_HAS_MATH_DIALECT
            SD_MLIR_HAS_GPU_DIALECT
            SD_MLIR_HAS_GPU_TRANSFORM_PASSES
            SD_MLIR_HAS_SPIRV_DIALECT
            SD_MLIR_HAS_SPIRV_OPS
            SD_MLIR_HAS_SPIRV_PASSES
            SD_MLIR_HAS_SPIRV_TARGET_ABI
            SD_MLIR_HAS_SPIRV_SERIALIZATION)
        if(NOT "${${_sd_required_capability}}")
            list(APPEND _sd_missing_spirv_capabilities "${_sd_required_capability}")
        endif()
    endforeach()

    if(NOT SD_MLIR_HAS_GPU_TO_SPIRV_PASS_HEADER
            AND NOT SD_MLIR_HAS_GPU_TO_SPIRV_LEGACY_HEADER)
        list(APPEND _sd_missing_spirv_capabilities
            "SD_MLIR_HAS_GPU_TO_SPIRV")
    endif()

    if(DEFINED LLVM_VERSION_MAJOR AND LLVM_VERSION_MAJOR VERSION_GREATER_EQUAL 22)
        if(NOT SD_MLIR_HAS_CONVERSION_PASSES)
            list(APPEND _sd_missing_spirv_capabilities
                "SD_MLIR_HAS_CONVERSION_PASSES (ArithToSPIRV)")
        endif()
    elseif(NOT SD_MLIR_HAS_ARITH_TO_SPIRV_HEADER)
        list(APPEND _sd_missing_spirv_capabilities
            "SD_MLIR_HAS_ARITH_TO_SPIRV_HEADER")
    endif()

    foreach(_sd_spirv_conversion IN ITEMS FUNC MEMREF SCF MATH)
        if(NOT SD_MLIR_HAS_${_sd_spirv_conversion}_TO_SPIRV_PASS_HEADER
                AND NOT SD_MLIR_HAS_${_sd_spirv_conversion}_TO_SPIRV_LEGACY_HEADER)
            list(APPEND _sd_missing_spirv_capabilities
                "SD_MLIR_HAS_${_sd_spirv_conversion}_TO_SPIRV")
        endif()
    endforeach()

    if(_sd_missing_spirv_capabilities)
        list(JOIN _sd_missing_spirv_capabilities ", " _sd_missing_spirv_capabilities_text)
        message(FATAL_ERROR
            "The selected shared MLIR package is incomplete for the Vulkan backend. "
            "Missing: ${_sd_missing_spirv_capabilities_text}. "
            "Selected MLIR include roots: ${MLIR_INCLUDE_DIRS}")
    endif()
endif()

list(REMOVE_DUPLICATES _sd_mlir_capability_definitions)

if(NOT TARGET MLIR::MLIR)
    add_library(MLIR::MLIR INTERFACE IMPORTED)
    set_target_properties(MLIR::MLIR PROPERTIES
        # The explicitly selected package must win over ambient SDK include roots.
        # Imported targets default to SYSTEM includes, which can otherwise put a
        # workstation-wide LLVM ahead of these matching headers.
        IMPORTED_NO_SYSTEM TRUE
        INTERFACE_INCLUDE_DIRECTORIES "${MLIR_INCLUDE_DIRS}"
        INTERFACE_COMPILE_DEFINITIONS "${_sd_mlir_capability_definitions}"
        INTERFACE_LINK_LIBRARIES "MLIR;MLIRExecutionEngineShared;LLVM")
endif()
set(MLIR_LIBRARIES MLIR MLIRExecutionEngineShared LLVM)

if(MLIR_ENABLE_GPU AND NOT TARGET MLIR::GPU)
    add_library(MLIR::GPU INTERFACE IMPORTED)
    set_target_properties(MLIR::GPU PROPERTIES
        INTERFACE_LINK_LIBRARIES MLIR::MLIR)
endif()

if(MLIR_ENABLE_VULKAN AND NOT TARGET MLIR::SPIRV)
    add_library(MLIR::SPIRV INTERFACE IMPORTED)
    set_target_properties(MLIR::SPIRV PROPERTIES
        INTERFACE_LINK_LIBRARIES MLIR::MLIR)
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
