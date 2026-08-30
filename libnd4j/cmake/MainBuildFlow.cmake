# =============================================================================
# UNIFIED BUILD ORCHESTRATION SCRIPT WITH CUDA TEMPLATE PARITY
#
# This enhanced version includes support for CUDA template processing
# to achieve complete parity between CPU and CUDA template systems.
# UPDATED with modern cuDNN integration
#
# CACHE INVALIDATION: Modified to force CMake reconfiguration after
# PCH disabling in session #1002. Resolves Makefile2 having stale
# progress count (99 steps) while build.make has correct count (1251 steps).
# This inconsistency caused build to stop at 98% missing files 48-64.
# =============================================================================

# Include partial linking support for large binaries (>2GB with debug/trace)
include(${CMAKE_CURRENT_LIST_DIR}/PartialLinking.cmake)

# =============================================================================
# SECTION 1: HELPER FUNCTION DEFINITIONS
# All functions are defined first to ensure they are available when called.
# =============================================================================



# --- Helper for colored status messages ---
function(print_status_colored type message)
    if(type STREQUAL "ERROR")
        message(FATAL_ERROR "❌ ${message}")
    elseif(type STREQUAL "WARNING")
        message(WARNING "⚠️  ${message}")
    elseif(type STREQUAL "SUCCESS")
        message(STATUS "✅ ${message}")
    elseif(type STREQUAL "INFO")
        message(STATUS "ℹ️  ${message}")
    else()
        message(STATUS "${message}")
    endif()
endfunction()


function(libnd4j_setup_target_with_types target_name)
    message(STATUS "🔧 Setting up libnd4j target with type definitions: ${target_name}")

    # Ensure target exists
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "Target ${target_name} does not exist")
    endif()

    # Apply type definitions immediately after target creation
    setup_type_definitions_for_target(${target_name})

    # Apply any additional target configuration
    set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    message(STATUS "✅ Target ${target_name} configured with type definitions")
endfunction()

function(orchestrate_enhanced_build target_name)
    message(STATUS "ℹ️  === ORCHESTRATING ENHANCED LIBND4J BUILD WITH TEMPLATE PARITY ===")
    message(STATUS "ℹ️  === 1. INITIALIZING BUILD CONFIGURATION ===")

    # Setup target with type definitions
    libnd4j_setup_target_with_types(${target_name})

    message(STATUS "ℹ️  === 2. INITIALIZING DEPENDENCIES & OPERATIONS ===")
    message(STATUS "Dependencies initialization complete.")

    message(STATUS "ℹ️  === 3. CONFIGURING ENHANCED SELECTIVE RENDERING ===")
    message(STATUS "✅ Enhanced CPU selective rendering setup complete")

    message(STATUS "ℹ️  === 4. FINALIZING BUILD WITH TYPE DEFINITIONS ===")
    finalize_build_with_type_definitions(${target_name})

    message(STATUS "✅ === ENHANCED BUILD ORCHESTRATION COMPLETE ===")
    message(STATUS "🖥️  Enhanced CPU build orchestration complete - System ready for compilation")
endfunction()

# Simplified function for immediate use after target creation
function(apply_libnd4j_type_definitions target_name)
    if(TARGET ${target_name})
        setup_type_definitions_for_target(${target_name})
        message(STATUS "✅ Applied type definitions to ${target_name}")
    else()
        message(FATAL_ERROR "❌ Target ${target_name} not found for type definitions")
    endif()
endfunction()


function(apply_libnd4j_type_definitions_auto)
    # Use the SD_LIBRARY_NAME variable set in CMakeLists.txt
    if(DEFINED SD_LIBRARY_NAME AND TARGET ${SD_LIBRARY_NAME})
        set(target_name ${SD_LIBRARY_NAME})
        message(STATUS "Using SD_LIBRARY_NAME target: ${target_name}")
        setup_type_definitions_for_target(${target_name})
        return()
    endif()

    # Fallback to detecting the target based on build type
    if(SD_CUDA AND TARGET nd4jcuda)
        setup_type_definitions_for_target(nd4jcuda)
        message(STATUS "Applied type definitions to nd4jcuda")
    elseif(SD_CPU AND TARGET nd4jcpu)
        setup_type_definitions_for_target(nd4jcpu)
        message(STATUS "Applied type definitions to nd4jcpu")
    elseif(TARGET nd4jcpu)
        setup_type_definitions_for_target(nd4jcpu)
        message(STATUS "Applied type definitions to nd4jcpu (fallback)")
    elseif(TARGET nd4jcuda)
        setup_type_definitions_for_target(nd4jcuda)
        message(STATUS "Applied type definitions to nd4jcuda (fallback)")
    else()
        message(WARNING "❌ No nd4j targets found for type definitions")
        get_property(all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
        message(STATUS "Available targets: ${all_targets}")
    endif()
endfunction()

# Function to apply to specific target by name
function(apply_libnd4j_type_definitions target_name)
    if(TARGET ${target_name})
        setup_type_definitions_for_target(${target_name})
        message(STATUS "✅ Applied type definitions to ${target_name}")
    else()
        message(FATAL_ERROR "❌ Target ${target_name} not found for type definitions")
    endif()
endfunction()

# Function to apply to all libnd4j targets
function(apply_libnd4j_type_definitions_all)
    set(possible_targets "nd4jcpu;nd4jcuda;nd4j;libnd4j")
    set(applied_count 0)

    foreach(target ${possible_targets})
        if(TARGET ${target})
            setup_type_definitions_for_target(${target})
            message(STATUS "✅ Applied type definitions to ${target}")
            math(EXPR applied_count "${applied_count} + 1")
        endif()
    endforeach()

    if(applied_count EQUAL 0)
        message(WARNING "❌ No libnd4j targets found")
        get_property(all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
        message(STATUS "Available targets: ${all_targets}")
    else()
        message(STATUS "✅ Applied type definitions to ${applied_count} targets")
    endif()
endfunction()

# --- Platform environment setup functions ---
function(setup_cpu_environment)
    message(STATUS "🔧 Setting up CPU environment")

    # ==========================================================================
    # TMPDIR Configuration - Use build directory instead of /tmp (tmpfs)
    # Large template instantiations can create significant temp files.
    # By using a directory under the build tree, we use disk storage instead.
    # ==========================================================================
    set(CPU_TMPDIR "${CMAKE_BINARY_DIR}/compiler_tmp")
    file(MAKE_DIRECTORY "${CPU_TMPDIR}")
    set(ENV{TMPDIR} "${CPU_TMPDIR}")
    set(ENV{TMP} "${CPU_TMPDIR}")
    set(ENV{TEMP} "${CPU_TMPDIR}")
    message(STATUS "📁 Compiler temp directory: ${CPU_TMPDIR}")

    add_definitions(-D__CPUBLAS__=true)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            message(STATUS "✅ OpenMP found and will be linked")
        else()
            message(STATUS "⚠️ OpenMP not found, will use manual configuration")
        endif()
    endif()
    if(SD_X86_BUILD)
        message(STATUS "🎯 Enabling x86 optimizations")
    endif()
    message(STATUS "✅ CPU environment setup complete")
endfunction()

# --- Enhanced source collection with CUDA template support ---
function(collect_all_sources out_source_list)
    set(ALL_SOURCES_LIST "")

    file(GLOB_RECURSE PERF_SOURCES ./include/performance/*.cpp)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES ./include/exceptions/*.cpp)
    file(GLOB_RECURSE TYPES_SOURCES ./include/types/*.cpp)
    file(GLOB_RECURSE GRAPH_SOURCES ./include/graph/*.cpp)
    if(SD_VULKAN)
        # Vulkan is an independent device backend. Keep only common graph control
        # plane sources plus the Vulkan implementation; never compile another
        # backend's execution or replay implementation into libnd4jvulkan.
        file(GLOB_RECURSE _non_vulkan_graph_sources
            ./include/graph/cpu/*.cpp
            ./include/graph/cuda/*.cpp
            ./include/graph/gpu/*.cpp
            ./include/graph/hexagon/*.cpp
            ./include/graph/hip/*.cpp
            ./include/graph/levelzero/*.cpp
            ./include/graph/tpu/*.cpp
        )
        list(REMOVE_ITEM GRAPH_SOURCES ${_non_vulkan_graph_sources})
        list(LENGTH _non_vulkan_graph_sources _non_vulkan_graph_count)
        message(STATUS "SD_VULKAN: excluded ${_non_vulkan_graph_count} foreign graph backend sources")
    elseif(NOT SD_CUDA)
        # GPU backend files use the CUDA host runtime and are not generic DSP
        # infrastructure. Non-CUDA backends provide their own recorder/dispatcher.
        file(GLOB_RECURSE _gpu_backend_sources ./include/graph/gpu/*.cpp)
        if(_gpu_backend_sources)
            list(REMOVE_ITEM GRAPH_SOURCES ${_gpu_backend_sources})
            list(LENGTH _gpu_backend_sources _gpu_count)
            message(STATUS "Excluded ${_gpu_count} GPU backend files (non-CUDA build)")
        endif()
    else()
        # CUDA build: only exclude Triton files when Triton is not available (they require MLIR headers)
        if(NOT HAVE_TRITON)
            file(GLOB_RECURSE _triton_gpu_sources
                ./include/graph/gpu/Triton*.cpp
            )
            if(_triton_gpu_sources)
                list(REMOVE_ITEM GRAPH_SOURCES ${_triton_gpu_sources})
                list(LENGTH _triton_gpu_sources _triton_count)
                message(STATUS "Excluded ${_triton_count} Triton GPU backend files (HAVE_TRITON=OFF)")
            endif()
        endif()
    endif()
    # Exclude MLIR CPU backend files when MLIR is not available (they require MLIR headers)
    if(NOT HAVE_MLIR)
        file(GLOB_RECURSE _mlir_cpu_sources
            ./include/graph/cpu/MlirCpu*.cpp
            ./include/graph/cpu/CpuIRBuilder.cpp
            ./include/graph/cpu/ArmHybridGraphBackend.cpp
        )
        if(_mlir_cpu_sources)
            list(REMOVE_ITEM GRAPH_SOURCES ${_mlir_cpu_sources})
            list(LENGTH _mlir_cpu_sources _mlir_count)
            message(STATUS "Excluded ${_mlir_count} MLIR CPU backend files (HAVE_MLIR=OFF)")
        endif()
    endif()
    # Exclude NNAPI backend files when not on Android (requires <android/NeuralNetworks.h>)
    if(NOT HAVE_NNAPI)
        file(GLOB_RECURSE _nnapi_sources
            ./include/graph/cpu/NnapiGraphBackend.cpp
        )
        if(_nnapi_sources)
            list(REMOVE_ITEM GRAPH_SOURCES ${_nnapi_sources})
            message(STATUS "Excluded NNAPI backend files (HAVE_NNAPI=OFF)")
        endif()
    endif()
    # Exclude MLX CPU backend files when MLX is not available (they require MLX headers + C++20)
    if(NOT HAVE_MLX)
        file(GLOB_RECURSE _mlx_cpu_sources
            ./include/graph/cpu/MlxGraph*.cpp
            ./include/graph/cpu/MlxIRBuilder.cpp
        )
        if(_mlx_cpu_sources)
            list(REMOVE_ITEM GRAPH_SOURCES ${_mlx_cpu_sources})
            list(LENGTH _mlx_cpu_sources _mlx_count)
            message(STATUS "Excluded ${_mlx_count} MLX CPU backend files (HAVE_MLX=OFF)")
        endif()
    endif()
    # Exclude OpenVINO CPU backend files when OpenVINO is not available
    if(NOT HAVE_OPENVINO)
        file(GLOB_RECURSE _openvino_cpu_sources
            ./include/graph/cpu/OpenVinoGraphBackend.cpp
        )
        if(_openvino_cpu_sources)
            list(REMOVE_ITEM GRAPH_SOURCES ${_openvino_cpu_sources})
            message(STATUS "Excluded OpenVINO backend files (HAVE_OPENVINO=OFF)")
        endif()
    endif()
    # Vulkan graph/replay sources belong only to the Vulkan native artifact.
    # HAVE_VULKAN describes SDK availability; it must not make CPU or CUDA
    # artifacts acquire a second backend implementation.
    if(NOT SD_VULKAN)
        file(GLOB_RECURSE _vulkan_sources
            ./include/graph/vulkan/*.cpp
        )
        if(_vulkan_sources)
            list(REMOVE_ITEM GRAPH_SOURCES ${_vulkan_sources})
            list(LENGTH _vulkan_sources _vulkan_count)
            message(STATUS "Excluded ${_vulkan_count} Vulkan backend files (not an SD_VULKAN build)")
        endif()
    endif()
    # MLX sources require C++20 (MLX API uses concepts, designated initializers, etc.)
    if(HAVE_MLX)
        file(GLOB_RECURSE _mlx_cxx20_sources ./include/graph/cpu/Mlx*.cpp)
        if(_mlx_cxx20_sources)
            set_source_files_properties(${_mlx_cxx20_sources} PROPERTIES COMPILE_FLAGS "-std=c++20")
            list(LENGTH _mlx_cxx20_sources _mlx_cxx20_count)
            message(STATUS "Set C++20 for ${_mlx_cxx20_count} MLX source files")
        endif()
    endif()
    file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp)
    file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp)

    list(APPEND ALL_SOURCES_LIST
            ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${TYPES_SOURCES} ${GRAPH_SOURCES}
            ${CUSTOMOPS_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES}
    )

    # Template processing is backend-scoped; Vulkan deliberately generates no CPU execution TUs.
    setup_template_processing()

    if(SD_CUDA)
        file(GLOB_RECURSE EXEC_SOURCES ./include/execution/impl/*.cpp ./include/execution/cuda/*.cu ./include/execution/*.cu)
        file(GLOB_RECURSE ARRAY_SOURCES ./include/array/cuda/*.cu ./include/array/impl/*.cpp  ./include/array/*.cpp)
        file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/impl/*.cpp ./include/memory/cuda/*.cu)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES ./include/ops/declarable/helpers/cuda/*.cu ./include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp ./include/helpers/cuda/*.cu)
        # helpers/vulkan/*.cpp are SD_VULKAN-only (guarded by #error otherwise) and the CUDA
        # backend ships its own helpers/cuda/*.cu variants. The recursive glob above pulls the
        # Vulkan sources in; drop them here (an SD_CUDA build is never an SD_VULKAN build).
        file(GLOB_RECURSE _vulkan_only_helpers ./include/helpers/vulkan/*.cpp)
        if(_vulkan_only_helpers)
            list(REMOVE_ITEM HELPERS_SOURCES ${_vulkan_only_helpers})
        endif()
        file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/impl/*.cpp)
        file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/*.cu ./include/legacy/cuda/*.cu)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA ./include/loops/*.cu ./include/loops/cuda/**/*.cu)
        # Split the heaviest monolithic BUILD_SINGLE/DOUBLE_TEMPLATE ops (full
        # SD_COMMON_TYPES matrix in one TU -> 7-15 MB objects, 100-200 s serial
        # long-poles) into one generated per-type TU each: small, parallel, low-RAM.
        # Type set unchanged (split on SD_COMMON_TYPES = 13 types -> indices 0..12).
        include(SplitHeavyInstantiations)
        split_heavy_op(LOOPS_SOURCES_CUDA transform_strict.cu  loops/cuda/transform/transform_strict.cu  12)
        split_heavy_op(LOOPS_SOURCES_CUDA transform_same.cu    loops/cuda/transform/transform_same.cu    12)
        split_heavy_op(LOOPS_SOURCES_CUDA transform_float.cu   loops/cuda/transform/transform_float.cu   12)
        split_heavy_op(LOOPS_SOURCES_CUDA transform_any.cu     loops/cuda/transform/transform_any.cu     12)
        split_heavy_op(LOOPS_SOURCES_CUDA broadcasting_bool.cu loops/cuda/broadcasting_bool.cu           12)
        split_heavy_op(LOOPS_SOURCES_CUDA scalar_bool.cu       loops/cuda/scalar_bool.cu                 12)
        split_heavy_op(LOOPS_SOURCES_CUDA pairwise_bool.cu     loops/cuda/pairwise_bool.cu               12)
        split_heavy_op(LOOPS_SOURCES_CUDA reduce_bool.cu       loops/cuda/reduce/reduce_bool.cu          12)
        split_heavy_op(LOOPS_SOURCES_CUDA reduce_long.cu       loops/cuda/reduce/reduce_long.cu          12)
        split_heavy_op(LOOPS_SOURCES_CUDA reduce_same.cu       loops/cuda/reduce/reduce_same.cu          12)
        split_heavy_op(LOOPS_SOURCES_CUDA bitonicSortStep.cu   loops/cuda/specials/bitonicSortStep.cu    12)
        split_heavy_op(LOOPS_SOURCES_CUDA bitonicArbitraryStep.cu loops/cuda/specials/bitonicArbitraryStep.cu 12)
        split_heavy_op(LOOPS_SOURCES_CUDA oesTad.cu            loops/cuda/specials/oesTad.cu             12)
        # NOTE: where.cu intentionally NOT split — its object weight is the shared
        # device-kernel impl (compiled into every TU), not the type instantiations
        # (a split TU is ~7 MB = the monolith), so splitting multiplies work for no
        # benefit. Only instantiation-dominated ops belong here.
        file(GLOB_RECURSE SYSTEM_CONFIG_SOURCES ./include/system/config/impl/*.cpp)
        file(GLOB_RECURSE GRAPH_CUDA_SOURCES ./include/graph/*.cu)
        # TritonGraphBackend_compile uses the CUDA host runtime but contains no device code.
        # Compile it as C++ so CUDA compilation units never parse Triton/MLIR headers.
        set(_TRITON_HOST_COMPILE_SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/gpu/TritonGraphBackend_compile.cu")
        set_source_files_properties("${_TRITON_HOST_COMPILE_SOURCE}" PROPERTIES LANGUAGE CXX)
        # GCC and Clang infer the language from the .cu suffix even after CMake assigns
        # the source to the CXX compiler. Without an explicit language flag they treat
        # it as linker input, report success, and emit no object for the final link.
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            set_property(SOURCE "${_TRITON_HOST_COMPILE_SOURCE}" APPEND PROPERTY
                COMPILE_OPTIONS "-x;c++")
        endif()

        # Exclude Triton .cu files when Triton is not available (they require MLIR headers)
        if(NOT HAVE_TRITON)
            file(GLOB_RECURSE _triton_cu_sources ./include/graph/gpu/Triton*.cu)
            if(_triton_cu_sources)
                list(REMOVE_ITEM GRAPH_CUDA_SOURCES ${_triton_cu_sources})
                list(LENGTH _triton_cu_sources _triton_cu_count)
                message(STATUS "Excluded ${_triton_cu_count} Triton .cu files (HAVE_TRITON=OFF)")
            endif()
        endif()
        file(GLOB_RECURSE VALIDATION_SOURCES ./include/array/DataTypeValidation.cpp)
        file(GLOB_RECURSE CUDA_FOREIGN_BACKEND_SOURCES
                ./include/legacy/cpu/*.cpp
                ./include/helpers/cpu/*.cpp
                ./include/helpers/vulkan/*.cpp
                ./include/array/cpu/*.cpp
                ./include/array/vulkan/*.cpp
                ./include/execution/cpu/*.cpp
                ./include/graph/cpu/*.cpp
                ./include/ops/declarable/helpers/cpu/*.cpp
        )

        list(APPEND ALL_SOURCES_LIST
                ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${CUSTOMOPS_HELPERS_SOURCES}
                ${HELPERS_SOURCES} ${LOOPS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES_CUDA}
                ${GRAPH_CUDA_SOURCES} ${VALIDATION_SOURCES} ${SYSTEM_CONFIG_SOURCES}
        )
        list(REMOVE_ITEM ALL_SOURCES_LIST ${CUDA_FOREIGN_BACKEND_SOURCES})

        if(HAVE_CUDNN)
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ./include/ops/declarable/platform/cudnn/*.cu)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_CUDNN_SOURCES})
        endif()

        # ZLUDA's CUDA ABI does not expose stream-ordered allocation, so
        # compile one SDK-isolated HIP bridge that forwards the underlying ZLUDA
        # stream handle to hipMallocAsync/hipFreeAsync.
        if(HAVE_ZLUDA AND ZLUDA_TARGET_BACKEND STREQUAL "AMD" AND ROCM_HIP_RUNTIME_LIBRARY)
            list(APPEND ALL_SOURCES_LIST
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/cuda/ZludaHipMemoryBridge.cpp")
            message(STATUS "✅ Added ROCm stream-ordered memory bridge for ZLUDA AMD")
        endif()

        # ZLUDA MIOpen support for AMD GPUs (cuDNN alternative)
        if(HAVE_ZLUDA AND ZLUDA_TARGET_BACKEND STREQUAL "AMD" AND
           ZLUDA_MIOPEN_HELPERS_ENABLED)
            file(GLOB_RECURSE CUSTOMOPS_MIOPEN_SOURCES ./include/ops/declarable/platform/miopen/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_MIOPEN_SOURCES})
            message(STATUS "✅ Added MIOpen platform sources for ZLUDA AMD")
        endif()

        message(STATUS "🚀 CUDA build: Enhanced template system will generate additional CUDA instantiations")
    elseif(SD_VULKAN)
        # Vulkan mirrors CUDA's backend boundary: common control-plane
        # implementations plus Vulkan-owned platform definitions. No CPU loop,
        # helper, workspace, NativeOps, or eager-execution translation unit is
        # admitted here.
        file(GLOB_RECURSE EXEC_SOURCES
            ./include/execution/impl/*.cpp
            ./include/execution/vulkan/*.cpp)
        file(GLOB_RECURSE ARRAY_SOURCES
            ./include/array/impl/*.cpp
            ./include/array/vulkan/*.cpp)
        file(GLOB_RECURSE MEMORY_SOURCES
            ./include/memory/impl/*.cpp
            ./include/memory/vulkan/*.cpp)

        # Declarable helper implementations under helpers/impl are host
        # algorithms or platform-selection placeholders. Vulkan implementations
        # belong under the Vulkan backend and remain absent until authored there.
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES
            ./include/ops/declarable/helpers/vulkan/*.cpp)

        # Autoregressive generation is a host-orchestrated control loop around
        # backend-owned plan execution. Reuse the canonical scalar controller,
        # sampling policy, and KV quantization implementation on Vulkan instead
        # of introducing an Android/Kotlin loop or a second Vulkan-specific
        # sampler. Tensor forwards and KV scatter remain Vulkan-owned; logits are
        # synchronized only at the sampling boundary. These sources retain their
        # historical cpu/ location until the CPU/Vulkan host-control split is
        # moved mechanically into a shared host/ directory.
        set(_vulkan_host_generation_helpers
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/autoregressive_decode.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/token_sample.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/sampling_penalties.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/kv_cache_quantize.cpp")
        list(APPEND CUSTOMOPS_HELPERS_SOURCES ${_vulkan_host_generation_helpers})

        file(GLOB_RECURSE _vulkan_excluded_host_op_helpers
            ./include/ops/declarable/helpers/impl/*.cpp)
        list(LENGTH _vulkan_excluded_host_op_helpers _vulkan_excluded_host_op_helper_count)

        # This is an ownership allowlist, not a denylist. These implementations
        # manipulate shape/index metadata, caches, allocation/control state, or
        # diagnostics; they do not implement tensor math on primary buffers.
        set(_vulkan_common_helper_relpaths
            ArrayUtils.cpp
            AttentionWorkspace.cpp
            BitwiseUtils.cpp
            BlasHelper.cpp
            ConstantShapeHelper.cpp
            ConstantTadHelper.cpp
            DirectShapeTrie.cpp
            DirectTadTrie.cpp
            DebugHelper.cpp
            DynamicKernelLoader.cpp
            EnumUtils.cpp
            HelperVersionRegistry.cpp
            KernelAutoTuner.cpp
            KernelPerformanceRegistry.cpp
            KernelSelectionEnvironment.cpp
            MmulHelper_dispatch.cpp
            RandomLauncher.cpp
            ModularHasher.cpp
            OmpLaunchHelper.cpp
            OpArgsHolder.cpp
            OpTracker.cpp
            ShapeBufferCreatorHelper.cpp
            ShapeBufferPlatformHelper.cpp
            ShapeBuilders.cpp
            ShapeUtils.cpp
            SimpleReadWriteLock.cpp
            StringUtils.cpp
            TAD.cpp
            TadCalculator.cpp
            TransferMetrics.cpp
            generic/TypedTrie.cpp
            helper_hash.cpp
            logger.cpp
            reshapeNoCopy.cpp)
        set(HELPERS_SOURCES "")
        foreach(_vulkan_common_helper_relpath IN LISTS _vulkan_common_helper_relpaths)
            set(_vulkan_common_helper
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/${_vulkan_common_helper_relpath}")
            if(NOT EXISTS "${_vulkan_common_helper}")
                message(FATAL_ERROR
                    "SD_VULKAN helper allowlist entry does not exist: ${_vulkan_common_helper}")
            endif()
            list(APPEND HELPERS_SOURCES "${_vulkan_common_helper}")
        endforeach()
        file(GLOB_RECURSE VULKAN_HELPERS_SOURCES
            ./include/helpers/vulkan/*.cpp)
        list(APPEND HELPERS_SOURCES
            ${VULKAN_HELPERS_SOURCES}
            "${CMAKE_CURRENT_SOURCE_DIR}/include/build_info.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ConstMessages.cpp")
        file(GLOB_RECURSE _vulkan_all_common_helper_sources
            ./include/helpers/impl/*.cpp)
        set(_vulkan_excluded_host_helpers ${_vulkan_all_common_helper_sources})
        list(REMOVE_ITEM _vulkan_excluded_host_helpers ${HELPERS_SOURCES})
        list(LENGTH _vulkan_excluded_host_helpers _vulkan_excluded_host_helper_count)
        message(STATUS
            "SD_VULKAN helper boundary: ${_vulkan_excluded_host_helper_count} host-compute helpers and "
            "${_vulkan_excluded_host_op_helper_count} non-Vulkan declarable helper TUs excluded")

        file(GLOB_RECURSE LEGACY_SOURCES
            ./include/legacy/impl/*.cpp
            ./include/legacy/vulkan/*.cpp)
        file(GLOB_RECURSE LOOPS_SOURCES
            ./include/loops/vulkan/*.cpp)
        file(GLOB_RECURSE _vulkan_excluded_host_loop_sources
            ./include/loops/impl/*.cpp)
        list(LENGTH _vulkan_excluded_host_loop_sources _vulkan_excluded_host_loop_count)
        message(STATUS
            "SD_VULKAN loop boundary: ${_vulkan_excluded_host_loop_count} host loop TUs excluded")
        file(GLOB_RECURSE SYSTEM_CONFIG_SOURCES
            ./include/system/config/impl/*.cpp
            ./include/system/config/vulkan/*.cpp)
        # MLIR dialect sources are compiler-only. AOT Vulkan runtimes must not
        # carry MLIR/LLVM objects or link dependencies onto mobile devices.
        set(MLIR_DIALECT_SOURCES "")
        if(HELPERS_mlir)
            file(GLOB_RECURSE MLIR_DIALECT_SOURCES
                ./include/mlir/dialect/*.cpp)
        endif()
        file(GLOB VALIDATION_SOURCES
            ./include/array/DataTypeValidation.cpp)

        # These files sit in nominally common directories but implement a
        # foreign backend, an explicit CPU functional fallback, or an ABI entry
        # point that Vulkan implements in legacy/vulkan/NativeOps.cpp. Filter the
        # final assembled list by source-tree-relative suffix: CMake glob results
        # are not guaranteed to have the same path spelling as hand-built absolute
        # paths, and GRAPH_SOURCES was already appended above.
        set(_vulkan_foreign_common_source_suffixes
            "include/execution/impl/DeviceManager[.]cpp"
            "include/graph/impl/FunctionalReplayHandle[.]cpp"
            "include/graph/cpu/NativeDynamicShapePlan_cuda_stubs[.]cpp"
            "include/helpers/impl/CudaLaunchHelper[.]cpp"
            "include/helpers/impl/CutlassGemmHelper[.]cpp"
            "include/helpers/impl/CutlassHelper[.]cpp"
            "include/legacy/impl/Environment_CudaConfig[.]cpp"
            "include/legacy/impl/NativeOpsHelpers_Execution[.]cpp"
            "include/legacy/impl/NativeOpsHelpers_Inspection[.]cpp"
            "include/system/config/impl/CudaDeviceConfig[.]cpp")

        list(APPEND ALL_SOURCES_LIST
            ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES}
            ${CUSTOMOPS_HELPERS_SOURCES} ${HELPERS_SOURCES}
            ${LEGACY_SOURCES} ${LOOPS_SOURCES} ${SYSTEM_CONFIG_SOURCES}
            ${MLIR_DIALECT_SOURCES} ${VALIDATION_SOURCES})
        foreach(_vulkan_foreign_suffix IN LISTS _vulkan_foreign_common_source_suffixes)
            list(FILTER ALL_SOURCES_LIST EXCLUDE REGEX "/${_vulkan_foreign_suffix}$")
        endforeach()

        message(STATUS "SD_VULKAN: selected common control plane and Vulkan-owned device sources")
    else()
        file(GLOB_RECURSE EXEC_SOURCES ./include/execution/*.cpp)
        file(GLOB_RECURSE VULKAN_EXEC_SOURCES ./include/execution/vulkan/*.cpp)
        list(REMOVE_ITEM EXEC_SOURCES ${VULKAN_EXEC_SOURCES})
        file(GLOB_RECURSE ARRAY_SOURCES ./include/array/*.cpp)
        # Backend-owned Vulkan array translation units must not enter generic CPU
        # artifacts merely because GLOB_RECURSE sees their directory.
        file(GLOB_RECURSE VULKAN_ARRAY_SOURCES ./include/array/vulkan/*.cpp)
        list(REMOVE_ITEM ARRAY_SOURCES ${VULKAN_ARRAY_SOURCES})
        # CPU/NNAPI owns only the common memory implementation and CPU
        # workspaces. Backend SDK bridges are admitted by their backend branch
        # (for example, the ZLUDA HIP bridge above), never by a recursive CPU glob.
        file(GLOB_RECURSE MEMORY_SOURCES
            ./include/memory/impl/*.cpp
            ./include/memory/cpu/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL_SOURCES ./include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU_SOURCES ./include/ops/declarable/helpers/cpu/*.cpp)
        file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp  ./include/helpers/cpu/*.cpp)
        file(GLOB_RECURSE VULKAN_HELPER_SOURCES ./include/helpers/vulkan/*.cpp)
        list(REMOVE_ITEM HELPERS_SOURCES ${VULKAN_HELPER_SOURCES})
        file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/cpu/*.cpp)
        file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/*.cpp)
        file(GLOB_RECURSE VULKAN_LOOP_SOURCES ./include/loops/vulkan/*.cpp)
        list(REMOVE_ITEM LOOPS_SOURCES ${VULKAN_LOOP_SOURCES})
        file(GLOB_RECURSE SYSTEM_CONFIG_SOURCES ./include/system/config/impl/*.cpp)

        list(APPEND ALL_SOURCES_LIST
                ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES}
                ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${HELPERS_SOURCES} ${LEGACY_SOURCES}
                ${LOOPS_SOURCES} ${SYSTEM_CONFIG_SOURCES}
        )

        # --- Multi-Helper Source Collection (CPU Build) ---
        # All enabled helpers are compiled simultaneously for dynamic kernel selection

        # OneDNN / MKL-DNN helper
        if(HAVE_ONEDNN)
            file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES ./include/ops/declarable/platform/mkldnn/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_ONEDNN_SOURCES})
            list(LENGTH CUSTOMOPS_ONEDNN_SOURCES onednn_count)
            message(STATUS "✅ Added OneDNN platform sources: ${onednn_count} files")
        endif()

        # ARM Compute Library helper
        if(HAVE_ARMCOMPUTE)
            file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES ./include/ops/declarable/platform/armcompute/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_ARMCOMPUTE_SOURCES})
            list(LENGTH CUSTOMOPS_ARMCOMPUTE_SOURCES armcompute_count)
            message(STATUS "✅ Added ARM Compute platform sources: ${armcompute_count} files")
        endif()

        # MLIR/LLVM JIT helper
        if(HAVE_MLIR)
            file(GLOB_RECURSE CUSTOMOPS_MLIR_SOURCES ./include/ops/declarable/platform/mlir/*.cpp)
            file(GLOB_RECURSE MLIR_DIALECT_SOURCES ./include/mlir/**/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_MLIR_SOURCES} ${MLIR_DIALECT_SOURCES})
            list(LENGTH CUSTOMOPS_MLIR_SOURCES mlir_count)
            message(STATUS "✅ Added MLIR platform sources: ${mlir_count} files")
        endif()

        # Metal Performance Shaders helper (macOS/iOS)
        if(HAVE_MPS)
            file(GLOB_RECURSE CUSTOMOPS_MPS_SOURCES ./include/ops/declarable/platform/mps/*.mm)
            file(GLOB_RECURSE METAL_REPLAY_SOURCES ./include/graph/metal/*.mm)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_MPS_SOURCES} ${METAL_REPLAY_SOURCES})
            # Set ObjC++ properties on .mm files for proper ARC compilation
            foreach(mm_source ${CUSTOMOPS_MPS_SOURCES} ${METAL_REPLAY_SOURCES})
                set_source_files_properties(${mm_source} PROPERTIES
                    LANGUAGE OBJCXX
                    COMPILE_FLAGS "-fobjc-arc"
                )
            endforeach()
            list(LENGTH CUSTOMOPS_MPS_SOURCES mps_count)
            list(LENGTH METAL_REPLAY_SOURCES metal_count)
            message(STATUS "✅ Added MPS platform sources: ${mps_count} helpers, ${metal_count} Metal replay files")
        endif()

        # Apple Accelerate framework helper (macOS/iOS)
        if(HAVE_ACCELERATE)
            file(GLOB_RECURSE CUSTOMOPS_ACCELERATE_SOURCES ./include/ops/declarable/platform/accelerate/*.cpp ./include/ops/declarable/platform/accelerate/*.mm)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_ACCELERATE_SOURCES})
            list(LENGTH CUSTOMOPS_ACCELERATE_SOURCES accelerate_count)
            message(STATUS "✅ Added Accelerate platform sources: ${accelerate_count} files")
        endif()

        # VLM (Vision-Language Model) operations
        if(HAVE_VLM)
            file(GLOB_RECURSE CUSTOMOPS_VLM_SOURCES ./include/ops/declarable/platform/vlm/*.cpp)
            if(CUSTOMOPS_VLM_SOURCES)
                list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_VLM_SOURCES})
                list(LENGTH CUSTOMOPS_VLM_SOURCES vlm_count)
                message(STATUS "✅ Added VLM platform sources: ${vlm_count} files")
            endif()
        endif()

        # --- Multi-Backend Dispatcher Sources ---
        # Always include multi-platform dispatcher for dynamic kernel selection
        if(SD_DYNAMIC_KERNEL_SELECTION)
            file(GLOB_RECURSE MULTI_PLATFORM_DISPATCHER_SOURCES
                ./include/ops/declarable/MultiPlatformDispatcher.h
                ./include/ops/declarable/impl/MultiPlatformDispatcher.cpp
                ./include/helpers/KernelSelectionEnvironment.h
                ./include/helpers/impl/KernelSelectionEnvironment.cpp
                ./include/helpers/KernelAutoTuner.h
                ./include/helpers/impl/KernelAutoTuner.cpp
                ./include/helpers/KernelPerformanceRegistry.h
                ./include/helpers/impl/KernelPerformanceRegistry.cpp
            )
            list(APPEND ALL_SOURCES_LIST ${MULTI_PLATFORM_DISPATCHER_SOURCES})
            message(STATUS "✅ Added Multi-Platform Dispatcher sources for dynamic kernel selection")
        endif()

        message(STATUS "🖥️  CPU build: Enhanced template system will generate optimized CPU instantiations")
    endif()

    # CUDA, ZLUDA/HIP, cuDNN, and MIOpen translation units require their
    # respective SDKs and belong only to the SD_CUDA backend branch. Enforce the
    # ownership boundary for every non-CUDA target (including Android CPU/NNAPI)
    # after all branch-local source lists have been assembled.
    if(NOT SD_CUDA)
        set(_non_cuda_backend_source_regex
            "/include/(array|execution|graph|helpers|legacy|loops|memory|system/config)/(cuda|gpu|hip)/|/include/ops/declarable/(helpers|platform)/(cuda|cudnn|miopen)/")
        set(_non_cuda_backend_sources "")
        foreach(_candidate_source IN LISTS ALL_SOURCES_LIST)
            if(_candidate_source MATCHES "${_non_cuda_backend_source_regex}")
                list(APPEND _non_cuda_backend_sources "${_candidate_source}")
            endif()
        endforeach()
        if(_non_cuda_backend_sources)
            list(REMOVE_ITEM ALL_SOURCES_LIST ${_non_cuda_backend_sources})
            list(LENGTH _non_cuda_backend_sources _non_cuda_backend_source_count)
            message(STATUS
                "Excluded ${_non_cuda_backend_source_count} CUDA/ZLUDA/HIP-only sources from non-CUDA target")
        endif()
    endif()

    # When SD_GCC_FUNCTRACE is ON, the binary exceeds 2GB and PLT relocations fail
    # libc_nonshared.a contains precompiled functions (like atexit) that use PLT
    # We provide our own implementations compiled with -fno-plt to override them
    if(SD_GCC_FUNCTRACE)
        list(APPEND ALL_SOURCES_LIST ${CMAKE_CURRENT_SOURCE_DIR}/include/platform/noplt_libc_stubs.c)
        message(STATUS "✅ Added no-PLT libc stubs for >2GB functrace build compatibility")
    endif()

    # Add only the template sources selected for this backend.
    list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_GENERIC_SOURCES})
    list(REMOVE_DUPLICATES ALL_SOURCES_LIST)

    # Keep the final source graph honest if later collection logic changes.
    if(NOT SD_CUDA)
        foreach(_candidate_source IN LISTS ALL_SOURCES_LIST)
            if(_candidate_source MATCHES "${_non_cuda_backend_source_regex}")
                message(FATAL_ERROR
                    "Non-CUDA source boundary admitted CUDA/ZLUDA/HIP source: ${_candidate_source}")
            endif()
        endforeach()
    endif()

    if(SD_VULKAN)
        # Make source ownership enforceable. A future broad glob must fail
        # configuration instead of silently reintroducing another backend.
        # Match implementation roots, not semantic op families such as
        # ops/declarable/generic/blas.
        set(_vulkan_foreign_backend_names
            "cpu|cuda|gpu|hexagon|hip|levelzero|metal|tpu|mkl|mkldnn|onednn|openvino|armcompute|cudnn|miopen|mps|accelerate")
        set(_vulkan_forbidden_source_regex
            "/include/(array|execution|graph|helpers|legacy|loops|memory|system/config)/(${_vulkan_foreign_backend_names})/|/include/ops/declarable/(helpers|platform)/(${_vulkan_foreign_backend_names})/")
        set(_vulkan_forbidden_sources "")
        foreach(_vulkan_source IN LISTS ALL_SOURCES_LIST)
            # The native generation controller and sampler are portable host
            # orchestration despite their historical cpu/ path. Only the
            # explicit allowlist assembled above may cross this path-based
            # ownership check; every other foreign-backend source still fails
            # configuration.
            list(FIND _vulkan_host_generation_helpers
                "${_vulkan_source}" _vulkan_generation_helper_index)
            if(_vulkan_source MATCHES "${_vulkan_forbidden_source_regex}" AND
               _vulkan_generation_helper_index EQUAL -1)
                list(APPEND _vulkan_forbidden_sources "${_vulkan_source}")
            endif()
        endforeach()
        if(_vulkan_forbidden_sources)
            list(JOIN _vulkan_forbidden_sources "\n  " _vulkan_forbidden_sources_text)
            message(FATAL_ERROR
                "SD_VULKAN source boundary admitted foreign backend sources:\n  "
                "${_vulkan_forbidden_sources_text}")
        endif()
        list(LENGTH ALL_SOURCES_LIST _vulkan_source_count)
        message(STATUS "SD_VULKAN: verified ${_vulkan_source_count} sources against the backend ownership boundary")
    endif()

    set(${out_source_list} ${ALL_SOURCES_LIST} PARENT_SCOPE)

    list(LENGTH CUSTOMOPS_GENERIC_SOURCES template_source_count)
    if(SD_CUDA)
        message(STATUS "📊 Total CUDA template-generated sources: ${template_source_count}")
    elseif(SD_VULKAN)
        message(STATUS "📊 Total Vulkan template-generated sources: ${template_source_count}")
    else()
        message(STATUS "📊 Total CPU template-generated sources: ${template_source_count}")
    endif()
endfunction()

function(configure_vulkan_linking main_target_name)
    if(NOT HAVE_VULKAN OR NOT DEFINED VULKAN_LIBRARY OR NOT VULKAN_LIBRARY)
        message(FATAL_ERROR
            "SD_VULKAN requires a resolved Vulkan loader before linking ${main_target_name}")
    endif()

    set(_vulkan_object_target "${main_target_name}_object")
    target_link_libraries(${main_target_name} PUBLIC
        flatbuffers_interface
        ${CMAKE_DL_LIBS}
        "${VULKAN_LIBRARY}")
    if(SD_JNI_ENABLED AND JVM_LIBRARY)
        target_link_libraries(${main_target_name} PUBLIC ${JVM_LIBRARY})
    endif()

    # Shared libraries on ELF normally permit unresolved symbols. Vulkan must
    # instead expose every missing backend implementation during this build.
    if(UNIX AND NOT APPLE)
        target_link_options(${main_target_name} PRIVATE "LINKER:-z,defs")
    endif()

    if(TARGET ${_vulkan_object_target})
        target_include_directories(${_vulkan_object_target} SYSTEM PRIVATE
            "${VULKAN_INCLUDE_DIR}")
    endif()

    # Vulkan consumes the validated MLIR::SPIRV target. On MinGW that target is
    # the upstream static component graph; supported hosts retain shared DSOs.
    if(HAVE_MLIR AND DEFINED MLIR)
        if(TARGET ${_vulkan_object_target})
            target_include_directories(${_vulkan_object_target} BEFORE PRIVATE
                ${MLIR_INCLUDE_DIRS})
            target_link_libraries(${_vulkan_object_target} PUBLIC ${MLIR})
            target_compile_definitions(${_vulkan_object_target} PUBLIC HAVE_MLIR=1)
        endif()
        target_link_libraries(${main_target_name} PUBLIC ${MLIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MLIR=1)
        message(STATUS "🔗 Vulkan: linking the selected MLIR-to-SPIR-V package")
    endif()

    # The dependency producer must finish before any Vulkan source includes its
    # generated MLIR headers. Linkage itself is owned only by MLIR::SPIRV above;
    # linking triton_interface here duplicated the same package graph.
    if(HAVE_MANAGED_LLVM_MLIR AND TARGET triton_external)
        if(TARGET ${_vulkan_object_target})
            add_dependencies(${_vulkan_object_target} triton_external)
        endif()
        add_dependencies(${main_target_name} triton_external)
        message(STATUS "🔒 Vulkan: waiting for managed LLVM/MLIR infrastructure")
    endif()

    # Common host coordination code uses OpenMP, but no CPU BLAS or CPU graph
    # backend is linked into the Vulkan artifact.
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${main_target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # SDZ archives are common backend metadata.
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        if(TARGET ${_vulkan_object_target})
            target_compile_definitions(${_vulkan_object_target} PUBLIC HAVE_ZLIB=1)
            target_link_libraries(${_vulkan_object_target} PUBLIC ZLIB::ZLIB)
        endif()
        target_link_libraries(${main_target_name} PUBLIC ZLIB::ZLIB)
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ZLIB=1)
    endif()

    if(SD_GCC_FUNCTRACE)
        target_link_libraries(${main_target_name} PUBLIC bfd dw dl elf pthread)
    endif()

    message(STATUS
        "🔗 Vulkan link boundary: flatbuffers + Vulkan + optional JNI/MLIR; "
        "no Triton DSP, BLAS, OneDNN, OpenVINO, MKL, CUDA, HIP, or vendor execution helper")
    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

function(configure_cpu_linking main_target_name)
    # Core libraries
    # CMAKE_DL_LIBS provides -ldl on Linux (needed for dlopen/dlsym in DynamicKernelLoader).
    # JVM_LIBRARY is intentionally conditional: Android/mobile native builds set
    # SD_BUILD_WITH_JAVA=OFF and must not inherit a NOTFOUND desktop JVM path.
    target_link_libraries(${main_target_name} PUBLIC
            ${OPENBLAS_LIBRARIES} ${BLAS_LIBRARIES} flatbuffers_interface ${CMAKE_DL_LIBS})
    if(SD_JNI_ENABLED AND JVM_LIBRARY)
        target_link_libraries(${main_target_name} PUBLIC ${JVM_LIBRARY})
    endif()

    # Android CPU-family builds include NnapiGraphBackend whenever HAVE_NNAPI is
    # enabled. Keep that system ABI explicit here just as the device-only linker
    # does, so -z,defs validates the graph backend and the final ELF records the
    # required libneuralnetworks dependency.
    if(HAVE_NNAPI)
        if(NOT ANDROID)
            message(FATAL_ERROR
                "HAVE_NNAPI requires an Android toolchain for ${main_target_name}")
        endif()
        find_library(_sd_cpu_nnapi_library neuralnetworks)
        if(NOT _sd_cpu_nnapi_library)
            message(FATAL_ERROR
                "Android system libneuralnetworks was not found for ${main_target_name}")
        endif()
        target_link_libraries(${main_target_name} PUBLIC "${_sd_cpu_nnapi_library}")
        message(STATUS
            "NNAPI CPU-family link boundary: ${main_target_name} -> ${_sd_cpu_nnapi_library}")
    endif()

    # --- Multi-Helper Library Linking ---
    # Link all enabled helper libraries for multi-backend support

    # OneDNN/MKL-DNN
    if(HAVE_ONEDNN AND DEFINED ONEDNN)
        target_link_libraries(${main_target_name} PUBLIC ${ONEDNN})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ONEDNN=1)
        message(STATUS "🔗 Linking OneDNN helper")
    endif()

    # ARM Compute Library
    if(HAVE_ARMCOMPUTE AND DEFINED ARMCOMPUTE_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${ARMCOMPUTE_LIBRARIES})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ARMCOMPUTE=1)
        message(STATUS "🔗 Linking ARM Compute helper")
    endif()

    # MLIR/LLVM JIT
    if(HAVE_MLIR AND DEFINED MLIR)
        # Compile and link against the same imported target. The backend sources live
        # in the object library, so linking MLIR only to the final shared library can
        # silently compile them against unrelated headers found elsewhere.
        set(_mlir_object_target "${main_target_name}_object")
        if(TARGET ${_mlir_object_target})
            # Put the selected package's headers on the ordinary, ordered include path.
            # Linking the imported target alone may classify them as SYSTEM includes,
            # after an ambient CPATH LLVM that does not match the selected shared DSOs.
            target_include_directories(${_mlir_object_target} BEFORE PRIVATE ${MLIR_INCLUDE_DIRS})
            target_link_libraries(${_mlir_object_target} PUBLIC ${MLIR})
            target_compile_definitions(${_mlir_object_target} PUBLIC HAVE_MLIR=1)
        endif()
        target_link_libraries(${main_target_name} PUBLIC ${MLIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MLIR=1)
        message(STATUS "🔗 Linking MLIR helper")
    endif()

    # Metal Performance Shaders (macOS/iOS) + Metal ICB Replay for DSP
    if(HAVE_MPS AND DEFINED MPS_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${MPS_LIBRARIES})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MPS=1 SD_METAL=1)
        message(STATUS "🔗 Linking MPS helper + Metal ICB replay")
    endif()

    # Triton GPU Compiler
    if(HAVE_TRITON AND DEFINED TRITON)
        target_link_libraries(${main_target_name} PUBLIC ${TRITON})
        # HAVE_TRITON is provided via generated config.h, not as a global -D flag.
        # Global -D flags change every file's compiler command line, breaking ccache
        # for the 276 generated .cu template instantiation files.
        message(STATUS "🔗 Linking Triton GPU compiler backend")
    endif()

    # OpenVINO CPU Graph Backend
    if(HAVE_OPENVINO AND TARGET openvino_interface)
        target_link_libraries(${main_target_name} PUBLIC openvino_interface)
        message(STATUS "🔗 Linking OpenVINO CPU graph backend")
    endif()

    # CUTLASS (Header-only CUDA GEMM Templates)
    if(HAVE_CUTLASS AND TARGET cutlass_interface)
        target_link_libraries(${main_target_name} PUBLIC cutlass_interface)
        # HAVE_CUTLASS is provided via generated config.h, not as a global -D flag.
        message(STATUS "Linking CUTLASS header-only templates")
    endif()

    # MLX (Apple Metal GPU Compute)
    if(HAVE_MLX AND DEFINED MLX)
        target_link_libraries(${main_target_name} PUBLIC ${MLX})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MLX=1)
        message(STATUS "🔗 Linking MLX Apple Silicon backend")
    endif()

    # OpenVINO (Intel CPU Graph Backend)
    if(HAVE_OPENVINO AND DEFINED OPENVINO_LIB)
        target_link_libraries(${main_target_name} PUBLIC ${OPENVINO_LIB})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_OPENVINO=1)
        message(STATUS "🔗 Linking OpenVINO CPU graph backend")

        # Bundle TBB shared library alongside libnd4jcpu.so so JavaCPP packaging
        # can include it in the native jar. libtbb.so.12 is OpenVINO's threading
        # runtime — it is always a shared library (no static variant in OpenVINO's build).
        # The pom.xml copy step globs all *.so / *.so.* from CMAKE_BINARY_DIR into
        # the jar. Without this step, libtbb.so.12 only exists at the absolute build
        # path that is baked into RUNPATH; after a clean build that path no longer exists.
        if(NOT WIN32)
            set(_OV_TBB_SRC_DIR "${CMAKE_BINARY_DIR}/openvino_install/runtime/3rdparty/tbb/lib64")
            if(EXISTS "${_OV_TBB_SRC_DIR}")
                # Copy at configure time for the "already built" reuse path
                file(GLOB _tbb_so_files "${_OV_TBB_SRC_DIR}/libtbb.so*"
                                        "${_OV_TBB_SRC_DIR}/libtbbmalloc.so*"
                                        "${_OV_TBB_SRC_DIR}/libtbbmalloc_proxy.so*")
                foreach(_f ${_tbb_so_files})
                    get_filename_component(_fname "${_f}" NAME)
                    if(NOT EXISTS "${CMAKE_BINARY_DIR}/${_fname}")
                        message(STATUS "  Copying TBB library for bundling: ${_fname}")
                        execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${_f}" "${CMAKE_BINARY_DIR}/${_fname}")
                    endif()
                endforeach()
            endif()
            # Also copy at build time (POST_BUILD) for the case where OpenVINO is
            # built as part of this cmake invocation (ExternalProject path).
            add_custom_command(TARGET ${main_target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                    -D_OV_TBB_SRC_DIR=${CMAKE_BINARY_DIR}/openvino_install/runtime/3rdparty/tbb/lib64
                    -D_DST_DIR=${CMAKE_BINARY_DIR}
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_tbb_libs.cmake"
                COMMENT "Copying OpenVINO TBB shared libraries to output directory for jar bundling"
            )
        endif()
    endif()

    # NCCL (Multi-GPU Collective Communications)
    if(HAVE_NCCL AND DEFINED NCCL_LIB)
        target_link_libraries(${main_target_name} PUBLIC ${NCCL_LIB})
        target_include_directories(${main_target_name} PUBLIC ${NCCL_INCLUDE_DIRS})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_NCCL=1)
        message(STATUS "🔗 Linking NCCL for multi-GPU collective communications")
    endif()

    # Apple Accelerate Framework (macOS/iOS)
    if(HAVE_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            target_link_libraries(${main_target_name} PUBLIC ${ACCELERATE_FRAMEWORK})
            target_compile_definitions(${main_target_name} PUBLIC HAVE_ACCELERATE=1)
            message(STATUS "🔗 Linking Accelerate helper")
        endif()
    endif()

    # Dynamic kernel selection compile definitions
    if(SD_DYNAMIC_KERNEL_SELECTION)
        target_compile_definitions(${main_target_name} PUBLIC SD_DYNAMIC_KERNEL_SELECTION=1)
        target_compile_definitions(${main_target_name} PUBLIC SD_KERNEL_STRATEGY="${SD_KERNEL_STRATEGY}")
        if(SD_KERNEL_AUTOTUNING)
            target_compile_definitions(${main_target_name} PUBLIC SD_KERNEL_AUTOTUNING=1)
        endif()
        if(SD_KERNEL_CACHING)
            target_compile_definitions(${main_target_name} PUBLIC SD_KERNEL_CACHING=1)
        endif()
        # Pass helper priority as compile definition
        if(SD_HELPER_PRIORITY)
            string(REPLACE ";" "," HELPER_PRIORITY_CSV "${SD_HELPER_PRIORITY}")
            target_compile_definitions(${main_target_name} PUBLIC SD_HELPER_PRIORITY="${HELPER_PRIORITY_CSV}")
        endif()
    endif()

    # Add debug libraries when SD_GCC_FUNCTRACE is enabled
    if(SD_GCC_FUNCTRACE)
        # FIX (Session #1045): Do NOT link libunwind!
        # - We removed --unwindlib=libunwind from compile flags in CompilerFlags.cmake
        # - Using system libgcc_s for exception handling (JVM compatible)
        # - libunwind would conflict with JVM's libgcc_s causing _Unwind_SetGR crashes
        # Both Clang and GCC now use the same debug libraries (no libunwind)
        target_link_libraries(${main_target_name} PUBLIC
                bfd dw dl elf pthread)
        message(STATUS "✅ Added debug libraries for SD_GCC_FUNCTRACE (using libgcc_s for unwinding, JVM compatible)")
    endif()

    # On macOS, Homebrew's libomp is keg-only and not found by default.
    # Set hints so find_package(OpenMP) can locate it.
    if(APPLE AND EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp" CACHE STRING "" FORCE)
        set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
        include_directories("/opt/homebrew/opt/libomp/include")
    elseif(APPLE AND EXISTS "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp" CACHE STRING "" FORCE)
        set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
        include_directories("/usr/local/opt/libomp/include")
    endif()

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${main_target_name} PUBLIC OpenMP::OpenMP_CXX)

        # Android NDK: OpenMP::OpenMP_CXX imported target may not propagate the
        # link flag correctly through the NDK toolchain file. The NDK bundles a
        # static libomp.a — explicitly add -fopenmp -static-openmp so the linker
        # resolves __kmpc_* symbols.
        if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android")
            target_link_libraries(${main_target_name} PUBLIC "-fopenmp" "-static-openmp")
            message(STATUS "✅ Android: added -fopenmp -static-openmp to linker for ${main_target_name}")
        endif()

        # Bundle OpenMP runtime library for cross-platform deployment
        if(NOT WIN32 AND NOT ANDROID AND OpenMP_omp_LIBRARY AND EXISTS "${OpenMP_omp_LIBRARY}")
            message(STATUS "📦 Bundling OpenMP library: ${OpenMP_omp_LIBRARY}")
            add_custom_command(TARGET ${main_target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OpenMP_omp_LIBRARY}" "${CMAKE_BINARY_DIR}/libomp.dylib"
                COMMENT "Copying libomp to output directory for bundling"
            )
        endif()
    elseif(NOT APPLE)
        # Only use -fopenmp fallback on non-Apple platforms (Apple Clang rejects it)
        target_link_libraries(${main_target_name} PUBLIC "-fopenmp")
    else()
        message(WARNING "⚠️ OpenMP not found on macOS — building without OpenMP support")
    endif()

    # SDZ SameDiff archives are ZIP/DEFLATE by default, so link zlib when available.
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        set(_sdx_object_target "${main_target_name}_object")
        if(TARGET ${_sdx_object_target})
            target_compile_definitions(${_sdx_object_target} PUBLIC HAVE_ZLIB=1)
            target_link_libraries(${_sdx_object_target} PUBLIC ZLIB::ZLIB)
        endif()
        target_link_libraries(${main_target_name} PUBLIC ZLIB::ZLIB)
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ZLIB=1)
        message(STATUS "🔗 Linking zlib for SDZ DEFLATE support")
    else()
        message(WARNING "⚠️ zlib not found - SDZ reader supports STORED ZIP entries only")
    endif()

    # Keep CPU JavaCPP bindings aligned with the native build's managed runtime
    # dependencies. Plain CPU builds have no project-managed runtime DSOs, but they
    # still must emit an explicit zero-entry manifest; JavaCPP treats a missing
    # manifest as an invalid native build. CPU Triton builds add the same pinned
    # LLVM/MLIR DSOs that are staged by the CUDA path.
    set(_cpu_shared_runtimes "")
    if(HAVE_TRITON_CPU)
        foreach(_triton_runtime_target IN ITEMS triton_mlir_shared triton_llvm_shared)
            if(NOT TARGET ${_triton_runtime_target})
                message(FATAL_ERROR
                    "Triton CPU requires normalized shared runtime target ${_triton_runtime_target}")
            endif()
            list(APPEND _cpu_shared_runtimes
                "$<TARGET_FILE:${_triton_runtime_target}>")
        endforeach()
    endif()
    list(JOIN _cpu_shared_runtimes "|" _cpu_shared_runtimes_pipe)
    add_custom_command(TARGET ${main_target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            "-DRUNTIME_LIBRARIES_PIPE=${_cpu_shared_runtimes_pipe}"
            "-DREADELF=${CMAKE_READELF}"
            "-DOTOOL=${CMAKE_OTOOL}"
            "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DOUTPUT_DIR=$<TARGET_FILE_DIR:${main_target_name}>"
            "-P${CMAKE_SOURCE_DIR}/cmake/StageSharedRuntime.cmake"
        VERBATIM)

    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

# NOTE: configure_cuda_linking() is defined in CudaConfiguration.cmake (included at line ~879).
# Do NOT redefine it here — cmake uses the last definition, and the CudaConfiguration version
# already handles: CUDA toolkit, nvrtc, cuda driver, cuDNN, OpenBLAS, Triton, JVM, OpenMP.

# --- Enhanced library creation function ---
function(create_and_link_library)
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    set(MAIN_LIB_NAME "${SD_LIBRARY_NAME}")

    if(NOT TARGET ${OBJECT_LIB_NAME})
        add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})
        if(WIN32 AND SD_ZLUDA)
            # Set these before any CUDA-generated build rule is emitted. The
            # ZLUDA classifier executes per-translation-unit PTX and must not
            # request a device-link step or a static CUDA device runtime.
            set_target_properties(${OBJECT_LIB_NAME} PROPERTIES
                CUDA_RESOLVE_DEVICE_SYMBOLS OFF
                CUDA_SEPARABLE_COMPILATION OFF
                CUDA_RUNTIME_LIBRARY Shared
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            )
        endif()
        add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)
        target_link_libraries(${OBJECT_LIB_NAME} PUBLIC flatbuffers_interface)

        # Capability-gated implementation files are compiled by the object target,
        # before the final shared-library target exists. Propagate NNAPI there so
        # NnapiGraphBackend.cpp emits its implementation instead of an empty
        # translation unit while callers are compiled with HAVE_NNAPI enabled.
        if(HAVE_NNAPI)
            if(NOT ANDROID)
                message(FATAL_ERROR
                    "HAVE_NNAPI requires an Android toolchain for ${OBJECT_LIB_NAME}")
            endif()
            target_compile_definitions(${OBJECT_LIB_NAME} PUBLIC HAVE_NNAPI=1)
            message(STATUS "NNAPI compile boundary: ${OBJECT_LIB_NAME} -> HAVE_NNAPI=1")
        endif()

        # OpenMP: add compile flags to the OBJECT target.
        # find_package(OpenMP) runs inside setup_cpu_environment() which has function
        # scope — OpenMP_CXX_FOUND isn't visible here. Re-find at this scope.
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            # Use generator expressions so -fopenmp only applies to C/CXX, not CUDA.
            # nvcc does not understand -fopenmp and fatals on Windows CUDA builds.
            # MSVC needs /openmp:experimental for #pragma omp simd support.
            if(MSVC)
                target_compile_options(${OBJECT_LIB_NAME} PUBLIC
                    $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>
                    $<$<COMPILE_LANGUAGE:C>:/openmp:experimental>)
            elseif(APPLE)
                # Apple Clang needs -Xclang -fopenmp (bare -fopenmp is unsupported).
                # OpenMP_CXX_FLAGS is set to "-Xclang -fopenmp" by the Homebrew hints
                # in configure_cpu_linking(). Use separate_arguments to split the string.
                separate_arguments(_omp_cxx_flags UNIX_COMMAND "${OpenMP_CXX_FLAGS}")
                separate_arguments(_omp_c_flags UNIX_COMMAND "${OpenMP_C_FLAGS}")
                target_compile_options(${OBJECT_LIB_NAME} PUBLIC
                    $<$<COMPILE_LANGUAGE:CXX>:${_omp_cxx_flags}>
                    $<$<COMPILE_LANGUAGE:C>:${_omp_c_flags}>)
            else()
                target_compile_options(${OBJECT_LIB_NAME} PUBLIC
                    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
                    $<$<COMPILE_LANGUAGE:C>:-fopenmp>)
            endif()
            message(STATUS "✅ OpenMP added to OBJECT target ${OBJECT_LIB_NAME} (language-guarded)")
        else()
            message(STATUS "⚠️ OpenMP NOT found — ${OBJECT_LIB_NAME} will be single-threaded")
        endif()

        # Generated instantiation sources compile into nested object directories
        # under CMakeFiles/<target>.dir. Create those paths up front so clean
        # parallel Makefile builds do not race the first generated compiles.
        set(_generated_object_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${OBJECT_LIB_NAME}.dir")
        file(MAKE_DIRECTORY
                "${_generated_object_dir}"
                "${_generated_object_dir}/compilation_units")
        if(SD_CUDA)
            file(MAKE_DIRECTORY "${_generated_object_dir}/cuda_instantiations")
        else()
            file(MAKE_DIRECTORY "${_generated_object_dir}/cpu_instantiations")
        endif()

        # =========================================================================
        # UNITY BUILD EXCLUSIONS
        # =========================================================================
        # Generated template-instantiation compilation units (e.g. specials_single_direct_*.cpp,
        # pairwise_instantiation_*.cpp, etc.) each #include the same impl header for
        # their type slice. In unity builds, multiple such .cpp files get merged into
        # one TU, causing redefinition errors — #pragma once fails on MinGW GCC due
        # to mixed-slash relative paths (..\..\include/...).
        # These files are intentionally split for parallel compilation and type-based
        # chunking, so excluding them from unity batching is correct.
        # =========================================================================
        # =========================================================================
        # EXCLUDE GENERATED FILES FROM UNITY BUILD AND PCH
        # =========================================================================
        # Generated instantiation and compilation-unit files are intentionally split
        # for parallel compilation and type-based chunking, so they must be excluded
        # from unity batching.
        #
        # They also only include their specific template headers (e.g.
        # <loops/cpu/broadcasting_bool.hpp>) and do NOT need the project-wide PCH.
        # Applying PCH to these files causes ccache to disable depend_mode
        # (ccache disables depend_mode when PCH is detected), falling back to
        # slower preprocessor-based caching with cache keys that change whenever
        # any PCH header changes — causing cache misses on every rebuild.
        # =========================================================================
        foreach(_src ${ALL_SOURCES})
            get_filename_component(_dir "${_src}" DIRECTORY)
            if(_dir MATCHES "compilation_units|cpu_instantiations|cuda_instantiations")
                set_source_files_properties(${_src} PROPERTIES
                    SKIP_PRECOMPILE_HEADERS ON)
                if(SD_UNITY_BUILD)
                    set_source_files_properties(${_src} PROPERTIES
                        SKIP_UNITY_BUILD_INCLUSION ON)
                endif()
            endif()
        endforeach()

        # =========================================================================
        # NOTE ON EXCEPTION TABLE REDUCTION (ABANDONED APPROACH)
        # =========================================================================
        # We attempted to use -fno-exceptions to reduce .gcc_except_table sections
        # that cause PC32 relocation overflow in >2GB binaries. However, this approach
        # was abandoned because DirectShapeTrie.h (which uses try/catch at line 106)
        # is included transitively by virtually ALL source files through the header
        # chain (via ConstantShapeHelper.h and other helpers).
        #
        # The relocation overflow issue is instead solved via:
        # 1. Partial linking (PartialLinking.cmake) - pre-links object files in small
        #    groups to resolve internal relocations before the final link
        # 2. Gold linker with appropriate flags for large binaries
        # 3. -mcmodel=medium for 64-bit data addressing
        # =========================================================================

        # =========================================================================
        # EXTERNAL DEPENDENCY BLOCKING
        # =========================================================================
        # External helper libraries MUST complete building before object files compile.
        # This prevents "header not found" errors (e.g., dnnl.hpp) when helpers are enabled.
        # =========================================================================

        # OneDNN helper - MUST complete before object files that include dnnl.hpp
        if(HELPERS_onednn STREQUAL "ON" AND TARGET onednn_external)
            message(STATUS "")
            message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  🔒 DEPENDENCY BLOCK: OneDNN                                      ║")
            message(STATUS "║  ${OBJECT_LIB_NAME} compilation will WAIT for OneDNN to finish    ║")
            message(STATUS "║  OneDNN build may take 5-15 minutes...                            ║")
            message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            add_dependencies(${OBJECT_LIB_NAME} onednn_external)
            if(TARGET onednn_interface)
                target_link_libraries(${OBJECT_LIB_NAME} PUBLIC onednn_interface)
            endif()
        endif()

        # ARM Compute helper
        if(HELPERS_armcompute STREQUAL "ON" AND TARGET armcompute_external)
            message(STATUS "")
            message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  🔒 DEPENDENCY BLOCK: ARM Compute Library                         ║")
            message(STATUS "║  ${OBJECT_LIB_NAME} compilation will WAIT for ARM Compute         ║")
            message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            add_dependencies(${OBJECT_LIB_NAME} armcompute_external)
            if(TARGET armcompute_interface)
                target_link_libraries(${OBJECT_LIB_NAME} PUBLIC armcompute_interface)
            endif()
        endif()

        # Triton GPU compiler helper - MUST complete before object files that include Triton headers
        if(HAVE_TRITON AND TARGET triton_external)
            message(STATUS "")
            message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  🔒 DEPENDENCY BLOCK: Triton                                       ║")
            message(STATUS "║  ${OBJECT_LIB_NAME} compilation will WAIT for Triton               ║")
            message(STATUS "║  Triton build may take 15-30 minutes (builds LLVM internally)...   ║")
            message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            add_dependencies(${OBJECT_LIB_NAME} triton_external)
            if(TARGET triton_interface)
                # The versioned interface target owns the matching Triton and LLVM
                # include roots, libraries, and external-project dependency.
                target_link_libraries(${OBJECT_LIB_NAME} PUBLIC triton_interface)
            endif()
        endif()

        # MIOpen helper
        if(HELPERS_miopen STREQUAL "ON" AND TARGET miopen_external)
            message(STATUS "")
            message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  🔒 DEPENDENCY BLOCK: MIOpen                                      ║")
            message(STATUS "║  ${OBJECT_LIB_NAME} compilation will WAIT for MIOpen              ║")
            message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            add_dependencies(${OBJECT_LIB_NAME} miopen_external)
            if(TARGET miopen_interface)
                target_link_libraries(${OBJECT_LIB_NAME} PUBLIC miopen_interface)
            endif()
        endif()

        # OpenVINO CPU graph backend
        if(HAVE_OPENVINO AND TARGET openvino_external)
            message(STATUS "")
            message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  🔒 DEPENDENCY BLOCK: OpenVINO                                     ║")
            message(STATUS "║  ${OBJECT_LIB_NAME} compilation will WAIT for OpenVINO             ║")
            message(STATUS "║  OpenVINO build may take 10-20 minutes...                          ║")
            message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            add_dependencies(${OBJECT_LIB_NAME} openvino_external)
            if(TARGET openvino_interface)
                target_link_libraries(${OBJECT_LIB_NAME} PUBLIC openvino_interface)
                set(_OV_INSTALL "${CMAKE_BINARY_DIR}/openvino_install")
                target_include_directories(${OBJECT_LIB_NAME} SYSTEM PUBLIC
                    "${_OV_INSTALL}/runtime/include"
                    "${_OV_INSTALL}/include"
                )
                message(STATUS "✅ OpenVINO include dirs added to ${OBJECT_LIB_NAME} (HAVE_OPENVINO via config.h)")
            endif()
        endif()

        # CUTLASS header-only templates
        if(HAVE_CUTLASS AND TARGET cutlass_external)
            add_dependencies(${OBJECT_LIB_NAME} cutlass_external)
            if(TARGET cutlass_interface)
                target_link_libraries(${OBJECT_LIB_NAME} PUBLIC cutlass_interface)
                message(STATUS "CUTLASS include dirs added to ${OBJECT_LIB_NAME} (HAVE_CUTLASS via config.h)")
            endif()
        endif()

        # MLX Apple Silicon backend — must build before Mlx*.cpp compiles
        # Mlx*.cpp includes <mlx/mlx.h> which lives in MLX_INSTALL_DIR/include;
        # that directory is exposed only via mlx_interface INTERFACE_INCLUDE_DIRECTORIES.
        # Without this block the object-library compile races mlx_external AND
        # gets "mlx/mlx.h: No such file or directory".
        if(HAVE_MLX AND TARGET mlx_external)
            message(STATUS "")
            message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  🔒 DEPENDENCY BLOCK: MLX Apple Silicon                           ║")
            message(STATUS "║  ${OBJECT_LIB_NAME} compilation will WAIT for MLX to finish       ║")
            message(STATUS "║  MLX build may take 3-8 minutes...                                ║")
            message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            add_dependencies(${OBJECT_LIB_NAME} mlx_external)
            if(TARGET mlx_interface)
                target_link_libraries(${OBJECT_LIB_NAME} PUBLIC mlx_interface)
                set(_MLX_INSTALL "${CMAKE_BINARY_DIR}/mlx_install")
                target_include_directories(${OBJECT_LIB_NAME} SYSTEM PUBLIC
                    "${_MLX_INSTALL}/include"
                )
                message(STATUS "✅ MLX include dirs added to ${OBJECT_LIB_NAME}")
            endif()
        elseif(HAVE_MLX AND TARGET mlx_interface)
            # Pre-installed MLX (found via FindMLX.cmake) — no mlx_external to depend on
            target_link_libraries(${OBJECT_LIB_NAME} PUBLIC mlx_interface)
            message(STATUS "✅ Pre-installed MLX linked to ${OBJECT_LIB_NAME}")
        endif()

        # picking up incomplete flatbuffers.h from libnd4j/include/flatbuffers/
        # The ExternalProject downloads full headers to flatbuffers-src/include
        get_target_property(FLATBUFFERS_INCLUDES flatbuffers_interface INTERFACE_INCLUDE_DIRECTORIES)
        if(FLATBUFFERS_INCLUDES)
            target_include_directories(${OBJECT_LIB_NAME} BEFORE PUBLIC ${FLATBUFFERS_INCLUDES})
        endif()

        if(SD_CUDA)
            # Find the CUDA Toolkit to make the CUDA::toolkit target available.
            find_package(CUDAToolkit REQUIRED)

            # THE FIX: For OBJECT libraries, you must get the include directories from
            # the modern CUDA::toolkit target and apply them DIRECTLY to the object library
            # that is being compiled.
            get_target_property(CUDA_INCLUDE_DIRS CUDA::toolkit INTERFACE_INCLUDE_DIRECTORIES)
            target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

            # UPDATED: Add cuDNN include directories if available
            if(HAVE_CUDNN AND CUDNN_INCLUDE_DIR)
                target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${CUDNN_INCLUDE_DIR})
                target_compile_definitions(${OBJECT_LIB_NAME} PUBLIC HAVE_CUDNN=1)
                message(STATUS "🔧 Added cuDNN includes to object library: ${CUDNN_INCLUDE_DIR}")
            else()
                target_compile_definitions(${OBJECT_LIB_NAME} PUBLIC HAVE_CUDNN=0)
            endif()

            message(STATUS "🚀 CUDA object library configured with enhanced template support")
        endif()

        target_include_directories(${OBJECT_LIB_NAME} PUBLIC
                "${CMAKE_CURRENT_BINARY_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/types"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/system"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/performance"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/indexing"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/generated"
                "${CMAKE_BINARY_DIR}/compilation_units"
        )

        if(SD_CUDA)
            target_include_directories(${OBJECT_LIB_NAME} PUBLIC
                    "${CMAKE_BINARY_DIR}/cuda_instantiations"
            )
        elseif(NOT SD_VULKAN)
            target_include_directories(${OBJECT_LIB_NAME} PUBLIC
                    "${OPENBLAS_PATH}/include"
                    "${CMAKE_BINARY_DIR}/cpu_instantiations"
            )
        endif ()

        if(SD_CUDA AND DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
            set_target_properties(${OBJECT_LIB_NAME} PROPERTIES
                    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        endif()

        message(STATUS "🔧 Applying type definitions to OBJECT library: ${OBJECT_LIB_NAME}")
        setup_type_definitions_for_target(${OBJECT_LIB_NAME})

        # Enable precompiled headers for large commonly-included headers
        # NOTE: PCH disabled for CUDA builds - nvcc doesn't support PCH well
        # This significantly speeds up builds by avoiding re-parsing 11,000+ lines of headers per file
        # PCH is ONLY enabled when SD_GCC_FUNCTRACE is OFF and NOT a CUDA build
        # SD_DISABLE_PCH env var: CI sets this because PCH breaks ccache across runner restores
        # (fresh SDK installs have new header mtimes, invalidating all PCH manifest entries)
        # PCH is now DISABLED by default because it breaks ccache depend_mode.
        # ccache falls back to preprocessor mode when PCH is detected, making
        # cache keys include ALL PCH headers — any change to shape.h, templatemath.h,
        # etc. invalidates all ~841 source files instead of just affected ones.
        # Set SD_ENABLE_PCH=1 to force-enable PCH (e.g., for cold builds without ccache).
        if(DEFINED ENV{SD_DISABLE_PCH})
            set(_SD_PCH_DISABLED_BY_ENV TRUE)
        elseif(NOT DEFINED ENV{SD_ENABLE_PCH})
            set(_SD_PCH_DISABLED_BY_ENV TRUE)
        else()
            set(_SD_PCH_DISABLED_BY_ENV FALSE)
        endif()
        if(NOT SD_GCC_FUNCTRACE AND NOT SD_CUDA AND NOT _SD_PCH_DISABLED_BY_ENV)
            message(STATUS "🚀 Enabling precompiled headers for ${OBJECT_LIB_NAME}")
            # On Windows, include <windows.h> first so that _WINDOWS_ is defined
            # before types.h processes its constexpr alias guards. Without this,
            # the PCH bakes in constexpr aliases (BOOL, INT32, etc.) that conflict
            # with Windows SDK typedefs when source files later include <windows.h>.
            if(WIN32)
                target_compile_definitions(${OBJECT_LIB_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
                target_precompile_headers(${OBJECT_LIB_NAME} PRIVATE
                    <windows.h>
                    <system/op_boilerplate.h>
                    <system/type_boilerplate.h>
                    <math/templatemath.h>
                    <helpers/shape.h>
                )
            else()
                target_precompile_headers(${OBJECT_LIB_NAME} PRIVATE
                    <system/op_boilerplate.h>
                    <system/type_boilerplate.h>
                    <math/templatemath.h>
                    <helpers/shape.h>
                )
            endif()
            message(STATUS "✅ Precompiled headers enabled (op_boilerplate.h, type_boilerplate.h, templatemath.h, shape.h)")
        elseif(SD_CUDA)
            message(STATUS "⚠️ Precompiled headers DISABLED for CUDA build (nvcc compatibility)")
        elseif(_SD_PCH_DISABLED_BY_ENV)
            message(STATUS "⚠️ Precompiled headers DISABLED (default — breaks ccache depend_mode). Set SD_ENABLE_PCH=1 to enable.")
        else()
            message(STATUS "⚠️ Precompiled headers DISABLED for functrace build (use clean build when switching)")
        endif()
    endif()

    if(NOT TARGET ${MAIN_LIB_NAME})
        # Check if partial linking is needed for large binary support
        sd_should_use_partial_linking(USE_PARTIAL_LINKING)

        if(USE_PARTIAL_LINKING)
            message(STATUS "")
            message(STATUS "╔══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  LARGE BINARY BUILD DETECTED (SD_GCC_FUNCTRACE + SD_CUDA)        ║")
            message(STATUS "║  Using partial linking to handle >2GB .eh_frame relocations      ║")
            message(STATUS "╚══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            sd_create_library_with_partial_linking(${MAIN_LIB_NAME} ${OBJECT_LIB_NAME})
        else()
            add_library(${MAIN_LIB_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
        endif()

        if(WIN32 AND SD_ZLUDA AND TARGET ${MAIN_LIB_NAME})
            # Keep the final DLL on the shared CUDA runtime path as well. The
            # target has no CUDA sources of its own, so set this explicitly rather
            # than relying on CMake to infer it from the object library.
            set_target_properties(${MAIN_LIB_NAME} PROPERTIES
                CUDA_RESOLVE_DEVICE_SYMBOLS OFF
                CUDA_SEPARABLE_COMPILATION OFF
                CUDA_RUNTIME_LIBRARY Shared
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            )
        endif()

        # For IMPORTED targets (created by partial linking), properties and includes
        # are already set in PartialLinking.cmake. Skip these operations.
        if(NOT SD_PARTIAL_LINKED_TARGET)
            set_target_properties(${MAIN_LIB_NAME} PROPERTIES OUTPUT_NAME ${MAIN_LIB_NAME})
            if(WIN32)
                if(MSVC)
                    # Public C/C++ APIs use SD_LIB_EXPORT/SDX_API explicitly. Do not
                    # synthesize an exports.def for every template-instantiated symbol:
                    # the generated export library exceeds MSVC's 65,535-member limit.
                    set_target_properties(${MAIN_LIB_NAME} PROPERTIES
                        WINDOWS_EXPORT_ALL_SYMBOLS OFF
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

                    # The monolithic target is also the SDX and NativeOps JavaCPP
                    # link target. Feed the explicit SDX C ABI to the DLL linker so
                    # it always emits an import library; SD_LIB_EXPORT directives
                    # from the object files add the complete NativeOps ABI without
                    # CMake's oversized all-symbol template export scan.
                    set(_sdx_windows_def
                        "${CMAKE_CURRENT_BINARY_DIR}/${MAIN_LIB_NAME}_sdx_exports.def")
                    file(WRITE "${_sdx_windows_def}" "EXPORTS\n")
                    file(STRINGS
                        "${CMAKE_CURRENT_SOURCE_DIR}/include/dsp/runtime/dsp_runtime_c.h"
                        _sdx_windows_api_declarations
                        REGEX "^[\\t ]*SDX_API[\\t ]+")
                    foreach(_sdx_windows_declaration IN LISTS
                            _sdx_windows_api_declarations)
                        string(REGEX MATCH
                            "sdx[A-Za-z0-9_]+[\\t ]*\\("
                            _sdx_windows_symbol_match
                            "${_sdx_windows_declaration}")
                        if(_sdx_windows_symbol_match)
                            string(REGEX REPLACE
                                ".*(sdx[A-Za-z0-9_]+)[\\t ]*\\(.*"
                                "\\1"
                                _sdx_windows_symbol
                                "${_sdx_windows_declaration}")
                            file(APPEND "${_sdx_windows_def}"
                                "    ${_sdx_windows_symbol}\n")
                        endif()
                    endforeach()
                    # Use the legacy MSVC target property as well as the dependency
                    # edge: CMake/Ninja has dropped target_link_options(/DEF:...) on
                    # this CUDA shared-library link path in hosted Windows builds.
                    set_target_properties(${MAIN_LIB_NAME} PROPERTIES
                        LINK_FLAGS "/DEF:${_sdx_windows_def}")
                    set_property(TARGET ${MAIN_LIB_NAME} APPEND PROPERTY
                        LINK_DEPENDS "${_sdx_windows_def}")
                    # Preserve the complete import library emitted by the DLL linker.
                    # Re-running lib.exe with the SDX-only definition would overwrite
                    # the SD_LIB_EXPORT NativeOps entries required by jnind4jcuda.
                elseif(SD_VULKAN)
                    # MinGW Vulkan embeds static LLVM/MLIR internals. Export only
                    # the explicit SD_LIB_EXPORT API to avoid PE's ordinal limit.
                    set_target_properties(${MAIN_LIB_NAME} PROPERTIES
                        WINDOWS_EXPORT_ALL_SYMBOLS OFF)
                else()
                    # MinGW understands CMake's export-all path but not the
                    # MSVC /DEF linker option used above.
                    set_target_properties(${MAIN_LIB_NAME} PROPERTIES
                        WINDOWS_EXPORT_ALL_SYMBOLS ON)
                endif()
            endif()

            # Code model configuration is now centralized in CompilerFlags.cmake to avoid conflicts
            # (CMAKE_SHARED_LINKER_FLAGS is set there with -mcmodel=large for sanitizer builds)
            message(STATUS "Code model configuration deferred to CompilerFlags.cmake (via CMAKE_SHARED_LINKER_FLAGS)")

            # No CUDA includes needed here, they are handled by the linking function
            target_include_directories(${MAIN_LIB_NAME} PUBLIC
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/blas"
                    "${CMAKE_CURRENT_BINARY_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/array"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/execution"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/graph"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/loops"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/memory"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/ops"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/types"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/system"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/performance"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/indexing"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/generated"
                    "${CMAKE_BINARY_DIR}/compilation_units"
            )

            if(SD_CUDA)
                target_include_directories(${MAIN_LIB_NAME} PUBLIC
                        "${CMAKE_BINARY_DIR}/cuda_instantiations"
                )
            elseif(NOT SD_VULKAN)
                target_include_directories(${MAIN_LIB_NAME} PUBLIC
                        "${OPENBLAS_PATH}/include"
                        "${CMAKE_BINARY_DIR}/cpu_instantiations"
                )
            endif ()

            # 🔧 ALSO apply type definitions to the shared library (for completeness)
            message(STATUS "🔧 Applying type definitions to SHARED library: ${MAIN_LIB_NAME}")
            setup_type_definitions_for_target(${MAIN_LIB_NAME})
        else()
            message(STATUS "Skipping target configuration for IMPORTED target (partial linking)")
        endif()
    endif()

    # Remove the old call since we're now applying to both targets explicitly
    # apply_libnd4j_type_definitions_auto()

    # For IMPORTED targets (partial linking), linking is handled by PartialLinking.cmake
    message(STATUS "DEBUG: SD_PARTIAL_LINKED_TARGET=${SD_PARTIAL_LINKED_TARGET}, SD_CUDA=${SD_CUDA}, MAIN_LIB_NAME=${MAIN_LIB_NAME}")
    if(NOT SD_PARTIAL_LINKED_TARGET)
        if(SD_CUDA)
            message(STATUS "DEBUG: About to call configure_cuda_linking(${MAIN_LIB_NAME})")
            configure_cuda_linking(${MAIN_LIB_NAME})
        elseif(SD_VULKAN)
            configure_vulkan_linking(${MAIN_LIB_NAME})
        else()
            configure_cpu_linking(${MAIN_LIB_NAME})
        endif()
    else()
        message(STATUS "Skipping linking configuration for IMPORTED target (handled by partial linking)")
    endif()
endfunction()

# =============================================================================
# SECTION 2: MAIN BUILD ORCHESTRATION WITH ENHANCED CUDA/CPU TEMPLATE SUPPORT
# The main, sequential build process starts here.
# =============================================================================

print_status_colored("INFO" "=== ORCHESTRATING ENHANCED LIBND4J BUILD WITH TEMPLATE PARITY ===")

# --- Phase 1: Initial Setup ---
print_status_colored("INFO" "=== 1. INITIALIZING BUILD CONFIGURATION ===")

function(setup_initial_configuration)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "❌ You need at least GCC 4.9")
    endif()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if(MSVC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()

    message(STATUS "🔧 Configuring Position Independent Code (PIC)...")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    message(STATUS "✅ Position Independent Code configuration complete")
endfunction()

setup_initial_configuration()
include(Dependencies)
setup_flatbuffers()

# --- Set library name and default engine first ---
if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
        set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
        add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CUDA)
        print_status_colored("INFO" "🚀 CUDA build mode: Enhanced template system will provide full CPU/CUDA parity")
    elseif(SD_TPU)
        set(SD_LIBRARY_NAME nd4jtpu)
        set(DEFAULT_ENGINE "samediff::ENGINE_TPU")
        add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_TPU)
        print_status_colored("INFO" "TPU build mode")
    elseif(SD_HEXAGON)
        set(SD_LIBRARY_NAME nd4jhexagon)
        set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
        add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CPU)
        print_status_colored("INFO" "Hexagon graph-backend build mode")
    elseif(SD_VULKAN)
        set(SD_LIBRARY_NAME nd4jvulkan)
        set(DEFAULT_ENGINE "samediff::ENGINE_VULKAN")
        add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_VULKAN)
        print_status_colored("INFO" "Vulkan device-backend build mode")
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
        set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
        add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CPU)
        print_status_colored("INFO" "🖥️  CPU build mode: Enhanced template system active")
    endif()
endif()

# =============================================================================
# MINIMAL TYPE VALIDATION INTEGRATION - ONLY WHAT'S NEEDED
# =============================================================================
print_status_colored("INFO" "=== INTEGRATING TYPE VALIDATION ===")

# Call the type validation setup
LIBND4J_SETUP_TYPE_VALIDATION()
print_status_colored("SUCCESS" "Type validation integration complete")
# =============================================================================

if(SD_CUDA)
    print_status_colored("INFO" "=== CUDA EARLY INITIALIZATION WITH TEMPLATE SUPPORT ===")
    include(CudaConfiguration)
    setup_cuda_build()
    print_status_colored("SUCCESS" "CUDA initialization complete with enhanced template support")
endif()

# --- ZLUDA Transpiler Configuration ---
if(SD_ZLUDA)
    print_status_colored("INFO" "=== ZLUDA TRANSPILER CONFIGURATION ===")
    include(ZludaConfiguration)
    setup_zluda()
    if(HAVE_ZLUDA)
        configure_zluda_cuda_flags()
        print_status_colored("SUCCESS" "ZLUDA configuration complete (target: ${ZLUDA_TARGET_BACKEND})")
    else()
        print_status_colored("WARNING" "ZLUDA configuration failed - falling back to standard CUDA if available")
    endif()
endif()

# --- Phase 2: Handle Dependencies & Operations ---
print_status_colored("INFO" "=== 2. INITIALIZING DEPENDENCIES & OPERATIONS ===")

include(DuplicateInstantiationDetection)
include(TemplateProcessing)
include(CompilerFlags)
include(HelperConfiguration)
setup_platform_optimizations()
if(SD_VULKAN)
    # Vulkan owns execution through its device API. CPU/vendor helper discovery
    # must not add sources, downloads, compile definitions, or staged runtimes.
    set(HAVE_ONEDNN FALSE)
    set(HAVE_MKL_VML FALSE)
    set(HAVE_MKL FALSE)
    set(HAVE_ARMCOMPUTE FALSE)
    set(HAVE_VLM FALSE)
    set(HAVE_MPS FALSE)
    set(HAVE_MLX FALSE)
    set(HAVE_OPENVINO FALSE)
else()
    setup_onednn()
    # Setup MKL VML for vectorized math operations (vsErf, vdErf, etc.)
    # Must be called after setup_onednn() since it depends on HELPERS_onednn.
    setup_mkl_vml()
    setup_armcompute()
    setup_vlm()
endif()

# UPDATED: Modern cuDNN setup
if(SD_CUDA)
    include(CudaConfiguration)
    setup_modern_cudnn()  # Use the modern cuDNN detection
    message(STATUS "🔧 Modern cuDNN configuration: HAVE_CUDNN=${HAVE_CUDNN}")
    if(HAVE_CUDNN)
        message(STATUS "✅ cuDNN found and configured")
        if(DEFINED CUDNN_VERSION_STRING)
            message(STATUS "   cuDNN version: ${CUDNN_VERSION_STRING}")
        endif()
    else()
        message(STATUS "ℹ️  Building without cuDNN support")
    endif()
else()
    # For CPU builds, set HAVE_CUDNN to false
    set(HAVE_CUDNN FALSE)
endif()

if(NOT SD_VULKAN)
    setup_blas()
    setup_mps()
endif()
# setup_triton owns both the optional Triton DSP compiler and the
# project-managed, patched shared LLVM/MLIR installation. Provision or restore
# the latter before the MLIR helper performs package discovery.
setup_triton()
setup_mlir()
if(NOT SD_VULKAN)
    setup_mlx()
    setup_cutlass()
    setup_nccl()
    setup_cusparse()
    setup_openvino()
else()
    set(HAVE_CUTLASS FALSE)
    set(HAVE_NCCL FALSE)
    set(HAVE_CUSPARSE FALSE)
endif()
# Vulkan SDK discovery and compile definitions belong to the Vulkan artifact.
# Separate native artifacts can coexist at runtime without making CPU/CUDA
# targets compile or link Vulkan backend code.
if(SD_VULKAN)
    setup_vulkan()
else()
    set(HAVE_VULKAN FALSE CACHE BOOL "Vulkan availability for this native artifact" FORCE)
endif()
if(SD_VULKAN)
    message(STATUS "SD_VULKAN dependency boundary: CPU/vendor execution helpers were not configured")
else()
    message(STATUS "🔍 DEBUG: After setup_blas() - OPENBLAS_PATH='${OPENBLAS_PATH}', HAVE_OPENBLAS='${HAVE_OPENBLAS}'")
endif()
message(STATUS "Dependencies initialization complete.")

# Print comprehensive helper configuration summary
message(STATUS "")
message(STATUS "🔧 === Helper Configuration ===")
message(STATUS "   ONEDNN:      ${HAVE_ONEDNN}")
message(STATUS "   MKL_VML:     ${HAVE_MKL_VML}")
message(STATUS "   MKL_BLAS:    ${HAVE_MKL}")
message(STATUS "   ARMCOMPUTE:  ${HAVE_ARMCOMPUTE}")
message(STATUS "   CUDNN:       ${HAVE_CUDNN}")
message(STATUS "   CUSPARSE:    ${HAVE_CUSPARSE}")
message(STATUS "   MLIR:        ${HAVE_MLIR}")
message(STATUS "   MPS:         ${HAVE_MPS}")
message(STATUS "   ACCELERATE:  ${HAVE_ACCELERATE}")
message(STATUS "   MIOPEN:      ${HAVE_MIOPEN}")
message(STATUS "   PJRT:        ${HAVE_PJRT}")
message(STATUS "   VLM:         ${HAVE_VLM}")
message(STATUS "   TRITON:      ${HAVE_TRITON}")
message(STATUS "   CUTLASS:     ${HAVE_CUTLASS}")
message(STATUS "   NCCL:        ${HAVE_NCCL}")
message(STATUS "   OPENVINO:    ${HAVE_OPENVINO}")
message(STATUS "   VULKAN:      ${HAVE_VULKAN}")
message(STATUS "")
message(STATUS "🔧 === Dynamic Kernel Selection ===")
message(STATUS "   Enabled:     ${SD_DYNAMIC_KERNEL_SELECTION}")
message(STATUS "   Strategy:    ${SD_KERNEL_STRATEGY}")
message(STATUS "   Auto-tuning: ${SD_KERNEL_AUTOTUNING}")
message(STATUS "   Priority:    ${SD_HELPER_PRIORITY}")
message(STATUS "")

# --- Build TYPE_DEFINES string before generating config.h ---
# This ensures config.h has all HAS_* macros defined from the start
set(TYPE_DEFINES "")

# Check if we're in selective types mode
if(DEFINED SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
    # Selective types mode - export only enabled types
    # The HAS_* macros were already set by apply_type_definitions_to_target
    # We need to capture them and add to TYPE_DEFINES

    # Get all HAS_* definitions that were set
    get_directory_property(COMPILE_DEFS COMPILE_DEFINITIONS)

    # Sort definitions to ensure deterministic order
    # This prevents config.h from changing during build-time CMake reconfiguration
    set(HAS_DEFS_LIST "")
    foreach(def IN LISTS COMPILE_DEFS)
        if(def MATCHES "^HAS_")
            list(APPEND HAS_DEFS_LIST "${def}")
        endif()
    endforeach()

    # Sort alphabetically for consistent output
    list(SORT HAS_DEFS_LIST)

    # Build TYPE_DEFINES string with sorted definitions
    foreach(def IN LISTS HAS_DEFS_LIST)
        string(APPEND TYPE_DEFINES "#define ${def}\n")
    endforeach()
else()
    # All types mode - define all HAS_* macros
    string(APPEND TYPE_DEFINES "#define HAS_BOOL 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_FLOAT 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_FLOAT32 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_DOUBLE 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_FLOAT64 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_FLOAT16 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_HALF 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_BFLOAT16 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_BFLOAT 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_FLOAT8 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_HALF2 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT8 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT8_T 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT16 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT16_T 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT32 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT32_T 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT64 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_INT64_T 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_LONG 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UINT8 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UINT8_T 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UINT16 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UINT16_T 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UINT32 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UINT32_T 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UINT64 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UINT64_T 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UNSIGNEDLONG 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UTF8 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_STD_STRING 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UTF16 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_STD_U16STRING 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_UTF32 1\n")
    string(APPEND TYPE_DEFINES "#define HAS_STD_U32STRING 1\n")
endif()

# --- config.h generation is handled solely by PostBuild.cmake ---
# PostBuild.cmake reads config.h.in directly and performs all #cmakedefine01,
# @VAR@, and TYPE_DEFINES substitutions, with write-if-different to preserve
# PCH timestamps. A redundant configure_file() call here can produce an
# intermediate config.h with unprocessed #cmakedefine01 directives that
# compilation sees before PostBuild.cmake runs.

set(DEFINITIONS_CONTENT "")
if(SD_ALL_OPS OR "${SD_OPS_LIST}" STREQUAL "")
    add_compile_definitions(SD_ALL_OPS=1)
    string(APPEND DEFINITIONS_CONTENT "#define SD_ALL_OPS 1\n")
else()
    foreach(OP ${SD_OPS_LIST})
        add_compile_definitions(OP_${OP}=1)
        string(APPEND DEFINITIONS_CONTENT "#define OP_${OP} 1\n")
    endforeach()
endif()
get_filename_component(OP_OUTPUT_DIRECTORY "${OP_OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${OP_OUTPUT_DIRECTORY}")
set(OP_DEFINITIONS_FILE_CONTENT
    "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")
set(EXISTING_OP_DEFINITIONS_FILE_CONTENT "")
if(EXISTS "${OP_OUTPUT_FILE}")
    file(READ "${OP_OUTPUT_FILE}" EXISTING_OP_DEFINITIONS_FILE_CONTENT)
endif()
# This header is transitively included by nearly every op translation unit.
# Preserve its timestamp when op selection is unchanged so a routine CMake
# configure does not invalidate the entire native object graph.
if(NOT "${EXISTING_OP_DEFINITIONS_FILE_CONTENT}" STREQUAL "${OP_DEFINITIONS_FILE_CONTENT}")
    file(WRITE "${OP_OUTPUT_FILE}" "${OP_DEFINITIONS_FILE_CONTENT}")
endif()

# --- Phase 3: Enhanced Selective Rendering Setup ---
print_status_colored("INFO" "=== 3. CONFIGURING ENHANCED SELECTIVE RENDERING ===")
include(SelectiveRenderingCore)
setup_selective_rendering_unified_safe(
        TYPE_PROFILE "${SD_TYPE_PROFILE}"
        OUTPUT_DIR "${CMAKE_BINARY_DIR}/include")
if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
    message(FATAL_ERROR "❌  Unified selective rendering setup failed!")
endif()
list(LENGTH UNIFIED_ACTIVE_TYPES type_count)
list(LENGTH UNIFIED_COMBINATIONS_3 combo_3_count)
if(SD_CUDA)
    message(STATUS "✅ Enhanced CUDA selective rendering setup complete: ${type_count} types, ${combo_3_count} combinations.")
elseif(SD_VULKAN)
    message(STATUS "✅ Vulkan descriptor type selection complete: ${type_count} types, ${combo_3_count} combinations.")
else()
    message(STATUS "✅ Enhanced CPU selective rendering setup complete: ${type_count} types, ${combo_3_count} combinations.")
endif()

if(DEFINED HAS_FLOAT16 AND HAS_FLOAT16)
    list(FIND UNIFIED_ACTIVE_TYPES "float16" float16_pos)
    if(float16_pos LESS 0)
        message(FATAL_ERROR "❌  float16 missing despite HAS_FLOAT16=TRUE! This will cause build errors.")
    else()
        message(STATUS "✅ VERIFIED: float16 present at position ${float16_pos}")
    endif()
endif()

# --- Phase 4: Create and Link Final Library with Enhanced Template System ---
print_status_colored("INFO" "=== 4. CREATING AND LINKING FINAL LIBRARY WITH ENHANCED TEMPLATES ===")
# ✅ DO NOT reset CUSTOMOPS_GENERIC_SOURCES - it's a CACHE variable populated by TemplateProcessing.cmake
# Resetting it here causes the cached template files to be lost, resulting in build errors
collect_all_sources(ALL_SOURCES)
create_and_link_library()
message(STATUS "Final library target created and linked with enhanced template support.")

# --- SDX Standalone Runtime (no JVM dependency) ---
if(SD_BUILD_SDX_STANDALONE)
    include(BuildSDX)
    build_sdx_library()
endif()

# --- Phase 5: Enhanced Final Reports and Summaries ---
if(DEFINED SD_GENERATE_TYPE_REPORT AND SD_GENERATE_TYPE_REPORT)
    print_status_colored("INFO" "=== GENERATING ENHANCED USAGE DOCUMENTATION ===")
    set(usage_doc "${CMAKE_BINARY_DIR}/enhanced_selective_rendering_usage.md")
    set(doc_content "# Enhanced Unified Selective Rendering Usage\n\nThis build was configured with backend-specific type and implementation selection.\n\n")
    if(DEFINED SD_TYPE_PROFILE)
        string(APPEND doc_content "**Type Profile**: ${SD_TYPE_PROFILE}\n\n")
    endif()
    if(SD_CUDA)
        string(APPEND doc_content "**Build Mode**: CUDA with enhanced template system\n")
        string(APPEND doc_content "**Template Parity**: Full parity achieved between CPU and CUDA template systems\n\n")
        if(HAVE_CUDNN)
            string(APPEND doc_content "**cuDNN Support**: Enabled\n")
            if(DEFINED CUDNN_VERSION_STRING)
                string(APPEND doc_content "**cuDNN Version**: ${CUDNN_VERSION_STRING}\n")
            endif()
        else()
            string(APPEND doc_content "**cuDNN Support**: Disabled\n")
        endif()
    elseif(SD_VULKAN)
        string(APPEND doc_content "**Build Mode**: Vulkan with descriptor-driven kernel emission\n\n")
    else()
        string(APPEND doc_content "**Build Mode**: CPU with enhanced template system\n\n")
    endif()
    file(WRITE "${usage_doc}" "${doc_content}")
    message(STATUS "📝 Enhanced usage documentation written to: ${usage_doc}")
endif()

print_status_colored("SUCCESS" "=== ENHANCED BUILD ORCHESTRATION COMPLETE ===")

message(STATUS "")
message(STATUS "🎯 ENHANCED BUILD SUMMARY:")
message(STATUS "   Library: ${SD_LIBRARY_NAME}")
message(STATUS "   Platform: ${CMAKE_SYSTEM_NAME}")
if(SD_CUDA)
    message(STATUS "   CUDA: Enabled with enhanced template parity")
    message(STATUS "   Template System: CUDA templates achieve full CPU parity")
    message(STATUS "   cuDNN: ${HAVE_CUDNN}")
    if(HAVE_CUDNN AND DEFINED CUDNN_VERSION_STRING)
        message(STATUS "   cuDNN Version: ${CUDNN_VERSION_STRING}")
    endif()
elseif(SD_VULKAN)
    message(STATUS "   Vulkan: Enabled")
    message(STATUS "   Kernel System: Descriptor-driven Vulkan/SPIR-V emission")
else()
    message(STATUS "   CUDA: Disabled")
    message(STATUS "   Template System: Enhanced CPU templates")
endif()
if(DEFINED SD_TYPE_PROFILE)
    message(STATUS "   Type Profile: ${SD_TYPE_PROFILE}")
endif()
if(DEFINED type_count)
    message(STATUS "   Active Types: ${type_count}")
endif()
if(DEFINED combo_3_count)
    message(STATUS "   Template Combinations: ${combo_3_count}")
endif()

include(TypeRegistryGenerator)
dump_type_macros_to_disk()

if(SD_CUDA)
    message(STATUS "🚀 Enhanced CUDA build orchestration complete - CUDA/CPU template parity achieved")
elseif(SD_VULKAN)
    message(STATUS "✅ Vulkan descriptor build orchestration complete - SPIR-V sources ready for compilation")
else()
    message(STATUS "🖥️  Enhanced CPU build orchestration complete - System ready for compilation")
endif()
message(STATUS "")

# === ENHANCED TEMPLATE SYSTEM VERIFICATION ===
print_status_colored("INFO" "=== VERIFYING ENHANCED TEMPLATE SYSTEM ===")

# Verify the backend's implementation-generation contract.
list(LENGTH CUSTOMOPS_GENERIC_SOURCES template_file_count)
if(SD_VULKAN)
    if(template_file_count EQUAL 0)
        message(STATUS "✅ Vulkan descriptor source set verified: no CPU/CUDA template instantiations")
    else()
        message(FATAL_ERROR
            "SD_VULKAN must not consume CPU/CUDA template instantiations: ${template_file_count} files found")
    endif()
elseif(template_file_count GREATER 0)
    if(SD_CUDA)
        message(STATUS "✅ CUDA template generation verified: ${template_file_count} files")

        # Check for CUDA-specific template files
        set(cuda_template_count 0)
        foreach(source_file ${CUSTOMOPS_GENERIC_SOURCES})
            if(source_file MATCHES "\\.cu$")
                math(EXPR cuda_template_count "${cuda_template_count} + 1")
            endif()
        endforeach()
        message(STATUS "🔍 CUDA template files: ${cuda_template_count}")

        if(cuda_template_count GREATER 0)
            print_status_colored("SUCCESS" "CUDA template parity system operational")
        else()
            print_status_colored("WARNING" "No CUDA template files detected - check template processing")
        endif()
    else()
        message(STATUS "✅ CPU template generation verified: ${template_file_count} files")
        print_status_colored("SUCCESS" "Enhanced CPU template system operational")
    endif()

endif()

# Final verification of directory structure
if(SD_CUDA)
    if(EXISTS "${CMAKE_BINARY_DIR}/cuda_instantiations")
        message(STATUS "✅ CUDA instantiation directory verified")
    else()
        message(WARNING "⚠️  CUDA instantiation directory not found")
    endif()
elseif(SD_VULKAN)
    message(STATUS "✅ Vulkan uses descriptor sources; no CPU/CUDA instantiation directory is expected")
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/cpu_instantiations")
        message(STATUS "✅ CPU instantiation directory verified")
    else()
        message(WARNING "⚠️  CPU instantiation directory not found")
    endif()
endif()

# Set OBJECT_LIB_NAME for PostBuild.cmake (matches pattern from create_and_link_library function)
if(NOT DEFINED OBJECT_LIB_NAME AND DEFINED SD_LIBRARY_NAME)
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    message(STATUS "Set OBJECT_LIB_NAME for PostBuild: ${OBJECT_LIB_NAME}")
endif()
# Check for instantiation extraction BEFORE PostBuild
if(SD_EXTRACT_INSTANTIATIONS)
    print_status_colored("INFO" "=== TEMPLATE INSTANTIATION EXTRACTION MODE ===")
    include(ExtractInstantiations)
    # ExtractInstantiations.cmake will handle everything and exit
    return()
endif()

# Enable precompiled headers for non-CUDA, non-functrace builds
# NOTE: PCH disabled for CUDA builds - nvcc doesn't support PCH well
# This dramatically reduces compile time by pre-compiling 11,000+ lines of headers
# Users switching between functrace and normal builds should do a clean build
if(NOT SD_GCC_FUNCTRACE AND NOT SD_CUDA AND NOT _SD_PCH_DISABLED_BY_ENV AND TARGET ${SD_LIBRARY_NAME})
    if(WIN32)
        target_compile_definitions(${SD_LIBRARY_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
        target_precompile_headers(${SD_LIBRARY_NAME} PRIVATE
                <windows.h>
                <vector>
                <memory>
                <algorithm>
                <functional>
                <cstring>
                "${CMAKE_CURRENT_SOURCE_DIR}/include/system/op_boilerplate.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/system/type_boilerplate.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/system/type_boiler_plate_expansions.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/math/templatemath.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/shape.h"
        )
    else()
        target_precompile_headers(${SD_LIBRARY_NAME} PRIVATE
                <vector>
                <memory>
                <algorithm>
                <functional>
                <cstring>
                "${CMAKE_CURRENT_SOURCE_DIR}/include/system/op_boilerplate.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/system/type_boilerplate.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/system/type_boiler_plate_expansions.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/math/templatemath.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/shape.h"
        )
    endif()
    message(STATUS "✅ Precompiled headers enabled for main library ${SD_LIBRARY_NAME}")
elseif(SD_CUDA)
    message(STATUS "⚠️ Precompiled headers DISABLED for CUDA build (nvcc compatibility)")
elseif(SD_GCC_FUNCTRACE)
    message(STATUS "⚠️ Precompiled headers DISABLED for functrace build")
endif()

# SDX C runtime SDK packaging: stage runtime library + public C header for embedding.
set(SDX_RUNTIME_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/dsp/runtime/dsp_runtime_c.h")
set(SDX_RUNTIME_README "${CMAKE_CURRENT_SOURCE_DIR}/include/dsp/runtime/README.md")
set(SDX_RUNTIME_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/../resources/dsp/manifest.schema.json")
set(SDX_RUNTIME_TEXT_GENERATION_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/../resources/dsp/text-generation.schema.json")
set(SDX_RUNTIME_BINDINGS_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SdxRuntimePackage.cmake")
set(SDX_RUNTIME_LANGUAGE_BINDINGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/dsp/runtime/bindings")
# Canonical Java wrapper source (nd4j-sdx module). Copied into wrappers/java/
# at staging time so the exported SDK is self-contained without duplicating
# the source in the libnd4j tree.
set(SDX_RUNTIME_JAVA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx/src/main/java")
set(SDX_RUNTIME_SDK_DIR "${CMAKE_BINARY_DIR}/sdx-runtime-sdk" CACHE PATH
    "Output directory for staged SDX C runtime SDK artifacts")

if(TARGET ${SD_LIBRARY_NAME} AND EXISTS "${SDX_RUNTIME_HEADER}")
    set(_sdx_sdk_include_dir "${SDX_RUNTIME_SDK_DIR}/include/dsp/runtime")
    set(_sdx_sdk_tokenizer_include_dir "${SDX_RUNTIME_SDK_DIR}/include/tokenizer")
    set(_sdx_sdk_lib_dir "${SDX_RUNTIME_SDK_DIR}/lib")
    set(_sdx_sdk_schema_dir "${SDX_RUNTIME_SDK_DIR}/share/dsp")
    set(_sdx_sdk_wrappers_dir "${SDX_RUNTIME_SDK_DIR}/wrappers")
    set(_sdx_schema_stage_cmds "")
    if(EXISTS "${SDX_RUNTIME_SCHEMA}")
        list(APPEND _sdx_schema_stage_cmds
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_sdx_sdk_schema_dir}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDX_RUNTIME_SCHEMA}" "${_sdx_sdk_schema_dir}/manifest.schema.json")
    endif()
    if(EXISTS "${SDX_RUNTIME_TEXT_GENERATION_SCHEMA}")
        list(APPEND _sdx_schema_stage_cmds
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_sdx_sdk_schema_dir}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDX_RUNTIME_TEXT_GENERATION_SCHEMA}" "${_sdx_sdk_schema_dir}/text-generation.schema.json")
    endif()
    set(_sdx_readme_stage_cmds "")
    if(EXISTS "${SDX_RUNTIME_README}")
        list(APPEND _sdx_readme_stage_cmds
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDX_RUNTIME_README}" "${_sdx_sdk_include_dir}/README.md")
    endif()
    set(_sdx_wrapper_stage_cmds "")
    if(EXISTS "${SDX_RUNTIME_LANGUAGE_BINDINGS_DIR}")
        list(APPEND _sdx_wrapper_stage_cmds
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_sdx_sdk_wrappers_dir}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${SDX_RUNTIME_LANGUAGE_BINDINGS_DIR}" "${_sdx_sdk_wrappers_dir}")
    endif()
    if(EXISTS "${SDX_RUNTIME_JAVA_SRC_DIR}")
        list(APPEND _sdx_wrapper_stage_cmds
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${SDX_RUNTIME_JAVA_SRC_DIR}" "${_sdx_sdk_wrappers_dir}/java/src/main/java")
    endif()

    # The tokenizer is built separately (Rust) so the portable runtime remains
    # usable without it. Mobile SDK generators pass both artifacts together.
    set(_sdx_tokenizer_stage_cmds "")
    set(_sdx_have_tokenizer_artifacts 0)
    if((DEFINED SDX_TOKENIZER_HEADER_FILE AND NOT "${SDX_TOKENIZER_HEADER_FILE}" STREQUAL "") OR
       (DEFINED SDX_TOKENIZER_LIBRARY_FILE AND NOT "${SDX_TOKENIZER_LIBRARY_FILE}" STREQUAL ""))
        if(NOT DEFINED SDX_TOKENIZER_HEADER_FILE OR "${SDX_TOKENIZER_HEADER_FILE}" STREQUAL "" OR
           NOT EXISTS "${SDX_TOKENIZER_HEADER_FILE}")
            message(FATAL_ERROR "SDX tokenizer header does not exist: ${SDX_TOKENIZER_HEADER_FILE}")
        endif()
        if(NOT DEFINED SDX_TOKENIZER_LIBRARY_FILE OR "${SDX_TOKENIZER_LIBRARY_FILE}" STREQUAL "" OR
           NOT EXISTS "${SDX_TOKENIZER_LIBRARY_FILE}")
            message(FATAL_ERROR "SDX tokenizer library does not exist: ${SDX_TOKENIZER_LIBRARY_FILE}")
        endif()
        set(_sdx_have_tokenizer_artifacts 1)
        list(APPEND _sdx_tokenizer_stage_cmds
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_sdx_sdk_tokenizer_include_dir}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_sdx_sdk_lib_dir}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDX_TOKENIZER_HEADER_FILE}" "${_sdx_sdk_tokenizer_include_dir}/tokenizers_ffi.h"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDX_TOKENIZER_LIBRARY_FILE}" "${_sdx_sdk_lib_dir}")
    endif()

    # Use standalone SDX library when available, otherwise fall back to monolithic.
    set(_sdx_sdk_uses_monolithic FALSE)
    if(SD_BUILD_SDX_STANDALONE AND DEFINED SDX_STANDALONE_TARGET AND TARGET ${SDX_STANDALONE_TARGET})
        set(_sdx_sdk_target ${SDX_STANDALONE_TARGET})
    else()
        set(_sdx_sdk_target ${SD_LIBRARY_NAME})
        set(_sdx_sdk_uses_monolithic TRUE)
    endif()

    # Compiler-enabled runtimes publish their normalized shared dependency
    # targets through a target property. Stage that exact closure into the SDK
    # and pass it to each platform package.
    get_target_property(_sdx_runtime_dependency_targets
        ${_sdx_sdk_target} SDX_RUNTIME_DEPENDENCY_TARGETS)
    if(_sdx_runtime_dependency_targets STREQUAL
       "_sdx_runtime_dependency_targets-NOTFOUND")
        set(_sdx_runtime_dependency_targets "")
    endif()

    # The normal monolithic target links Triton through its object/final link
    # configuration rather than BuildSDX, so it does not pass through the
    # standalone helper that publishes this runtime closure. Publish the same
    # normalized targets here before the generic SDK packaging path consumes it.
    if(_sdx_sdk_uses_monolithic AND HAVE_TRITON AND
       NOT _sdx_runtime_dependency_targets)
        set(_sdx_runtime_dependency_targets
            triton_mlir_shared triton_llvm_shared)
        set_property(TARGET ${_sdx_sdk_target} PROPERTY
            SDX_RUNTIME_DEPENDENCY_TARGETS
            "${_sdx_runtime_dependency_targets}")
    endif()

    set(_sdx_sdk_shared_runtime_files "")
    foreach(_sdx_runtime_dependency_target IN LISTS
            _sdx_runtime_dependency_targets)
        if(NOT TARGET ${_sdx_runtime_dependency_target})
            message(FATAL_ERROR
                "SDX runtime dependency target is missing: ${_sdx_runtime_dependency_target}")
        endif()
        list(APPEND _sdx_sdk_shared_runtime_files
            "$<TARGET_FILE:${_sdx_runtime_dependency_target}>")
    endforeach()
    list(JOIN _sdx_sdk_shared_runtime_files "|"
        _sdx_sdk_shared_runtime_files_pipe)

    set(_sdx_shared_runtime_stage_cmds "")
    if(_sdx_sdk_shared_runtime_files)
        list(APPEND _sdx_shared_runtime_stage_cmds
            COMMAND ${CMAKE_COMMAND}
                "-DRUNTIME_LIBRARIES_PIPE=${_sdx_sdk_shared_runtime_files_pipe}"
                "-DREADELF=${CMAKE_READELF}"
                "-DOTOOL=${CMAKE_OTOOL}"
                "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DOUTPUT_DIR=${_sdx_sdk_lib_dir}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageSharedRuntime.cmake")
    endif()
    set(_sdx_sdk_dependencies
        ${_sdx_sdk_target} ${_sdx_runtime_dependency_targets})

    add_custom_target(sdx_runtime_sdk
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_sdx_sdk_include_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_sdx_sdk_lib_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDX_RUNTIME_HEADER}" "${_sdx_sdk_include_dir}/dsp_runtime_c.h"
        ${_sdx_readme_stage_cmds}
        COMMAND ${CMAKE_COMMAND}
            "-DINPUT_FILE=$<TARGET_FILE:${_sdx_sdk_target}>"
            "-DOUTPUT_FILE=${_sdx_sdk_lib_dir}/$<TARGET_FILE_NAME:${_sdx_sdk_target}>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageArtifact.cmake"
        COMMAND ${CMAKE_COMMAND}
            "-DINPUT_FILE=$<TARGET_LINKER_FILE:${_sdx_sdk_target}>"
            "-DOUTPUT_FILE=${_sdx_sdk_lib_dir}/$<TARGET_LINKER_FILE_NAME:${_sdx_sdk_target}>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageArtifact.cmake"
        ${_sdx_shared_runtime_stage_cmds}
        ${_sdx_schema_stage_cmds}
        ${_sdx_wrapper_stage_cmds}
        ${_sdx_tokenizer_stage_cmds}
        DEPENDS ${_sdx_sdk_dependencies}
        COMMENT "Staging SDX C runtime SDK artifacts in ${SDX_RUNTIME_SDK_DIR} (target: ${_sdx_sdk_target})"
        VERBATIM
    )

    # Platform-specific binding packaging (zip/AAR/XCFramework where applicable).
    set(_sdx_os "")
    set(_sdx_arch_raw "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _sdx_arch_raw)
        set(_sdx_arch_raw "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    string(TOLOWER "${_sdx_arch_raw}" _sdx_arch)
    if(_sdx_arch MATCHES "^(x86_64|amd64)$")
        set(_sdx_arch "x86_64")
    elseif(_sdx_arch MATCHES "^(aarch64|arm64)$")
        set(_sdx_arch "arm64")
    elseif(_sdx_arch MATCHES "^(armv7l|armv7)$")
        set(_sdx_arch "armv7")
    endif()

    set(_sdx_android_abi "")
    set(_sdx_android_api "")
    set(_sdx_android_ndk_major "")
    set(_sdx_android_stl "")
    if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android")
        set(_sdx_os "android")

        if(DEFINED CMAKE_ANDROID_API AND NOT "${CMAKE_ANDROID_API}" STREQUAL "")
            set(_sdx_android_api "${CMAKE_ANDROID_API}")
        elseif(DEFINED ANDROID_NATIVE_API_LEVEL AND
               NOT "${ANDROID_NATIVE_API_LEVEL}" STREQUAL "")
            string(REGEX REPLACE "^android-" "" _sdx_android_api
                "${ANDROID_NATIVE_API_LEVEL}")
        elseif(DEFINED ANDROID_PLATFORM AND NOT "${ANDROID_PLATFORM}" STREQUAL "")
            string(REGEX REPLACE "^android-" "" _sdx_android_api "${ANDROID_PLATFORM}")
        elseif(DEFINED CMAKE_SYSTEM_VERSION AND
               "${CMAKE_SYSTEM_VERSION}" MATCHES "^[0-9]+$" AND
               CMAKE_SYSTEM_VERSION GREATER_EQUAL 16)
            # Some Android toolchains expose a generic system version of "1";
            # never publish that as an Android API level.
            set(_sdx_android_api "${CMAKE_SYSTEM_VERSION}")
        endif()
        if(_sdx_android_api STREQUAL "")
            # Vulkan and the current SDX Android runtime use API 24 as their
            # conservative package floor when the toolchain omits its API.
            set(_sdx_android_api "24")
        endif()

        if(DEFINED CMAKE_ANDROID_NDK_VERSION AND NOT "${CMAKE_ANDROID_NDK_VERSION}" STREQUAL "")
            string(REGEX MATCH "^[0-9]+" _sdx_android_ndk_major "${CMAKE_ANDROID_NDK_VERSION}")
        elseif(DEFINED ANDROID_NDK_REVISION AND NOT "${ANDROID_NDK_REVISION}" STREQUAL "")
            string(REGEX MATCH "^[0-9]+" _sdx_android_ndk_major "${ANDROID_NDK_REVISION}")
        endif()
        if(_sdx_android_ndk_major STREQUAL "")
            set(_sdx_android_ndk_major "0")
        endif()

        if(DEFINED CMAKE_ANDROID_STL_TYPE AND NOT "${CMAKE_ANDROID_STL_TYPE}" STREQUAL "")
            set(_sdx_android_stl "${CMAKE_ANDROID_STL_TYPE}")
        elseif(DEFINED ANDROID_STL AND NOT "${ANDROID_STL}" STREQUAL "")
            set(_sdx_android_stl "${ANDROID_STL}")
        else()
            set(_sdx_android_stl "c++_shared")
        endif()
        if(DEFINED CMAKE_ANDROID_ARCH_ABI AND NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "")
            set(_sdx_android_abi "${CMAKE_ANDROID_ARCH_ABI}")
        elseif(DEFINED ANDROID_ABI AND NOT ANDROID_ABI STREQUAL "")
            set(_sdx_android_abi "${ANDROID_ABI}")
        endif()

        if(_sdx_android_abi STREQUAL "arm64-v8a")
            set(_sdx_arch "arm64")
        elseif(_sdx_android_abi STREQUAL "armeabi-v7a")
            set(_sdx_arch "armv7")
        elseif(_sdx_android_abi STREQUAL "x86_64")
            set(_sdx_arch "x86_64")
        elseif(_sdx_android_abi STREQUAL "x86")
            set(_sdx_arch "x86")
        endif()

        if(_sdx_android_abi STREQUAL "")
            if(_sdx_arch STREQUAL "arm64")
                set(_sdx_android_abi "arm64-v8a")
            elseif(_sdx_arch STREQUAL "armv7")
                set(_sdx_android_abi "armeabi-v7a")
            elseif(_sdx_arch STREQUAL "x86_64")
                set(_sdx_android_abi "x86_64")
            elseif(_sdx_arch STREQUAL "x86")
                set(_sdx_android_abi "x86")
            endif()
        endif()
    elseif(IOS)
        set(_sdx_os "ios")
    elseif(APPLE)
        set(_sdx_os "macos")
    elseif(WIN32)
        set(_sdx_os "windows")
    else()
        set(_sdx_os "linux")
    endif()

    set(_sdx_platform_id "${_sdx_os}-${_sdx_arch}")
    if(IOS AND DEFINED CMAKE_OSX_SYSROOT AND CMAKE_OSX_SYSROOT MATCHES "iphonesimulator")
        set(_sdx_platform_id "ios-simulator-${_sdx_arch}")
    endif()

    set(_sdx_have_cuda 0)
    if(SD_CUDA)
        set(_sdx_have_cuda 1)
    endif()
    set(_sdx_have_zluda 0)
    if(HAVE_ZLUDA)
        set(_sdx_have_zluda 1)
    endif()
    set(_sdx_have_triton 0)
    if(HAVE_TRITON)
        set(_sdx_have_triton 1)
    endif()
    set(_sdx_have_vulkan 0)
    if(SD_VULKAN AND HAVE_VULKAN)
        set(_sdx_have_vulkan 1)
    endif()
    set(_sdx_have_mlir 0)
    if(HAVE_MLIR)
        set(_sdx_have_mlir 1)
    endif()
    set(_sdx_have_mlx 0)
    if(HAVE_MLX)
        set(_sdx_have_mlx 1)
    endif()
    set(_sdx_have_nnapi 0)
    if(HAVE_NNAPI)
        set(_sdx_have_nnapi 1)
    endif()
    set(_sdx_have_onednn 0)
    if(HAVE_ONEDNN)
        set(_sdx_have_onednn 1)
    endif()
    set(_sdx_have_armcompute 0)
    if(HAVE_ARMCOMPUTE)
        set(_sdx_have_armcompute 1)
    endif()
    set(_sdx_have_hexagon 0)
    if(SD_HEXAGON)
        # The backend and stable adapter ABI are compiled independently of the
        # optional vendor SDK, which is loaded on-device via dlopen.
        set(_sdx_have_hexagon 1)
    endif()
    set(_sdx_have_tpu 0)
    if(SD_TPU AND HAVE_PJRT)
        set(_sdx_have_tpu 1)
    endif()

    set(_sdx_enable_android_aar 0)
    if(_sdx_os STREQUAL "android")
        set(_sdx_enable_android_aar 1)
    endif()

    set(_sdx_enable_apple_xcframework 0)
    if(APPLE)
        set(_sdx_enable_apple_xcframework 1)
    endif()

    if(SD_NNAPI_ACCELERATOR_ONLY AND SD_NNAPI_TENSOR_G3_HYBRID)
        set(_sdx_runtime_variants "tensor-g3")
    elseif(SD_HEXAGON)
        set(_sdx_runtime_variants "hexagon")
    elseif(SD_TPU)
        set(_sdx_runtime_variants "tpu")
    elseif(SD_VULKAN)
        set(_sdx_runtime_variants "vulkan")
    else()
        set(_sdx_runtime_variants "cpu")
        if(_sdx_have_cuda)
            list(APPEND _sdx_runtime_variants "cuda")
        endif()
        if(_sdx_have_zluda)
            list(APPEND _sdx_runtime_variants "amd")
        endif()
        list(REMOVE_DUPLICATES _sdx_runtime_variants)
    endif()

    set(_sdx_standalone 0)
    if(SD_BUILD_SDX_STANDALONE AND DEFINED SDX_STANDALONE_TARGET AND TARGET ${SDX_STANDALONE_TARGET})
        set(_sdx_standalone 1)
    endif()

    set(_sdx_strip_tool "")
    if(CMAKE_STRIP AND NOT APPLE)
        set(_sdx_strip_tool "${CMAKE_STRIP}")
    endif()

    # Keep mobile SDKs self-contained when the runtime links or dlopens
    # separately distributed native dependencies. Pass resolved files, not
    # linker tokens, so packaging fails before an incomplete AAR is published.
    set(_sdx_runtime_dependency_files
        ${_sdx_sdk_shared_runtime_files})
    if(_sdx_enable_android_aar)
        # Host BLAS is valid only for the Android CPU/reference runtime. Device
        # variants must never acquire a host execution path through packaging.
        if(NOT SD_HEXAGON AND NOT SD_TPU AND NOT SD_VULKAN AND
           NOT SD_NNAPI_ACCELERATOR_ONLY AND DEFINED OPENBLAS_LIBRARIES)
            foreach(_sdx_dependency_candidate IN LISTS OPENBLAS_LIBRARIES)
                if(IS_ABSOLUTE "${_sdx_dependency_candidate}" AND
                   EXISTS "${_sdx_dependency_candidate}")
                    list(APPEND _sdx_runtime_dependency_files
                        "${_sdx_dependency_candidate}")
                endif()
            endforeach()
        endif()
        if(DEFINED OpenMP_omp_LIBRARY AND
           IS_ABSOLUTE "${OpenMP_omp_LIBRARY}" AND
           EXISTS "${OpenMP_omp_LIBRARY}")
            list(APPEND _sdx_runtime_dependency_files "${OpenMP_omp_LIBRARY}")
        endif()
        # The main Tensor G3 runtime uses c++_static, but the pinned ARM
        # Compute DSO is a c++_shared consumer. Package the NDK runtime whenever
        # either the provider itself or its deferred ARM dependency requires it.
        if("${_sdx_android_stl}" STREQUAL "c++_shared" OR
           TARGET armcompute_external)
            set(_sdx_cxx_shared_library "")
            foreach(_sdx_implicit_link_directory IN LISTS
                    CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES)
                if(EXISTS
                   "${_sdx_implicit_link_directory}/libc++_shared.so")
                    set(_sdx_cxx_shared_library
                        "${_sdx_implicit_link_directory}/libc++_shared.so")
                    break()
                endif()
            endforeach()
            if(NOT IS_ABSOLUTE "${_sdx_cxx_shared_library}" OR
               NOT EXISTS "${_sdx_cxx_shared_library}")
                message(FATAL_ERROR
                    "Android libc++_shared.so was not found in the C++ "
                    "toolchain's implicit link directories: "
                    "${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}")
            endif()
            list(APPEND _sdx_runtime_dependency_files
                "${_sdx_cxx_shared_library}")
        endif()

        # Vendor accelerator adapters are supplied by the build environment and
        # copied into the same jni/<abi> directory as the SDX runtime. A pipe is
        # accepted as a path-list separator because CMake uses semicolons for lists.
        set(_sdx_extra_runtime_dependency_files "")
        if(DEFINED SDX_EXTRA_RUNTIME_DEPENDENCY_FILES AND
           NOT "${SDX_EXTRA_RUNTIME_DEPENDENCY_FILES}" STREQUAL "")
            set(_sdx_extra_runtime_dependency_files
                "${SDX_EXTRA_RUNTIME_DEPENDENCY_FILES}")
        elseif(DEFINED ENV{SDX_EXTRA_RUNTIME_DEPENDENCY_FILES} AND
               NOT "$ENV{SDX_EXTRA_RUNTIME_DEPENDENCY_FILES}" STREQUAL "")
            set(_sdx_extra_runtime_dependency_files
                "$ENV{SDX_EXTRA_RUNTIME_DEPENDENCY_FILES}")
        endif()
        string(REPLACE "|" ";" _sdx_extra_runtime_dependency_files
            "${_sdx_extra_runtime_dependency_files}")
        foreach(_sdx_dependency_candidate IN LISTS
                _sdx_extra_runtime_dependency_files)
            get_filename_component(_sdx_dependency_name
                "${_sdx_dependency_candidate}" NAME)
            string(TOLOWER "${_sdx_dependency_name}" _sdx_dependency_name_lower)
            # ARM Compute is an ExternalProject artifact. Its install directory
            # is created by the build before sdx_runtime_bindings runs, so allow
            # the two known Tensor G3 shared objects to be absent at configure
            # time while retaining strict validation for every other dependency.
            set(_sdx_deferred_armcompute_dependency FALSE)
            if(TARGET armcompute_external AND
               _sdx_dependency_name_lower MATCHES
               "^libarm_compute(_graph)?\\.so$")
                set(_sdx_deferred_armcompute_dependency TRUE)
            endif()
            if(NOT IS_ABSOLUTE "${_sdx_dependency_candidate}" OR
               IS_DIRECTORY "${_sdx_dependency_candidate}" OR
               (NOT EXISTS "${_sdx_dependency_candidate}" AND
                NOT _sdx_deferred_armcompute_dependency))
                message(FATAL_ERROR
                    "Invalid SDX extra runtime dependency: ${_sdx_dependency_candidate}")
            endif()
            if((SD_HEXAGON OR SD_TPU OR SD_VULKAN OR SD_NNAPI_ACCELERATOR_ONLY) AND
               _sdx_dependency_name_lower MATCHES
               "openblas|mkl|gfortran|quadmath")
                message(FATAL_ERROR
                    "Device accelerator dependency cannot be host BLAS: ${_sdx_dependency_candidate}")
            endif()
            list(APPEND _sdx_runtime_dependency_files
                "${_sdx_dependency_candidate}")
        endforeach()
    endif()
    list(REMOVE_DUPLICATES _sdx_runtime_dependency_files)
    string(JOIN "|" _sdx_runtime_dependency_files_encoded
        ${_sdx_runtime_dependency_files})

    set(_sdx_binding_package_cmds "")
    foreach(_sdx_variant IN LISTS _sdx_runtime_variants)
        set(_sdx_default_gpu_target "AUTO")
        set(_sdx_default_accelerator "NONE")
        if(_sdx_variant STREQUAL "cuda")
            set(_sdx_default_gpu_target "CUDA")
        elseif(_sdx_variant STREQUAL "amd")
            set(_sdx_default_gpu_target "AMD")
        elseif(_sdx_variant STREQUAL "vulkan")
            set(_sdx_default_gpu_target "VULKAN")
        elseif(_sdx_variant STREQUAL "hexagon")
            set(_sdx_default_accelerator "QUALCOMM_HEXAGON_HTP")
        elseif(_sdx_variant STREQUAL "tpu")
            set(_sdx_default_accelerator "PJRT_TPU")
        elseif(_sdx_variant STREQUAL "tensor-g3")
            set(_sdx_default_accelerator "NNAPI_ACCELERATOR_ONLY")
        endif()

        set(_sdx_required_accelerator_device "")
        if(SD_NNAPI_ACCELERATOR_ONLY AND DEFINED SD_NNAPI_REQUIRED_DEVICE_NAME)
            set(_sdx_required_accelerator_device "${SD_NNAPI_REQUIRED_DEVICE_NAME}")
        endif()

        list(APPEND _sdx_binding_package_cmds
            COMMAND ${CMAKE_COMMAND}
                "-DSDX_SDK_DIR=${SDX_RUNTIME_SDK_DIR}"
                "-DSDX_PLATFORM_ID=${_sdx_platform_id}"
                "-DSDX_OS=${_sdx_os}"
                "-DSDX_ARCH=${_sdx_arch}"
                "-DSDX_ANDROID_ABI=${_sdx_android_abi}"
                "-DSDX_ANDROID_API=${_sdx_android_api}"
                "-DSDX_ANDROID_NDK_MAJOR=${_sdx_android_ndk_major}"
                "-DSDX_ANDROID_STL=${_sdx_android_stl}"
                "-DSDX_VARIANT=${_sdx_variant}"
                "-DSDX_DEFAULT_GPU_TARGET=${_sdx_default_gpu_target}"
                "-DSDX_DEFAULT_ACCELERATOR=${_sdx_default_accelerator}"
                "-DSDX_REQUIRED_ACCELERATOR_DEVICE=${_sdx_required_accelerator_device}"
                "-DSDX_LIBRARY_FILE=$<TARGET_FILE:${_sdx_sdk_target}>"
                "-DSDX_LINKER_FILE=$<TARGET_LINKER_FILE:${_sdx_sdk_target}>"
                "-DSDX_HEADER_FILE=${SDX_RUNTIME_HEADER}"
                "-DSDX_TOKENIZER_HEADER_FILE=${SDX_TOKENIZER_HEADER_FILE}"
                "-DSDX_TOKENIZER_LIBRARY_FILE=${SDX_TOKENIZER_LIBRARY_FILE}"
                "-DSDX_RUNTIME_DEPENDENCY_FILES=${_sdx_runtime_dependency_files_encoded}"
                "-DSDX_README_FILE=${SDX_RUNTIME_README}"
                "-DSDX_SCHEMA_FILE=${SDX_RUNTIME_SCHEMA}"
                "-DSDX_TEXT_GENERATION_SCHEMA_FILE=${SDX_RUNTIME_TEXT_GENERATION_SCHEMA}"
                "-DSDX_BINDINGS_TEMPLATE_DIR=${SDX_RUNTIME_LANGUAGE_BINDINGS_DIR}"
                "-DSDX_JAVA_SRC_DIR=${SDX_RUNTIME_JAVA_SRC_DIR}"
                "-DSDX_STANDALONE=${_sdx_standalone}"
                "-DSDX_STRIP_TOOL=${_sdx_strip_tool}"
                "-DSDX_RUNTIME_ABI=1"
                "-DSDX_HAVE_CUDA=${_sdx_have_cuda}"
                "-DSDX_HAVE_ZLUDA=${_sdx_have_zluda}"
                "-DSDX_HAVE_TRITON=${_sdx_have_triton}"
                "-DSDX_HAVE_VULKAN=${_sdx_have_vulkan}"
                "-DSDX_HAVE_MLIR=${_sdx_have_mlir}"
                "-DSDX_HAVE_MLX=${_sdx_have_mlx}"
                "-DSDX_HAVE_NNAPI=${_sdx_have_nnapi}"
                "-DSDX_HAVE_ONEDNN=${_sdx_have_onednn}"
                "-DSDX_HAVE_ARMCOMPUTE=${_sdx_have_armcompute}"
                "-DSDX_HAVE_HEXAGON=${_sdx_have_hexagon}"
                "-DSDX_HAVE_TPU=${_sdx_have_tpu}"
                "-DSDX_ENABLE_ANDROID_AAR=${_sdx_enable_android_aar}"
                "-DSDX_ENABLE_APPLE_XCFRAMEWORK=${_sdx_enable_apple_xcframework}"
                -P "${SDX_RUNTIME_BINDINGS_SCRIPT}")
    endforeach()

    if(EXISTS "${SDX_RUNTIME_BINDINGS_SCRIPT}")
        add_custom_target(sdx_runtime_bindings ALL
            ${_sdx_binding_package_cmds}
            DEPENDS sdx_runtime_sdk
            COMMENT "Packaging platform-specific SDX runtime bindings for ${_sdx_platform_id}"
            VERBATIM
        )
        if(TARGET armcompute_external)
            add_dependencies(sdx_runtime_bindings armcompute_external)
        endif()
    endif()

    install(FILES "${SDX_RUNTIME_HEADER}" DESTINATION include/dsp/runtime)
    if(_sdx_have_tokenizer_artifacts)
        install(FILES "${SDX_TOKENIZER_HEADER_FILE}" DESTINATION include/tokenizer)
        install(FILES "${SDX_TOKENIZER_LIBRARY_FILE}" DESTINATION lib)
    endif()
    if(EXISTS "${SDX_RUNTIME_README}")
        install(FILES "${SDX_RUNTIME_README}" DESTINATION share/dsp/runtime)
    endif()
    if(EXISTS "${SDX_RUNTIME_SCHEMA}")
        install(FILES "${SDX_RUNTIME_SCHEMA}" DESTINATION share/dsp)
    endif()
    if(EXISTS "${SDX_RUNTIME_TEXT_GENERATION_SCHEMA}")
        install(FILES "${SDX_RUNTIME_TEXT_GENERATION_SCHEMA}" DESTINATION share/dsp)
    endif()
endif()

include(PostBuild)

# =============================================================================
# BUILD STAMP - per-build unique fingerprint for DSP plan disk cache
# Generates include/build_stamp.h with a timestamp that changes every rebuild
# so the Java DspPlanDiskCache invalidates stale .bin/.meta entries.
#
# Strategy: write once at configure time (for the very first compilation), then
# add an ALL custom target that regenerates it unconditionally every build via a
# small inline CMake script.  The generated file lives in
# ${CMAKE_CURRENT_BINARY_DIR}/include/ which PostBuild.cmake already adds to
# the include path via include_directories(${CMAKE_CURRENT_BINARY_DIR}/include).
# build_info.cpp then #includes it and appends LIBND4J_BUILD_STAMP to the
# buildInfo() string.
# =============================================================================

# Ensure the output directory exists
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

# Write a placeholder at configure time so build_info.cpp can always find the
# header even if the ALL target hasn't run yet in this CMake invocation.
# The real per-build timestamp is written by the custom target below.
set(_build_stamp_file "${CMAKE_CURRENT_BINARY_DIR}/include/build_stamp.h")
if(NOT EXISTS "${_build_stamp_file}")
    file(WRITE "${_build_stamp_file}"
        "/* Auto-generated -- regenerated every build. Do NOT edit. */\n"
        "#ifndef LIBND4J_BUILD_STAMP_H\n"
        "#define LIBND4J_BUILD_STAMP_H\n"
        "#define LIBND4J_BUILD_STAMP \"configure-time-placeholder\"\n"
        "#endif /* LIBND4J_BUILD_STAMP_H */\n"
    )
endif()

# Custom target that runs on every build and rewrites build_stamp.h with the
# current wall-clock timestamp.  It depends on nothing so CMake always runs it.
add_custom_target(generate_build_stamp ALL
    COMMAND ${CMAKE_COMMAND}
        "-DOUT_FILE=${_build_stamp_file}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuildStamp.cmake"
    COMMENT "Regenerating build_stamp.h (per-build fingerprint)"
    VERBATIM
)

# Make the object library wait for build_stamp.h before any TU compiles.
if(TARGET ${SD_LIBRARY_NAME}_object)
    add_dependencies(${SD_LIBRARY_NAME}_object generate_build_stamp)
endif()

print_status_colored("SUCCESS" "=== ENHANCED TEMPLATE SYSTEM VERIFICATION COMPLETE ===")
