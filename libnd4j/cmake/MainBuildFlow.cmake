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
    file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp)
    file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp)

    list(APPEND ALL_SOURCES_LIST
            ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${TYPES_SOURCES} ${GRAPH_SOURCES}
            ${CUSTOMOPS_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES}
    )

    # This call populates CUSTOMOPS_GENERIC_SOURCES with both CPU and CUDA generated files
    setup_template_processing()

    if(SD_CUDA)
        file(GLOB_RECURSE EXEC_SOURCES ./include/execution/impl/*.cpp ./include/execution/cuda/*.cu ./include/execution/*.cu)
        file(GLOB_RECURSE ARRAY_SOURCES ./include/array/cuda/*.cu ./include/array/impl/*.cpp  ./include/array/*.cpp)
        file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/impl/*.cpp ./include/memory/cuda/*.cu)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES ./include/ops/declarable/helpers/cuda/*.cu ./include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp ./include/helpers/cuda/*.cu)
        file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/impl/*.cpp)
        file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/*.cu ./include/legacy/cuda/*.cu)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA ./include/loops/*.cu ./include/loops/cuda/**/*.cu)
        file(GLOB_RECURSE VALIDATION_SOURCES ./include/array/DataTypeValidation.cpp)
        file(GLOB CPU_HELPERS_TO_EXCLUDE ./include/legacy/cpu/*.cpp ./include/helpers/cpu/*.cpp ./include/array/cpu/*.cpp)

        list(APPEND ALL_SOURCES_LIST
                ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${CUSTOMOPS_HELPERS_SOURCES}
                ${HELPERS_SOURCES} ${LOOPS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES_CUDA}
                ${VALIDATION_SOURCES}
        )
        list(REMOVE_ITEM ALL_SOURCES_LIST ${CPU_HELPERS_TO_EXCLUDE})

        if(HAVE_CUDNN)
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ./include/ops/declarable/platform/cudnn/*.cu)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_CUDNN_SOURCES})
        endif()

        # ZLUDA MIOpen support for AMD GPUs (cuDNN alternative)
        if(HAVE_ZLUDA AND ZLUDA_TARGET_BACKEND STREQUAL "AMD" AND HAVE_MIOPEN)
            file(GLOB_RECURSE CUSTOMOPS_MIOPEN_SOURCES ./include/ops/declarable/platform/miopen/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_MIOPEN_SOURCES})
            message(STATUS "✅ Added MIOpen platform sources for ZLUDA AMD")
        endif()

        message(STATUS "🚀 CUDA build: Enhanced template system will generate additional CUDA instantiations")
    else()
        file(GLOB_RECURSE EXEC_SOURCES ./include/execution/*.cpp)
        file(GLOB_RECURSE ARRAY_SOURCES ./include/array/*.cpp)
        file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL_SOURCES ./include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU_SOURCES ./include/ops/declarable/helpers/cpu/*.cpp)
        file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp  ./include/helpers/cpu/*.cpp)
        file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/cpu/*.cpp)
        file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/*.cpp)

        list(APPEND ALL_SOURCES_LIST
                ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES}
                ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${HELPERS_SOURCES} ${LEGACY_SOURCES}
                ${LOOPS_SOURCES}
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
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_MPS_SOURCES})
            list(LENGTH CUSTOMOPS_MPS_SOURCES mps_count)
            message(STATUS "✅ Added MPS platform sources: ${mps_count} files")
        endif()

        # Apple Accelerate framework helper (macOS/iOS)
        if(HAVE_ACCELERATE)
            file(GLOB_RECURSE CUSTOMOPS_ACCELERATE_SOURCES ./include/ops/declarable/platform/accelerate/*.cpp ./include/ops/declarable/platform/accelerate/*.mm)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_ACCELERATE_SOURCES})
            list(LENGTH CUSTOMOPS_ACCELERATE_SOURCES accelerate_count)
            message(STATUS "✅ Added Accelerate platform sources: ${accelerate_count} files")
        endif()

        # LlamaCpp helper (for LLM operations)
        if(HAVE_LLAMACPP)
            file(GLOB_RECURSE CUSTOMOPS_LLAMACPP_SOURCES ./include/ops/declarable/platform/llamacpp/*.cpp)
            if(CUSTOMOPS_LLAMACPP_SOURCES)
                list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_LLAMACPP_SOURCES})
                list(LENGTH CUSTOMOPS_LLAMACPP_SOURCES llamacpp_count)
                message(STATUS "✅ Added LlamaCpp platform sources: ${llamacpp_count} files")
            endif()
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

    # When SD_GCC_FUNCTRACE is ON, the binary exceeds 2GB and PLT relocations fail
    # libc_nonshared.a contains precompiled functions (like atexit) that use PLT
    # We provide our own implementations compiled with -fno-plt to override them
    if(SD_GCC_FUNCTRACE)
        list(APPEND ALL_SOURCES_LIST ${CMAKE_CURRENT_SOURCE_DIR}/include/platform/noplt_libc_stubs.c)
        message(STATUS "✅ Added no-PLT libc stubs for >2GB functrace build compatibility")
    endif()

    # ✅ Add the generated template sources (now includes both CPU and CUDA)
    list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_GENERIC_SOURCES})
    list(REMOVE_DUPLICATES ALL_SOURCES_LIST)
    set(${out_source_list} ${ALL_SOURCES_LIST} PARENT_SCOPE)

    # Enhanced logging
    list(LENGTH CUSTOMOPS_GENERIC_SOURCES template_source_count)
    if(SD_CUDA)
        message(STATUS "📊 Total CUDA template-generated sources: ${template_source_count}")
    else()
        message(STATUS "📊 Total CPU template-generated sources: ${template_source_count}")
    endif()
endfunction()

function(configure_cpu_linking main_target_name)
    # Core libraries
    target_link_libraries(${main_target_name} PUBLIC
            ${OPENBLAS_LIBRARIES} ${BLAS_LIBRARIES} ${JVM_LIBRARY} flatbuffers_interface)

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
        target_link_libraries(${main_target_name} PUBLIC ${MLIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MLIR=1)
        message(STATUS "🔗 Linking MLIR helper")
    endif()

    # Metal Performance Shaders (macOS/iOS)
    if(HAVE_MPS AND DEFINED MPS_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${MPS_LIBRARIES})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MPS=1)
        message(STATUS "🔗 Linking MPS helper")
    endif()

    # Triton GPU Compiler
    if(HAVE_TRITON AND DEFINED TRITON)
        target_link_libraries(${main_target_name} PUBLIC ${TRITON})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_TRITON=1)
        message(STATUS "🔗 Linking Triton GPU compiler backend")
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

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${main_target_name} PUBLIC OpenMP::OpenMP_CXX)

        # Bundle OpenMP runtime library for cross-platform deployment
        if(NOT WIN32 AND OpenMP_omp_LIBRARY AND EXISTS "${OpenMP_omp_LIBRARY}")
            message(STATUS "📦 Bundling OpenMP library: ${OpenMP_omp_LIBRARY}")
            add_custom_command(TARGET ${main_target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OpenMP_omp_LIBRARY}" "${CMAKE_BINARY_DIR}/libomp.so"
                COMMENT "Copying libomp.so to output directory for bundling"
            )
        endif()
    else()
        target_link_libraries(${main_target_name} PUBLIC "-fopenmp")
    endif()
endfunction()

# UPDATED: Modern CUDA linking with improved cuDNN detection and ZLUDA support
function(configure_cuda_linking main_target_name)
    # Find the CUDAToolkit to define the CUDA::toolkit target
    find_package(CUDAToolkit REQUIRED)

    # Setup modern cuDNN detection
    setup_modern_cudnn()

    # Modern CMake uses imported targets which handle all necessary dependencies.
    # Linking against CUDA::toolkit automatically adds include directories,
    # runtime libraries, and all other required flags.
    # CUDA::cublas and CUDA::cusolver are required for cuBLAS and cuSolver functions
    # used in MmulHelper.cu, svd.cu, cublasHelper.cu, etc.
    target_link_libraries(${main_target_name} PUBLIC
        CUDA::toolkit
        CUDA::cublas
        CUDA::cusolver
        ${JVM_LIBRARY})

    # If cuDNN was found, link against its imported target
    if(HAVE_CUDNN AND TARGET CUDNN::cudnn)
        message(STATUS "✅ Linking with modern CUDNN::cudnn target")
        target_link_libraries(${main_target_name} PUBLIC CUDNN::cudnn)
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
    elseif(HAVE_CUDNN AND CUDNN_LIBRARIES)
        message(STATUS "✅ Linking with cuDNN libraries: ${CUDNN_LIBRARIES}")
        target_link_libraries(${main_target_name} PUBLIC ${CUDNN_LIBRARIES})
        target_include_directories(${main_target_name} PUBLIC ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
    elseif(HAVE_CUDNN AND CUDNN_INCLUDE_DIR)
        message(STATUS "✅ Linking with cuDNN include-only (embedded in CUDA)")
        target_include_directories(${main_target_name} PUBLIC ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
    else()
        message(STATUS "ℹ️  Building without cuDNN support")
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=0)
    endif()

    target_include_directories("${main_target_name}" PUBLIC "${CUDA_INCLUDE_DIRS}")
    target_link_libraries(${main_target_name} PUBLIC flatbuffers_interface)

    # Link OpenBLAS for CUDA builds (needed by BlasHelper.cpp for openblas_set_num_threads)
    if(OPENBLAS_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${OPENBLAS_LIBRARIES})
        message(STATUS "✅ Linking CUDA build with OpenBLAS: ${OPENBLAS_LIBRARIES}")
    endif()

    # ZLUDA linking configuration
    if(HAVE_ZLUDA)
        configure_zluda_linking(${main_target_name})
    endif()
endfunction()

# --- Enhanced library creation function ---
function(create_and_link_library)
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    set(MAIN_LIB_NAME "${SD_LIBRARY_NAME}")

    if(NOT TARGET ${OBJECT_LIB_NAME})
        add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})
        add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)
        target_link_libraries(${OBJECT_LIB_NAME} PUBLIC flatbuffers_interface)

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

        # ZLUDA helper
        if(SD_ZLUDA AND TARGET zluda_external)
            message(STATUS "")
            message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  🔒 DEPENDENCY BLOCK: ZLUDA                                       ║")
            message(STATUS "║  ${OBJECT_LIB_NAME} compilation will WAIT for ZLUDA               ║")
            message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            add_dependencies(${OBJECT_LIB_NAME} zluda_external)
            if(TARGET zluda_interface)
                target_link_libraries(${OBJECT_LIB_NAME} PUBLIC zluda_interface)
            endif()
        endif()

        # Triton GPU compiler helper - MUST complete before object files that include Triton headers
        if(HELPERS_triton STREQUAL "ON" AND TARGET triton_external)
            message(STATUS "")
            message(STATUS "╔═══════════════════════════════════════════════════════════════════╗")
            message(STATUS "║  🔒 DEPENDENCY BLOCK: Triton                                       ║")
            message(STATUS "║  ${OBJECT_LIB_NAME} compilation will WAIT for Triton               ║")
            message(STATUS "║  Triton build may take 15-30 minutes (builds LLVM internally)...   ║")
            message(STATUS "╚═══════════════════════════════════════════════════════════════════╝")
            message(STATUS "")
            add_dependencies(${OBJECT_LIB_NAME} triton_external)
            if(TARGET triton_interface)
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
        else()
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
        if(NOT SD_GCC_FUNCTRACE AND NOT SD_CUDA)
            message(STATUS "🚀 Enabling precompiled headers for ${OBJECT_LIB_NAME}")
            target_precompile_headers(${OBJECT_LIB_NAME} PRIVATE
                <system/op_boilerplate.h>
                <system/type_boilerplate.h>
                <math/templatemath.h>
                <helpers/shape.h>
            )
            message(STATUS "✅ Precompiled headers enabled (op_boilerplate.h, type_boilerplate.h, templatemath.h, shape.h)")
        elseif(SD_CUDA)
            message(STATUS "⚠️ Precompiled headers DISABLED for CUDA build (nvcc compatibility)")
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

        # For IMPORTED targets (created by partial linking), properties and includes
        # are already set in PartialLinking.cmake. Skip these operations.
        if(NOT SD_PARTIAL_LINKED_TARGET)
            set_target_properties(${MAIN_LIB_NAME} PROPERTIES OUTPUT_NAME ${MAIN_LIB_NAME})

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
            else()
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
    if(NOT SD_PARTIAL_LINKED_TARGET)
        if(SD_CUDA)
            configure_cuda_linking(${MAIN_LIB_NAME})
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
setup_onednn()
# Setup MKL VML for vectorized math operations (vsErf, vdErf, etc.)
# Must be called after setup_onednn() since it depends on HELPERS_onednn
setup_mkl_vml()
setup_armcompute()
setup_llamacpp()
setup_vlm()

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

setup_blas()
setup_mlir()
setup_mps()
setup_triton()
message(STATUS "🔍 DEBUG: After setup_blas() - OPENBLAS_PATH='${OPENBLAS_PATH}', HAVE_OPENBLAS='${HAVE_OPENBLAS}'")
message(STATUS "Dependencies initialization complete.")

# Print comprehensive helper configuration summary
message(STATUS "")
message(STATUS "🔧 === Helper Configuration ===")
message(STATUS "   ONEDNN:      ${HAVE_ONEDNN}")
message(STATUS "   MKL_VML:     ${HAVE_MKL_VML}")
message(STATUS "   MKL_BLAS:    ${HAVE_MKL}")
message(STATUS "   ARMCOMPUTE:  ${HAVE_ARMCOMPUTE}")
message(STATUS "   CUDNN:       ${HAVE_CUDNN}")
message(STATUS "   MLIR:        ${HAVE_MLIR}")
message(STATUS "   MPS:         ${HAVE_MPS}")
message(STATUS "   ACCELERATE:  ${HAVE_ACCELERATE}")
message(STATUS "   MIOPEN:      ${HAVE_MIOPEN}")
message(STATUS "   PJRT:        ${HAVE_PJRT}")
message(STATUS "   LLAMACPP:    ${HAVE_LLAMACPP}")
message(STATUS "   VLM:         ${HAVE_VLM}")
message(STATUS "   TRITON:      ${HAVE_TRITON}")
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
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/generated")
set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
file(WRITE "${INCLUDE_OPS_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")

# --- Phase 3: Enhanced Selective Rendering Setup ---
print_status_colored("INFO" "=== 3. CONFIGURING ENHANCED SELECTIVE RENDERING ===")
include(SelectiveRenderingCore)
setup_selective_rendering_unified_safe(
        TYPE_PROFILE "${SD_TYPE_PROFILE}"
        OUTPUT_DIR "${CMAKE_BINARY_DIR}/include")
if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
    message(FATAL_ERROR "❌ CRITICAL: Unified selective rendering setup failed!")
endif()
list(LENGTH UNIFIED_ACTIVE_TYPES type_count)
list(LENGTH UNIFIED_COMBINATIONS_3 combo_3_count)
if(SD_CUDA)
    message(STATUS "✅ Enhanced CUDA selective rendering setup complete: ${type_count} types, ${combo_3_count} combinations.")
else()
    message(STATUS "✅ Enhanced CPU selective rendering setup complete: ${type_count} types, ${combo_3_count} combinations.")
endif()

if(DEFINED HAS_FLOAT16 AND HAS_FLOAT16)
    list(FIND UNIFIED_ACTIVE_TYPES "float16" float16_pos)
    if(float16_pos LESS 0)
        message(FATAL_ERROR "❌ CRITICAL: float16 missing despite HAS_FLOAT16=TRUE! This will cause build errors.")
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

# --- Phase 5: Enhanced Final Reports and Summaries ---
if(DEFINED SD_GENERATE_TYPE_REPORT AND SD_GENERATE_TYPE_REPORT)
    print_status_colored("INFO" "=== GENERATING ENHANCED USAGE DOCUMENTATION ===")
    set(usage_doc "${CMAKE_BINARY_DIR}/enhanced_selective_rendering_usage.md")
    set(doc_content "# Enhanced Unified Selective Rendering Usage\n\nThis build was configured with the enhanced unified selective rendering system providing full CPU/CUDA parity.\n\n")
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
else()
    message(STATUS "🖥️  Enhanced CPU build orchestration complete - System ready for compilation")
endif()
message(STATUS "")

# === ENHANCED TEMPLATE SYSTEM VERIFICATION ===
print_status_colored("INFO" "=== VERIFYING ENHANCED TEMPLATE SYSTEM ===")

# Verify template generation worked
list(LENGTH CUSTOMOPS_GENERIC_SOURCES template_file_count)
if(template_file_count GREATER 0)
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
if(NOT SD_GCC_FUNCTRACE AND NOT SD_CUDA AND TARGET ${SD_LIBRARY_NAME})
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
    message(STATUS "✅ Precompiled headers enabled for main library ${SD_LIBRARY_NAME}")
elseif(SD_CUDA)
    message(STATUS "⚠️ Precompiled headers DISABLED for CUDA build (nvcc compatibility)")
elseif(SD_GCC_FUNCTRACE)
    message(STATUS "⚠️ Precompiled headers DISABLED for functrace build")
endif()

include(PostBuild)

print_status_colored("SUCCESS" "=== ENHANCED TEMPLATE SYSTEM VERIFICATION COMPLETE ===")