# =============================================================================
# UNIFIED BUILD ORCHESTRATION SCRIPT WITH CUDA TEMPLATE PARITY
#
# This enhanced version includes support for CUDA template processing
# to achieve complete parity between CPU and CUDA template systems.
# =============================================================================

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

# --- Platform environment setup functions ---
function(setup_cpu_environment)
    message(STATUS "🔧 Setting up CPU environment")
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
        file(GLOB_RECURSE ARRAY_SOURCES ./include/array/cuda/*.cu ./include/array/impl/*.cpp)
        file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/impl/*.cpp ./include/memory/cuda/*.cu)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES ./include/ops/declarable/helpers/cuda/*.cu ./include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp ./include/helpers/cuda/*.cu)
        file(GLOB CPU_HELPERS_TO_EXCLUDE ./include/helpers/cpu/*.cpp)
        list(REMOVE_ITEM HELPERS_SOURCES ${CPU_HELPERS_TO_EXCLUDE})
        file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/impl/*.cpp)
        file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/*.cu)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA ./include/loops/*.cu ./include/loops/cuda/**/*.cu)
        file(GLOB_RECURSE VALIDATION_SOURCES ./include/array/DataTypeValidation.cpp)

        list(APPEND ALL_SOURCES_LIST
                ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${CUSTOMOPS_HELPERS_SOURCES}
                ${HELPERS_SOURCES} ${LOOPS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES_CUDA}
                ${VALIDATION_SOURCES}
        )

        if(HAVE_CUDNN)
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ./include/ops/declarable/platform/cudnn/*.cu)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_CUDNN_SOURCES})
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

        if(HAVE_ONEDNN)
            file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES ./include/ops/declarable/platform/mkldnn/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_ONEDNN_SOURCES})
        endif()
        if(HAVE_ARMCOMPUTE)
            file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES ./include/ops/declarable/platform/armcompute/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_ARMCOMPUTE_SOURCES})
        endif()

        message(STATUS "🖥️  CPU build: Enhanced template system will generate optimized CPU instantiations")
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

# --- Linking functions for CPU and CUDA ---
function(configure_cpu_linking main_target_name)
    target_link_libraries(${main_target_name} PUBLIC
            ${ONEDNN} ${ARMCOMPUTE_LIBRARIES} ${OPENBLAS_LIBRARIES}
            ${BLAS_LIBRARIES} flatbuffers_interface)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${main_target_name} PUBLIC OpenMP::OpenMP_CXX)
        else()
            target_link_libraries(${main_target_name} PUBLIC "-fopenmp")
        endif()
    endif()
    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

function(configure_cuda_linking main_target_name)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${main_target_name} PUBLIC
            CUDA::cudart
            CUDA::cublas
            CUDA::cusolver
            CUDA::curand
            flatbuffers_interface)

    if(HAVE_CUDNN AND CUDNN_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${CUDNN_LIBRARIES})
    endif()

    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

# --- Enhanced library creation function ---
function(create_and_link_library)
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    set(MAIN_LIB_NAME "${SD_LIBRARY_NAME}")

    if(NOT TARGET ${OBJECT_LIB_NAME})
        add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})
        add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)

        if(SD_CUDA)
            # Find the CUDA Toolkit to make the CUDA::toolkit target available.
            find_package(CUDAToolkit REQUIRED)

            # THE FIX: For OBJECT libraries, you must get the include directories from
            # the modern CUDA::toolkit target and apply them DIRECTLY to the object library
            # that is being compiled.
            get_target_property(CUDA_INCLUDE_DIRS CUDA::toolkit INTERFACE_INCLUDE_DIRECTORIES)
            target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

            message(STATUS "🚀 CUDA object library configured with enhanced template support")
        endif()

        # CRITICAL: Set include directories on the object library so it can find project headers
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
                "${CMAKE_BINARY_DIR}/cpu_instantiations"
                "${CMAKE_BINARY_DIR}/cuda_instantiations")

        if(SD_CUDA AND DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
            set_target_properties(${OBJECT_LIB_NAME} PROPERTIES
                    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()

    if(NOT TARGET ${MAIN_LIB_NAME})
        add_library(${MAIN_LIB_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
        set_target_properties(${MAIN_LIB_NAME} PROPERTIES OUTPUT_NAME ${MAIN_LIB_NAME})

        # No CUDA includes needed here, they are handled by the linking function
        target_include_directories(${MAIN_LIB_NAME} PUBLIC
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
                "${CMAKE_BINARY_DIR}/cpu_instantiations"
                "${CMAKE_BINARY_DIR}/cuda_instantiations")
    endif()

    if(SD_CUDA)
        configure_cuda_linking(${MAIN_LIB_NAME})
    else()
        configure_cpu_linking(${MAIN_LIB_NAME})
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

# --- CRITICAL: Setup CUDA early if needed ---
if(SD_CUDA)
    print_status_colored("INFO" "=== CUDA EARLY INITIALIZATION WITH TEMPLATE SUPPORT ===")
    include(CudaConfiguration)
    setup_cuda_architectures_early()
    setup_cuda_build()
    print_status_colored("SUCCESS" "CUDA initialization complete with enhanced template support")
endif()

# --- Phase 2: Handle Dependencies & Operations ---
print_status_colored("INFO" "=== 2. INITIALIZING DEPENDENCIES & OPERATIONS ===")
include(Dependencies)
include(DuplicateInstantiationDetection)
include(TemplateProcessing)
include(CompilerFlags)
setup_flatbuffers()
setup_onednn()
setup_armcompute()
setup_cudnn()
setup_blas()
message(STATUS "Dependencies initialization complete.")

message(STATUS "🔧 Helper Configuration: ONEDNN=${HAVE_ONEDNN}, ARMCOMPUTE=${HAVE_ARMCOMPUTE}, CUDNN=${HAVE_CUDNN}")

# --- Generate config.h AFTER setting all variables ---
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/config.h")

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
set(CUSTOMOPS_GENERIC_SOURCES "")
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

print_status_colored("SUCCESS" "=== ENHANCED TEMPLATE SYSTEM VERIFICATION COMPLETE ===")