# cmake/MainBuildFlow.cmake - UPDATED VERSION
# Simplified build flow using the unified selective rendering system
# This replaces the complex multi-phase setup with a single, reliable call

# =============================================================================
# FUNCTION DEFINITIONS (unchanged for backwards compatibility)
# =============================================================================
include(Dependencies)
include(TemplateProcessing)
# Display helper configuration from Options.cmake
message(STATUS "🔧 Helper Configuration:")
message(STATUS "  HELPERS_onednn: ${HELPERS_onednn}")
message(STATUS "  HELPERS_armcompute: ${HELPERS_armcompute}")
message(STATUS "  HELPERS_cudnn: ${HELPERS_cudnn}")
message(STATUS "  HAVE_ONEDNN: ${HAVE_ONEDNN}")
message(STATUS "  HAVE_ARMCOMPUTE: ${HAVE_ARMCOMPUTE}")
message(STATUS "  HAVE_CUDNN: ${HAVE_CUDNN}")


# Compile Definitions for Operations (unchanged)
set(DEFINITIONS_CONTENT "")
if(SD_ALL_OPS OR "${SD_OPS_LIST}" STREQUAL "")
    message("Adding all ops due to empty op list or SD_ALL_OPS definition: SD_ALL_OPS=${SD_ALL_OPS} SD_OPS_LIST=${SD_OPS_LIST}")
    add_compile_definitions(SD_ALL_OPS=1)
    string(APPEND DEFINITIONS_CONTENT "#define SD_ALL_OPS 1\n")
else()
    message("_OPS: ${SD_OPS_LIST}")
    foreach(OP ${SD_OPS_LIST})
        add_compile_definitions(OP_${OP}=1)
        message(STATUS "OP: ${OP}")
        string(APPEND DEFINITIONS_CONTENT "#define OP_${OP} 1\n")
    endforeach()
endif()


# Create the directory for generated files if it doesn't exist
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/generated")

# Write Definitions to include_ops.h file
set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
message("Generating include_ops.h at: ${INCLUDE_OPS_FILE}")
file(WRITE "${INCLUDE_OPS_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")

# Add the generated directory to include paths
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/generated")


if(NOT HAVE_ONEDNN)
    add_compile_definitions(HAVE_ONEDNN=0)
    message(STATUS "🚫 Globally disabled HAVE_ONEDNN")
endif()

if(NOT HAVE_ARMCOMPUTE)
    add_compile_definitions(HAVE_ARMCOMPUTE=0)
    message(STATUS "🚫 Globally disabled HAVE_ARMCOMPUTE")
endif()

if(NOT HAVE_CUDNN)
    add_compile_definitions(HAVE_CUDNN=0)
    message(STATUS "🚫 Globally disabled HAVE_CUDNN")
endif()



function(collect_all_sources_with_selective_rendering out_source_list)
    set(ALL_SOURCES_LIST "")

    # Common sources for both CPU and CUDA builds
    file(GLOB_RECURSE PERF_SOURCES ./include/performance/*.cpp ./include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES ./include/exceptions/*.cpp ./include/exceptions/*.h)
    file(GLOB_RECURSE TYPES_SOURCES ./include/types/*.cpp ./include/types/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES ./include/graph/*.cpp ./include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp ./include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp ./include/indexing/*.h)

    list(APPEND ALL_SOURCES_LIST
            ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${TYPES_SOURCES} ${GRAPH_SOURCES}
            ${CUSTOMOPS_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES}
    )

    if(SD_CUDA)
        message(STATUS "DEBUG: Before setup_template_processing() - CUSTOMOPS_GENERIC_SOURCES count: ${CMAKE_MATCH_COUNT}")
        setup_template_processing()
        list(LENGTH CUSTOMOPS_GENERIC_SOURCES template_count)
        message(STATUS "DEBUG: After setup_template_processing() - CUSTOMOPS_GENERIC_SOURCES count: ${template_count}")

        # CUDA-specific sources
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

        message(STATUS "DEBUG: Adding CUSTOMOPS_GENERIC_SOURCES to CUDA build")
        if(CUSTOMOPS_GENERIC_SOURCES)
            message(STATUS "DEBUG: CUSTOMOPS_GENERIC_SOURCES contains ${template_count} files:")
            set(debug_count 0)
            foreach(template_file ${CUSTOMOPS_GENERIC_SOURCES})
                if(debug_count LESS 3)
                    message(STATUS "DEBUG:   Template file: ${template_file}")
                    math(EXPR debug_count "${debug_count} + 1")
                endif()
            endforeach()
            if(template_count GREATER 3)
                math(EXPR remaining_count "${template_count} - 3")
                message(STATUS "DEBUG:   ... and ${remaining_count} more template files")
            endif()
        else()
            message(STATUS "DEBUG: CUSTOMOPS_GENERIC_SOURCES is EMPTY!")
        endif()

        list(APPEND ALL_SOURCES_LIST
                ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${CUSTOMOPS_HELPERS_SOURCES}
                ${HELPERS_SOURCES} ${LOOPS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES_CUDA} ${VALIDATION_SOURCES}
                ${CUSTOMOPS_GENERIC_SOURCES}
        )
        if(HAVE_CUDNN)
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ./include/ops/declarable/platform/cudnn/*.cu)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_CUDNN_SOURCES})
        endif()
    else()
        message(STATUS "DEBUG: Before setup_template_processing() - CUSTOMOPS_GENERIC_SOURCES count: ${CMAKE_MATCH_COUNT}")
        setup_template_processing()
        list(LENGTH CUSTOMOPS_GENERIC_SOURCES template_count)
        message(STATUS "DEBUG: After setup_template_processing() - CUSTOMOPS_GENERIC_SOURCES count: ${template_count}")

        # CPU-specific sources
        file(GLOB_RECURSE EXEC_SOURCES ./include/execution/*.cpp)
        file(GLOB_RECURSE ARRAY_SOURCES ./include/array/*.cpp)
        file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL_SOURCES ./include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU_SOURCES ./include/ops/declarable/helpers/cpu/*.cpp)
        file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp  ./include/helpers/cpu/*.cpp)
        file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/cpu/*.cpp)
        file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/*.cpp)

        message(STATUS "DEBUG: Adding CUSTOMOPS_GENERIC_SOURCES to CPU build")
        if(CUSTOMOPS_GENERIC_SOURCES)
            message(STATUS "DEBUG: CUSTOMOPS_GENERIC_SOURCES contains ${template_count} files:")
            set(debug_count 0)
            foreach(template_file ${CUSTOMOPS_GENERIC_SOURCES})
                if(debug_count LESS 3)
                    message(STATUS "DEBUG:   Template file: ${template_file}")
                    math(EXPR debug_count "${debug_count} + 1")
                endif()
            endforeach()
            if(template_count GREATER 3)
                math(EXPR remaining_count "${template_count} - 3")
                message(STATUS "DEBUG:   ... and ${remaining_count} more template files")
            endif()
        else()
            message(STATUS "DEBUG: CUSTOMOPS_GENERIC_SOURCES is EMPTY!")
        endif()

        list(APPEND ALL_SOURCES_LIST
                ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES}
                ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${HELPERS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES}
                ${CUSTOMOPS_GENERIC_SOURCES}
        )
        if(HAVE_ONEDNN)
            file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES ./include/ops/declarable/platform/mkldnn/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_ONEDNN_SOURCES})
        endif()
        if(HAVE_ARMCOMPUTE)
            file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES ./include/ops/declarable/platform/armcompute/*.cpp)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_ARMCOMPUTE_SOURCES})
        endif()

        if(NOT HAVE_ONEDNN)
            list(FILTER EXEC_SOURCES EXCLUDE REGEX ".*LaunchContext\\.cpp$")
            message(STATUS "🚫 Excluded LaunchContext.cpp (OneDNN disabled)")
        endif()
    endif()

    # Add custom ops generic sources and remove duplicates
    list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_GENERIC_SOURCES})
    list(REMOVE_DUPLICATES ALL_SOURCES_LIST)

    list(LENGTH ALL_SOURCES_LIST final_source_count)
    message(STATUS "DEBUG: Final ALL_SOURCES_LIST count: ${final_source_count}")

    # Write all sources to debug file
    set(DEBUG_LOG_FILE "${CMAKE_BINARY_DIR}/sources_debug_log.txt")
    file(WRITE "${DEBUG_LOG_FILE}" "=== ALL SOURCES BEING COMPILED ===\n")
    file(APPEND "${DEBUG_LOG_FILE}" "Total source files: ${final_source_count}\n\n")

    file(APPEND "${DEBUG_LOG_FILE}" "=== TEMPLATE GENERATED SOURCES ===\n")
    if(CUSTOMOPS_GENERIC_SOURCES)
        foreach(template_file ${CUSTOMOPS_GENERIC_SOURCES})
            file(APPEND "${DEBUG_LOG_FILE}" "TEMPLATE: ${template_file}\n")
        endforeach()
    else()
        file(APPEND "${DEBUG_LOG_FILE}" "NO TEMPLATE SOURCES FOUND!\n")
    endif()
    file(APPEND "${DEBUG_LOG_FILE}" "\n")

    file(APPEND "${DEBUG_LOG_FILE}" "=== ALL SOURCES (COMPLETE LIST) ===\n")
    foreach(source_file ${ALL_SOURCES_LIST})
        file(APPEND "${DEBUG_LOG_FILE}" "${source_file}\n")
    endforeach()

    message(STATUS "DEBUG: Complete source list written to: ${DEBUG_LOG_FILE}")

    set(ALL_SOURCES ${ALL_SOURCES_LIST} CACHE INTERNAL "All source files for build")
    set(${out_source_list} ${ALL_SOURCES_LIST} PARENT_SCOPE)
endfunction()

# Links the final CUDA library against all its dependencies.
function(configure_cuda_linking main_target_name object_target_name)
    target_link_libraries(${main_target_name} PUBLIC
            ${CUDA_LIBRARIES}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_CURAND_LIBRARIES}
            ${CUDNN_LIBRARIES}
            flatbuffers_interface
    )

    if(CUDA_cusolver_LIBRARY)
        target_link_libraries(${main_target_name} PUBLIC ${CUDA_cusolver_LIBRARY})
    endif()

    if(CUDA_cusparse_LIBRARY)
        target_link_libraries(${main_target_name} PUBLIC ${CUDA_cusparse_LIBRARY})
    endif()

    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

# Setup CUDA build environment and configuration
function(setup_cuda_environment)
    message(STATUS "🔧 Setting up CUDA environment")

    # Enable CUDA language
    enable_language(CUDA)

    # Find CUDA toolkit
    find_package(CUDA REQUIRED)

    # Set CUDA-specific definitions
    add_definitions(-D__CUDABLAS__=true)
    add_definitions(-DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})
    add_definitions(-DCUDA_VERSION_MINOR=${CUDA_VERSION_MINOR})

    # Configure CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86" CACHE STRING "CUDA architectures")
    endif()

    # Set CUDA compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

    # Set library output directory
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

    message(STATUS "✅ CUDA environment setup complete")
    message(STATUS "   CUDA Version: ${CUDA_VERSION}")
    message(STATUS "   CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endfunction()

# Setup CPU build environment and configuration
function(setup_cpu_environment)
    message(STATUS "🔧 Setting up CPU environment")

    # Set CPU-specific definitions
    add_definitions(-D__CPUBLAS__=true)

    # Set library output directory
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

    # Configure OpenMP if available
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            message(STATUS "✅ OpenMP found and will be linked")
        else()
            message(STATUS "⚠️ OpenMP not found, will use manual configuration")
        endif()
    endif()

    # Platform-specific optimizations
    if(SD_X86_BUILD)
        message(STATUS "🎯 Enabling x86 optimizations")
        # Will be applied to specific files later
    endif()

    message(STATUS "✅ CPU environment setup complete")
endfunction()

# Setup build configuration based on platform and options
function(setup_build_configuration)
    message(STATUS "🔧 Setting up build configuration")

    # Set default build type if not specified
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    endif()

    # Configure compiler-specific flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            message(FATAL_ERROR "❌ You need at least GCC 4.9")
        endif()
    endif()

    # Set C++ standard
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Configure runtime library for MSVC
    if(MSVC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()

    message(STATUS "✅ Build configuration complete")
    message(STATUS "   Build Type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "   C++ Standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "   Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endfunction()


# Function to print colored status messages
function(print_status_colored level message)
    if(level STREQUAL "SUCCESS")
        message(STATUS "✅ ${message}")
    elseif(level STREQUAL "INFO")
        message(STATUS "ℹ️ ${message}")
    elseif(level STREQUAL "WARNING")
        message(WARNING "⚠️ ${message}")
    elseif(level STREQUAL "ERROR")
        message(FATAL_ERROR "❌ ${message}")
    else()
        message(STATUS "${message}")
    endif()
endfunction()

# Links the final CPU library against all its dependencies.
function(configure_cpu_linking main_target_name object_target_name)
    target_link_libraries(${main_target_name} PUBLIC
            ${ONEDNN}
            ${ARMCOMPUTE_LIBRARIES}
            ${OPENBLAS_LIBRARIES}
            ${BLAS_LIBRARIES}
            flatbuffers_interface
    )
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

# Creates the object and shared libraries, and calls the correct linking function.
function(create_and_link_library)
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    set(MAIN_LIB_NAME "${SD_LIBRARY_NAME}")

    # A target guard prevents these from ever running twice.
    if(NOT TARGET ${OBJECT_LIB_NAME})
        message(STATUS "Creating object library: ${OBJECT_LIB_NAME}")
        add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})

        # UPDATED: Add include directories including generated selective rendering headers
        target_include_directories(${OBJECT_LIB_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include/array
                ${CMAKE_CURRENT_SOURCE_DIR}/include/execution
                ${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions
                ${CMAKE_CURRENT_SOURCE_DIR}/include/graph
                ${CMAKE_CURRENT_SOURCE_DIR}/include/helpers
                ${CMAKE_CURRENT_SOURCE_DIR}/include/loops
                ${CMAKE_CURRENT_SOURCE_DIR}/include/memory
                ${CMAKE_CURRENT_SOURCE_DIR}/include/ops
                ${CMAKE_CURRENT_SOURCE_DIR}/include/types
                ${CMAKE_CURRENT_SOURCE_DIR}/include/system
                ${CMAKE_CURRENT_SOURCE_DIR}/include/legacy
                ${CMAKE_CURRENT_SOURCE_DIR}/include/performance
                ${CMAKE_CURRENT_SOURCE_DIR}/include/indexing
                ${CMAKE_CURRENT_SOURCE_DIR}/include/generated
                ${CMAKE_BINARY_DIR}/include                     # Generated selective rendering headers
                ${CMAKE_BINARY_DIR}/compilation_units
                ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include
        )

        # Set preprocessor definitions based on Options.cmake settings
        if(HAVE_ONEDNN)
            target_compile_definitions(${OBJECT_LIB_NAME} PRIVATE HAVE_ONEDNN=1)
            add_compile_definitions(HAVE_ONEDNN=1)
            message(STATUS "✅ Enabling HAVE_ONEDNN preprocessor macro")
        else()
            target_compile_definitions(${OBJECT_LIB_NAME} PRIVATE HAVE_ONEDNN=0)
            add_compile_definitions(HAVE_ONEDNN=0)
            message(STATUS "🚫 Disabling HAVE_ONEDNN preprocessor macro (HAVE_ONEDNN=0)")
        endif()

        if(HAVE_ARMCOMPUTE)
            target_compile_definitions(${OBJECT_LIB_NAME} PRIVATE HAVE_ARMCOMPUTE=1)
            add_compile_definitions(HAVE_ARMCOMPUTE=1)
            message(STATUS "✅ Enabling HAVE_ARMCOMPUTE preprocessor macro")
        else()
            target_compile_definitions(${OBJECT_LIB_NAME} PRIVATE HAVE_ARMCOMPUTE=0)
            add_compile_definitions(HAVE_ARMCOMPUTE=0)
            message(STATUS "🚫 Disabling HAVE_ARMCOMPUTE preprocessor macro")
        endif()

        if(HAVE_CUDNN)
            target_compile_definitions(${OBJECT_LIB_NAME} PRIVATE HAVE_CUDNN=1)
            add_compile_definitions(HAVE_CUDNN=1)
            message(STATUS "✅ Enabling HAVE_CUDNN preprocessor macro")
        else()
            target_compile_definitions(${OBJECT_LIB_NAME} PRIVATE HAVE_CUDNN=0)
            add_compile_definitions(HAVE_CUDNN=0)
            message(STATUS "🚫 Disabling HAVE_CUDNN preprocessor macro")
        endif()

        add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)
        if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
            add_dependencies(${OBJECT_LIB_NAME} generate_flatbuffers_headers)
        endif()

        # UPDATED: Add selective rendering dependency (handled by unified system)
        # The unified system ensures headers are generated before this target builds
    endif()

    if(NOT TARGET ${MAIN_LIB_NAME})
        message(STATUS "Creating shared library: ${MAIN_LIB_NAME}")
        add_library(${MAIN_LIB_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
        set_target_properties(${MAIN_LIB_NAME} PROPERTIES OUTPUT_NAME ${MAIN_LIB_NAME})

        # Also set include directories for the main library
        target_include_directories(${MAIN_LIB_NAME} PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_BINARY_DIR}/include                     # Generated selective rendering headers
        )
    endif()

    message(STATUS "Configuring target dependencies for ${MAIN_LIB_NAME}")
    if(SD_CUDA)
        configure_cuda_linking(${MAIN_LIB_NAME} ${OBJECT_LIB_NAME})
    else()
        configure_cpu_linking(${MAIN_LIB_NAME} ${OBJECT_LIB_NAME})
    endif()
    message(STATUS "Target dependencies configured.")
endfunction()

# =============================================================================
# SIMPLIFIED BUILD EXECUTION - USING UNIFIED SELECTIVE RENDERING
# =============================================================================

print_status_colored("INFO" "=== ORCHESTRATING LIBND4J BUILD (UNIFIED SELECTIVE RENDERING) ===")
include(BasicSetup)

# --- Phase 1: Initialize Library Name ---
print_status_colored("INFO" "=== INITIALIZING BUILD CONFIGURATION ===")
if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
    endif()
endif()

# --- Phase 2: Setup Platform and Compiler ---
print_status_colored("INFO" "=== SETTING UP PLATFORM AND COMPILER ===")
if(SD_CUDA)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
    setup_cuda_environment()
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
    setup_cpu_environment()
endif()
message(STATUS "Platform and compiler setup complete.")

# --- Phase 3: Initialize Dependencies ---
print_status_colored("INFO" "=== INITIALIZING DEPENDENCIES ===")
setup_flatbuffers()
setup_onednn()
setup_armcompute()
setup_cudnn()
setup_blas()
message(STATUS "Dependencies initialization complete.")

# ============================================================================
# UNIFIED SELECTIVE RENDERING SETUP - SINGLE CALL REPLACES COMPLEX LOGIC
# ============================================================================

print_status_colored("INFO" "=== SETTING UP UNIFIED SELECTIVE RENDERING ===")

# Include the unified core system
include(SelectiveRenderingCore)

# Single call to set up everything - replaces all the complex multi-phase setup
setup_selective_rendering_unified_safe(
        TYPE_PROFILE "${SD_TYPE_PROFILE}"
        OUTPUT_DIR "${CMAKE_BINARY_DIR}/include"
)


# Verify that the setup worked
if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
    message(FATAL_ERROR "❌ CRITICAL: Unified selective rendering setup failed!")
endif()

# Display results
list(LENGTH UNIFIED_ACTIVE_TYPES type_count)
list(LENGTH UNIFIED_COMBINATIONS_2 combo_2_count)
list(LENGTH UNIFIED_COMBINATIONS_3 combo_3_count)

message(STATUS "✅ Unified selective rendering setup complete:")
message(STATUS "   🔢 Active types: ${type_count}")
message(STATUS "   📊 2-type combinations: ${combo_2_count}")
message(STATUS "   📊 3-type combinations: ${combo_3_count}")
message(STATUS "   🎯 Type list: ${UNIFIED_ACTIVE_TYPES}")

# Map to legacy variables for backwards compatibility (done automatically by wrapper)
if(DEFINED COMBINATIONS_3)
    list(LENGTH COMBINATIONS_3 legacy_combo_count)
    message(STATUS "   🔄 Legacy compatibility: ${legacy_combo_count} combinations mapped")
endif()

# ============================================================================
# FINAL VALIDATION - ENSURE CRITICAL TYPES ARE PRESENT
# ============================================================================

# Verify float16 is present if it should be (this was the original issue)
if(DEFINED HAS_FLOAT16 AND HAS_FLOAT16)
    list(FIND UNIFIED_ACTIVE_TYPES "float16" float16_pos)
    if(float16_pos LESS 0)
        message(FATAL_ERROR "❌ CRITICAL: float16 missing despite HAS_FLOAT16=TRUE! This will cause build errors.")
    else()
        message(STATUS "✅ VERIFIED: float16 present at position ${float16_pos}")

        # Also verify it appears in combinations
        set(float16_in_combos FALSE)
        foreach(combo ${UNIFIED_COMBINATIONS_3})
            if(combo MATCHES "${float16_pos}")
                set(float16_in_combos TRUE)
                break()
            endif()
        endforeach()

        if(float16_in_combos)
            message(STATUS "✅ VERIFIED: float16 appears in combinations")
        else()
            message(FATAL_ERROR "❌ CRITICAL: float16 not found in any combinations!")
        endif()
    endif()
endif()

print_status_colored("SUCCESS" "=== UNIFIED SELECTIVE RENDERING COMPLETE ===")

# --- Phase 4: Process Templates (Simplified) ---
print_status_colored("INFO" "=== PROCESSING TEMPLATES ===")
set(CUSTOMOPS_GENERIC_SOURCES "")

# Template processing is now automatically handled by the unified system
# The generated headers are already available and combinations are ready

# Display statistics if available
if(SD_ENABLE_SEMANTIC_FILTERING AND DEFINED combo_3_count AND combo_3_count GREATER 0)
    math(EXPR total_possible "${type_count} * ${type_count} * ${type_count}")
    if(total_possible GREATER 0)
        math(EXPR savings_percent "100 - (100 * ${combo_3_count} / ${total_possible})")
        message(STATUS "📊 Template Optimization Statistics:")
        message(STATUS "   Total possible combinations: ${total_possible}")
        message(STATUS "   Active combinations: ${combo_3_count}")
        message(STATUS "   Memory savings: ~${savings_percent}% reduction in template instantiations")
        if(DEFINED SD_TYPE_PROFILE)
            message(STATUS "   Type profile: ${SD_TYPE_PROFILE}")
        endif()
    endif()
endif()

message(STATUS "Template processing complete.")

# --- Phase 5 & 6: Create and Link Final Library ---
print_status_colored("INFO" "=== CREATING AND LINKING FINAL LIBRARY ===")
collect_all_sources_with_selective_rendering(ALL_SOURCES)
setup_build_configuration()

create_and_link_library()
message(STATUS "Final library target created and linked.")

# --- Phase 7: Generate Usage Documentation (Optional) ---
if(DEFINED SD_GENERATE_TYPE_REPORT AND SD_GENERATE_TYPE_REPORT)
    print_status_colored("INFO" "=== GENERATING USAGE DOCUMENTATION ===")

    set(usage_doc "${CMAKE_BINARY_DIR}/selective_rendering_usage.md")
    set(doc_content "# Unified Selective Rendering Usage\n\n")
    string(APPEND doc_content "This build was configured with the unified selective rendering system.\n\n")

    if(DEFINED SD_TYPE_PROFILE)
        string(APPEND doc_content "**Type Profile**: ${SD_TYPE_PROFILE}\n\n")
    endif()

    string(APPEND doc_content "## Statistics\n\n")
    string(APPEND doc_content "- Active types: ${type_count}\n")
    string(APPEND doc_content "- Active combinations: ${combo_3_count}\n")
    if(SD_ENABLE_SEMANTIC_FILTERING AND DEFINED combo_3_count)
        math(EXPR total_possible "${type_count} * ${type_count} * ${type_count}")
        if(total_possible GREATER 0)
            math(EXPR savings "100 - (100 * ${combo_3_count} / ${total_possible})")
            string(APPEND doc_content "- Memory savings: ~${savings}% reduction\n")
        endif()
    endif()

    string(APPEND doc_content "\n## Active Types\n\n")
    foreach(type_name ${UNIFIED_ACTIVE_TYPES})
        string(APPEND doc_content "- ${type_name}\n")
    endforeach()

    string(APPEND doc_content "\n## System Information\n\n")
    string(APPEND doc_content "- Selective rendering: ENABLED (Unified System)\n")
    string(APPEND doc_content "- Semantic filtering: ${SD_ENABLE_SEMANTIC_FILTERING}\n")
    if(DEFINED SD_TYPE_PROFILE)
        string(APPEND doc_content "- Type profile: ${SD_TYPE_PROFILE}\n")
    endif()
    string(APPEND doc_content "- Platform: ${CMAKE_SYSTEM_NAME}\n")
    if(SD_CUDA)
        string(APPEND doc_content "- CUDA: Enabled\n")
    else()
        string(APPEND doc_content "- CUDA: Disabled\n")
    endif()

    string(APPEND doc_content "\n## Usage\n\n")
    string(APPEND doc_content "Source files automatically use selective rendering when they include:\n")
    string(APPEND doc_content "```cpp\n#include <system/selective_rendering.h>\n```\n\n")
    string(APPEND doc_content "The unified system ensures optimal type combinations while preserving all essential patterns.\n\n")

    string(APPEND doc_content "## Migration to Unified API\n\n")
    string(APPEND doc_content "For new code, consider using the unified API:\n")
    string(APPEND doc_content "```cmake\nsetup_selective_rendering_unified(\n")
    string(APPEND doc_content "    TYPE_PROFILE \${SD_TYPE_PROFILE}\n")
    string(APPEND doc_content "    OUTPUT_DIR \${CMAKE_BINARY_DIR}/include\n")
    string(APPEND doc_content ")\n```\n\n")

    file(WRITE "${usage_doc}" "${doc_content}")
    message(STATUS "📝 Usage documentation written to: ${usage_doc}")
endif()

print_status_colored("SUCCESS" "=== BUILD ORCHESTRATION COMPLETE (UNIFIED SELECTIVE RENDERING) ===")

# ============================================================================
# FINAL BUILD SUMMARY WITH UNIFIED SYSTEM STATUS
# ============================================================================

message(STATUS "")
message(STATUS "🎯 BUILD SUMMARY:")
message(STATUS "   Library: ${SD_LIBRARY_NAME}")
message(STATUS "   Platform: ${CMAKE_SYSTEM_NAME}")
if(SD_CUDA)
    message(STATUS "   CUDA: Enabled")
else()
    message(STATUS "   CUDA: Disabled")
endif()

# Unified system status
message(STATUS "   Selective Rendering: UNIFIED SYSTEM")
if(DEFINED SD_TYPE_PROFILE)
    message(STATUS "   Type Profile: ${SD_TYPE_PROFILE}")
endif()
if(DEFINED type_count)
    message(STATUS "   Active Types: ${type_count}")
endif()
if(DEFINED combo_3_count)
    message(STATUS "   Template Combinations: ${combo_3_count}")
endif()

# Memory optimization summary
if(SD_ENABLE_SEMANTIC_FILTERING AND DEFINED combo_3_count AND DEFINED type_count)
    math(EXPR total_possible "${type_count} * ${type_count} * ${type_count}")
    if(total_possible GREATER combo_3_count)
        math(EXPR filtered_count "${total_possible} - ${combo_3_count}")
        message(STATUS "   Template Instantiations: Reduced by ${filtered_count} combinations")
        math(EXPR savings_percent "100 * ${filtered_count} / ${total_possible}")
        message(STATUS "   Memory Optimization: ~${savings_percent}% reduction")
    endif()
endif()

# System health check
set(health_issues "")
if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
    list(APPEND health_issues "No combinations generated")
endif()
if(DEFINED HAS_FLOAT16 AND HAS_FLOAT16)
    list(FIND UNIFIED_ACTIVE_TYPES "float16" float16_check)
    if(float16_check LESS 0)
        list(APPEND health_issues "float16 missing")
    endif()
endif()

list(LENGTH health_issues issue_count)
if(issue_count GREATER 0)
    message(STATUS "   Health Issues: ${health_issues}")
    message(WARNING "⚠️ Build completed but health issues detected!")
else()
    message(STATUS "   System Health: ✅ All checks passed")
endif()

# Performance metrics (if available)
if(DEFINED SRCORE_COMBINATIONS_2 AND DEFINED SRCORE_COMBINATIONS_3)
    list(LENGTH SRCORE_COMBINATIONS_2 perf_combo_2)
    list(LENGTH SRCORE_COMBINATIONS_3 perf_combo_3)
    math(EXPR total_combinations "${perf_combo_2} + ${perf_combo_3}")
    message(STATUS "   Total Combinations: ${total_combinations} (2-type: ${perf_combo_2}, 3-type: ${perf_combo_3})")
endif()


# ============================================================================
# COMPATIBILITY AND MIGRATION NOTICES
# ============================================================================

# Show migration notice if this is the first time using unified system
if(NOT DEFINED SRCORE_MIGRATION_NOTICE_SHOWN)
    message(STATUS "")
    message(STATUS "ℹ️  UNIFIED SELECTIVE RENDERING SYSTEM")
    message(STATUS "========================================")
    message(STATUS "Your build is now using the new unified selective rendering system.")
    message(STATUS "")
    message(STATUS "BENEFITS:")
    message(STATUS "  ✅ Single source of truth - no more conflicts")
    message(STATUS "  ✅ Better error handling and diagnostics")
    message(STATUS "  ✅ Improved performance with result caching")
    message(STATUS "  ✅ Automatic compatibility with existing code")
    message(STATUS "")
    message(STATUS "BACKWARDS COMPATIBILITY:")
    message(STATUS "  All existing function calls continue to work unchanged.")
    message(STATUS "  Legacy variables (COMBINATIONS_2, COMBINATIONS_3, etc.) are automatically set.")
    message(STATUS "")
    message(STATUS "NEW FEATURES AVAILABLE:")
    message(STATUS "  - Enhanced type profiles (MINIMAL, ESSENTIAL, QUANTIZATION, LLM, COMPREHENSIVE)")
    message(STATUS "  - Intelligent semantic filtering")
    message(STATUS "  - Automatic error recovery")
    message(STATUS "  - Detailed diagnostic reporting")
    message(STATUS "")
    message(STATUS "To disable this notice, set: SRCORE_MIGRATION_NOTICE_SHOWN=TRUE")
    message(STATUS "========================================")
    message(STATUS "")

    set(SRCORE_MIGRATION_NOTICE_SHOWN TRUE CACHE INTERNAL "Migration notice shown flag")
endif()

# Diagnostic summary for troubleshooting
if(DEFINED SRCORE_ENABLE_DIAGNOSTICS AND SRCORE_ENABLE_DIAGNOSTICS)
    message(STATUS "")
    message(STATUS "🔍 DIAGNOSTIC SUMMARY:")
    message(STATUS "========================================")
    message(STATUS "Unified System Status: ACTIVE")
    if(DEFINED SRCORE_SETUP_COMPLETE)
        message(STATUS "Setup Completion: ${SRCORE_SETUP_COMPLETE}")
    endif()
    if(DEFINED SRCORE_CACHE_VALID)
        message(STATUS "Result Caching: ${SRCORE_CACHE_VALID}")
    endif()

    # Show legacy compatibility status
    set(legacy_vars_ok TRUE)
    foreach(var_name "COMBINATIONS_2" "COMBINATIONS_3" "SD_COMMON_TYPES_COUNT")
        if(NOT DEFINED ${var_name})
            set(legacy_vars_ok FALSE)
            message(STATUS "Missing Legacy Variable: ${var_name}")
        endif()
    endforeach()

    if(legacy_vars_ok)
        message(STATUS "Legacy Compatibility: ✅ All variables mapped correctly")
    else()
        message(STATUS "Legacy Compatibility: ⚠️ Some variables missing")
    endif()

    # Show type name mapping status
    if(DEFINED SD_COMMON_TYPES_COUNT AND SD_COMMON_TYPES_COUNT GREATER 0)
        math(EXPR max_type_index "${SD_COMMON_TYPES_COUNT} - 1")
        set(type_mapping_ok TRUE)
        foreach(i RANGE 0 ${max_type_index})
            if(NOT DEFINED TYPE_NAME_${i})
                set(type_mapping_ok FALSE)
                break()
            endif()
        endforeach()

        if(type_mapping_ok)
            message(STATUS "Type Name Mapping: ✅ All TYPE_NAME_X variables set")
        else()
            message(STATUS "Type Name Mapping: ⚠️ Some TYPE_NAME_X variables missing")
        endif()
    endif()

    message(STATUS "Diagnostic Report: ${CMAKE_BINARY_DIR}/selective_rendering_diagnostic_report.txt")
    message(STATUS "========================================")
    message(STATUS "")
endif()

# Final verification that critical functionality is working
if(COMMAND verify_wrapper_compatibility)
    verify_wrapper_compatibility()
endif()
include(TypeRegistryGenerator)
dump_type_macros_to_disk()


message(STATUS "🚀 Build orchestration complete - System ready for compilation")
message(STATUS "")