# cmake/MainBuildFlow.cmake
# This file defines the functions that orchestrate the build and then executes them procedurally.

# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================
# Add global include directory for header resolution


# Display helper configuration from Options.cmake
message(STATUS "🔧 Helper Configuration:")
message(STATUS "  HELPERS_onednn: ${HELPERS_onednn}")
message(STATUS "  HELPERS_armcompute: ${HELPERS_armcompute}")
message(STATUS "  HELPERS_cudnn: ${HELPERS_cudnn}")
message(STATUS "  HAVE_ONEDNN: ${HAVE_ONEDNN}")
message(STATUS "  HAVE_ARMCOMPUTE: ${HAVE_ARMCOMPUTE}")
message(STATUS "  HAVE_CUDNN: ${HAVE_CUDNN}")

# CRITICAL FIX: Set global compile definitions early
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
# Gathers all .cpp and .cu source files for the build.
function(collect_all_sources out_source_list)
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
        list(APPEND ALL_SOURCES_LIST
                ${EXEC_SOURCES} ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${CUSTOMOPS_HELPERS_SOURCES}
                ${HELPERS_SOURCES} ${LOOPS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES_CUDA} ${VALIDATION_SOURCES}
        )
        if(HAVE_CUDNN)
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ./include/ops/declarable/platform/cudnn/*.cu)
            list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_CUDNN_SOURCES})
        endif()
    else()
        # CPU-specific sources
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
                ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${HELPERS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES}
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

    list(APPEND ALL_SOURCES_LIST ${CUSTOMOPS_GENERIC_SOURCES})
    list(REMOVE_DUPLICATES ALL_SOURCES_LIST)
    set(${out_source_list} ${ALL_SOURCES_LIST} PARENT_SCOPE)
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
# Creates the object and shared libraries, and calls the correct linking function.
function(create_and_link_library)
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    set(MAIN_LIB_NAME "${SD_LIBRARY_NAME}")
    # CRITICAL FIX: Add comprehensive include directories for proper header resolution
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
            ${CMAKE_BINARY_DIR}/compilation_units
            ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include
    )

    # Also set include directories for the main library
    target_include_directories(${MAIN_LIB_NAME} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    # A target guard prevents these from ever running twice.
    if(NOT TARGET ${OBJECT_LIB_NAME})
        message(STATUS "Creating object library: ${OBJECT_LIB_NAME}")
        add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})

        # CRITICAL FIX: Add include directories for proper header resolution
        target_include_directories(${OBJECT_LIB_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        # Set preprocessor definitions based on Options.cmake settings
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
    endif()

    if(NOT TARGET ${MAIN_LIB_NAME})
        message(STATUS "Creating shared library: ${MAIN_LIB_NAME}")
        add_library(${MAIN_LIB_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
        set_target_properties(${MAIN_LIB_NAME} PROPERTIES OUTPUT_NAME ${MAIN_LIB_NAME})
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
# PROCEDURAL BUILD EXECUTION
# =============================================================================

print_status_colored("INFO" "=== ORCHESTRATING LIBND4J BUILD (PROCEDURAL) ===")
include(BasicSetup)

# --- Phase 1: Initialize Type System ---
print_status_colored("INFO" "=== INITIALIZING TYPE SYSTEM ===")
if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
    endif()
endif()
validate_and_process_types_failfast()
build_indexed_type_lists()
if(SD_ENABLE_SEMANTIC_FILTERING AND COMMAND setup_enhanced_semantic_validation)
    setup_enhanced_semantic_validation()
endif()
initialize_dynamic_combinations()
message(STATUS "Type system initialization complete.")


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


# --- Phase 4: Process Templates & Generate Sources ---
print_status_colored("INFO" "=== PROCESSING TEMPLATES AND GENERATING SOURCES ===")
set(CUSTOMOPS_GENERIC_SOURCES "")

# CRITICAL FIX: Ensure combinations are initialized before template processing
if(NOT DEFINED COMBINATIONS_2 OR NOT DEFINED COMBINATIONS_3)
    message(STATUS "Re-initializing dynamic combinations for template processing")
    initialize_dynamic_combinations()

    # Debug: Check if combinations were actually created
    if(DEFINED COMBINATIONS_2 AND DEFINED COMBINATIONS_3)
        list(LENGTH COMBINATIONS_2 combo2_count)
        list(LENGTH COMBINATIONS_3 combo3_count)
        message(STATUS "✅ Combinations initialized: 2-type=${combo2_count}, 3-type=${combo3_count}")
    else()
        message(FATAL_ERROR "❌ Failed to initialize type combinations!")
    endif()
endif()

if(SD_CUDA)
     process_cuda_templates()
else()
    process_cpu_templates()
endif()
if(SD_ENABLE_SEMANTIC_FILTERING AND COMMAND process_pairwise_templates_semantic)
    process_pairwise_templates_semantic() 
endif()
message(STATUS "Template processing complete.")


# --- Phase 5 & 6: Create and Link Final Library ---
print_status_colored("INFO" "=== CREATING AND LINKING FINAL LIBRARY ===")
collect_all_sources(ALL_SOURCES)
setup_build_configuration()

create_and_link_library()
message(STATUS "Final library target created and linked.")

print_status_colored("SUCCESS" "=== BUILD ORCHESTRATION COMPLETE ===")
