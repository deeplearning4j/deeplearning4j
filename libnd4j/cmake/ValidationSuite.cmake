# =============================================================================
# ValidationSuite.cmake - Comprehensive Build Validation for libnd4j
# =============================================================================

if(NOT COMMAND print_status_colored)
    include(${CMAKE_CURRENT_LIST_DIR}/PrintingUtilities.cmake)
endif()

# Master validation function
function(run_comprehensive_validation)
    print_status_colored("INFO" "=== RUNNING COMPREHENSIVE BUILD VALIDATION ===")

    validate_build_environment_comprehensive()
    validate_cmake_variables_comprehensive()
    validate_type_system_comprehensive()
    validate_template_system_comprehensive()
    validate_dependencies_comprehensive()
    validate_generated_files_comprehensive()
    validate_platform_specific_configuration()

    print_status_colored("SUCCESS" "All validation checks passed")
endfunction()

# Environment validation
function(validate_build_environment_comprehensive)
    print_status_colored("INFO" "Phase 1: Environment validation")

    set(validation_errors "")
    set(validation_warnings "")

    if(CMAKE_VERSION VERSION_LESS "3.15")
        list(APPEND validation_errors "CMake 3.15+ required, found ${CMAKE_VERSION}")
    endif()

    set(REQUIRED_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    foreach(dir ${REQUIRED_DIRS})
        if(NOT EXISTS "${dir}")
            list(APPEND validation_errors "Required directory not found: ${dir}")
        endif()
    endforeach()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9")
        list(APPEND validation_errors "GCC 4.9+ required, found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()

    list(LENGTH validation_errors error_count)
    if(error_count GREATER 0)
        foreach(error ${validation_errors})
            print_status_colored("ERROR" "  - ${error}")
        endforeach()
        message(FATAL_ERROR "Environment validation failed")
    endif()

    print_status_colored("SUCCESS" "Environment validation passed")
endfunction()

# CMake variables validation
function(validate_cmake_variables_comprehensive)
    print_status_colored("INFO" "Phase 2: CMake variables validation")

    set(validation_warnings "")

    if(SD_CUDA AND SD_CPU AND SD_CPU STREQUAL "ON")
        list(APPEND validation_warnings "Both SD_CUDA and SD_CPU enabled, SD_CUDA takes precedence")
    endif()

    if(DEFINED SD_TYPE_LIST AND NOT DEFINED SD_TYPES_LIST)
        list(APPEND validation_warnings "Found SD_TYPE_LIST, converting to SD_TYPES_LIST")
        set(SD_TYPES_LIST "${SD_TYPE_LIST}" PARENT_SCOPE)
    endif()

    if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST MATCHES ".*,.*")
        list(APPEND validation_warnings "Converting comma-separated to semicolon-separated types")
        string(REPLACE "," ";" SD_TYPES_LIST "${SD_TYPES_LIST}")
        set(SD_TYPES_LIST "${SD_TYPES_LIST}" PARENT_SCOPE)
    endif()

    list(LENGTH validation_warnings warning_count)
    if(warning_count GREATER 0)
        foreach(warning ${validation_warnings})
            print_status_colored("WARNING" "  - ${warning}")
        endforeach()
    endif()

    print_status_colored("SUCCESS" "CMake variables validation passed")
endfunction()

# Type system validation
function(validate_type_system_comprehensive)
    print_status_colored("INFO" "Phase 3: Type system validation")

    set(validation_errors "")

    if(NOT DEFINED SD_COMMON_TYPES_COUNT OR SD_COMMON_TYPES_COUNT EQUAL 0)
        list(APPEND validation_errors "Type system not initialized")
    endif()

    if(DEFINED COMBINATIONS_3 AND DEFINED COMBINATIONS_2)
        list(LENGTH COMBINATIONS_3 combo_3_count)
        list(LENGTH COMBINATIONS_2 combo_2_count)
        if(combo_3_count EQUAL 0 AND combo_2_count EQUAL 0)
            list(APPEND validation_errors "No type combinations generated")
        endif()
    endif()

    list(LENGTH validation_errors error_count)
    if(error_count GREATER 0)
        foreach(error ${validation_errors})
            print_status_colored("ERROR" "  - ${error}")
        endforeach()
        message(FATAL_ERROR "Type system validation failed")
    endif()

    print_status_colored("SUCCESS" "Type system validation passed")
endfunction()

# Template system validation
function(validate_template_system_comprehensive)
    print_status_colored("INFO" "Phase 4: Template system validation")

    set(validation_errors "")

    if(DEFINED CUSTOMOPS_GENERIC_SOURCES)
        list(LENGTH CUSTOMOPS_GENERIC_SOURCES generated_count)
        if(generated_count GREATER 0)
            set(check_count 0)
            foreach(generated_file ${CUSTOMOPS_GENERIC_SOURCES})
                if(check_count LESS 3 AND EXISTS "${generated_file}")
                    file(READ "${generated_file}" file_content)
                    if(file_content MATCHES "#cmakedefine|@[A-Za-z_]+@")
                        list(APPEND validation_errors "Unprocessed template: ${generated_file}")
                    endif()
                    math(EXPR check_count "${check_count} + 1")
                endif()
            endforeach()
        endif()
    endif()

    list(LENGTH validation_errors error_count)
    if(error_count GREATER 0)
        foreach(error ${validation_errors})
            print_status_colored("ERROR" "  - ${error}")
        endforeach()
        message(FATAL_ERROR "Template system validation failed")
    endif()

    print_status_colored("SUCCESS" "Template system validation passed")
endfunction()

# Dependency validation
function(validate_dependencies_comprehensive)
    print_status_colored("INFO" "Phase 5: Dependency validation")

    set(validation_warnings "")

    if(NOT TARGET flatbuffers_interface)
        list(APPEND validation_warnings "FlatBuffers interface target not found")
    endif()

    if(HELPERS_onednn AND (NOT DEFINED HAVE_ONEDNN OR NOT HAVE_ONEDNN))
        list(APPEND validation_warnings "OneDNN enabled but not configured")
    endif()

    if(HELPERS_cudnn AND SD_CUDA AND (NOT DEFINED HAVE_CUDNN OR NOT HAVE_CUDNN))
        list(APPEND validation_warnings "cuDNN enabled but not configured")
    endif()

    if(SD_CUDA AND NOT CMAKE_CUDA_COMPILER)
        list(APPEND validation_warnings "CUDA build but no CUDA compiler")
    endif()

    list(LENGTH validation_warnings warning_count)
    if(warning_count GREATER 0)
        foreach(warning ${validation_warnings})
            print_status_colored("WARNING" "  - ${warning}")
        endforeach()
    endif()

    print_status_colored("SUCCESS" "Dependency validation passed")
endfunction()

# Generated files validation
function(validate_generated_files_comprehensive)
    print_status_colored("INFO" "Phase 6: Generated files validation")

    set(validation_warnings "")

    set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
    if(NOT EXISTS "${INCLUDE_OPS_FILE}")
        list(APPEND validation_warnings "Generated include_ops.h not found")
    else()
        file(READ "${INCLUDE_OPS_FILE}" ops_content)
        if(NOT ops_content MATCHES "#define")
            list(APPEND validation_warnings "include_ops.h appears empty or malformed")
        endif()
    endif()

    list(LENGTH validation_warnings warning_count)
    if(warning_count GREATER 0)
        foreach(warning ${validation_warnings})
            print_status_colored("WARNING" "  - ${warning}")
        endforeach()
    endif()

    print_status_colored("SUCCESS" "Generated files validation passed")
endfunction()

# Platform-specific validation
function(validate_platform_specific_configuration)
    print_status_colored("INFO" "Phase 7: Platform-specific validation")

    set(validation_warnings "")

    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(NOT CMAKE_CXX_FLAGS MATCHES "-Wa,-mbig-obj")
            list(APPEND validation_warnings "Windows GCC build may need -Wa,-mbig-obj flag")
        endif()
    endif()

    if(SD_ARM_BUILD AND NOT DEFINED SD_ARCH)
        list(APPEND validation_warnings "ARM build without SD_ARCH specification")
    endif()

    if(ANDROID AND NOT ANDROID_ABI)
        list(APPEND validation_warnings "Android build without ABI specification")
    endif()

    list(LENGTH validation_warnings warning_count)
    if(warning_count GREATER 0)
        foreach(warning ${validation_warnings})
            print_status_colored("WARNING" "  - ${warning}")
        endforeach()
    endif()

    print_status_colored("SUCCESS" "Platform validation passed")
endfunction()

# Quick validation for development
function(run_quick_validation)
    print_status_colored("INFO" "=== RUNNING QUICK VALIDATION ===")

    if(CMAKE_VERSION VERSION_LESS "3.15")
        message(FATAL_ERROR "CMake 3.15+ required")
    endif()

    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
        message(FATAL_ERROR "Include directory not found")
    endif()

    if(SD_CUDA AND NOT CMAKE_CUDA_COMPILER)
        print_status_colored("WARNING" "CUDA build requested but compiler not found")
    endif()

    print_status_colored("SUCCESS" "Quick validation passed")
endfunction()

# Validation for CI/CD
function(run_ci_validation)
    print_status_colored("INFO" "=== RUNNING CI/CD VALIDATION ===")

    run_comprehensive_validation()

    # Additional CI-specific checks
    if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        print_status_colored("WARNING" "BUILD_TYPE not specified for CI build")
    endif()

    if(DEFINED ENV{CI} AND SD_GCC_FUNCTRACE STREQUAL "ON")
        print_status_colored("WARNING" "Function tracing enabled in CI environment")
    endif()

    print_status_colored("SUCCESS" "CI validation completed")
endfunction()