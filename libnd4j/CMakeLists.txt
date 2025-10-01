cmake_minimum_required(VERSION 3.15)

if(WIN32 AND SD_CUDA)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    set(CMAKE_VERBOSE_MAKEFILE OFF)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Disable response files and dependencies
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_DEPFILE_FORMAT "" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_DEPENDS_USE_COMPILER OFF CACHE BOOL "" FORCE)

    set(CMAKE_CUDA_COMPILE_OBJECT
            "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE> --compiler-options \"/bigobj /EHsc /MD /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH\""
            CACHE STRING "" FORCE)
endif()

if(SD_CUDA)
    project(libnd4j LANGUAGES CXX CUDA)
else()
    project(libnd4j LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# Enable ccache if available for faster builds
include(cmake/Ccache.cmake)
setup_ccache()


if(NOT SD_CUDA)
    if(NOT SD_CPU)
        set(SD_CUDA FALSE)
        set(SD_CPU TRUE)
    endif()
endif()

if(SD_CUDA)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CUDA)
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
    add_compile_definitions(DEFAULT_ENGINE=samediff::ENGINE_CPU)
endif()

if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
    endif()
endif()

include(PrintingUtilities)
include(Options)
include(PlatformDetection)
include(TypeSystem)
include(TypeCombinationEngine)
include(TypeValidation)
include(TypeProfiles)
include(SemanticTypeFiltering)
include(SelectiveRenderingIntegration)
include(Dependencies)
include(CompilerOptimizations)
include(PlatformOptimizations)

LIBND4J_SETUP_TYPE_VALIDATION()

message(STATUS "=== TYPE SETUP VERIFICATION ===")
if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST)
    message(STATUS "SD_TYPES_LIST: ${SD_TYPES_LIST}")
    list(LENGTH SD_TYPES_LIST type_count)
    message(STATUS "Type count: ${type_count}")
else()
    message(STATUS "SD_TYPES_LIST: Not defined (ALL TYPES mode)")
endif()

if(DEFINED SD_TYPES_LIST_COUNT)
    message(STATUS "SD_TYPES_LIST_COUNT: ${SD_TYPES_LIST_COUNT}")
endif()

get_directory_property(COMPILE_DEFS COMPILE_DEFINITIONS)
message(STATUS "Some compile definitions: ${COMPILE_DEFS}")
message(STATUS "==============================")

include(cmake/MainBuildFlow.cmake)

if(SD_CUDA)
    include(cmake/CudaConfiguration.cmake)
    setup_cuda_build()
else()
    setup_cpu_environment()
endif()

if(WIN32 AND SD_CUDA)
    function(apply_cuda_target_fixes)
        get_property(all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target ${all_targets})
            if(TARGET ${target})
                get_target_property(target_type ${target} TYPE)
                if(target_type STREQUAL "OBJECT_LIBRARY" OR target_type STREQUAL "SHARED_LIBRARY")
                    get_target_property(compile_options ${target} COMPILE_OPTIONS)
                    if(compile_options)
                        list(FILTER compile_options EXCLUDE REGEX "/FS")
                        list(FILTER compile_options EXCLUDE REGEX "/Fd")
                        set_target_properties(${target} PROPERTIES COMPILE_OPTIONS "${compile_options}")
                    endif()
                    set_property(TARGET ${target} PROPERTY CUDA_FLAGS "")
                    set_target_properties(${target} PROPERTIES
                            CUDA_RESOLVE_DEVICE_SYMBOLS ON
                            CUDA_SEPARABLE_COMPILATION ON
                            MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
                    )
                endif()
            endif()
        endforeach()
    endfunction()
    apply_cuda_target_fixes()
endif()

if(NOT TARGET analyze_types)
    add_custom_target(analyze_types
            COMMAND ${CMAKE_COMMAND} -E echo "Analyzing type usage..."
            COMMENT "Analyzing type usage patterns in codebase"
            VERBATIM
    )
endif()

add_custom_target(verify_types
        COMMAND ${CMAKE_COMMAND} -E echo "Verifying type configuration..."
        COMMAND ${CMAKE_COMMAND} -E echo "Check build logs above for type setup verification"
        COMMENT "Verify that type definitions are properly set"
        VERBATIM
)

if(BUILD_PPSTEP)
include(cmake/Ppstep.cmake)
endif()