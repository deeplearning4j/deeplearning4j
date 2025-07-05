
################################################################################
# CUDA Configuration Functions - FIXED VERSION
# Functions for CUDA-specific build configuration and optimization
# FIXED: Proper /FS flag handling for CUDA compilation
################################################################################

if(WIN32 AND MSVC)
    # More comprehensive /FS flag removal - also handle CUDA-specific flags
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")

    # Also disable for CUDA if it exists
    if(DEFINED CMAKE_CUDA_FLAGS)
        string(REPLACE "/FS" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    endif()

    # CRITICAL FIX: Disable automatic /FS injection by CMake for CUDA
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded "/MT")
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "/MD")
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug "/MTd")
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "/MDd")

    # Prevent CMake from adding /FS automatically for CUDA compilation
    set(CMAKE_CUDA_COMPILE_OBJECT_DEPENDS "")

    message(STATUS "Disabled /FS flag to prevent nvcc compilation errors")
endif()

# Enhanced CUDA toolkit detection with proper include path setup
function(setup_cuda_toolkit_paths)
    message(STATUS "🔍 Setting up CUDA toolkit paths...")

    # Find CUDA toolkit first
    find_package(CUDAToolkit REQUIRED)

    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDA toolkit not found. Please install CUDA toolkit or set CUDA_PATH environment variable.")
    endif()

    # Get CUDA include directories
    get_target_property(CUDA_INCLUDE_DIRS CUDA::toolkit INTERFACE_INCLUDE_DIRECTORIES)

    # If the above doesn't work, try alternative methods
    if(NOT CUDA_INCLUDE_DIRS)
        set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    endif()

    # Still not found? Try environment variables and common paths
    if(NOT CUDA_INCLUDE_DIRS)
        set(CUDA_SEARCH_PATHS
                $ENV{CUDA_PATH}
                $ENV{CUDA_HOME}
                $ENV{CUDA_ROOT}
                ${CUDAToolkit_ROOT}
        )

        if(WIN32)
            list(APPEND CUDA_SEARCH_PATHS
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
                    "C:/tools/cuda"
            )
        else()
            list(APPEND CUDA_SEARCH_PATHS
                    /usr/local/cuda
                    /opt/cuda
                    /usr/cuda
            )
        endif()

        foreach(search_path ${CUDA_SEARCH_PATHS})
            if(EXISTS "${search_path}/include/cuda.h")
                set(CUDA_INCLUDE_DIRS "${search_path}/include")
                message(STATUS "✅ Found CUDA include directory: ${CUDA_INCLUDE_DIRS}")
                break()
            endif()
        endforeach()
    endif()

    # Verify we found CUDA headers
    if(NOT CUDA_INCLUDE_DIRS)
        message(FATAL_ERROR "❌ CUDA include directories not found. Please ensure CUDA toolkit is properly installed.")
    endif()

    # Verify cuda.h exists
    set(CUDA_H_FOUND FALSE)
    foreach(include_dir ${CUDA_INCLUDE_DIRS})
        if(EXISTS "${include_dir}/cuda.h")
            set(CUDA_H_FOUND TRUE)
            message(STATUS "✅ Found cuda.h in: ${include_dir}")
            break()
        endif()
    endforeach()

    if(NOT CUDA_H_FOUND)
        message(FATAL_ERROR "❌ cuda.h not found in CUDA include directories: ${CUDA_INCLUDE_DIRS}")
    endif()

    # Set variables for parent scope
    set(CUDA_INCLUDE_DIRS "${CUDA_INCLUDE_DIRS}" PARENT_SCOPE)
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}" PARENT_SCOPE)

    message(STATUS "✅ CUDA toolkit paths configured:")
    message(STATUS "   Root: ${CUDAToolkit_ROOT}")
    message(STATUS "   Include: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "   Version: ${CUDAToolkit_VERSION}")
endfunction()

function(configure_windows_cuda_build)
    if(NOT WIN32)
        return()
    endif()

    message(STATUS "Configuring Windows CUDA build with proper response files...")

    # Enable response file support for long command lines
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS ON PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES ON PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES ON PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LINK_OBJECTS ON PARENT_SCOPE)
    set(CMAKE_CUDA_RESPONSE_FILE_LINK_FLAG "@" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_USE_RESPONSE_FILE ON PARENT_SCOPE)

    # CRITICAL FIX: Completely override the CUDA compilation command to exclude /FS
    set(CMAKE_CUDA_COMPILE_OBJECT
            "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -x cu -c <SOURCE> -o <OBJECT>"
            PARENT_SCOPE)

    # Alternative: Set custom CUDA host compiler flags that exclude /FS
    set(CMAKE_CUDA_HOST_COMPILER_OPTIONS "" PARENT_SCOPE)

    # Clean runtime library settings without /FS
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded "/MT" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "/MD" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug "/MTd" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "/MDd" PARENT_SCOPE)

    message(STATUS "Windows CUDA: Configured with clean compilation command (no /FS)")
endfunction()

function(build_cuda_compiler_flags CUDA_ARCH_FLAGS)
    set(LOCAL_CUDA_FLAGS "")

    if(WIN32 AND MSVC)
        message(STATUS "Configuring CUDA for Windows MSVC...")
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} PARENT_SCOPE)
        set(LOCAL_CUDA_FLAGS "-maxrregcount=128")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/nologo")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/EHsc")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/std:c++17")

        # CRITICAL: Explicitly exclude /FS from host compiler flags
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/bigobj")

        if(MSVC_RT_LIB STREQUAL "MultiThreadedDLL")
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/MD")
        else()
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/MT")
        endif()

        # Ensure no /FS gets through
        string(REPLACE "/FS" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")

    else()
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -maxrregcount=128")

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(SD_GCC_FUNCTRACE STREQUAL "ON")
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-fPIC --device-debug -lineinfo -G")
            else()
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-fPIC")
            endif()
        endif()
    endif()

    if("${SD_PTXAS}" STREQUAL "ON")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --ptxas-options=-v")
    endif()

    if(SD_KEEP_NVCC_OUTPUT)
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --keep")
    endif()

    if(DEFINED CUDA_ARCH_FLAGS)
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
    endif()
    set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all")

    if(CMAKE_CUDA_COMPILER_VERSION)
        string(REGEX MATCH "^([0-9]+)" CUDA_VERSION_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}")
    endif()

    # Final cleanup to ensure no /FS gets through
    string(REPLACE "/FS" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
    string(REGEX REPLACE "  +" " " LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
    string(STRIP "${LOCAL_CUDA_FLAGS}" LOCAL_CUDA_FLAGS)

    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON PARENT_SCOPE)
    set(CMAKE_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}" PARENT_SCOPE)

    message(STATUS "Final CMAKE_CUDA_FLAGS: ${LOCAL_CUDA_FLAGS}")
endfunction()

# Enhanced function to clean up CUDA compilation command
function(fix_cuda_compilation_command)
    if(WIN32 AND MSVC)
        # Get the current CUDA compile object command
        get_property(CUDA_COMPILE_OBJECT GLOBAL PROPERTY RULE_LAUNCH_COMPILE)

        # Override with a clean version that excludes problematic flags
        set(CMAKE_CUDA_COMPILE_OBJECT
                "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -x cu -c <SOURCE> -o <OBJECT>"
                PARENT_SCOPE)

        # Also set a custom compile command that filters out /FS
        set(CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE
                "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>"
                PARENT_SCOPE)

        message(STATUS "✅ Fixed CUDA compilation command to exclude /FS flag")
    endif()
endfunction()

# MAIN CUDA SETUP FUNCTION - UPDATED
function(setup_cuda_build)
    message(STATUS "=== CUDA BUILD CONFIGURATION ===")

    if(NOT DEFINED _CMAKE_CUDA_WHOLE_FLAG)
        message(STATUS "Setting _CMAKE_CUDA_WHOLE_FLAG (was missing)")
        if(WIN32)
            set(_CMAKE_CUDA_WHOLE_FLAG "/WHOLEARCHIVE:" CACHE INTERNAL "CUDA whole archive flag")
        else()
            set(_CMAKE_CUDA_WHOLE_FLAG "-Wl,--whole-archive" CACHE INTERNAL "CUDA whole archive flag")
        endif()
    endif()

    # Setup CUDA toolkit paths and include directories early
    setup_cuda_include_directories()

    if(NOT DEFINED COMPUTE)
        set(COMPUTE "auto")
    endif()
    configure_cuda_architecture_flags("${COMPUTE}")
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

    setup_cuda_language()

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found after enabling CUDA language")
    endif()

    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDAToolkit Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "Host CXX Compiler: ${CMAKE_CXX_COMPILER_ID}")

    # CRITICAL: Configure Windows CUDA build BEFORE setting flags
    configure_windows_cuda_build()

    # CRITICAL: Fix the compilation command
    fix_cuda_compilation_command()

    build_cuda_compiler_flags("${CUDA_ARCH_FLAGS}")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)

    # Also set the toolkit include directories for global access
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}" PARENT_SCOPE)

    debug_cuda_configuration()

    add_compile_definitions(SD_CUDA=true)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA" PARENT_SCOPE)

    message(STATUS "=== CUDA BUILD CONFIGURATION COMPLETE ===")
endfunction()

# Additional function to be called after project() to ensure flags are clean
function(finalize_cuda_configuration)
    if(WIN32 AND MSVC AND SD_CUDA)
        # Final cleanup of any /FS flags that might have been added
        get_property(COMPILE_RULES GLOBAL PROPERTY RULE_LAUNCH_COMPILE)

        # Clean up any remaining /FS flags in CUDA-related variables
        foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
            string(REPLACE "/FS" "" CMAKE_CUDA_FLAGS_${config} "${CMAKE_CUDA_FLAGS_${config}}")
            set(CMAKE_CUDA_FLAGS_${config} "${CMAKE_CUDA_FLAGS_${config}}" PARENT_SCOPE)
        endforeach()

        message(STATUS "✅ Finalized CUDA configuration - all /FS flags removed")
    endif()
endfunction()


# Enhanced function to ensure CUDA paths are available at configure time
function(ensure_cuda_paths_available)
    if(NOT SD_CUDA)
        return()
    endif()

    message(STATUS "🔍 Ensuring CUDA paths are available...")

    # Find CUDA early
    find_package(CUDAToolkit REQUIRED)

    # Set up paths immediately
    setup_cuda_toolkit_paths()

    # Export to parent scope for immediate use
    set(CUDA_INCLUDE_DIRS "${CUDA_INCLUDE_DIRS}" PARENT_SCOPE)
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}" PARENT_SCOPE)

    message(STATUS "✅ CUDA paths configured and available")
endfunction()

# Legacy function for backward compatibility - now calls modern version
function(setup_cudnn)
    setup_modern_cudnn()

    # Set legacy variables for backward compatibility
    set(HAVE_CUDNN ${HAVE_CUDNN} PARENT_SCOPE)
    if(HAVE_CUDNN)
        set(CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR} PARENT_SCOPE)
        set(CUDNN ${CUDNN_LIBRARIES} PARENT_SCOPE)
    endif()
endfunction()
