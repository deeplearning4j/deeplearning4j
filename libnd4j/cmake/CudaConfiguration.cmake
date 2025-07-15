################################################################################
# CUDA Configuration Functions
# Enhanced functions for CUDA-specific build configuration with clean toolchain paths
################################################################################

# Enhanced CUDA toolkit detection with proper include path setup
function(setup_cuda_toolkit_paths)
    message(STATUS "🔍 Setting up CUDA toolkit paths...")

    # Find CUDA toolkit first
    find_package(CUDAToolkit REQUIRED)

    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDA toolkit not found. Please install CUDA toolkit or set CUDA_PATH environment variable.")
    endif()

    # Get CUDA include directories using modern CMake approach
    get_target_property(CUDA_INCLUDE_DIRS CUDA::toolkit INTERFACE_INCLUDE_DIRECTORIES)

    # Fallback to CUDAToolkit variables if target property fails
    if(NOT CUDA_INCLUDE_DIRS)
        set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    endif()

    # Final fallback to environment-based search
    if(NOT CUDA_INCLUDE_DIRS)
        set(CUDA_SEARCH_PATHS
                $ENV{CUDA_PATH}/include
                $ENV{CUDA_HOME}/include
                $ENV{CUDA_ROOT}/include
                ${CUDAToolkit_ROOT}/include
        )

        if(WIN32)
            list(APPEND CUDA_SEARCH_PATHS
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include"
            )
        else()
            list(APPEND CUDA_SEARCH_PATHS
                    /usr/local/cuda/include
                    /opt/cuda/include
            )
        endif()

        foreach(search_path ${CUDA_SEARCH_PATHS})
            if(EXISTS "${search_path}/cuda.h")
                get_filename_component(CUDA_INCLUDE_DIRS "${search_path}" ABSOLUTE)
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
    if(IS_LIST CUDA_INCLUDE_DIRS)
        foreach(include_dir ${CUDA_INCLUDE_DIRS})
            if(EXISTS "${include_dir}/cuda.h")
                set(CUDA_H_FOUND TRUE)
                message(STATUS "✅ Found cuda.h in: ${include_dir}")
                break()
            endif()
        endforeach()
    else()
        if(EXISTS "${CUDA_INCLUDE_DIRS}/cuda.h")
            set(CUDA_H_FOUND TRUE)
            message(STATUS "✅ Found cuda.h in: ${CUDA_INCLUDE_DIRS}")
        endif()
    endif()

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

# Simplified cuDNN detection with clean paths
function(setup_modern_cudnn)
    set(HAVE_CUDNN FALSE PARENT_SCOPE)

    if(NOT (HELPERS_cudnn AND SD_CUDA))
        message(STATUS "🔍 cuDNN: Skipped (HELPERS_cudnn=${HELPERS_cudnn}, SD_CUDA=${SD_CUDA})")
        return()
    endif()

    message(STATUS "🔍 Searching for cuDNN...")

    # Find the CUDA toolkit first to get the proper paths
    find_package(CUDAToolkit REQUIRED)

    # Clean search paths prioritizing environment variables
    set(CUDNN_SEARCH_PATHS)

    # Environment variables first
    if(DEFINED ENV{CUDNN_ROOT_DIR} AND EXISTS "$ENV{CUDNN_ROOT_DIR}")
        list(APPEND CUDNN_SEARCH_PATHS "$ENV{CUDNN_ROOT_DIR}")
    endif()
    if(DEFINED ENV{CUDNN_ROOT} AND EXISTS "$ENV{CUDNN_ROOT}")
        list(APPEND CUDNN_SEARCH_PATHS "$ENV{CUDNN_ROOT}")
    endif()
    if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}")
        list(APPEND CUDNN_SEARCH_PATHS "$ENV{CUDA_PATH}")
    endif()

    # CMake variables
    if(DEFINED CUDNN_ROOT_DIR AND EXISTS "${CUDNN_ROOT_DIR}")
        list(APPEND CUDNN_SEARCH_PATHS "${CUDNN_ROOT_DIR}")
    endif()
    if(CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}")
        list(APPEND CUDNN_SEARCH_PATHS "${CUDAToolkit_ROOT}")
    endif()

    # Platform-specific standard paths
    if(WIN32)
        list(APPEND CUDNN_SEARCH_PATHS
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
        )
    else()
        list(APPEND CUDNN_SEARCH_PATHS
                /usr/local/cuda
                /usr/include
                /usr/local/include
                /opt/cuda
        )
    endif()

    message(STATUS "🔍 Searching for cuDNN headers...")

    # Search for cuDNN headers
    find_path(CUDNN_INCLUDE_DIR
            NAMES cudnn.h
            HINTS ${CUDNN_SEARCH_PATHS}
            PATH_SUFFIXES include targets/x86_64-linux/include
            NO_DEFAULT_PATH
    )

    # Fallback to system paths if not found
    if(NOT CUDNN_INCLUDE_DIR)
        find_path(CUDNN_INCLUDE_DIR
                NAMES cudnn.h
                PATHS /usr/include /usr/local/include
        )
    endif()

    message(STATUS "🔍 Searching for cuDNN libraries...")

    # Search for cuDNN libraries
    find_library(CUDNN_LIBRARY
            NAMES cudnn libcudnn
            HINTS ${CUDNN_SEARCH_PATHS}
            PATH_SUFFIXES lib64 lib lib/x64 targets/x86_64-linux/lib
            NO_DEFAULT_PATH
    )

    # Fallback to system paths if not found
    if(NOT CUDNN_LIBRARY)
        find_library(CUDNN_LIBRARY
                NAMES cudnn libcudnn
                PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
        )
    endif()

    message(STATUS "🔍 cuDNN search results:")
    message(STATUS "   CUDNN_INCLUDE_DIR: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "   CUDNN_LIBRARY: ${CUDNN_LIBRARY}")

    # Check if we found both header and library
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        message(STATUS "✅ cuDNN found!")

        # Extract version information from cudnn.h
        if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h")
            file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_HEADER_CONTENTS)

            string(REGEX MATCH "#define CUDNN_MAJOR[ \t]+([0-9]+)" CUDNN_VERSION_MAJOR_MATCH "${CUDNN_HEADER_CONTENTS}")
            if(CUDNN_VERSION_MAJOR_MATCH)
                string(REGEX REPLACE "#define CUDNN_MAJOR[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR_MATCH}")
                string(REGEX MATCH "#define CUDNN_MINOR[ \t]+([0-9]+)" CUDNN_VERSION_MINOR_MATCH "${CUDNN_HEADER_CONTENTS}")
                string(REGEX MATCH "#define CUDNN_PATCHLEVEL[ \t]+([0-9]+)" CUDNN_VERSION_PATCH_MATCH "${CUDNN_HEADER_CONTENTS}")
                string(REGEX REPLACE "#define CUDNN_MINOR[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR_MATCH}")
                string(REGEX REPLACE "#define CUDNN_PATCHLEVEL[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH_MATCH}")
                set(CUDNN_VERSION_STRING "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
            else()
                set(CUDNN_VERSION_STRING "Unknown")
            endif()
        else()
            set(CUDNN_VERSION_STRING "Unknown")
        endif()

        # Create imported target
        if(NOT TARGET CUDNN::cudnn)
            add_library(CUDNN::cudnn UNKNOWN IMPORTED)
            set_target_properties(CUDNN::cudnn PROPERTIES
                    IMPORTED_LOCATION "${CUDNN_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
            )
        endif()

        message(STATUS "✅ cuDNN configuration:")
        message(STATUS "   Include: ${CUDNN_INCLUDE_DIR}")
        message(STATUS "   Library: ${CUDNN_LIBRARY}")
        message(STATUS "   Version: ${CUDNN_VERSION_STRING}")

        # Set variables for parent scope
        set(HAVE_CUDNN TRUE PARENT_SCOPE)
        set(CUDNN_FOUND TRUE PARENT_SCOPE)
        set(CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE_DIR}" PARENT_SCOPE)
        set(CUDNN_LIBRARIES "${CUDNN_LIBRARY}" PARENT_SCOPE)
        set(CUDNN_LIBRARY "${CUDNN_LIBRARY}" PARENT_SCOPE)
        set(CUDNN_VERSION_STRING "${CUDNN_VERSION_STRING}" PARENT_SCOPE)

        return()
    endif()

    # Check if cuDNN is embedded in CUDA installation
    if(CUDAToolkit_FOUND AND CUDA_INCLUDE_DIRS)
        set(cuda_dirs_to_check)
        if(IS_LIST CUDA_INCLUDE_DIRS)
            set(cuda_dirs_to_check ${CUDA_INCLUDE_DIRS})
        else()
            set(cuda_dirs_to_check ${CUDA_INCLUDE_DIRS})
        endif()

        foreach(cuda_include_dir ${cuda_dirs_to_check})
            if(EXISTS "${cuda_include_dir}/cudnn.h")
                message(STATUS "✅ Found cuDNN embedded in CUDA installation")
                set(HAVE_CUDNN TRUE PARENT_SCOPE)
                set(CUDNN_INCLUDE_DIR "${cuda_include_dir}" PARENT_SCOPE)
                set(CUDNN_VERSION_STRING "Embedded" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endif()

    message(STATUS "❌ cuDNN not found.")
    message(STATUS "💡 To fix this issue:")
    message(STATUS "   1. Install cuDNN development libraries")
    message(STATUS "   2. Set CUDNN_ROOT_DIR to your cuDNN installation")
    message(STATUS "   3. Ensure cuDNN headers are in CUDA_PATH/include")
    message(STATUS "   4. Or disable cuDNN with -DHELPERS_cudnn=OFF")
endfunction()

# Clean Windows CUDA host compiler detection
function(setup_windows_cuda_compiler)
    if(NOT WIN32 OR NOT MSVC)
        return()
    endif()

    message(STATUS "🔧 Setting up Windows CUDA host compiler...")

    # First try CMAKE_CXX_COMPILER if it's already set and valid
    if(CMAKE_CXX_COMPILER AND EXISTS "${CMAKE_CXX_COMPILER}")
        get_filename_component(compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        if(compiler_dir MATCHES "x64" OR compiler_dir MATCHES "amd64")
            set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" PARENT_SCOPE)
            message(STATUS "✅ Using existing x64 compiler: ${CMAKE_CXX_COMPILER}")
            return()
        endif()
    endif()

    # Look for x64 cl.exe in PATH
    find_program(CL_EXECUTABLE cl.exe)
    if(CL_EXECUTABLE)
        get_filename_component(cl_dir "${CL_EXECUTABLE}" DIRECTORY)
        if(cl_dir MATCHES "x64" OR cl_dir MATCHES "amd64")
            set(CMAKE_CUDA_HOST_COMPILER "${CL_EXECUTABLE}" PARENT_SCOPE)
            message(STATUS "✅ Found x64 cl.exe in PATH: ${CL_EXECUTABLE}")
            return()
        else()
            # Try to find x64 version relative to this cl.exe
            get_filename_component(cl_parent "${cl_dir}" DIRECTORY)
            set(x64_candidates
                    "${cl_parent}/x64/cl.exe"
                    "${cl_parent}/../x64/cl.exe"
                    "${cl_parent}/Hostx64/x64/cl.exe"
            )
            foreach(candidate ${x64_candidates})
                if(EXISTS "${candidate}")
                    set(CMAKE_CUDA_HOST_COMPILER "${candidate}" PARENT_SCOPE)
                    message(STATUS "✅ Found x64 cl.exe relative to PATH: ${candidate}")
                    return()
                endif()
            endforeach()
        endif()
    endif()

    # Look in standard VS installation paths
    set(VS_SEARCH_PATHS
            "C:/Program Files/Microsoft Visual Studio/2022/BuildTools"
            "C:/Program Files/Microsoft Visual Studio/2022/Community"
            "C:/Program Files/Microsoft Visual Studio/2022/Professional"
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise"
    )

    foreach(vs_path ${VS_SEARCH_PATHS})
        if(EXISTS "${vs_path}")
            file(GLOB msvc_versions "${vs_path}/VC/Tools/MSVC/*")
            if(msvc_versions)
                list(SORT msvc_versions)
                list(REVERSE msvc_versions)
                list(GET msvc_versions 0 latest_msvc)

                set(candidate_compiler "${latest_msvc}/bin/Hostx64/x64/cl.exe")
                if(EXISTS "${candidate_compiler}")
                    set(CMAKE_CUDA_HOST_COMPILER "${candidate_compiler}" PARENT_SCOPE)
                    message(STATUS "✅ Found x64 compiler in VS installation: ${candidate_compiler}")
                    return()
                endif()
            endif()
        endif()
    endforeach()

    message(WARNING "❌ Could not find x64 CUDA host compiler")
    message(STATUS "💡 Suggestions:")
    message(STATUS "   1. Use x64 Native Tools Command Prompt")
    message(STATUS "   2. Set manually: -DCMAKE_CUDA_HOST_COMPILER=\"path/to/x64/cl.exe\"")
    message(STATUS "   3. Ensure x64 cl.exe is in your PATH")
endfunction()

# Clean CUDA architecture configuration
function(configure_cuda_architectures)
    if(NOT SD_CUDA)
        return()
    endif()

    message(STATUS "🔧 Configuring CUDA architectures...")

    if(DEFINED COMPUTE)
        string(TOLOWER "${COMPUTE}" compute_lower)
        if(compute_lower STREQUAL "all")
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" PARENT_SCOPE)
            message(STATUS "   Using all architectures: 75;80;86;89")
        elseif(compute_lower STREQUAL "auto")
            set(CMAKE_CUDA_ARCHITECTURES "86" PARENT_SCOPE)
            message(STATUS "   Using auto-detected architecture: 86")
        else()
            # Parse custom compute capabilities
            string(REPLACE "," ";" arch_list "${COMPUTE}")
            set(parsed_archs "")
            foreach(arch ${arch_list})
                string(REPLACE "." "" arch_clean "${arch}")
                if(arch_clean MATCHES "^[0-9][0-9]$")
                    list(APPEND parsed_archs "${arch_clean}")
                endif()
            endforeach()
            if(parsed_archs)
                set(CMAKE_CUDA_ARCHITECTURES "${parsed_archs}" PARENT_SCOPE)
                message(STATUS "   Using custom architectures: ${parsed_archs}")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "86" PARENT_SCOPE)
                message(STATUS "   Using default architecture: 86")
            endif()
        endif()
    else()
        set(CMAKE_CUDA_ARCHITECTURES "86" PARENT_SCOPE)
        message(STATUS "   Using default architecture: 86")
    endif()
endfunction()

# Clean CUDA flags configuration
function(configure_cuda_flags)
    message(STATUS "🔧 Configuring CUDA compilation flags...")

    set(CUDA_FLAGS "")

    # Basic CUDA flags
    list(APPEND CUDA_FLAGS
            "-w"
            "--cudart=shared"
            "--expt-extended-lambda"
            "-Xfatbin" "-compress-all"
    )

    # Platform-specific flags
    if(WIN32 AND MSVC)
        setup_windows_cuda_compiler()

        list(APPEND CUDA_FLAGS
                "-Xcompiler=/std:c++17"
                "-Xcompiler=/bigobj"
                "-Xcompiler=/EHsc"
                "-Xcompiler=/Zc:preprocessor-"
        )

        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND CUDA_FLAGS "-g" "-G")
        endif()
    else()
        list(APPEND CUDA_FLAGS "-Xcompiler=-fPIC")

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            list(APPEND CUDA_FLAGS "-Xcompiler=-fpermissive")
            if(NOT SD_GCC_FUNCTRACE)
                list(APPEND CUDA_FLAGS "-Xcompiler=-fno-implicit-templates")
            endif()
        endif()

        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND CUDA_FLAGS "--device-debug" "-lineinfo" "-G")
        endif()
    endif()

    # Add CUDA version definition
    if(CMAKE_CUDA_COMPILER_VERSION)
        string(REGEX MATCH "^([0-9]+)" cuda_major "${CMAKE_CUDA_COMPILER_VERSION}")
        list(APPEND CUDA_FLAGS "-DCUDA_VERSION_MAJOR=${cuda_major}")
    endif()

    # Convert list to string
    string(REPLACE ";" " " cuda_flags_string "${CUDA_FLAGS}")

    # Set CMake variables
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON PARENT_SCOPE)
    set(CMAKE_CUDA_FLAGS "${cuda_flags_string}" PARENT_SCOPE)

    message(STATUS "✅ CUDA flags configured: ${cuda_flags_string}")
endfunction()

# Main CUDA configuration function
function(configure_cuda_linking main_target_name)
    message(STATUS "🔧 Configuring CUDA linking for target: ${main_target_name}")

    # Setup CUDA toolkit paths
    setup_cuda_toolkit_paths()

    # Find the CUDAToolkit to define targets
    find_package(CUDAToolkit REQUIRED)

    # Setup cuDNN
    setup_modern_cudnn()

    # Add CUDA include directories to target
    if(CUDA_INCLUDE_DIRS)
        target_include_directories(${main_target_name} PUBLIC ${CUDA_INCLUDE_DIRS})
        message(STATUS "✅ Added CUDA include directories: ${CUDA_INCLUDE_DIRS}")
    endif()

    # Link with CUDA toolkit
    target_link_libraries(${main_target_name} PUBLIC CUDA::toolkit)

    # Link with cuDNN if available
    if(HAVE_CUDNN AND TARGET CUDNN::cudnn)
        target_link_libraries(${main_target_name} PUBLIC CUDNN::cudnn)
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
        message(STATUS "✅ Linked with cuDNN target")
    elseif(HAVE_CUDNN AND CUDNN_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${CUDNN_LIBRARIES})
        target_include_directories(${main_target_name} PUBLIC ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
        message(STATUS "✅ Linked with cuDNN libraries")
    elseif(HAVE_CUDNN AND CUDNN_INCLUDE_DIR)
        target_include_directories(${main_target_name} PUBLIC ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
        message(STATUS "✅ Linked with cuDNN headers only")
    else()
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=0)
        message(STATUS "ℹ️  Building without cuDNN support")
    endif()

    # Link with flatbuffers if available
    if(TARGET flatbuffers_interface)
        target_link_libraries(${main_target_name} PUBLIC flatbuffers_interface)
    endif()

    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

# Main CUDA setup function
function(setup_cuda_build)
    message(STATUS "=== CUDA BUILD CONFIGURATION ===")

    if(NOT SD_CUDA)
        message(STATUS "CUDA disabled, skipping CUDA setup")
        return()
    endif()

    # Configure CUDA architectures early
    configure_cuda_architectures()

    # Enable CUDA language
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        find_program(CMAKE_CUDA_COMPILER nvcc)
        if(NOT CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR "CUDA compiler not found. Please install CUDA toolkit.")
        endif()
    endif()

    enable_language(CUDA)
    message(STATUS "✅ CUDA language enabled with compiler: ${CMAKE_CUDA_COMPILER}")

    # Setup paths and includes
    setup_cuda_toolkit_paths()

    # Configure CUDA flags
    configure_cuda_flags()

    # Set build definitions
    add_compile_definitions(SD_CUDA=true)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA" PARENT_SCOPE)

    message(STATUS "=== CUDA BUILD CONFIGURATION COMPLETE ===")
    message(STATUS "   Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "   Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "   Flags: ${CMAKE_CUDA_FLAGS}")
    message(STATUS "   Include dirs: ${CUDA_INCLUDE_DIRS}")
endfunction()

# Legacy compatibility functions
function(setup_cudnn)
    setup_modern_cudnn()
    # Set legacy variables for backward compatibility
    set(HAVE_CUDNN ${HAVE_CUDNN} PARENT_SCOPE)
    if(HAVE_CUDNN)
        set(CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR} PARENT_SCOPE)
        set(CUDNN ${CUDNN_LIBRARIES} PARENT_SCOPE)
    endif()
endfunction()

function(ensure_cuda_paths_available)
    if(SD_CUDA)
        setup_cuda_toolkit_paths()
        set(CUDA_INCLUDE_DIRS "${CUDA_INCLUDE_DIRS}" PARENT_SCOPE)
        set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" PARENT_SCOPE)
    endif()
endfunction()