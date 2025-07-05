################################################################################
# CUDA Configuration Functions
# Functions for CUDA-specific build configuration and optimization
# UPDATED VERSION - Fixed CUDA path discovery and include directory setup
################################################################################

if(WIN32 AND MSVC)
    # Disable the /FS flag that's causing nvcc to fail
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/FS" "" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")

    # Also disable for CUDA if it exists
    if(DEFINED CMAKE_CUDA_FLAGS)
        string(REPLACE "/FS" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    endif()

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

# Modern cuDNN detection using updated FindCUDNN.cmake practices
function(setup_modern_cudnn)
    set(HAVE_CUDNN FALSE PARENT_SCOPE)

    if(NOT (HELPERS_cudnn AND SD_CUDA))
        message(STATUS "🔍 cuDNN: Skipped (HELPERS_cudnn=${HELPERS_cudnn}, SD_CUDA=${SD_CUDA})")
        return()
    endif()

    message(STATUS "🔍 Searching for cuDNN...")

    # Find the CUDA toolkit first to get the proper paths
    find_package(CUDAToolkit REQUIRED)

    # Enhanced search paths for CI environments and common installations
    set(CUDNN_SEARCH_PATHS
            # Environment variables
            $ENV{CUDNN_ROOT_DIR}
            $ENV{CUDNN_ROOT}
            $ENV{CUDA_PATH}
            $ENV{CUDA_HOME}

            # CMake variables
            ${CUDNN_ROOT_DIR}
            ${CUDAToolkit_ROOT}
    )

    # Add platform-specific paths
    if(WIN32)
        # Windows-specific paths
        list(APPEND CUDNN_SEARCH_PATHS
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.7"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.5"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.4"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.3"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.1"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.0"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA"
                "C:/tools/cuda" # Common CI path on Windows
        )
    else()
        # Linux/Unix-specific paths
        list(APPEND CUDNN_SEARCH_PATHS
                # CI-specific paths (GitHub Actions, etc.)
                /usr/local/cuda-12.6
                /usr/local/cuda-12.5
                /usr/local/cuda-12.4
                /usr/local/cuda-12.3
                /usr/local/cuda-12.2
                /usr/local/cuda-12.1
                /usr/local/cuda-12.0
                /usr/local/cuda-11.8
                /usr/local/cuda-11.7
                /usr/local/cuda-11.6
                /usr/local/cuda-11.5
                /usr/local/cuda-11.4
                /usr/local/cuda-11.3
                /usr/local/cuda-11.2
                /usr/local/cuda-11.1
                /usr/local/cuda-11.0
                /usr/local/cuda

                # Package manager paths
                /usr/include/cudnn
                /usr/local/include/cudnn
                /opt/cuda
                /opt/cudnn

                # System paths
                /usr
                /usr/local
        )
    endif()

    message(STATUS "🔍 Searching for cuDNN headers...")

    # Search for cuDNN headers with comprehensive path coverage
    find_path(CUDNN_INCLUDE_DIR
            NAMES cudnn.h
            HINTS ${CUDNN_SEARCH_PATHS}
            PATHS ${CUDNN_SEARCH_PATHS}
            PATH_SUFFIXES
            include
            targets/x86_64-linux/include
            targets/aarch64-linux/include
            include/cudnn
            cudnn/include
            x86_64-linux/include
            aarch64-linux/include
            NO_DEFAULT_PATH
    )

    # If not found, try system paths
    if(NOT CUDNN_INCLUDE_DIR)
        find_path(CUDNN_INCLUDE_DIR
                NAMES cudnn.h
                PATHS /usr/include /usr/local/include /opt/include
                PATH_SUFFIXES cudnn
        )
    endif()

    message(STATUS "🔍 Searching for cuDNN libraries...")

    # Search for cuDNN libraries
    find_library(CUDNN_LIBRARY
            NAMES cudnn libcudnn cudnn8 libcudnn8
            HINTS ${CUDNN_SEARCH_PATHS}
            PATHS ${CUDNN_SEARCH_PATHS}
            PATH_SUFFIXES
            lib64
            lib
            lib/x64
            targets/x86_64-linux/lib
            targets/aarch64-linux/lib
            lib64/cudnn
            lib/cudnn
            cudnn/lib64
            cudnn/lib
            x86_64-linux/lib
            aarch64-linux/lib
            NO_DEFAULT_PATH
    )

    # If not found, try system paths
    if(NOT CUDNN_LIBRARY)
        find_library(CUDNN_LIBRARY
                NAMES cudnn libcudnn cudnn8 libcudnn8
                PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /opt/lib64 /opt/lib
                PATH_SUFFIXES cudnn
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

            # Try different version detection methods
            string(REGEX MATCH "define CUDNN_MAJOR[ \t]+([0-9]+)" CUDNN_VERSION_MAJOR_MATCH "${CUDNN_HEADER_CONTENTS}")
            string(REGEX MATCH "define CUDNN_MINOR[ \t]+([0-9]+)" CUDNN_VERSION_MINOR_MATCH "${CUDNN_HEADER_CONTENTS}")
            string(REGEX MATCH "define CUDNN_PATCHLEVEL[ \t]+([0-9]+)" CUDNN_VERSION_PATCH_MATCH "${CUDNN_HEADER_CONTENTS}")

            if(CUDNN_VERSION_MAJOR_MATCH)
                string(REGEX REPLACE "define CUDNN_MAJOR[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR_MATCH}")
                string(REGEX REPLACE "define CUDNN_MINOR[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR_MATCH}")
                string(REGEX REPLACE "define CUDNN_PATCHLEVEL[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH_MATCH}")

                set(CUDNN_VERSION_STRING "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
            else()
                # Try alternative version detection
                string(REGEX MATCH "#define CUDNN_MAJOR ([0-9]+)" CUDNN_VERSION_MAJOR_MATCH2 "${CUDNN_HEADER_CONTENTS}")
                if(CUDNN_VERSION_MAJOR_MATCH2)
                    string(REGEX REPLACE "#define CUDNN_MAJOR ([0-9]+)" "\\1" CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR_MATCH2}")
                    string(REGEX MATCH "#define CUDNN_MINOR ([0-9]+)" CUDNN_VERSION_MINOR_MATCH2 "${CUDNN_HEADER_CONTENTS}")
                    string(REGEX MATCH "#define CUDNN_PATCHLEVEL ([0-9]+)" CUDNN_VERSION_PATCH_MATCH2 "${CUDNN_HEADER_CONTENTS}")
                    string(REGEX REPLACE "#define CUDNN_MINOR ([0-9]+)" "\\1" CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR_MATCH2}")
                    string(REGEX REPLACE "#define CUDNN_PATCHLEVEL ([0-9]+)" "\\1" CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH_MATCH2}")
                    set(CUDNN_VERSION_STRING "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
                else()
                    set(CUDNN_VERSION_STRING "Unknown")
                endif()
            endif()
        else()
            message(WARNING "⚠️  cuDNN header found but cannot read version")
            set(CUDNN_VERSION_STRING "Unknown")
        endif()

        # Create imported target if it doesn't exist
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

        # Set all the variables that might be needed
        set(HAVE_CUDNN TRUE PARENT_SCOPE)
        set(CUDNN_FOUND TRUE PARENT_SCOPE)
        set(CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE_DIR}" PARENT_SCOPE)
        set(CUDNN_LIBRARIES "${CUDNN_LIBRARY}" PARENT_SCOPE)
        set(CUDNN_LIBRARY "${CUDNN_LIBRARY}" PARENT_SCOPE)
        set(CUDNN_VERSION_STRING "${CUDNN_VERSION_STRING}" PARENT_SCOPE)

        return()
    endif()

    # Try package manager detection as fallback
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_CUDNN QUIET cudnn)
        if(PC_CUDNN_FOUND)
            message(STATUS "✅ Found cuDNN via pkg-config")
            set(HAVE_CUDNN TRUE PARENT_SCOPE)
            set(CUDNN_INCLUDE_DIR "${PC_CUDNN_INCLUDE_DIRS}" PARENT_SCOPE)
            set(CUDNN_LIBRARIES "${PC_CUDNN_LIBRARIES}" PARENT_SCOPE)
            set(CUDNN_VERSION_STRING "${PC_CUDNN_VERSION}" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Final attempt: check if cuDNN is embedded in CUDA installation
    if(CUDAToolkit_FOUND AND CUDAToolkit_INCLUDE_DIRS)
        foreach(cuda_include_dir ${CUDAToolkit_INCLUDE_DIRS})
            if(EXISTS "${cuda_include_dir}/cudnn.h")
                message(STATUS "✅ Found cuDNN embedded in CUDA installation")
                set(HAVE_CUDNN TRUE PARENT_SCOPE)
                set(CUDNN_INCLUDE_DIR "${cuda_include_dir}" PARENT_SCOPE)
                set(CUDNN_LIBRARIES "" PARENT_SCOPE)  # May be linked with CUDA
                set(CUDNN_VERSION_STRING "Embedded" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endif()

    message(STATUS "❌ cuDNN not found. Searched extensively in:")
    message(STATUS "   Environment variables: CUDNN_ROOT_DIR, CUDNN_ROOT, CUDA_PATH, CUDA_HOME")
    message(STATUS "   CUDA installation: ${CUDAToolkit_ROOT}")
    if(WIN32)
        message(STATUS "   System paths: $ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit")
    else()
        message(STATUS "   System paths: /usr, /usr/local, /opt")
    endif()
    message(STATUS "   Package manager: pkg-config")
    message(STATUS "")
    message(STATUS "💡 To fix this issue:")
    message(STATUS "   1. Install cuDNN development libraries")
    message(STATUS "   2. Set CUDNN_ROOT_DIR to your cuDNN installation")
    message(STATUS "   3. Ensure cuDNN headers are in CUDA_PATH/include")
    message(STATUS "   4. Or disable cuDNN with -DHELPERS_cudnn=OFF")

    # Debug information
    message(STATUS "")
    message(STATUS "🔍 Debug information:")
    message(STATUS "   CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
    message(STATUS "   CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "   ENV CUDNN_ROOT_DIR: $ENV{CUDNN_ROOT_DIR}")
    message(STATUS "   ENV CUDA_PATH: $ENV{CUDA_PATH}")
    message(STATUS "   ENV CUDA_HOME: $ENV{CUDA_HOME}")
endfunction()

function(configure_cuda_linking main_target_name)
    # Setup CUDA toolkit paths first
    setup_cuda_toolkit_paths()

    # Find the CUDAToolkit to define the CUDA::toolkit target
    find_package(CUDAToolkit REQUIRED)

    # Setup modern cuDNN detection
    setup_modern_cudnn()

    # CRITICAL: Explicitly add CUDA include directories to the target
    # This ensures cuda.h and other CUDA headers are found
    if(CUDA_INCLUDE_DIRS)
        target_include_directories(${main_target_name} PUBLIC ${CUDA_INCLUDE_DIRS})
        message(STATUS "✅ Added CUDA include directories to ${main_target_name}: ${CUDA_INCLUDE_DIRS}")
    endif()

    # Modern CMake uses imported targets which handle all necessary dependencies.
    # Linking against CUDA::toolkit automatically adds include directories,
    # runtime libraries, and all other required flags.
    target_link_libraries(${main_target_name} PUBLIC CUDA::toolkit)

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

    target_link_libraries(${main_target_name} PUBLIC flatbuffers_interface)
    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

function(setup_cuda_architectures_early)
    if(NOT SD_CUDA)
        return()
    endif()

    # Fix missing _CMAKE_CUDA_WHOLE_FLAG (note the underscore prefix)
    if(NOT DEFINED _CMAKE_CUDA_WHOLE_FLAG)
        message(STATUS "Setting _CMAKE_CUDA_WHOLE_FLAG (was missing)")
        if(WIN32)
            set(_CMAKE_CUDA_WHOLE_FLAG "/WHOLEARCHIVE:" CACHE INTERNAL "CUDA whole archive flag")
        else()
            set(_CMAKE_CUDA_WHOLE_FLAG "-Wl,--whole-archive" CACHE INTERNAL "CUDA whole archive flag")
        endif()
    endif()

    if(NOT DEFINED CMAKE_CUDA_WHOLE_FLAG)
        if(WIN32)
            set(CMAKE_CUDA_WHOLE_FLAG "/WHOLEARCHIVE:" CACHE STRING "CUDA whole archive flag")
        else()
            set(CMAKE_CUDA_WHOLE_FLAG "-Wl,--whole-archive" CACHE STRING "CUDA whole archive flag")
        endif()
    endif()

    message(STATUS "🔧 Early CUDA: Configuring architectures before project() call")

    if(DEFINED COMPUTE)
        string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
        if(COMPUTE_CMP STREQUAL "all")
            set(CUDA_ARCHITECTURES "75;80;86;89" PARENT_SCOPE)
            message(STATUS "   CUDA architectures (all): 75;80;86;89")
        elseif(COMPUTE_CMP STREQUAL "auto")
            set(CMAKE_CUDA_ARCHITECTURES "86" PARENT_SCOPE)
            message(STATUS "   CUDA architectures (auto): 86")
        else()
            string(REPLACE "," ";" ARCH_LIST "${COMPUTE}")
            set(PARSED_ARCHS "")
            foreach(ARCH ${ARCH_LIST})
                string(REPLACE "." "" ARCH_CLEAN "${ARCH}")
                if(ARCH_CLEAN MATCHES "^[0-9][0-9]$")
                    list(APPEND PARSED_ARCHS "${ARCH_CLEAN}")
                endif()
            endforeach()
            if(PARSED_ARCHS)
                set(CUDA_ARCHITECTURES "${PARSED_ARCHS}" PARENT_SCOPE)
                message(STATUS "   CUDA architectures (custom): ${PARSED_ARCHS}")
            else()
                set(CUDA_ARCHITECTURES "86" PARENT_SCOPE)
                message(STATUS "   CUDA architectures (default): 86")
            endif()
        endif()
    else()
        set(CUDA_ARCHITECTURES "86" PARENT_SCOPE)
        message(STATUS "   CUDA architectures (no COMPUTE): 86")
    endif()
endfunction()

function(setup_cuda_language)
    if(NOT DEFINED _CMAKE_CUDA_WHOLE_FLAG)
        message(STATUS "Setting _CMAKE_CUDA_WHOLE_FLAG (was missing)")
        if(WIN32)
            set(_CMAKE_CUDA_WHOLE_FLAG "/WHOLEARCHIVE:" CACHE INTERNAL "CUDA whole archive flag")
        else()
            set(_CMAKE_CUDA_WHOLE_FLAG "-Wl,--whole-archive" CACHE INTERNAL "CUDA whole archive flag")
        endif()
    endif()

    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        find_program(NVCC_EXECUTABLE nvcc)
        if(NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE} PARENT_SCOPE)
            message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
        else()
            message(FATAL_ERROR "CUDA compiler not found. Please ensure CUDA toolkit is installed and nvcc is in PATH.")
        endif()
    endif()

    message(STATUS "CUDA language enabled successfully with compiler: ${CMAKE_CUDA_COMPILER}")
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

    # SIMPLE FIX: Just disable the automatic /FS flag that's causing the conflict
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded "/MT" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "/MD" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug "/MTd" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "/MDd" PARENT_SCOPE)

    message(STATUS "Windows CUDA: Configured with clean runtime flags (no /FS)")
endfunction()

function(configure_cuda_architecture_flags COMPUTE)
    string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
    if(COMPUTE_CMP STREQUAL "all")
        set(CUDA_ARCH_FLAGS "-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89" PARENT_SCOPE)
        message(STATUS "Building for all CUDA architectures (gencode flags)")
    elseif(COMPUTE_CMP STREQUAL "auto")
        set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86" PARENT_SCOPE)
        message(STATUS "Auto-detecting CUDA architectures (gencode flags)")
    else()
        string(REPLACE "," ";" ARCH_LIST "${COMPUTE}")
        set(ARCH_FLAGS "")
        foreach(ARCH ${ARCH_LIST})
            string(REPLACE "." "" ARCH_CLEAN "${ARCH}")
            if(ARCH_CLEAN MATCHES "^[0-9][0-9]$")
                set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_${ARCH_CLEAN},code=sm_${ARCH_CLEAN}")
            endif()
        endforeach()
        string(STRIP "${ARCH_FLAGS}" ARCH_FLAGS)
        if(ARCH_FLAGS)
            set(CUDA_ARCH_FLAGS "${ARCH_FLAGS}" PARENT_SCOPE)
            message(STATUS "Using custom CUDA architectures (gencode flags): ${ARCH_FLAGS}")
        else()
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86" PARENT_SCOPE)
            message(STATUS "Using default CUDA architecture (gencode flags)")
        endif()
    endif()
endfunction()

# CRITICAL FIX: Updated CUDA compiler flags function
function(build_cuda_compiler_flags CUDA_ARCH_FLAGS)
    set(LOCAL_CUDA_FLAGS "")

    if(WIN32 AND MSVC)
        message(STATUS "Configuring CUDA for Windows MSVC...")
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} PARENT_SCOPE)
        set(LOCAL_CUDA_FLAGS "-maxrregcount=128")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/nologo")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/EHsc")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/std:c++17")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/D__NVCC_ALLOW_UNSUPPORTED_COMPILER__")

        # CRITICAL FIX: DO NOT add /FS flag - this causes the "single input file" error
        # The /FS flag conflicts with CMake's automatic /Fd flag generation
        # Let CMake handle PDB file generation automatically
        # set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/FS")  # REMOVED

        if(MSVC_RT_LIB STREQUAL "MultiThreadedDLL")
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/MD")
        else()
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/MT")
        endif()

        message(STATUS "CUDA Windows flags configured WITHOUT /FS to prevent nvcc errors")
    else()
        set(LOCAL_CUDA_FLAGS "--allow-unsupported-compiler -Xcompiler -D__NVCC_ALLOW_UNSUPPORTED_COMPILER__")
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

    string(REGEX REPLACE "  +" " " LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
    string(STRIP "${LOCAL_CUDA_FLAGS}" LOCAL_CUDA_FLAGS)

    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON PARENT_SCOPE)
    set(CMAKE_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}" PARENT_SCOPE)

    message(STATUS "Final CMAKE_CUDA_FLAGS: ${LOCAL_CUDA_FLAGS}")
endfunction()

# Debug configuration function
function(debug_cuda_configuration)
    message(STATUS "=== CUDA Configuration Debug Info ===")
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message(STATUS "CMAKE_CUDA_COMPILE_OBJECT: ${CMAKE_CUDA_COMPILE_OBJECT}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    if(HAVE_CUDNN)
        message(STATUS "CUDNN_INCLUDE_DIR: ${CUDNN_INCLUDE_DIR}")
        message(STATUS "CUDNN_LIBRARIES: ${CUDNN_LIBRARIES}")
        message(STATUS "CUDNN_VERSION: ${CUDNN_VERSION_STRING}")
    endif()
    message(STATUS "=== End CUDA Debug Info ===")
endfunction()

# Enhanced CUDA include directory setup for global configuration
function(setup_cuda_include_directories)
    setup_cuda_toolkit_paths()

    # Set global include directories that will be inherited by all targets
    if(CUDA_INCLUDE_DIRS)
        include_directories(${CUDA_INCLUDE_DIRS})
        message(STATUS "✅ Added global CUDA include directories: ${CUDA_INCLUDE_DIRS}")
    endif()

    # Also set the CMAKE variable for compatibility
    if(CUDAToolkit_INCLUDE_DIRS)
        set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE)
    endif()
endfunction()

# MAIN CUDA SETUP FUNCTION
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

    configure_windows_cuda_build()
    build_cuda_compiler_flags("${CUDA_ARCH_FLAGS}")

    # CRITICAL: Set CMAKE_CUDA_FLAGS to parent scope so it propagates
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)

    # Also set the toolkit include directories for global access
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}" PARENT_SCOPE)

    debug_cuda_configuration()

    add_compile_definitions(SD_CUDA=true)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA" PARENT_SCOPE)

    message(STATUS "=== CUDA BUILD CONFIGURATION COMPLETE ===")
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

# Additional helper function to clean up any remaining problematic flags
function(fix_cuda_compile_flags_post_setup)
    if(WIN32 AND MSVC)
        # Clean up any remaining problematic flag combinations that might have been added
        string(REPLACE "-Xcompiler=/FS" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
        string(REPLACE "-Xcompiler=-FS" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
        string(REPLACE "/FS" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
        string(REPLACE "-Xcompiler=/Fd" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

        # Clean up multiple spaces and commas
        string(REGEX REPLACE "  +" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
        string(REGEX REPLACE ",-" " -" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
        string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)
        message(STATUS "🔧 Cleaned problematic CUDA flags: ${CMAKE_CUDA_FLAGS}")
    endif()
endfunction()