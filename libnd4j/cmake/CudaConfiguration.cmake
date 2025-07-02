################################################################################
# CUDA Configuration Functions
# Functions for CUDA-specific build configuration and optimization
# FIXED VERSION - Resolves NVCC include directory flags issue
################################################################################


function(configure_cuda_linking main_target_name)
    # Find the CUDAToolkit to define the CUDA::toolkit target
    find_package(CUDAToolkit REQUIRED)

    # Find CUDNN to define the CUDNN::cudnn target (if HELPERS_cudnn is enabled)
    if(HELPERS_cudnn)
        find_package(CUDNN)
    endif()

    # Modern CMake uses imported targets which handle all necessary dependencies.
    # Linking against CUDA::toolkit automatically adds include directories,
    # runtime libraries, and all other required flags.
    target_link_libraries(${main_target_name} PUBLIC CUDA::toolkit)

    # If CUDNN was found, link against its official imported target.
    if(CUDNN_FOUND AND TARGET CUDNN::cudnn)
        message(STATUS "✅ Linking with modern CUDNN::cudnn target")
        target_link_libraries(${main_target_name} PUBLIC CUDNN::cudnn)
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
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS ON PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES ON PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES ON PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LINK_OBJECTS ON PARENT_SCOPE)
    set(CMAKE_CUDA_RESPONSE_FILE_LINK_FLAG "@" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_USE_RESPONSE_FILE ON PARENT_SCOPE)
    message(STATUS "Windows CUDA: Enabled response file support for long command lines")
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

        if(MSVC_RT_LIB STREQUAL "MultiThreadedDLL")
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/MD")
        else()
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=/MT")
        endif()
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

# Debug configuration function - was missing
function(debug_cuda_configuration)
    message(STATUS "=== CUDA Configuration Debug Info ===")
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message(STATUS "CMAKE_CUDA_COMPILE_OBJECT: ${CMAKE_CUDA_COMPILE_OBJECT}")
    message(STATUS "=== End CUDA Debug Info ===")
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
    message(STATUS "CUDA Include Dirs: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message(STATUS "Host CXX Compiler: ${CMAKE_CXX_COMPILER_ID}")

    configure_windows_cuda_build()
    build_cuda_compiler_flags("${CUDA_ARCH_FLAGS}")

    # CRITICAL: Set CMAKE_CUDA_FLAGS to parent scope so it propagates
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)

    debug_cuda_configuration()

    add_compile_definitions(SD_CUDA=true)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA" PARENT_SCOPE)

    message(STATUS "=== CUDA BUILD CONFIGURATION COMPLETE ===")
endfunction()

function(setup_cudnn)
    set(HAVE_CUDNN FALSE PARENT_SCOPE)
    if(NOT (HELPERS_cudnn AND SD_CUDA))
        return()
    endif()
    find_package(CUDNN)
    if(CUDNN_FOUND)
        set(HAVE_CUDNN TRUE PARENT_SCOPE)
        set(CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR} PARENT_SCOPE)
        set(CUDNN ${CUDNN_LIBRARIES} PARENT_SCOPE)
    endif()
endfunction()

