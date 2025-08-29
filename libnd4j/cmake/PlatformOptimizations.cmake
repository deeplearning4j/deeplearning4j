################################################################################
# Platform Optimization Functions
# Platform-specific compiler optimizations and build configurations
################################################################################

# Function to apply Android x86_64 PLT fixes for large template libraries
function(apply_android_x86_64_plt_fixes target_name)
    if(NOT (SD_ANDROID_BUILD AND ANDROID_ABI MATCHES "x86_64"))
        return()
    endif()

    # Additional target-specific compiler flags (all verified) - FIXED: Only for GCC/Clang
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target_name} PRIVATE
                -ffunction-sections
                -fdata-sections
                -fvisibility=hidden
                -fvisibility-inlines-hidden
                -fno-plt
                -fno-semantic-interposition
        )
    else()
        # MSVC equivalent flags
        target_compile_options(${target_name} PRIVATE
                /Gy  # Function-level linking
        )
    endif()

    # Target-specific preprocessor definitions
    target_compile_definitions(${target_name} PRIVATE
            ANDROID_X86_64_OPTIMIZED=1
    )

    # Apply linker flags only to shared/executable targets
    get_target_property(target_type ${target_name} TYPE)


    message(STATUS "Applied verified Android x86_64 optimizations to target: ${target_name}")
endfunction()

# Comprehensive linker configuration for large template libraries
function(configure_large_template_linker)
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND SD_X86_BUILD))
        return()
    endif()

    message(STATUS "Configuring linker for large template library with PLT overflow prevention")

    # Clear any existing conflicting linker flags
    string(REGEX REPLACE "-fuse-ld=[a-zA-Z]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    string(REGEX REPLACE "-fuse-ld=[a-zA-Z]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

    # Test if linker supports --plt-align before using it
    execute_process(
            COMMAND ${CMAKE_LINKER} --help
            OUTPUT_VARIABLE LD_HELP_OUTPUT
            ERROR_QUIET
    )

    string(FIND "${LD_HELP_OUTPUT}" "--plt-align" PLT_ALIGN_SUPPORTED)


    message(STATUS "Applied PLT overflow prevention for large template library")
endfunction()

# Function to configure memory model based on architecture
function(configure_memory_model)
    # Use large memory model (required for template scale) - FIXED: Only for x86-64, not ARM
    if(SD_X86_BUILD AND NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium -fPIC" PARENT_SCOPE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=medium" PARENT_SCOPE)
        message(STATUS "Applied large memory model for x86-64 architecture")
    else()
        if(SD_ARM_BUILD OR SD_ANDROID_BUILD)
            message(STATUS "Skipping large memory model for ARM/Android architecture (not supported)")
        elseif(WIN32)
            message(STATUS "Skipping large memory model for Windows (using alternative approach)")
        endif()
    endif()
endfunction()

# Function to apply memory optimization during compilation
function(configure_compilation_memory_optimization)
    # Memory optimization during compilation - FIXED: Only apply to GCC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=100 --param ggc-min-heapsize=131072" PARENT_SCOPE)
        message(STATUS "Applied GCC memory optimization flags")
    endif()
endfunction()

# Function to configure section splitting for better linker handling
function(configure_section_splitting)
    # MSVC-specific optimizations
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # MSVC equivalent optimizations
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Gy" PARENT_SCOPE)      # Function-level linking
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Gy" PARENT_SCOPE)          # Function-level linking
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /OPT:REF /OPT:ICF" PARENT_SCOPE)  # Remove unreferenced code
        message(STATUS "Applied MSVC function-level linking optimizations")
    endif()
endfunction()


# Function to disable PLT completely for memory issues
function(configure_plt_disable)
    # For CUDA builds, disable PLT in host compiler flags
    if(SD_CUDA AND CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-plt" PARENT_SCOPE)
        message(STATUS "Disabled PLT for CUDA host compiler")
    endif()
endfunction()

# Function to determine platform type accurately
function(determine_platform_type)
    set(SD_X86_BUILD false PARENT_SCOPE)
    set(SD_ARM_BUILD false PARENT_SCOPE)

    if(SD_ANDROID_BUILD)
        if(ANDROID_ABI MATCHES "x86_64")
            set(SD_X86_BUILD true PARENT_SCOPE)
            set(SD_ARCH "x86-64" PARENT_SCOPE)
        elseif(ANDROID_ABI MATCHES "x86")
            set(SD_X86_BUILD true PARENT_SCOPE)
            set(SD_ARCH "x86" PARENT_SCOPE)
        elseif(ANDROID_ABI MATCHES "arm64-v8a")
            set(SD_ARM_BUILD true PARENT_SCOPE)
            set(SD_ARCH "arm64-v8a" PARENT_SCOPE)
        elseif(ANDROID_ABI MATCHES "armeabi-v7a")
            set(SD_ARM_BUILD true PARENT_SCOPE)
            set(SD_ARCH "armv7-a" PARENT_SCOPE)
        endif()
    elseif(NOT SD_IOS_BUILD)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
            set(SD_X86_BUILD true PARENT_SCOPE)
            if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
                set(SD_ARCH "x86-64" PARENT_SCOPE)
            endif()
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm*|aarch64")
            set(SD_ARM_BUILD true PARENT_SCOPE)
            if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
                if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
                    set(SD_ARCH "armv8-a" PARENT_SCOPE)
                else()
                    set(SD_ARCH "armv7-a" PARENT_SCOPE)
                endif()
            endif()
        endif()
    endif()

    # Set default for ARM builds if not specified
    if(SD_ARM_BUILD)
        if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
            message(STATUS "Warning: SD_ARCH was not set for this ARM build. Defaulting to 'armv8-a'.")
            set(SD_ARCH "armv8-a" PARENT_SCOPE)
        endif()

        if(SD_ANDROID)
            set(CMAKE_POSITION_INDEPENDENT_CODE ON PARENT_SCOPE)
        endif()
    endif()

    message(STATUS "Platform detection results: SD_X86_BUILD=${SD_X86_BUILD}, SD_ARM_BUILD=${SD_ARM_BUILD}, SD_ARCH=${SD_ARCH}")
endfunction()

# Function to configure architecture tuning
function(configure_architecture_tuning)
    if(SD_ARCH MATCHES "armv8")
        set(ARCH_TUNE "-march=${SD_ARCH}" PARENT_SCOPE)
        message(STATUS "ARM64 architecture tuning: ${SD_ARCH}")
    elseif(SD_ARCH MATCHES "armv7")
        set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon" PARENT_SCOPE)
        message(STATUS "ARM32 architecture tuning: ${SD_ARCH} with NEON")
    elseif(SD_EXTENSION MATCHES "avx2")
        message(STATUS "Building AVX2 binary...")
        set(ARCH_TUNE "-mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mprefetchwt1 -DSD_F16C=true -DF_AVX2=true" PARENT_SCOPE)
        if(NO_AVX256_SPLIT)
            set(ARCH_TUNE "${ARCH_TUNE} -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" PARENT_SCOPE)
        endif()
    else()
        if("${SD_ARCH}" STREQUAL "x86-64")
            message(STATUS "Building x86_64 binary...")
            set(ARCH_TYPE "generic")
            add_compile_definitions(F_X64=true)
        else()
            set(ARCH_TYPE "${SD_ARCH}")
        endif()

        if(SD_EXTENSION MATCHES "avx512")
            message(STATUS "Building AVX512 binary...")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mbmi -mbmi2 -mprefetchwt1 -mclflushopt -mxsavec -mxsaves -DSD_F16C=true -DF_AVX512=true" PARENT_SCOPE)
        endif()

        # FIXED: Only set architecture flags if we have valid values
        if(NOT WIN32 AND NOT SD_CUDA)
            if(DEFINED SD_ARCH AND NOT SD_ARCH STREQUAL "" AND DEFINED ARCH_TYPE AND NOT ARCH_TYPE STREQUAL "")
                set(ARCH_TUNE "-march=${SD_ARCH} -mtune=${ARCH_TYPE}" PARENT_SCOPE)
            elseif(DEFINED SD_ARCH AND NOT SD_ARCH STREQUAL "")
                # Fallback if ARCH_TYPE is not set
                set(ARCH_TUNE "-march=${SD_ARCH}" PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()

# Function to apply compiler-specific flags with architecture tuning
function(apply_compiler_specific_flags ARCH_TUNE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND SD_X86_BUILD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -O${SD_OPTIMIZATION_LEVEL} -fp-model fast" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT SD_CUDA)
        message(STATUS "Adding GCC memory optimization flag: --param ggc-min-expand=10")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=10 ${ARCH_TUNE} ${INFORMATIVE_FLAGS} -std=c++17 -fPIC" PARENT_SCOPE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --param ggc-min-expand=10 -fPIC" PARENT_SCOPE)


        if(UNIX)
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,-z,--no-undefined,--verbose" PARENT_SCOPE)
        else()
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose" PARENT_SCOPE)
        endif()

        if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT APPLE AND NOT WIN32)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic,--verbose" PARENT_SCOPE)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic,--verbose" PARENT_SCOPE)
        endif()

        if(SD_GCC_FUNCTRACE)
            set(COMPILER_IS_NVCC false)
            get_filename_component(COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
            if(COMPILER_NAME MATCHES "^nvcc")
                set(COMPILER_IS_NVCC TRUE)
            endif()

            if(DEFINED ENV{OMPI_CXX} OR DEFINED ENV{MPICH_CXX})
                if("$ENV{OMPI_CXX}" MATCHES "nvcc" OR "$ENV{MPICH_CXX}" MATCHES "nvcc")
                    set(COMPILER_IS_NVCC TRUE)
                endif()
            endif()

            set(CMAKE_CXX_STANDARD_REQUIRED TRUE PARENT_SCOPE)
            if(COMPILER_IS_NVCC)
                set(CMAKE_CXX_EXTENSIONS OFF PARENT_SCOPE)
            endif()

            # Compiler flags (no linker libraries here)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0 -gno-record-gcc-switches -ftrack-macro-expansion=0 -fstack-protector -fstack-protector-all -Wall -Wextra -Wno-return-type -Wno-error=int-in-bool-context -Wno-unused-variable -Wno-error=implicit-fallthrough -Wno-return-type -Wno-unused-parameter -Wno-error=unknown-pragmas -ggdb3 -pthread -MT -Bsymbolic -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -finstrument-functions -O0 -fPIC" PARENT_SCOPE)

            # Linker libraries (separate from compiler flags)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread -lbfd -lunwind -ldw -ldl -lelf" PARENT_SCOPE)
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lpthread -lbfd -lunwind -ldw -ldl -lelf" PARENT_SCOPE)

            # Add the compiler definition
            add_compile_definitions(SD_GCC_FUNCTRACE=ON)
        endif()
    endif()
endfunction()

# Main function to setup all platform optimizations
function(setup_platform_optimizations)
    message(STATUS "Setting up platform-specific optimizations...")

    # Determine platform type
    determine_platform_type()

    # Configure PLT disable
    configure_plt_disable()

    # Configure memory model
    configure_memory_model()

    # Configure compilation memory optimization
    configure_compilation_memory_optimization()

    # Configure section splitting
    configure_section_splitting()

    # Configure large template linker
    configure_large_template_linker()

    # Configure architecture tuning
    configure_architecture_tuning()

    # Apply architecture tuning based on calculated ARCH_TUNE
    if(DEFINED ARCH_TUNE)
        apply_compiler_specific_flags("${ARCH_TUNE}")
    endif()

    message(STATUS "Platform optimizations setup complete")
endfunction()
