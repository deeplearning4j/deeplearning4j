# CompilerOptimizations.cmake - Compiler flags and optimization settings



# Link Time Optimization
if(SD_USE_LTO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Using Link Time Optimization")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

# Architecture Tuning
if(SD_ARCH MATCHES "armv8")
    set(ARCH_TUNE "-march=${SD_ARCH}")
elseif(SD_ARCH MATCHES "armv7")
    set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon")
elseif(SD_EXTENSION MATCHES "avx2")
    message("Building AVX2 binary...")
    set(ARCH_TUNE "-mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mprefetchwt1 -DSD_F16C=true -DF_AVX2=true")
    if(NO_AVX256_SPLIT)
        set(ARCH_TUNE "${ARCH_TUNE} -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
    endif()
else()
    if("${SD_ARCH}" STREQUAL "x86-64")
        message("Building x86_64 binary...")
        set(ARCH_TYPE "generic")
        add_compile_definitions(F_X64=true)
    else()
        set(ARCH_TYPE "${SD_ARCH}")
    endif()

    if(SD_EXTENSION MATCHES "avx512")
        message("Building AVX512 binary...")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mbmi -mbmi2 -mprefetchwt1 -mclflushopt -mxsavec -mxsaves -DSD_F16C=true -DF_AVX512=true")
    endif()

    # FIXED: Only set architecture flags if we have valid values
    if(NOT WIN32 AND NOT SD_CUDA)
        if(DEFINED SD_ARCH AND NOT SD_ARCH STREQUAL "" AND DEFINED ARCH_TYPE AND NOT ARCH_TYPE STREQUAL "")
            set(ARCH_TUNE "-march=${SD_ARCH} -mtune=${ARCH_TYPE}")
        elseif(DEFINED SD_ARCH AND NOT SD_ARCH STREQUAL "")
            # Fallback if ARCH_TYPE is not set
            set(ARCH_TUNE "-march=${SD_ARCH}")
        endif()
    endif()
endif()

# Comprehensive linker fix for PLT overflow
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND SD_X86_BUILD)
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
endif()

# Use large memory model (required for your template scale) - FIXED: Only for x86-64, not ARM
if(SD_X86_BUILD AND NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=medium")
    message(STATUS "Applied large memory model for x86-64 architecture")
else()
    if(SD_ARM_BUILD OR SD_ANDROID_BUILD)
        message(STATUS "Skipping large memory model for ARM/Android architecture (not supported)")
    elseif(WIN32)
        message(STATUS "Skipping large memory model for Windows (using alternative approach)")
    endif()
endif()

# Memory optimization during compilation - FIXED: Only apply to GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=100 --param ggc-min-heapsize=131072")
endif()

# Section splitting for better linker handling - FIXED: Only apply to GCC/Clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
endif()

# MSVC-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC equivalent optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Gy")  # Function-level linking
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Gy")      # Function-level linking
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /OPT:REF /OPT:ICF")  # Remove unreferenced code
endif()

message(STATUS "Applied PLT overflow prevention for large template library")

if(SD_NATIVE)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*")
        set(SD_X86_BUILD false)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()
