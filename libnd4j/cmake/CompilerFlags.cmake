# cmake/CompilerFlags.cmake
# Configures compiler and linker flags for optimization and correctness.

# ===== DISABLE PLT COMPLETELY =====
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-plt")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-plt")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,now")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,now")
endif()
if(SD_CUDA AND CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-plt")
endif()

# --- Link Time Optimization (LTO) ---
if(SD_USE_LTO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Using Link Time Optimization")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

# --- Memory Model for large binaries ---
if(SD_X86_BUILD AND NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=large")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=large")
    message(STATUS "Applied large memory model for x86-64 architecture")
else()
    if(SD_ARM_BUILD OR SD_ANDROID_BUILD)
        message(STATUS "Skipping large memory model for ARM/Android architecture (not supported)")
    elseif(WIN32)
        message(STATUS "Skipping large memory model for Windows (using alternative approach)")
    endif()
endif()

# --- Memory Optimization during compilation ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=100 --param ggc-min-heapsize=131072")
endif()

# --- Section splitting for better linker handling ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
endif()

# --- MSVC-specific optimizations ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/Gy)  # Function-level linking
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /OPT:REF /OPT:ICF")
    add_compile_options(/bigobj /EHsc /Zc:preprocessor)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# --- Windows Specific Configurations ---
if(WIN32 AND NOT ANDROID)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()
    set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE INTERNAL "")
endif()

# --- GCC/Clang Specific Flags ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT SD_CUDA)
    message(STATUS "Adding GCC memory optimization flag: --param ggc-min-expand=10")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=10 ${INFORMATIVE_FLAGS} -std=c++17 -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --param ggc-min-expand=10 -fPIC")
    if(SD_X86_BUILD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-relax")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--no-relax")
    endif()
    if(UNIX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,-z,--no-undefined,--verbose")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose")
    endif()
endif()

# --- Comprehensive linker fix for PLT overflow on GCC ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND SD_X86_BUILD)
    message(STATUS "Configuring linker for large template library with PLT overflow prevention")
    string(REGEX REPLACE "-fuse-ld=[a-zA-Z]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    string(REGEX REPLACE "-fuse-ld=[a-zA-Z]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
    execute_process(COMMAND ${CMAKE_LINKER} --help OUTPUT_VARIABLE LD_HELP_OUTPUT ERROR_QUIET)
    string(FIND "${LD_HELP_OUTPUT}" "--plt-align" PLT_ALIGN_SUPPORTED)
    if(PLT_ALIGN_SUPPORTED GREATER -1)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--plt-align=32 -Wl,--hash-style=both -Wl,-z,max-page-size=0x200000")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--plt-align=32 -Wl,--hash-style=both -Wl,-z,max-page-size=0x200000")
        message(STATUS "✓ Using GNU LD with PLT overflow prevention")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--hash-style=both -Wl,-z,max-page-size=0x200000")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--hash-style=both -Wl,-z,max-page-size=0x200000")
        message(STATUS "✓ Using GNU LD with basic optimizations (--plt-align not supported)")
    endif()
endif()

# --- Build Type Specific Flags ---
if(SD_ANDROID_BUILD)
    # ... flags for android ...
elseif(APPLE)
    # ... flags for apple ...
elseif(WIN32)
    # ... flags for windows ...
else() # Generic Linux/Unix
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -g")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")
endif()

# --- Sanitizer Configuration ---
if(SD_SANITIZE)
    set(SANITIZE_FLAGS " -Wall -Wextra -fPIE -lpthread -ftls-model=local-dynamic -static-libasan -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all")
    message("Using sanitizers: ${SD_SANITIZERS}...")
    if(SD_CPU)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${SANITIZE_FLAGS}")
    endif()
    if(SD_CUDA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${SANITIZE_FLAGS} -lpthread -ftls-model=local-dynamic --relocatable-device-code=true")
    endif()
endif()