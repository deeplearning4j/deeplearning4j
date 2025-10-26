# cmake/CompilerFlags.cmake
# Configures compiler and linker flags for optimization and correctness.

# ===== DISABLE PLT COMPLETELY =====
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-plt")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-plt")
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

# --- Memory Optimization during compilation ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors --param ggc-min-expand=100 --param ggc-min-heapsize=131072")
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
    if(UNIX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose")
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
# In CompilerFlags.cmake, change the sanitizer section:
# --- Sanitizer Configuration ---
if(SD_SANITIZE)
    # Use global-dynamic TLS model for shared libraries
    set(SANITIZE_FLAGS " -Wall -Wextra -fPIC -ftls-model=global-dynamic -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all")
    set(SANITIZE_LINK_FLAGS "-fsanitize=${SD_SANITIZERS}")
    
    message("Using sanitizers: ${SD_SANITIZERS}...")
    if(SD_CPU)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZE_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZE_LINK_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZE_LINK_FLAGS}")
    endif()
    if(SD_CUDA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS} --relocatable-device-code=true")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SANITIZE_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZE_LINK_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZE_LINK_FLAGS}")
    endif()
endif()

if(SD_GCC_FUNCTRACE)
    message(STATUS "✅ Applying SD_GCC_FUNCTRACE debug flags for line number information")

    # Override any optimization flags with debug-friendly ones
    set(CMAKE_CXX_FLAGS_RELEASE "-O0 -ggdb3 -fPIC -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb3 -fPIC")
    set(CMAKE_C_FLAGS_RELEASE "-O0 -ggdb3 -fPIC -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb3 -fPIC")

    # Add comprehensive debug flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -finstrument-functions -gdwarf-4 -fno-eliminate-unused-debug-types")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb3 -fno-omit-frame-pointer -gdwarf-4")

        # Override any conflicting optimization
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "-O[0-9s]" "-O0" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    endif()

    # Ensure debug info is preserved in linker
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")

    # Prevent stripping
    set(CMAKE_STRIP "/bin/true")

    # Add the compiler definition
    add_compile_definitions(SD_GCC_FUNCTRACE=ON)
endif()