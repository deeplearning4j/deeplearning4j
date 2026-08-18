cmake_minimum_required(VERSION 3.16)
project(tokenizers-cpp-wrapper VERSION 1.0.0 LANGUAGES C CXX)

# Standard installation directories
include(GNUInstallDirs)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform detection
if(NOT DEFINED PLATFORM)
    if(WIN32)
        set(PLATFORM "windows")
    elseif(APPLE)
        set(PLATFORM "darwin")
    else()
        set(PLATFORM "linux")
    endif()
endif()

if(NOT DEFINED ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(ARCH "x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(ARCH "arm64")
    else()
        set(ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()

# JavaCPP platform naming convention
if(NOT DEFINED JAVACPP_PLATFORM)
    if(PLATFORM STREQUAL "linux")
        if(ARCH STREQUAL "arm64" OR ARCH STREQUAL "aarch64")
            set(JAVACPP_PLATFORM "linux-arm64")
        else()
            set(JAVACPP_PLATFORM "linux-x86_64")
        endif()
    elseif(PLATFORM STREQUAL "darwin")
        if(ARCH STREQUAL "arm64")
            set(JAVACPP_PLATFORM "macosx-arm64")
        else()
            set(JAVACPP_PLATFORM "macosx-x86_64")
        endif()
    elseif(PLATFORM STREQUAL "windows")
        set(JAVACPP_PLATFORM "windows-x86_64")
    else()
        set(JAVACPP_PLATFORM "${PLATFORM}-${ARCH}")
    endif()
endif()

message(STATUS "Building for platform: ${PLATFORM}-${ARCH}")
message(STATUS "JavaCPP platform: ${JAVACPP_PLATFORM}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find Rust and Cargo
find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_program(RUSTC_EXECUTABLE rustc REQUIRED)

message(STATUS "Found Cargo: ${CARGO_EXECUTABLE}")
message(STATUS "Found Rustc: ${RUSTC_EXECUTABLE}")

# Local build directory used by the existing desktop build script.
# Mobile builds normally pass TOKENIZERS_FFI_LIB explicitly.
set(TOKENIZERS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/tokenizers-build")

# Platform-specific library names and extensions
if(PLATFORM STREQUAL "windows")
    set(WRAPPER_LIB_NAME "tokenizers_wrapper")
    set(LIB_SUFFIX ".dll")
    # GNU-style Windows toolchains (MinGW or clang with GNU-like
    # command-line semantics) consume Rust's .a archive. MSVC consumes .lib.
    if(MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(STATIC_LIB_SUFFIX ".a")
    else()
        set(STATIC_LIB_SUFFIX ".lib")
    endif()
elseif(PLATFORM STREQUAL "darwin")
    set(WRAPPER_LIB_NAME "tokenizers_wrapper")
    set(LIB_SUFFIX ".dylib")
    set(STATIC_LIB_SUFFIX ".a")
else()
    set(WRAPPER_LIB_NAME "tokenizers_wrapper")
    set(LIB_SUFFIX ".so")
    set(STATIC_LIB_SUFFIX ".a")
endif()

# Set build profile for Rust
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_BUILD_TYPE "debug")
    set(CARGO_BUILD_FLAG "")
else()
    set(CARGO_BUILD_TYPE "release")
    set(CARGO_BUILD_FLAG "--release")
endif()

# The Rust tokenizer crate is the single source of HuggingFace tokenizer logic.
# Its build scripts produce a static archive first; this wrapper only exposes the
# stable tokenizers_c.h ABI. An explicit path makes Android/iOS cross-builds
# hermetic and avoids a configure-time network clone.
set(TOKENIZERS_FFI_LIB "" CACHE FILEPATH "Prebuilt tokenizers Rust static library")
set(TOKENIZERS_FFI_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/tokenizers-ffi")
if(NOT TOKENIZERS_FFI_LIB)
    set(TOKENIZERS_FFI_LIB
        "${TOKENIZERS_FFI_BINARY_DIR}/release/libtokenizers_ffi${STATIC_LIB_SUFFIX}")
endif()
if(NOT EXISTS "${TOKENIZERS_FFI_LIB}")
    message(FATAL_ERROR
        "Prebuilt tokenizers FFI library not found: ${TOKENIZERS_FFI_LIB}. "
        "Run buildnativetokenizers.sh or build-mobile-tokenizers.sh first.")
endif()
message(STATUS "Using prebuilt tokenizers FFI library: ${TOKENIZERS_FFI_LIB}")

# Check if header was generated
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/tokenizers_ffi.h")
    message(STATUS "Found tokenizers_ffi.h header")
else()
    message(WARNING "tokenizers_ffi.h header not found - build may fail")
endif()

# Source files for the C++ wrapper (including the new C FFI layer)
set(WRAPPER_SOURCES
    src/model_manager.cpp
    src/tokenizer_wrapper.cpp
    src/tokenizers_c.cpp
)

# Header files
set(WRAPPER_HEADERS
    include/model_manager.h
    include/tokenizer_wrapper.h
    include/tokenizers_c.h
)

# Create the wrapper library
add_library(${WRAPPER_LIB_NAME} SHARED ${WRAPPER_SOURCES} ${WRAPPER_HEADERS})

# Note: tokenizers-ffi is built by the build script before CMake runs
# No external project dependencies needed here

# Include directories
target_include_directories(${WRAPPER_LIB_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compiler-specific options
# Clang can target the MSVC ABI without CMake setting MSVC=TRUE. Treat that
# mode like MSVC as well: clang-cl/clang++ reject GCC's -fPIC option for the
# x86_64-pc-windows-msvc target. CMake may omit CMAKE_CXX_COMPILER_TARGET
# for the host compiler, so the non-MinGW Clang case is covered explicitly.
set(TOKENIZERS_MSVC_ABI FALSE)
if(MSVC OR
   (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
    (CMAKE_CXX_COMPILER_TARGET MATCHES "windows-msvc" OR NOT MINGW)))
    set(TOKENIZERS_MSVC_ABI TRUE)
endif()
if(TOKENIZERS_MSVC_ABI)
    target_compile_options(${WRAPPER_LIB_NAME} PRIVATE /W3 /EHsc)
    target_compile_definitions(${WRAPPER_LIB_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(${WRAPPER_LIB_NAME} PRIVATE -Wall -Wextra -fPIC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${WRAPPER_LIB_NAME} PRIVATE -g -O0)
    else()
        target_compile_options(${WRAPPER_LIB_NAME} PRIVATE -O3 -DNDEBUG)
    endif()
endif()

# Link the Rust FFI library (built by buildnativetokenizers.sh)
target_link_libraries(${WRAPPER_LIB_NAME} PRIVATE
    ${TOKENIZERS_FFI_LIB}
)

# Platform-specific linking options
if(PLATFORM STREQUAL "linux")
    target_link_libraries(${WRAPPER_LIB_NAME} PRIVATE dl pthread)
    set_target_properties(${WRAPPER_LIB_NAME} PROPERTIES
        LINK_FLAGS "-Wl,-rpath,$ORIGIN"
    )
elseif(PLATFORM STREQUAL "darwin")
    target_link_libraries(${WRAPPER_LIB_NAME} PRIVATE dl pthread)
    set_target_properties(${WRAPPER_LIB_NAME} PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(PLATFORM STREQUAL "windows")
    # Windows-specific settings: Rust static libs need Windows system libraries
    target_link_libraries(${WRAPPER_LIB_NAME} PRIVATE ws2_32 userenv bcrypt ntdll)
    set_target_properties(${WRAPPER_LIB_NAME} PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
endif()

# Output library naming
set_target_properties(${WRAPPER_LIB_NAME} PROPERTIES
    OUTPUT_NAME "${WRAPPER_LIB_NAME}"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Note: Headers are now generated by buildnativetokenizers.sh using cbindgen
# before CMake is invoked. The tokenizers_ffi.h header should already exist
# in the include directory.

# JavaCPP-specific installation paths
# Install to platform-specific directory for JavaCPP
install(TARGETS ${WRAPPER_LIB_NAME}
    LIBRARY DESTINATION ${JAVACPP_PLATFORM}
    ARCHIVE DESTINATION ${JAVACPP_PLATFORM}
    RUNTIME DESTINATION ${JAVACPP_PLATFORM}
)

# Install headers to common include directory for JavaCPP
install(FILES ${WRAPPER_HEADERS}
    DESTINATION include
)

# Install the generated FFI header if it exists
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/tokenizers_ffi.h
    DESTINATION include
    OPTIONAL
)

# Install to legacy paths for backward compatibility
install(TARGETS ${WRAPPER_LIB_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${WRAPPER_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tokenizers
)

# Create a pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizers.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tokenizers.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# Preserve the historical target name without configuring any network-backed
# external project.
add_custom_target(clean-external
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${TOKENIZERS_BINARY_DIR}
    COMMENT "Cleaning local tokenizer build directories"
)

# Print build summary
message(STATUS "")
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Platform: ${PLATFORM}")
message(STATUS "Architecture: ${ARCH}")
message(STATUS "JavaCPP Platform: ${JAVACPP_PLATFORM}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Cargo Build Type: ${CARGO_BUILD_TYPE}")
message(STATUS "Wrapper Library: ${WRAPPER_LIB_NAME}${LIB_SUFFIX}")
message(STATUS "Tokenizers FFI Library: ${TOKENIZERS_FFI_LIB}")
message(STATUS "Tokenizers Binary: ${TOKENIZERS_BINARY_DIR}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "JavaCPP Platform Dir: ${CMAKE_INSTALL_PREFIX}/${JAVACPP_PLATFORM}")
message(STATUS "===================================")
