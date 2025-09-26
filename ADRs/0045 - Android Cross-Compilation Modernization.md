# ADR-0045: Android Cross-Compilation Toolchain Modernization

## Status

Proposed

Proposed by: Adam Gibson (27-09-2025)

## Context

Android cross-compilation uses outdated toolchain configuration.
Old android-x86_64.cmake uses environment variables for everything.
No proper error checking or validation.
Hardcoded paths assume specific NDK structure.

## Decision

Modernize Android cross-compilation toolchain files.
Add comprehensive error checking and validation.
Support flexible NDK path detection.
Explicitly use LLVM toolchain components.

## Changes Made

### Path Detection
```cmake
# Before - Required specific env vars
set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK}")

# After - Flexible detection
if(NOT DEFINED CMAKE_ANDROID_NDK)
   if(DEFINED ENV{ANDROID_NDK_ROOT})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_ROOT})
   elseif(DEFINED ENV{ANDROID_NDK_HOME})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_HOME})
   elseif(DEFINED ENV{ANDROID_NDK})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK})
   else()
      message(FATAL_ERROR "Android NDK not found...")
   endif()
endif()
```

### Toolchain Specification
```cmake
# Before - Environment variable compiler
set(CMAKE_C_COMPILER "$ENV{ANDROID_CC}")

# After - Explicit LLVM toolchain
set(CMAKE_C_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang")
set(CMAKE_AR "${NDK_TOOLCHAIN_PATH}/bin/llvm-ar")
set(CMAKE_STRIP "${NDK_TOOLCHAIN_PATH}/bin/llvm-strip")
set(CMAKE_RANLIB "${NDK_TOOLCHAIN_PATH}/bin/llvm-ranlib")
```

### Error Checking
Added validation for:
- NDK directory existence
- Toolchain path existence
- Sysroot existence
- Compiler binary existence

### API Level Configuration
```cmake
# Before - Only environment variable
set(CMAKE_SYSTEM_VERSION "$ENV{ANDROID_VERSION}")

# After - Flexible with default
if(NOT DEFINED ANDROID_NATIVE_API_LEVEL AND DEFINED ENV{ANDROID_VERSION})
   set(ANDROID_NATIVE_API_LEVEL $ENV{ANDROID_VERSION})
elseif(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
   set(ANDROID_NATIVE_API_LEVEL 21)  # Default API level
endif()
```

### Linker Flags
Added modern Android linker flags:
- `-Wl,--build-id=sha1`
- `-Wl,--no-rosegment`
- `-Wl,--fatal-warnings`
- `-Wl,--gc-sections`
- `-Wl,--no-undefined`

## Consequences

- More robust cross-compilation
- Better error messages when misconfigured
- Supports multiple NDK installation methods
- Enforces LLVM toolchain usage
- Default API 21 (Android 5.0) minimum

## References
- Android NDK CMake documentation
- LLVM cross-compilation guide