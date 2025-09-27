# ADR: Android Cross-Compilation Toolchain Modernization

## Status

Proposed

Proposed by: Adam Gibson (September 2025)

Discussed with: Development Team

## Context

Our Android cross-compilation toolchain configuration has grown outdated and brittle. The current `android-x86_64.cmake` relies heavily on environment variables for everything, lacks proper error checking, and assumes a specific NDK directory structure. This creates frequent build failures that are difficult to diagnose.

When developers encounter issues, they often see cryptic CMake errors rather than helpful messages about what's actually wrong. The toolchain file doesn't validate that paths exist, compilers are present, or that the NDK version is compatible. This wastes developer time and makes it harder for new contributors to build the project.

Additionally, with the NDK r27d migration, we need to ensure our toolchain configuration explicitly uses LLVM components rather than assuming GNU tools are available.

## Decision

We will comprehensively modernize our Android cross-compilation toolchain files with proper error checking, flexible path detection, and explicit LLVM toolchain usage.

### Key Improvements

**1. Flexible NDK Path Detection**

Instead of requiring a specific environment variable, we'll check multiple common locations:

```cmake
if(NOT DEFINED CMAKE_ANDROID_NDK)
   if(DEFINED ENV{ANDROID_NDK_ROOT})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_ROOT})
   elseif(DEFINED ENV{ANDROID_NDK_HOME})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_HOME})
   elseif(DEFINED ENV{ANDROID_NDK})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK})
   else()
      message(FATAL_ERROR "Android NDK not found. Please set ANDROID_NDK_ROOT, ANDROID_NDK_HOME, or ANDROID_NDK")
   endif()
endif()
```

This supports various NDK installation methods and CI/CD configurations without requiring specific environment setup.

**2. Comprehensive Validation**

We'll validate each critical path and binary:

```cmake
if(NOT EXISTS "${CMAKE_ANDROID_NDK}")
    message(FATAL_ERROR "Android NDK path does not exist: ${CMAKE_ANDROID_NDK}")
endif()

if(NOT EXISTS "${NDK_TOOLCHAIN_PATH}")
    message(FATAL_ERROR "NDK toolchain path not found: ${NDK_TOOLCHAIN_PATH}")
endif()

if(NOT EXISTS "${CMAKE_SYSROOT}")
    message(FATAL_ERROR "Android sysroot not found: ${CMAKE_SYSROOT}")
endif()

if(NOT EXISTS "${CMAKE_C_COMPILER}")
    message(FATAL_ERROR "C compiler not found: ${CMAKE_C_COMPILER}")
endif()
```

**3. Explicit LLVM Toolchain Configuration**

Rather than relying on environment variables or PATH, we'll explicitly specify LLVM tools:

```cmake
set(CMAKE_C_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang")
set(CMAKE_CXX_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang++")
set(CMAKE_AR "${NDK_TOOLCHAIN_PATH}/bin/llvm-ar")
set(CMAKE_STRIP "${NDK_TOOLCHAIN_PATH}/bin/llvm-strip")
set(CMAKE_RANLIB "${NDK_TOOLCHAIN_PATH}/bin/llvm-ranlib")
```

**4. Flexible API Level Configuration**

Support both environment variables and CMake variables with sensible defaults:

```cmake
if(NOT DEFINED ANDROID_NATIVE_API_LEVEL AND DEFINED ENV{ANDROID_VERSION})
   set(ANDROID_NATIVE_API_LEVEL $ENV{ANDROID_VERSION})
elseif(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
   set(ANDROID_NATIVE_API_LEVEL 21)  # Minimum for NDK r27d
endif()
```

**5. Modern Android Linker Flags**

Add flags that improve debugging, security, and binary size:

```cmake
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=sha1")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-rosegment")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--fatal-warnings")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
```

## Implementation Details

### Affected Files

We'll update all architecture-specific toolchain files:
- `android-arm.cmake`
- `android-arm64.cmake`
- `android-x86.cmake`
- `android-x86_64.cmake`

### Testing Strategy

1. **Local Testing**: Verify builds work with NDK in various locations
2. **CI/CD Testing**: Ensure GitHub Actions workflows continue to function
3. **Error Testing**: Deliberately misconfigure to verify error messages
4. **Cross-Platform**: Test on Linux, macOS, and Windows hosts

## Consequences

### Advantages

**Developer Experience**: Clear error messages dramatically reduce debugging time. When something goes wrong, developers immediately know whether it's a missing NDK, wrong path, or incompatible version.

**Flexibility**: Supporting multiple NDK path conventions means developers can use their preferred installation method. This is especially helpful for CI/CD systems with different conventions.

**Robustness**: Explicit validation catches problems early in the CMake configuration phase rather than during compilation or linking.

**Future-Proof**: Explicitly using LLVM tools ensures compatibility with newer NDK versions that have removed GNU toolchain components.

**Documentation**: The toolchain file itself becomes self-documenting with clear error messages explaining requirements.

### Disadvantages

**Complexity**: The toolchain files become longer and more complex with all the validation logic.

**Maintenance**: More code to maintain and keep synchronized across architecture variants.

**Breaking Changes**: Developers who relied on specific environment variable names may need to adjust their setup.

### Migration Impact

Most developers won't notice any change if their environment is correctly configured. Those with non-standard setups will see helpful error messages guiding them to the correct configuration.

CI/CD pipelines should continue to work unchanged, as we're adding support for new conventions while maintaining backward compatibility with existing ones.

## Conclusion

These improvements transform our Android toolchain from a source of frustration into a helpful part of the build system. By investing in proper error handling and validation, we save developer time and make the project more accessible to new contributors.

## References

- Android NDK CMake Toolchain Documentation
- CMake Cross-Compilation Guide
- LLVM Toolchain Documentation