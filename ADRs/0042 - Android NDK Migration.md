# ADR: Android NDK Migration from r21d to r27d

## Status

Proposed

Proposed by: Adam Gibson (September 2025)

Discussed with: Development Team

## Context

We've been using Android NDK r21d since October 2019, when it was released as the first LTS version. After 5 years, it's time to evaluate whether we should upgrade to the latest r27d release.

The version gap is substantial - we're looking at jumping 9 major LLVM versions (from LLVM 9 to LLVM 18). This isn't just a routine update; it represents fundamental changes in the Android build ecosystem:

**Toolchain Evolution**: The Android team has completely removed GNU binutils (as, ld, ar, strip) in favor of LLVM alternatives. GDB is gone, replaced entirely by LLDB. Even GNU Make has been upgraded from 4.2 to 4.4.1. The toolchain is now purely LLVM-based with LLD as the only linker option.

**Platform Support Changes**: Perhaps most significantly, the minimum API level has increased from 16 to 21. This means dropping support for KitKat (APIs 19-20) and Jelly Bean (APIs 16-18). Android 5.0 Lollipop becomes our new baseline. Additionally, 32-bit Windows host support has been removed entirely.

**Why This Matters**: Our Android builds have been stable, but we're missing out on years of compiler optimizations, C++20/23 features, and security improvements. The old toolchain also makes it harder to integrate with modern Android development practices.

## Decision

We will upgrade to Android NDK r27d across all our Android builds. This means accepting the increased minimum API requirement and fully embracing the LLVM toolchain.

### Technical Migration

The toolchain changes require several updates:

**Compiler Tooling**:
- Replace GNU assembler with Clang's integrated assembler
- Remove any `-fno-integrated-as` flags from our builds
- Switch from `ar` to `llvm-ar`
- Switch from `strip` to `llvm-strip`

**Build System Updates**:
- Update all CMAKE_TOOLCHAIN_FILE paths
- Remove GDB debugging configurations
- Set minimum API to 21 in all Android builds
- Remove 32-bit Windows cross-compilation support

### CMake Toolchain Improvements

Beyond just updating versions, we're modernizing our toolchain files to be more robust:

**Flexible Path Detection**:
```cmake
# Old approach - rigid environment variable
set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK}")

# New approach - check multiple common locations
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

**Explicit Toolchain Specification**:
```cmake
# Old approach - environment-dependent
set(CMAKE_C_COMPILER "$ENV{ANDROID_CC}")

# New approach - explicit LLVM toolchain
set(CMAKE_C_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang")
set(CMAKE_AR "${NDK_TOOLCHAIN_PATH}/bin/llvm-ar")
set(CMAKE_STRIP "${NDK_TOOLCHAIN_PATH}/bin/llvm-strip")
set(CMAKE_RANLIB "${NDK_TOOLCHAIN_PATH}/bin/llvm-ranlib")
```

**Comprehensive Validation**:
We're adding checks for:
- NDK directory existence
- Toolchain path validity
- Sysroot presence
- Compiler binary availability

**Modern Linker Configuration**:
Adding contemporary Android linker flags:
- `-Wl,--build-id=sha1` for better debugging
- `-Wl,--no-rosegment` for compatibility
- `-Wl,--fatal-warnings` to catch issues early
- `-Wl,--gc-sections` for smaller binaries
- `-Wl,--no-undefined` to prevent symbol issues

## Consequences

### Advantages

**Build Quality**: Clang 18 brings substantial improvements:
- Better optimization passes
- Full C++20 support with preview C++23 features
- Clearer, more actionable error messages
- Faster linking with LLD
- Modern compiler warnings catch more bugs

**Security**: The new toolchain includes:
- Fortify enabled by default
- Enhanced stack protection mechanisms
- Modern exploit mitigations
- No more maintaining patches for the old toolchain

**Development Experience**: 
- More robust error handling in toolchain files
- Better error messages when misconfigured
- Support for multiple NDK installation methods
- Consistent LLVM toolchain reduces surprises

### Disadvantages

**Device Compatibility**: We lose support for approximately 2-3% of Android devices:
- Android 4.4 KitKat and older
- These are primarily older, low-end devices
- Many already struggle with modern ML workloads
- No migration path for these users

**Migration Effort**:
- Update all GitHub Actions workflows
- Modify CMake configurations across the project
- Extensive testing on API 21 devices
- Documentation updates
- Potential issues with third-party dependencies

**Cross-Compilation Complexity**:
- Windows 32-bit builds no longer possible
- May affect some contributor workflows
- Requires all Windows developers to use 64-bit systems

### Implementation Timeline

1. **Phase 1 - Testing** (2 weeks):
   - Update CI/CD pipelines with r27d
   - Run parallel builds with both NDKs
   - Performance comparison

2. **Phase 2 - Migration** (1 week):
   - Switch primary builds to r27d
   - Update all documentation
   - Notify users of minimum API change

3. **Phase 3 - Cleanup** (1 week):
   - Remove r21d from CI/CD
   - Archive old toolchain files
   - Final compatibility testing

## References

- Android NDK r21 Release Notes
- Android NDK r27 Release Notes  
- Android NDK Revision History
- LLVM 18 Release Notes
- Android Platform Version Distribution