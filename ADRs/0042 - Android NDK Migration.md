# ADR-0042: Android NDK r21d to r27d Migration

## Status

Proposed

Proposed by: Adam Gibson (27-09-2025)

## Context

Android NDK r21 was released in October 2019 as the first LTS release.
It uses Clang r365631 (based on LLVM 9).
NDK r27 was released in July 2024 using Clang r522817d (based on LLVM 18).
This represents a 5-year gap and 9 major LLVM versions.

Major changes between r21d and r27d:
- Minimum API level increased from 16 to 21
- GNU binutils completely removed (as, ld, ar, strip)
- GDB removed in favor of LLDB only
- GNU Make upgraded from 4.2 to 4.4.1
- LLD is now the only linker option
- 32-bit Windows host support dropped

KitKat (APIs 19-20) support removed in r26.
Jelly Bean (APIs 16-18) support removed in r24.
This means Android 5.0 Lollipop (API 21) is now the minimum.

## Decision

Upgrade to Android NDK r27d for all Android builds.
Accept the increased minimum API requirement.
Update build scripts to use LLVM tooling exclusively.
Modernize cross-compilation toolchain files.

### Technical Changes

**Toolchain Updates**
- Replace GNU as with integrated Clang assembler
- Remove any `-fno-integrated-as` flags
- Use `llvm-ar` instead of `ar`
- Use `llvm-strip` instead of `strip`

**Build System Updates**
- Update CMAKE_TOOLCHAIN_FILE paths
- Remove any GDB debugging configurations
- Update minimum API to 21 in all builds

**Cross-compilation Changes**
- Windows 32-bit builds no longer possible
- Updated sysroot structures
- New compiler flags and optimizations

### CMake Toolchain File Modernization

**Path Detection**
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

**Toolchain Specification**
```cmake
# Before - Environment variable compiler
set(CMAKE_C_COMPILER "$ENV{ANDROID_CC}")

# After - Explicit LLVM toolchain
set(CMAKE_C_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang")
set(CMAKE_AR "${NDK_TOOLCHAIN_PATH}/bin/llvm-ar")
set(CMAKE_STRIP "${NDK_TOOLCHAIN_PATH}/bin/llvm-strip")
set(CMAKE_RANLIB "${NDK_TOOLCHAIN_PATH}/bin/llvm-ranlib")
```

**Error Checking**
Added validation for:
- NDK directory existence
- Toolchain path existence
- Sysroot existence
- Compiler binary existence

**API Level Configuration**
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

**Linker Flags**
Added modern Android linker flags:
- `-Wl,--build-id=sha1`
- `-Wl,--no-rosegment`
- `-Wl,--fatal-warnings`
- `-Wl,--gc-sections`
- `-Wl,--no-undefined`

## Consequences

### Build Improvements
- Clang 18 provides better optimization
- Improved C++20/23 support
- Better error messages
- Faster compilation with LLD

### Compatibility Impact
- Android 4.4 KitKat devices unsupported
- Approximately 2-3% of Android devices affected
- These are primarily older, low-end devices
- Already struggle with modern ML workloads

### Security Benefits
- Fortify enabled by default
- Better stack protection
- Modern compiler security features
- No longer maintaining old toolchain patches

### Migration Effort
- Update all GitHub Actions workflows
- Modify CMake configurations
- Test on minimum API devices
- Update documentation

### Cross-Compilation Improvements
- More robust error handling
- Better error messages when misconfigured
- Supports multiple NDK installation methods
- Enforces LLVM toolchain usage

## Implementation

Change workflow configurations:
```yaml
# Before
- uses: ndk/setup-ndk@v1
  with:
    ndk-version: r21d

# After
- uses: ndk/setup-ndk@v1
  with:
    ndk-version: r27d
```

Update CMake minimum API:
```cmake
# Before
set(ANDROID_NATIVE_API_LEVEL 16)

# After
set(ANDROID_NATIVE_API_LEVEL 21)
```

## References
- Android NDK r21 release notes
- Android NDK Revision History
- LLVM 18 release notes
- Android cross-compilation documentation