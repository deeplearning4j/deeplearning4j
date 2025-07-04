# android-arm64.cmake - Modern and simplified toolchain for NDK r27b+

# 1. Activate CMake's native Android support. This is the most important step.
set(CMAKE_SYSTEM_NAME Android)

# 2. Specify the target architecture.
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)

# 3. Specify the target Android API level.
#    CMake will use this to select the correct compilers and libraries.
#    Default to 21 if the ANDROID_VERSION environment variable isn't set.
if(NOT DEFINED CMAKE_ANDROID_API AND DEFINED ENV{ANDROID_VERSION})
   set(CMAKE_ANDROID_API $ENV{ANDROID_VERSION})
else()
   set(CMAKE_ANDROID_API 21)
endif()

# 4. Point CMake to your NDK installation.
#    CMake will automatically find the toolchain files within the NDK.
set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_ROOT})

# 5. Select the C++ standard library.
set(CMAKE_ANDROID_STL_TYPE c++_shared)

