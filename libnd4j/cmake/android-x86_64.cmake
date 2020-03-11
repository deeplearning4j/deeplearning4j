# CMake toolchain to build for Android 5.0 or newer. Sample usage:

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK}")
set(CMAKE_ANDROID_STL_TYPE c++_shared)
set(CMAKE_SYSTEM_VERSION  "$ENV{ANDROID_VERSION}")
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)

set(ANDROID TRUE)
if (WIN32)
   set(CMAKE_C_COMPILER   "$ENV{ANDROID_CC}.exe")
   set(CMAKE_CXX_COMPILER "$ENV{ANDROID_CC}++.exe")
   else()
   set(CMAKE_C_COMPILER   "$ENV{ANDROID_CC}")
   set(CMAKE_CXX_COMPILER "$ENV{ANDROID_CC}++")
endif (WIN32)



add_definitions(-D__ANDROID_API__=$ENV{ANDROID_VERSION} -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -target aarch64-none-linux-android -march=armv8-a)
