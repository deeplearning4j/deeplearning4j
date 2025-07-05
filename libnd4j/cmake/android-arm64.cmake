set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK}")
set(CMAKE_ANDROID_STL_TYPE c++_static)
set(CMAKE_SYSTEM_VERSION 21)
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)

set(CMAKE_C_COMPILER   "$ENV{ANDROID_NDK}/toolchains/llvm/prebuilt/linux-aarch64/bin/aarch64-linux-android21-clang")
set(CMAKE_CXX_COMPILER "$ENV{ANDROID_NDK}/toolchains/llvm/prebuilt/linux-aarch64/bin/aarch64-linux-android21-clang++")

add_definitions(-D__ANDROID_API__=21 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong)