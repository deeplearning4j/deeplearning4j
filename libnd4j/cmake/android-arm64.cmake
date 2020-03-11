# CMake toolchain to build for Android 5.0 or newer. Sample usage:
#
# ANDROID_BIN="/path/to/android-ndk/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/" \
# ANDROID_CPP="/path/to/android-ndk/sources/cxx-stl/llvm-libc++/" \
# ANDROID_CC="/path/to/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" \
# ANDROID_ROOT="/path/to/android-ndk/platforms/android-21/arch-arm64/" \
# cmake -DCMAKE_TOOLCHAIN_FILE=android-arm64.cmake -DCMAKE_INSTALL_PREFIX=..

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
# /c/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669//sysroot/usr/lib/aarch64-linux-android/
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -v")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")



add_definitions(-D__ANDROID_API__=$ENV{ANDROID_VERSION} -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -target aarch64-none-linux-android -march=armv8-a)

#

# Probably delete
#
#
#
#

#include_directories("$ENV{ANDROID_NDK}/toolchains/llvm/prebuilt/$KERNEL/lib64/clang/9.0.8/include/")
#include_directories("$ENV{ANDROID_NDK}/sysroot/usr/include/linux" )
#include_directories("$ENV{ANDROID_NDK}/sources/cxx-stl/llvm-libc++/include/")
#include_directories( "$ENV{ANDROID_CPP}/include/")
#include_directories( "$ENV{ANDROID_CPP}/../llvm-libc++abi/include/")
#include_directories( "$ENV{ANDROID_NDK}/sysroot/usr/include/aarch64-linux-android/")
#include_directories(  "$ENV{ANDROID_NDK}/sources/android/support/include/")
#include_directories("$ENV{ANDROID_CPP}/libs/arm64-v8a/include/")
# include_directories(  "$ENV{ANDROID_NDK}/sysroot/usr/include/" )

#C:/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669/toolchains/aarch64-linux-android-4.9/prebuilt/windows-x86_64/aarch64-linux-android/bin/ld.exe
#C:/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669/toolchains/arm-linux-androideabi-4.9/prebuilt/windows-x86_64/arm-linux-androideabi/bin/ld.exe
#C:/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/windows-x86_64/aarch64-linux-android/bin/ld.exe
#C:/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/windows-x86_64/arm-linux-androideabi/bin/ld.exe
#C:/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/windows-x86_64/i686-linux-android/bin/ld.exe
#C:/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/windows-x86_64/x86_64-linux-android/bin/ld.exe
#C:/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669/toolchains/x86-4.9/prebuilt/windows-x86_64/i686-linux-android/bin/ld.exe
#C:/Users/agibs/AppData/Local/Android/Sdk/ndk/21.0.6113669/toolchains/x86_64-4.9/prebuilt/windows-x86_64/x86_64-linux-android/bin/ld.exe


#include_directories( "$ENV{ANDROID_ROOT}/usr/include/")