# CMake toolchain to build libnd4j for Android 4.0 or newer. Sample usage:
#
# ANDROID_BIN="/path/to/android-ndk/toolchains/x86_64-4.9/prebuilt/linux-x86_64/" \
# ANDROID_CPP="/path/to/android-ndk/sources/cxx-stl/llvm-libc++/" \
# ANDROID_LLVM="/path/to/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/" \
# ANDROID_ROOT="/path/to/android-ndk/platforms/android-21/arch-x86_64/" \
# cmake -DCMAKE_TOOLCHAIN_FILE=android-x86_64.cmake -DCMAKE_INSTALL_PREFIX=..

set(CMAKE_SYSTEM_NAME UnixPaths)
set(CMAKE_SYSTEM_PROCESSOR atom64)
set(ANDROID TRUE)

set(CMAKE_C_COMPILER   "$ENV{ANDROID_LLVM}/bin/clang")
set(CMAKE_CXX_COMPILER "$ENV{ANDROID_LLVM}/bin/clang++")

set(CMAKE_C_LINK_EXECUTABLE    "<CMAKE_C_COMPILER>   <FLAGS> <CMAKE_C_LINK_FLAGS>   <LINK_FLAGS> <OBJECTS> -target x86_64-none-linux-android -Wl,--no-undefined -z text -o <TARGET> <LINK_LIBRARIES> -gcc-toolchain $ENV{ANDROID_BIN} --sysroot=$ENV{ANDROID_ROOT} -lc -lm")
set(CMAKE_CXX_LINK_EXECUTABLE  "<CMAKE_CXX_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -target x86_64-none-linux-android -Wl,--no-undefined -z text -o <TARGET> <LINK_LIBRARIES> -gcc-toolchain $ENV{ANDROID_BIN} --sysroot=$ENV{ANDROID_ROOT} -L$ENV{ANDROID_CPP}/libs/x86_64/ -static-libstdc++ -lc++_static -lc++abi -landroid_support -lc -lm")

set(CMAKE_C_CREATE_SHARED_LIBRARY    "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_C_FLAG><TARGET_SONAME> -target x86_64-none-linux-android -Wl,--no-undefined -z text -o <TARGET> <OBJECTS> <LINK_LIBRARIES> -gcc-toolchain $ENV{ANDROID_BIN} --sysroot=$ENV{ANDROID_ROOT} -lc -lm")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY  "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG><TARGET_SONAME> -target x86_64-none-linux-android -Wl,--no-undefined -z text -o <TARGET> <OBJECTS> <LINK_LIBRARIES> -gcc-toolchain $ENV{ANDROID_BIN} --sysroot=$ENV{ANDROID_ROOT} -L$ENV{ANDROID_CPP}/libs/x86_64/ -static-libstdc++ -lc++_static -lc++abi -landroid_support -lc -lm")

add_definitions(-D__ANDROID_API__=21 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -target x86_64-none-linux-android -march=x86-64 -mtune=atom)

include_directories("$ENV{ANDROID_CPP}/include/" "$ENV{ANDROID_CPP}/../llvm-libc++abi/include/" "$ENV{ANDROID_NDK}/sources/android/support/include/" "$ENV{ANDROID_CPP}/libs/x86_64/include/" "$ENV{ANDROID_NDK}/sysroot/usr/include/" "$ENV{ANDROID_NDK}/sysroot/usr/include/x86_64-linux-android/" "$ENV{ANDROID_ROOT}/usr/include/")
