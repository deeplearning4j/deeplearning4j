# CMake toolchain to build libnd4j for iOS armv7. Sample usage:
#
# cmake -DCMAKE_TOOLCHAIN_FILE=ios-armv7.cmake -DCMAKE_INSTALL_PREFIX=..
#
# If you really need to use libnd4j on a CPU with no FPU, replace "libs/armeabi-v7a" by "libs/armeabi" and
# "-march=armv7-a  with "-march=armv5te -mtune=xscale -msoft-float"

set(CMAKE_SYSTEM_NAME UnixPaths)
set(CMAKE_SYSTEM_PROCESSOR armv)
set(IOS TRUE)
set(CFLAGS, "-miphoneos-version-min=6.0 -arch armv7")


set(CMAKE_C_COMPILER   "clang-omp")
set(CMAKE_CXX_COMPILER "clang-omp")

# set(CMAKE_C_LINK_EXECUTABLE    "<CMAKE_C_COMPILER>  <FLAGS> <CMAKE_C_LINK_FLAGS>   <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> ")
# set(CMAKE_CXX_LINK_EXECUTABLE  "<CMAKE_CXX_COMPILER>  <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> ")
set(CMAKE_C_LINK_EXECUTABLE    "libtool -static <FLAGS> <CMAKE_C_LINK_FLAGS>   <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -L$ENV{IOS_SDK}/usr/lib/ -syslibroot $ENV{IOS_SDK}")
set(CMAKE_CXX_LINK_EXECUTABLE    "libtool -static <FLAGS> <CMAKE_CXX_LINK_FLAGS>   <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -L$ENV{IOS_SDK}/usr/lib/ -syslibroot $ENV{IOS_SDK}")

set(CMAKE_C_CREATE_STATIC_LIBRARY    "libtool  <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_C_FLAG><TARGET_SONAME> -Wl,--no-undefined -z text -o <TARGET> <OBJECTS> <LINK_LIBRARIES> -L$ENV{ANDROID_ROOT}/usr/lib/ --sysroot=$ENV{ANDROID_ROOT}")
# set(CMAKE_CXX_CREATE_STATIC_LIBRARY  "libtool <CMAKE_STATIC_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_STATIC_LIBRARY_CREATE_CXX_FLAGS> <CMAKE_STATIC_LIBRARY_SONAME_CXX_FLAG><TARGET_SONAME>  -Wl,--no-undefined -z text -o <TARGET> <OBJECTS> <LINK_LIBRARIES> -L$ENV{IOS_ROOT}/usr/lib/ --sysroot=$ENV{IOS_ROOT}")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY  "libtool -o <TARGET> <OBJECTS> <LINK_LIBRARIES> -syslibroot $ENV{IOS_SDK} -L$ENV{IOS_SDK}/usr/lib/")

add_definitions("-DIOS -miphoneos-version-min=6.0 -arch armv7 -isysroot $ENV{IOS_SDK} -I/usr/local/Cellar/llvm/4.0.0/include/c++/v1 -I/usr/local/Cellar/llvm/4.0.0/lib/clang/4.0.0/include -fPIC -ffunction-sections -funwind-tables -fstack-protector -fomit-frame-pointer -fstrict-aliasing")

# include_directories("$ENV{ANDROID_CPP}/include/" "$ENV{ANDROID_CPP}/libs/armeabi-v7a/include/" "$ENV{ANDROID_ROOT}/usr/include/")
