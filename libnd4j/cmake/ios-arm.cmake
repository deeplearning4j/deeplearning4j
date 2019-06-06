# CMake toolchain to build libnd4j for iOS for 32-bit architecture cpu armv7. Sample usage:
#
# cmake -DCMAKE_TOOLCHAIN_FILE=ios-arm.cmake -DCMAKE_INSTALL_PREFIX=..
#

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR armv7)
set(IOS TRUE)
set(CFLAGS, "-miphoneos-version-min=6.0 -arch armv7")
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang")
set(CMAKE_C_LINK_EXECUTABLE "libtool -static <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -L$ENV{IOS_SDK}/usr/lib/ -syslibroot $ENV{IOS_SDK}")
set(CMAKE_CXX_LINK_EXECUTABLE "libtool -static <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -L$ENV{IOS_SDK}/usr/lib/ -syslibroot $ENV{IOS_SDK}")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "libtool -o <TARGET> <OBJECTS> <LINK_LIBRARIES> -syslibroot $ENV{IOS_SDK} -L$ENV{IOS_SDK}/usr/lib/")
add_definitions("-DIOS -stdlib=libc++ -miphoneos-version-min=6.0 -arch armv7 -isysroot $ENV{IOS_SDK} -I/usr/local/opt/llvm/4.0.0/include/c++/v1 -I/usr/local/opt/llvm/4.0.0/lib/clang/4.0.0/include -fPIC -ffunction-sections -funwind-tables -fstack-protector -fomit-frame-pointer -fstrict-aliasing")
