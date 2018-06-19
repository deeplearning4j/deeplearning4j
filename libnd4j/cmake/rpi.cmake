# CMake toolchain to build libnd4j for Raspberry PI. Sample usage:
#
# PI_BIN=/path/to/raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf \
# cmake -DCMAKE_TOOLCHAIN_FILE=rpi.cmake -DCMAKE_INSTALL_PREFIX=..

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)

set(CMAKE_C_COMPILER   "$ENV{RPI_BIN}-gcc")
set(CMAKE_CXX_COMPILER "$ENV{RPI_BIN}-g++")

SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)