if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
endif()


if(APPLE)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
endif()
include_directories(../include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -ftree-vectorize -msse2")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -ftree-vectorizer-verbose=2 -fopt-info-vec -fopt-info-vec-missed")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -ffast-math -ftree-vectorize -msse2 -ftree-vectorizer-verbose=2 -fopt-info-vec -fopt-info-vec-missed")

    # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
endif()
IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

add_library(buffer SHARED NativeBuffer.cpp)
target_link_libraries(buffer)
message("Installing buffer in  $ENV{ND4J_HOME}/nd4j-buffer/src/main/resources")
install(TARGETS buffer DESTINATION  $ENV{ND4J_HOME}/nd4j-buffer/src/main/resources)






