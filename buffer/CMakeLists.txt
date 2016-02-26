if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
endif()

include_directories(../include)
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -ffast-math -ftree-vectorize -msse2 -ftree-vectorizer-verbose=2 -fopt-info-vec -fopt-info-vec-missed")
IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

add_library(buffer SHARED NativeBuffer.cpp)
target_link_libraries(buffer)






