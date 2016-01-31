if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
endif()

include_directories(../include)
#used for determining if we should include jni components or not
find_package(Java COMPONENTS Development)
if(Java_FOUND)
    if(EXISTS "$ENV{JAVA_HOME}")
        message("INCLUDING JAVA $ENV{JAVA_HOME}include")
        include_directories("$ENV{JAVA_HOME}include" "$ENV{JAVA_HOME}include/linux")
        add_definitions(-D__JNI__=true)

    else()
        message(FATAL_ERROR "Could not find JAVA_HOME environment variable")
    endif(EXISTS "$ENV{JAVA_HOME}")
endif()
SET(BLAS_LIBRARIES "/opt/OpenBLAS/lib")
set(LAPACK_LIBRARIES "/opt/OpenBLAS/lib")
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}" )

add_library(libnd4j SHARED blasimpl.cpp NativeOps.cpp)
target_link_libraries(libnd4j ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})