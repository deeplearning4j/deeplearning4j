include_directories(tests)
if(LINUX)
    link_directories(/usr/local/lib)
endif()


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

FIND_PACKAGE(BLAS)
FIND_PACKAGE(LAPACK)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}" )
add_library(libnd4j blasimpl.cpp)
