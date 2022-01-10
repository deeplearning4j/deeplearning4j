
message("<FindVEDNN.cmake>")

### Find vednn STATIC libraries

SET (VEDNN_INCLUDE_DIRS 
    /opt/nec/ve/include
    ${VEDNN_ROOT}/include
    $ENV{VEDNN_ROOT}/include
)

SET (VEDNN_LIB_DIRS  
    /opt/nec/ve/lib
    ${VEDNN_ROOT}
    $ENV{VEDNN_ROOT}
    ${VEDNN_ROOT}/lib
    $ENV{VEDNN_ROOT}/lib
)

find_path(VEDNN_INCLUDE vednn.h
            PATHS ${VEDNN_INCLUDE_DIRS}
            NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

find_path(VEDNN_INCLUDE vednn.h)

if (NOT DEFINED VEDNN_LIBRARIES)
 
    find_library(VEDNN_OPENMP NAMES vednn_openmp
                    PATHS ${VEDNN_LIB_DIRS}
                    PATH_SUFFIXES "Release"
                    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH) 
    find_library(VEDNN_OPENMP NAMES vednn_openmp)  

    set(VEDNN_LIBRARIES  ${VEDNN_OPENMP})
endif()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(VEDNN REQUIRED_VARS VEDNN_INCLUDE VEDNN_LIBRARIES)

message("</FindVEDNN.cmake>")