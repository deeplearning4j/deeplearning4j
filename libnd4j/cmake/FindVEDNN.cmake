
message("<Findvednn.cmake>")

### Find vednn STATIC libraries

SET (vednn_INCLUDE_DIRS 
    /opt/nec/ve/include
    ${vednn_ROOT}/include
    $ENV{vednn_ROOT}/include
)


SET (vednn_LIB_DIRS  
    /opt/nec/ve/lib
    ${vednn_ROOT}
    $ENV{vednn_ROOT}
    ${vednn_ROOT}/lib
    $ENV{vednn_ROOT}/lib
)

find_path(VEDNN_INCLUDE vednn.h
            PATHS ${vednn_INCLUDE_DIRS}
            NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

find_path(VEDNN_INCLUDE vednn.h)

if (NOT DEFINED vednn_LIBRARIES)
 
    find_library(vednn_OPENMP NAMES vednn_openmp
                    PATHS ${vednn_LIB_DIRS}
                    PATH_SUFFIXES "Release"
                    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH) 
    find_library(vednn_OPENMP NAMES vednn_openmp)  
    
    set(VEDNN_LIBRARIES  ${vednn_OPENMP})
endif()
 
 
INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(vednn REQUIRED_VARS VEDNN_INCLUDE VEDNN_LIBRARIES)

message("</Findvednn.cmake>")