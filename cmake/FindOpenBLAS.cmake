if(WIN32)
    if(DEFINED ENV{MINGW_HOME})
        set(MINGW_HOME ENV{MINGW_HOME})
        else()
          message("NO MINGW HOME DEFINED USING C:\\MinGW")
        set(MINGW_HOME C:/MinGW)
    endif()

    if(DEFINED ENV{MSYS_HOME})
        set(MSYS_HOME ENV{MSYS_HOME})
    else()
        message("NO MSYS HOME DEFINED USING ${MINGW_HOME}/msys")
        set(MSYS_HOME ${MINGW_HOME}/msys)
    endif()

    if(DEFINED ENV{MSYS_HOME})
        set(MSYS_HOME ENV{MSYS_HOME})
    else()
        message("NO MSYS HOME DEFINED USING ${MINGW_HOME}/msys")
        set(MSYS_HOME ${MINGW_HOME}/msys)
    endif()


    if(DEFINED ENV{MSYS_VERSION})
        set(MSYS_VERSION ENV{MSYS_VERSION})
    else()
        message("NO MSYS VERSION DEFINED USING 1.0")
        set(MSYS_VERSION 1.0)
    endif()


    if(DEFINED ENV{MSYS_ROOT})
        set(MSYS_ROOT ENV{MSYS_ROOT})
    else()
        message("NO MSYS ROOT DEFINED USING ${MSYS_HOME}/${MSYS_VERSION}")
        set(MSYS_ROOT ${MSYS_HOME}/${MSYS_VERSION})
    endif()
    SET(Open_BLAS_INCLUDE_SEARCH_PATHS
            ${MSYS_ROOT}/usr/include
            ${MSYS_ROOT}/usr/include/openblas
            ${MSYS_ROOT}/usr/include/openblas-base
            ${MSYS_ROOT}/usr/local/include
            ${MSYS_ROOT}/usr/local/include/openblas
            ${MSYS_ROOT}/usr/local/include/openblas-base
            ${MSYS_ROOT}/opt/OpenBLAS/include
            ${MSYS_ROOT}$ENV{OpenBLAS_HOME}
            ${MSYS_ROOT}$ENV{OpenBLAS_HOME}/include
            )

    SET(Open_BLAS_LIB_SEARCH_PATHS
            ${MSYS_ROOT}/lib/
            ${MSYS_ROOT}/lib/openblas-base
            ${MSYS_ROOT}/lib64/
            ${MSYS_ROOT}/usr/lib
            ${MSYS_ROOT}/usr/lib/openblas-base
            ${MSYS_ROOT}/usr/lib64
            ${MSYS_ROOT}/usr/local/lib
            ${MSYS_ROOT}/usr/local/lib64
            ${MSYS_ROOT}/opt/OpenBLAS/lib
            ${MSYS_ROOT}/$ENV{OpenBLAS}cd
            ${MSYS_ROOT}/$ENV{OpenBLAS}/lib
#           ${MSYS_ROOT}/$ENV{OpenBLAS_HOME}
            ${MSYS_ROOT}/$ENV{OpenBLAS_HOME}/lib
            )
    else()
    SET(Open_BLAS_INCLUDE_SEARCH_PATHS
            /usr/include
            /usr/include/openblas
            /usr/include/openblas-base
            /usr/local/include
            /usr/local/include/openblas
            /usr/local/include/openblas-base
            /opt/OpenBLAS/include
            $ENV{OpenBLAS_HOME}
            $ENV{OpenBLAS_HOME}/include
            )

    SET(Open_BLAS_LIB_SEARCH_PATHS
            /lib/
            /lib/openblas-base
            /lib64/
            /usr/lib
            /usr/lib/openblas-base
            /usr/lib64
            /usr/local/lib
            /usr/local/lib64
            /opt/OpenBLAS/lib
            $ENV{OpenBLAS}cd
            $ENV{OpenBLAS}/lib
            $ENV{OpenBLAS_HOME}
            $ENV{OpenBLAS_HOME}/lib
            )
endif()
message("SEARCHING INCLUDE  ${Open_BLAS_INCLUDE_SEARCH_PATHS} and LIB ${Open_BLAS_LIB_SEARCH_PATHS}a")

FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
    IF (NOT OpenBLAS_FIND_QUIETLY)
        MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
        MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
    ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
    IF (OpenBLAS_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
    ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
        OpenBLAS_INCLUDE_DIR
        OpenBLAS_LIB
        OpenBLAS
)
