FIND_PATH(CHOLMOD_INCLUDE_DIR NAMES cholmod.h amd.h camd.h
    PATHS
    ${SUITE_SPARSE_ROOT}/include
    /usr/include/suitesparse
    /usr/include/ufsparse
    /opt/local/include/ufsparse
    /usr/local/include/ufsparse
    /sw/include/ufsparse
  )

FIND_LIBRARY(CHOLMOD_LIBRARY NAMES cholmod
     PATHS
     ${SUITE_SPARSE_ROOT}/lib
     /usr/lib
     /usr/local/lib
     /opt/local/lib
     /sw/lib
   )

FIND_LIBRARY(AMD_LIBRARY NAMES SHARED NAMES amd
  PATHS
  ${SUITE_SPARSE_ROOT}/lib
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  )

FIND_LIBRARY(CAMD_LIBRARY NAMES camd
  PATHS
  ${SUITE_SPARSE_ROOT}/lib
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  )

FIND_LIBRARY(SUITESPARSECONFIG_LIBRARY NAMES suitesparseconfig
  PATHS
  ${SUITE_SPARSE_ROOT}/lib
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  )


# Different platforms seemingly require linking against different sets of libraries
IF(CYGWIN)
  FIND_PACKAGE(PkgConfig)
  FIND_LIBRARY(COLAMD_LIBRARY NAMES colamd
    PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    )
  PKG_CHECK_MODULES(LAPACK lapack)

  SET(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} ${AMD_LIBRARY} ${CAMD_LIBRARY} ${COLAMD_LIBRARY} ${CCOLAMD_LIBRARY} ${LAPACK_LIBRARIES})

# MacPorts build of the SparseSuite requires linking against extra libraries

ELSEIF(APPLE)

  FIND_LIBRARY(COLAMD_LIBRARY NAMES colamd
    PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    )

  FIND_LIBRARY(CCOLAMD_LIBRARY NAMES ccolamd
    PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    )

  FIND_LIBRARY(METIS_LIBRARY NAMES metis
    PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    )

  SET(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} ${AMD_LIBRARY} ${CAMD_LIBRARY} ${COLAMD_LIBRARY} ${CCOLAMD_LIBRARY} ${METIS_LIBRARY} "-framework Accelerate")
ELSE(APPLE)
  SET(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} ${AMD_LIBRARY})
ENDIF(CYGWIN)

IF(CHOLMOD_INCLUDE_DIR AND CHOLMOD_LIBRARIES)
  SET(CHOLMOD_FOUND TRUE)
ELSE(CHOLMOD_INCLUDE_DIR AND CHOLMOD_LIBRARIES)
  SET(CHOLMOD_FOUND FALSE)
ENDIF(CHOLMOD_INCLUDE_DIR AND CHOLMOD_LIBRARIES)

# Look for csparse; note the difference in the directory specifications!
FIND_PATH(CSPARSE_INCLUDE_DIR NAMES cs.h
  PATHS
  /usr/include/suitesparse
  /usr/include
  /opt/local/include
  /usr/local/include
  /sw/include
  /usr/include/ufsparse
  /opt/local/include/ufsparse
  /usr/local/include/ufsparse
  /sw/include/ufsparse
  )

FIND_LIBRARY(CSPARSE_LIBRARY NAMES cxsparse
  PATHS
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  )

IF(CSPARSE_INCLUDE_DIR AND CSPARSE_LIBRARY)
  SET(CSPARSE_FOUND TRUE)
ELSE(CSPARSE_INCLUDE_DIR AND CSPARSE_LIBRARY)
  SET(CSPARSE_FOUND FALSE)
ENDIF(CSPARSE_INCLUDE_DIR AND CSPARSE_LIBRARY)
