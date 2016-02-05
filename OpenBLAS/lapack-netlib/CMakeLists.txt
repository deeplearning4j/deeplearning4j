cmake_minimum_required(VERSION 2.8.10)
project(LAPACK Fortran)

set(LAPACK_MAJOR_VERSION 3)
set(LAPACK_MINOR_VERSION 6)
set(LAPACK_PATCH_VERSION 0)
set(
  LAPACK_VERSION
  ${LAPACK_MAJOR_VERSION}.${LAPACK_MINOR_VERSION}.${LAPACK_PATCH_VERSION}
  )

# Updated OSX RPATH settings
# In response to CMake 3.0 generating warnings regarding policy CMP0042,
# the OSX RPATH settings have been updated per recommendations found
# in the CMake Wiki:
#  http://www.cmake.org/Wiki/CMake_RPATH_handling#Mac_OS_X_and_the_RPATH
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif()


# Configure the warning and code coverage suppression file
configure_file(
  "${LAPACK_SOURCE_DIR}/CTestCustom.cmake.in"
  "${LAPACK_BINARY_DIR}/CTestCustom.cmake"
  COPYONLY
)

# Add the CMake directory for custon CMake modules
set(CMAKE_MODULE_PATH "${LAPACK_SOURCE_DIR}/CMAKE" ${CMAKE_MODULE_PATH})

if (UNIX)
   if ( "${CMAKE_Fortran_COMPILER}" MATCHES "ifort" )
  set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fltconsistency -fp_port" )
   endif ()
   if ( "${CMAKE_Fortran_COMPILER}" MATCHES "xlf" )
  set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -qnosave -qstrict=none" )
   endif ()
# Delete libmtsk in linking sequence for Sun/Oracle Fortran Compiler.
# This library is not present in the Sun package SolarisStudio12.3-linux-x86-bin
   STRING(REPLACE \;mtsk\; \; CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")
endif ()

# Get Python
find_package(PythonInterp)
message(STATUS "Looking for Python found - ${PYTHONINTERP_FOUND}")
if (PYTHONINTERP_FOUND)
   message(STATUS "Using Python version ${PYTHON_VERSION_STRING}")
endif()
# --------------------------------------------------

set(LAPACK_INSTALL_EXPORT_NAME lapack-targets)

if (UNIX)
  include(GNUInstallDirs)
  set(ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR})
  set(LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR})
  set(RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
else()
  set(ARCHIVE_DIR lib${LIB_SUFFIX})
  set(LIBRARY_DIR lib${LIB_SUFFIX})
  set(RUNTIME_DIR bin)
endif()

macro(lapack_install_library lib)
  install(TARGETS ${lib}
    EXPORT ${LAPACK_INSTALL_EXPORT_NAME}
    ARCHIVE DESTINATION ${ARCHIVE_DIR}
    LIBRARY DESTINATION ${LIBRARY_DIR}
    RUNTIME DESTINATION ${RUNTIME_DIR}
  )
endmacro()

# --------------------------------------------------
# Testing

enable_testing()
include(CTest)
enable_testing()
# --------------------------------------------------

# Organize output files.  On Windows this also keeps .dll files next
# to the .exe files that need them, making tests easy to run.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LAPACK_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LAPACK_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LAPACK_BINARY_DIR}/lib)

# --------------------------------------------------
# Check for any necessary platform specific compiler flags
include( CheckLAPACKCompilerFlags )
CheckLAPACKCompilerFlags()

# --------------------------------------------------
# Check second function

include(CheckTimeFunction)
set(TIME_FUNC NONE ${TIME_FUNC})
CHECK_TIME_FUNCTION(NONE TIME_FUNC)
CHECK_TIME_FUNCTION(INT_CPU_TIME TIME_FUNC)
CHECK_TIME_FUNCTION(EXT_ETIME TIME_FUNC)
CHECK_TIME_FUNCTION(EXT_ETIME_ TIME_FUNC)
CHECK_TIME_FUNCTION(INT_ETIME TIME_FUNC)
message(STATUS "--> Will use second_${TIME_FUNC}.f and dsecnd_${TIME_FUNC}.f as timing function.")

set(SECOND_SRC  ${LAPACK_SOURCE_DIR}/INSTALL/second_${TIME_FUNC}.f)
set(DSECOND_SRC  ${LAPACK_SOURCE_DIR}/INSTALL/dsecnd_${TIME_FUNC}.f)
set(PKG_CONFIG_DIR ${LIBRARY_DIR}/pkgconfig)

# --------------------------------------------------
# Precision to build
# By default all precisions are generated


# --------------------------------------------------
# Subdirectories that need to be processed

option(USE_OPTIMIZED_BLAS "Whether or not to use an optimized BLAS library instead of included netlib BLAS" OFF)


# Check the usage of the user provided BLAS libraries
if(BLAS_LIBRARIES)
  include(CheckFortranFunctionExists)
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  CHECK_FORTRAN_FUNCTION_EXISTS("dgemm" BLAS_FOUND)
  unset( CMAKE_REQUIRED_LIBRARIES )
  if(BLAS_FOUND)
    message(STATUS "--> BLAS supplied by user is WORKING, will use ${BLAS_LIBRARIES}.")
  else(BLAS_FOUND)
    message(ERROR "--> BLAS supplied by user is not WORKING, CANNOT USE ${BLAS_LIBRARIES}.")
    message(ERROR "-->     Will use REFERENCE BLAS (by default)")
    message(ERROR "-->     Or Correct your BLAS_LIBRARIES entry ")
    message(ERROR "-->     Or Consider checking USE_OPTIMIZED_BLAS")
  endif(BLAS_FOUND)

# User did not provide a BLAS Library but specified to search for one
elseif( USE_OPTIMIZED_BLAS )
  find_package( BLAS )
endif (BLAS_LIBRARIES)

# Neither user specified or optimized BLAS libraries can be used
if(NOT BLAS_FOUND)
  message(STATUS "Using supplied NETLIB BLAS implementation")
  add_subdirectory(BLAS)
  set( BLAS_LIBRARIES blas )
else()
  set( CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}" 
    CACHE STRING "Linker flags for executables" FORCE)
  set( CMAKE_MODULE_LINKER_FLAGS 
    "${CMAKE_MODULE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}" 
    CACHE STRING "Linker flags for modules" FORCE)
  set( CMAKE_SHARED_LINKER_FLAGS 
    "${CMAKE_SHARED_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}" 
    CACHE STRING "Linker flags for shared libs" FORCE)
endif( NOT BLAS_FOUND )


# --------------------------------------------------
# CBLAS
option(CBLAS "Build CBLAS" OFF)

if(CBLAS)
  add_subdirectory(CBLAS)
endif(CBLAS)

# --------------------------------------------------
# XBLAS

option(USE_XBLAS "Build extended precision (needs XBLAS)" OFF)
if (USE_XBLAS)
  find_library(XBLAS_LIBRARY NAMES xblas)
endif(USE_XBLAS)
   
option(USE_OPTIMIZED_LAPACK "Whether or not to use an optimized LAPACK library instead of included netlib LAPACK" OFF)

# --------------------------------------------------
# LAPACK
# User did not provide a LAPACK Library but specified to search for one
if( USE_OPTIMIZED_LAPACK )
  find_package( LAPACK )
endif (USE_OPTIMIZED_LAPACK)

# Check the usage of the user provided or automatically found LAPACK libraries
if(LAPACK_LIBRARIES)
  include(CheckFortranFunctionExists)
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
  # Check if new routine of 3.4.0 is in LAPACK_LIBRARIES
  CHECK_FORTRAN_FUNCTION_EXISTS("dgeqrt" LATESTLAPACK_FOUND)
  unset( CMAKE_REQUIRED_LIBRARIES )
  if(LATESTLAPACK_FOUND)
    message(STATUS "--> LAPACK supplied by user is WORKING, will use ${LAPACK_LIBRARIES}.")
  else(LAPACK_FOUND)
    message(ERROR "--> LAPACK supplied by user is not WORKING or is older than LAPACK 3.4.0, 	CANNOT USE ${LAPACK_LIBRARIES}.")
    message(ERROR "-->     Will use REFERENCE LAPACK (by default)")
    message(ERROR "-->     Or Correct your LAPACK_LIBRARIES entry ")
    message(ERROR "-->     Or Consider checking USE_OPTIMIZED_LAPACK")
  endif(LATESTLAPACK_FOUND)
endif (LAPACK_LIBRARIES)

# Neither user specified or optimized LAPACK libraries can be used
if(NOT LATESTLAPACK_FOUND)
  message(STATUS "Using supplied NETLIB LAPACK implementation")
  set( LAPACK_LIBRARIES lapack )
  option(BUILD_SINGLE "Build LAPACK Single Precision" ON)
  option(BUILD_DOUBLE "Build LAPACK Double Precision" ON)
  option(BUILD_COMPLEX "Build LAPACK Complex Precision" ON)
  option(BUILD_COMPLEX16 "Build LAPACK Double Complex Precision" ON)
  add_subdirectory(SRC)
else()
  set( CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}" 
    CACHE STRING "Linker flags for executables" FORCE)
  set( CMAKE_MODULE_LINKER_FLAGS 
    "${CMAKE_MODULE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}" 
    CACHE STRING "Linker flags for modules" FORCE)
  set( CMAKE_SHARED_LINKER_FLAGS 
    "${CMAKE_SHARED_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}" 
    CACHE STRING "Linker flags for shared libs" FORCE)
endif( NOT LATESTLAPACK_FOUND )

message(STATUS "BUILD TESTING : ${BUILD_TESTING}" )
if(BUILD_TESTING)
  add_subdirectory(TESTING)
endif(BUILD_TESTING)

# deprecated LAPACK routines
option(BUILD_DEPRECATED "Build deprecated routines" OFF)

# --------------------------------------------------
# LAPACKE
option(LAPACKE "Build LAPACKE" OFF)

# LAPACKE has also the interface to some routines from tmglib,
# if LAPACKE_WITH_TMG is selected, we need to add those routines to LAPACKE
option(LAPACKE_WITH_TMG "Build LAPACKE with tmglib routines" OFF)
if (LAPACKE_WITH_TMG)
  set(LAPACKE ON)
  if(NOT BUILD_TESTING)
     add_subdirectory(TESTING/MATGEN)
   endif(NOT BUILD_TESTING)
endif(LAPACKE_WITH_TMG)

if(LAPACKE)
  add_subdirectory(LAPACKE)
endif(LAPACKE)

# --------------------------------------------------
# CPACK Packaging 

SET(CPACK_PACKAGE_NAME "LAPACK")
SET(CPACK_PACKAGE_VENDOR "University of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "LAPACK- Linear Algebra Package")
set(CPACK_PACKAGE_VERSION_MAJOR 3)
set(CPACK_PACKAGE_VERSION_MINOR 5)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "LAPACK")
IF(WIN32 AND NOT UNIX)
  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
  SET(CPACK_NSIS_HELP_LINK "http:\\\\\\\\http://icl.cs.utk.edu/lapack-forum")
  SET(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\www.netlib.org/lapack")
  SET(CPACK_NSIS_CONTACT "lapack@eecs.utk.edu")
  SET(CPACK_NSIS_MODIFY_PATH ON)
  SET(CPACK_NSIS_DISPLAY_NAME "LAPACK-${LAPACK_VERSION}")
  set(CPACK_PACKAGE_RELOCATABLE "true")
ELSE(WIN32 AND NOT UNIX)
  SET(CPACK_GENERATOR "TGZ")
  SET(CPACK_SOURCE_GENERATOR TGZ)
  SET(CPACK_SOURCE_PACKAGE_FILE_NAME "lapack-${LAPACK_VERSION}" )
  SET(CPACK_SOURCE_IGNORE_FILES ~$ .svn ${CPACK_SOURCE_IGNORE_FILES} )
ENDIF(WIN32 AND NOT UNIX)
INCLUDE(CPack)


# --------------------------------------------------
# By default static library
OPTION(BUILD_SHARED_LIBS "Build shared libraries" OFF )
OPTION(BUILD_STATIC_LIBS "Build static libraries" ON )
#OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON )

if(NOT BLAS_FOUND)
  set(ALL_TARGETS ${ALL_TARGETS} blas)
endif(NOT BLAS_FOUND)

if(NOT LATESTLAPACK_FOUND)
  set(ALL_TARGETS ${ALL_TARGETS} lapack)
endif(NOT LATESTLAPACK_FOUND)

if(BUILD_TESTING OR LAPACKE_WITH_TMG)
  set(ALL_TARGETS ${ALL_TARGETS} tmglib)
endif(BUILD_TESTING OR LAPACKE_WITH_TMG)

# Export lapack targets, not including lapacke, from the
# install tree, if any.
set(_lapack_config_install_guard_target "")
if(ALL_TARGETS)
  install(EXPORT lapack-targets
    DESTINATION ${LIBRARY_DIR}/cmake/lapack-${LAPACK_VERSION})

  # Choose one of the lapack targets to use as a guard for
  # lapack-config.cmake to load targets from the install tree.
  list(GET ALL_TARGETS 0 _lapack_config_install_guard_target)
endif()

# Include cblas in targets exported from the build tree.
if(CBLAS)
  set(ALL_TARGETS ${ALL_TARGETS} cblas)
endif(CBLAS)

# Include lapacke in targets exported from the build tree.
if(LAPACKE)
  set(ALL_TARGETS ${ALL_TARGETS} lapacke)
endif(LAPACKE)

# Export lapack and lapacke targets from the build tree, if any.
set(_lapack_config_build_guard_target "")
if(ALL_TARGETS)
  export(TARGETS ${ALL_TARGETS} FILE lapack-targets.cmake)

  # Choose one of the lapack or lapacke targets to use as a guard
  # for lapack-config.cmake to load targets from the build tree.
  list(GET ALL_TARGETS 0 _lapack_config_build_guard_target)
endif()

configure_file(${LAPACK_SOURCE_DIR}/CMAKE/lapack-config-build.cmake.in
  ${LAPACK_BINARY_DIR}/lapack-config.cmake @ONLY)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lapack.pc.in ${CMAKE_CURRENT_BINARY_DIR}/lapack.pc)
  install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lapack.pc
  DESTINATION ${PKG_CONFIG_DIR}
   )

configure_file(${LAPACK_SOURCE_DIR}/CMAKE/lapack-config-install.cmake.in
  ${LAPACK_BINARY_DIR}/CMakeFiles/lapack-config.cmake @ONLY)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${LAPACK_BINARY_DIR}/lapack-config-version.cmake
  VERSION ${LAPACK_VERSION}
  COMPATIBILITY SameMajorVersion
  )

install(FILES
  ${LAPACK_BINARY_DIR}/CMakeFiles/lapack-config.cmake
  ${LAPACK_BINARY_DIR}/lapack-config-version.cmake
  DESTINATION ${LIBRARY_DIR}/cmake/lapack-${LAPACK_VERSION}
  )
