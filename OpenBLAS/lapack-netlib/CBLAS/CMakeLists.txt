message(STATUS "CBLAS enable")
enable_language(C)

set(LAPACK_INSTALL_EXPORT_NAME cblas-targets)

# Create a header file cblas.h for the routines called in my C programs
include(FortranCInterface)
FortranCInterface_HEADER( ${CMAKE_CURRENT_SOURCE_DIR}/include/cblas_mangling.h 
                          MACRO_NAMESPACE "F77_"
                          SYMBOL_NAMESPACE "F77_" )

# Old way to detect mangling
#include(FortranMangling)
#FORTRAN_MANGLING(CDEFS)
#set(CDEFS ${CDEFS} CACHE STRING "Fortran Mangling" FORCE)
#MESSAGE(STATUS "=========")

# --------------------------------------------------
# Compiler Flags
#ADD_DEFINITIONS( "-D${CDEFS}")


include_directories( include )
add_subdirectory(include)
add_subdirectory(src)

macro(append_subdir_files variable dirname)
get_directory_property(holder DIRECTORY ${dirname} DEFINITION ${variable})
foreach(depfile ${holder})
  list(APPEND ${variable} "${dirname}/${depfile}")
endforeach()
endmacro()

append_subdir_files(CBLAS_INCLUDE "include")
INSTALL( FILES ${CBLAS_INCLUDE} DESTINATION include )

# --------------------------------------------------
if(BUILD_TESTING)
   add_subdirectory(testing)
   add_subdirectory(examples)
endif(BUILD_TESTING)

if(NOT BLAS_FOUND)
  set(ALL_TARGETS ${ALL_TARGETS} blas)
endif(NOT BLAS_FOUND)

# Export cblas targets from the
# install tree, if any.
set(_cblas_config_install_guard_target "")
if(ALL_TARGETS)
  install(EXPORT cblas-targets
    DESTINATION lib/cmake/cblas-${LAPACK_VERSION})
  # Choose one of the cblas targets to use as a guard for
  # cblas-config.cmake to load targets from the install tree.
  list(GET ALL_TARGETS 0 _cblas_config_install_guard_target)
endif()

# Export cblas targets from the build tree, if any.
set(_cblas_config_build_guard_target "")
if(ALL_TARGETS)
  export(TARGETS ${ALL_TARGETS} FILE cblas-targets.cmake)

  # Choose one of the cblas targets to use as a guard
  # for cblas-config.cmake to load targets from the build tree.
  list(GET ALL_TARGETS 0 _cblas_config_build_guard_target)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMAKE/cblas-config-version.cmake.in
  ${LAPACK_BINARY_DIR}/cblas-config-version.cmake @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMAKE/cblas-config-build.cmake.in
  ${LAPACK_BINARY_DIR}/cblas-config.cmake @ONLY)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cblas.pc.in ${CMAKE_CURRENT_BINARY_DIR}/cblas.pc)
  install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cblas.pc
  DESTINATION ${PKG_CONFIG_DIR}
   )

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cblas-config-install.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/cblas-config.cmake @ONLY)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/cblas-config.cmake
  ${LAPACK_BINARY_DIR}/cblas-config-version.cmake
  DESTINATION lib/cmake/cblas-${LAPACK_VERSION}
  )

#install(EXPORT cblas-targets
#  DESTINATION lib/cmake/cblas-${LAPACK_VERSION})

