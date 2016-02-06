#######################################################################
#  This CMakeLists.txt creates the test programs for the CBLAS routines.
#
#######################################################################

macro(add_cblas_test output input target)
  set(TEST_INPUT "${LAPACK_SOURCE_DIR}/cblas/testing/${input}")
  set(TEST_OUTPUT "${LAPACK_BINARY_DIR}/cblas/testing/${output}")
  set(testName "${target}")

  if(EXISTS "${TEST_INPUT}")
    add_test(NAME CBLAS-${testName} COMMAND "${CMAKE_COMMAND}"
      -DTEST=$<TARGET_FILE:${target}>
      -DINPUT=${TEST_INPUT}
      -DOUTPUT=${TEST_OUTPUT} 
      -DINTDIR=${CMAKE_CFG_INTDIR}
      -P "${LAPACK_SOURCE_DIR}/TESTING/runtest.cmake")
    else()
      string(REPLACE "." "_" input_name ${input})
      add_test(NAME CBLAS-${testName} COMMAND "${CMAKE_COMMAND}" 
        -DTEST=$<TARGET_FILE:${target}>
        -DOUTPUT=${TEST_OUTPUT} 
        -DINTDIR=${CMAKE_CFG_INTDIR}
        -P "${LAPACK_SOURCE_DIR}/TESTING/runtest.cmake")
    endif()
endmacro(add_cblas_test)


# Object files for single real precision
SET( STESTL1O c_sblas1.c)

SET( STESTL2O c_sblas2.c c_s2chke.c auxiliary.c c_xerbla.c)
SET( STESTL2O c_sblas2.c c_s2chke.c auxiliary.c c_xerbla.c)
SET( STESTL3O c_sblas3.c c_s3chke.c auxiliary.c c_xerbla.c)

# Object files for double real precision
SET( DTESTL1O  c_dblas1.c)
SET( DTESTL2O  c_dblas2.c c_d2chke.c auxiliary.c c_xerbla.c)
SET( DTESTL3O  c_dblas3.c c_d3chke.c auxiliary.c c_xerbla.c)

# Object files for single complex precision
SET( CTESTL1O  c_cblat1.f c_cblas1.c)
SET( CTESTL2O  c_cblas2.c c_c2chke.c auxiliary.c c_xerbla.c)
SET( CTESTL3O  c_cblas3.c c_c3chke.c auxiliary.c c_xerbla.c)

# Object files for double complex precision
SET( ZTESTL1O  c_zblas1.c)
SET( ZTESTL2O  c_zblas2.c c_z2chke.c auxiliary.c c_xerbla.c)
SET( ZTESTL3O  c_zblas3.c c_z3chke.c auxiliary.c c_xerbla.c)



if(BUILD_SINGLE)
  add_executable(xscblat1 c_sblat1.f ${STESTL1O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  add_executable(xscblat2 c_sblat2.f ${STESTL2O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  add_executable(xscblat3 c_sblat3.f ${STESTL3O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  
  target_link_libraries(xscblat1 cblas ${BLAS_LIBRARIES})
  target_link_libraries(xscblat2 cblas ${BLAS_LIBRARIES})
  target_link_libraries(xscblat3 cblas ${BLAS_LIBRARIES})
 
  add_cblas_test(stest1.out ""   xscblat1)
  add_cblas_test(stest2.out sin2 xscblat2)
  add_cblas_test(stest3.out sin3 xscblat3)
 
endif()

if(BUILD_DOUBLE)

  add_executable(xdcblat1 c_dblat1.f ${DTESTL1O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  add_executable(xdcblat2 c_dblat2.f ${DTESTL2O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  add_executable(xdcblat3 c_dblat3.f ${DTESTL3O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  
  target_link_libraries(xdcblat1 cblas ${BLAS_LIBRARIES})
  target_link_libraries(xdcblat2 cblas ${BLAS_LIBRARIES})
  target_link_libraries(xdcblat3 cblas ${BLAS_LIBRARIES})
 
  add_cblas_test(dtest1.out ""   xdcblat1)
  add_cblas_test(dtest2.out din2 xdcblat2)
  add_cblas_test(dtest3.out din3 xdcblat3)

endif()

if(BUILD_COMPLEX)

  add_executable(xccblat1 c_cblat1.f ${CTESTL1O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  add_executable(xccblat2 c_cblat2.f ${CTESTL2O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  add_executable(xccblat3 c_cblat3.f ${CTESTL3O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  
  target_link_libraries(xccblat1 cblas ${BLAS_LIBRARIES})
  target_link_libraries(xccblat2 cblas ${BLAS_LIBRARIES})
  target_link_libraries(xccblat3 cblas ${BLAS_LIBRARIES})
 
  add_cblas_test(ctest1.out ""   xccblat1)
  add_cblas_test(ctest2.out cin2 xccblat2)
  add_cblas_test(ctest3.out cin3 xccblat3)

endif()

if(BUILD_COMPLEX16)

  add_executable(xzcblat1 c_zblat1.f ${ZTESTL1O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  add_executable(xzcblat2 c_zblat2.f ${ZTESTL2O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  add_executable(xzcblat3 c_zblat3.f ${ZTESTL3O} ${LAPACK_BINARY_DIR}/include/cblas_test.h)
  
  target_link_libraries(xzcblat1 cblas ${BLAS_LIBRARIES})
  target_link_libraries(xzcblat2 cblas ${BLAS_LIBRARIES})
  target_link_libraries(xzcblat3 cblas ${BLAS_LIBRARIES})
 
  add_cblas_test(ztest1.out ""   xzcblat1)
  add_cblas_test(ztest2.out zin2 xzcblat2)
  add_cblas_test(ztest3.out zin3 xzcblat3)

endif()
