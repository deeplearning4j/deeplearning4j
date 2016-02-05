SET (CBLAS_INCLUDE cblas.h cblas_f77.h cblas_test.h cblas_mangling.h)

file(COPY ${CBLAS_INCLUDE} DESTINATION ${LAPACK_BINARY_DIR}/include)
