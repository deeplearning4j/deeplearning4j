SET (LAPACKE_INCLUDE lapacke.h lapacke_config.h lapacke_utils.h lapacke_mangling.h)

file(COPY ${LAPACKE_INCLUDE} DESTINATION ${LAPACK_BINARY_DIR}/include)
