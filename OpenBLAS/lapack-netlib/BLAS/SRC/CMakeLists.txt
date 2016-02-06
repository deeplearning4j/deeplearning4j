#######################################################################
#  This is the makefile to create a library for the BLAS.
#  The files are grouped as follows:
#
#       SBLAS1 -- Single precision real BLAS routines
#       CBLAS1 -- Single precision complex BLAS routines
#       DBLAS1 -- Double precision real BLAS routines
#       ZBLAS1 -- Double precision complex BLAS routines
#
#       CB1AUX -- Real BLAS routines called by complex routines
#       ZB1AUX -- D.P. real BLAS routines called by d.p. complex
#                 routines
#
#      ALLBLAS -- Auxiliary routines for Level 2 and 3 BLAS
#
#       SBLAS2 -- Single precision real BLAS2 routines
#       CBLAS2 -- Single precision complex BLAS2 routines
#       DBLAS2 -- Double precision real BLAS2 routines
#       ZBLAS2 -- Double precision complex BLAS2 routines
#
#       SBLAS3 -- Single precision real BLAS3 routines
#       CBLAS3 -- Single precision complex BLAS3 routines
#       DBLAS3 -- Double precision real BLAS3 routines
#       ZBLAS3 -- Double precision complex BLAS3 routines
#
#  The library can be set up to include routines for any combination
#  of the four precisions.  To create or add to the library, enter make
#  followed by one or more of the precisions desired.  Some examples:
#       make single
#       make single complex
#       make single double complex complex16
#  Note that these commands are not safe for parallel builds.
#
#  Alternatively, the commands
#       make all
#  or
#       make
#  without any arguments creates a library of all four precisions.
#  The name of the library is held in BLASLIB, which is set in the
#  top-level make.inc
#
#  To remove the object files after the library is created, enter
#       make clean
#  To force the source files to be recompiled, enter, for example,
#       make single FRC=FRC
#
#---------------------------------------------------------------------
#
#  Edward Anderson, University of Tennessee
#  March 26, 1990
#  Susan Ostrouchov, Last updated September 30, 1994
#  ejr, May 2006.
#
#######################################################################

#---------------------------------------------------------
#  Comment out the next 6 definitions if you already have
#  the Level 1 BLAS.
#---------------------------------------------------------
set(SBLAS1 isamax.f sasum.f saxpy.f scopy.f sdot.f snrm2.f 
	srot.f srotg.f sscal.f sswap.f sdsdot.f srotmg.f srotm.f)

set(CBLAS1 scabs1.f scasum.f scnrm2.f icamax.f caxpy.f ccopy.f 
	cdotc.f cdotu.f csscal.f crotg.f cscal.f cswap.f csrot.f)

set(DBLAS1 idamax.f dasum.f daxpy.f dcopy.f ddot.f dnrm2.f 
	drot.f drotg.f dscal.f dsdot.f dswap.f drotmg.f drotm.f)

set(ZBLAS1 dcabs1.f dzasum.f dznrm2.f izamax.f zaxpy.f zcopy.f 
	zdotc.f zdotu.f zdscal.f zrotg.f zscal.f zswap.f zdrot.f)

set(CB1AUX  isamax.f sasum.f saxpy.f scopy.f snrm2.f sscal.f)

set(ZB1AUX idamax.f dasum.f daxpy.f dcopy.f dnrm2.f dscal.f)

#---------------------------------------------------------------------
#  The following line defines auxiliary routines needed by both the
#  Level 2 and Level 3 BLAS.  Comment it out only if you already have
#  both the Level 2 and 3 BLAS.
#---------------------------------------------------------------------
set(ALLBLAS  lsame.f xerbla.f xerbla_array.f)

#---------------------------------------------------------
#  Comment out the next 4 definitions if you already have
#  the Level 2 BLAS.
#---------------------------------------------------------
set(SBLAS2 sgemv.f sgbmv.f ssymv.f ssbmv.f sspmv.f 
	strmv.f stbmv.f stpmv.f strsv.f stbsv.f stpsv.f 
	sger.f ssyr.f sspr.f ssyr2.f sspr2.f)

set(CBLAS2 cgemv.f cgbmv.f chemv.f chbmv.f chpmv.f 
	ctrmv.f ctbmv.f ctpmv.f ctrsv.f ctbsv.f ctpsv.f 
	cgerc.f cgeru.f cher.f chpr.f cher2.f chpr2.f)

set(DBLAS2 dgemv.f dgbmv.f dsymv.f dsbmv.f dspmv.f 
	dtrmv.f dtbmv.f dtpmv.f dtrsv.f dtbsv.f dtpsv.f 
	dger.f dsyr.f dspr.f dsyr2.f dspr2.f)

set(ZBLAS2 zgemv.f zgbmv.f zhemv.f zhbmv.f zhpmv.f 
	ztrmv.f ztbmv.f ztpmv.f ztrsv.f ztbsv.f ztpsv.f 
	zgerc.f zgeru.f zher.f zhpr.f zher2.f zhpr2.f)

#---------------------------------------------------------
#  Comment out the next 4 definitions if you already have
#  the Level 3 BLAS.
#---------------------------------------------------------
set(SBLAS3 sgemm.f ssymm.f ssyrk.f ssyr2k.f strmm.f strsm.f )

set(CBLAS3 cgemm.f csymm.f csyrk.f csyr2k.f ctrmm.f ctrsm.f 
	chemm.f cherk.f cher2k.f)

set(DBLAS3 dgemm.f dsymm.f dsyrk.f dsyr2k.f dtrmm.f dtrsm.f)

set(ZBLAS3 zgemm.f zsymm.f zsyrk.f zsyr2k.f ztrmm.f ztrsm.f 
	zhemm.f zherk.f zher2k.f)
# default build all of it
set(ALLOBJ ${SBLAS1} ${SBLAS2} ${SBLAS3} ${DBLAS1} ${DBLAS2} ${DBLAS3}	
	${CBLAS1} ${CBLAS2} ${CBLAS3} ${ZBLAS1} 
	${ZBLAS2} ${ZBLAS3} ${ALLBLAS})

if(BLAS_SINGLE)
  set(ALLOBJ ${SBLAS1} ${ALLBLAS} 
	${SBLAS2} ${SBLAS3})
endif()
if(BLAS_DOUBLE)
  set(ALLOBJ ${DBLAS1} ${ALLBLAS} 
	${DBLAS2} ${DBLAS3})
endif()
if(BLAS_COMPLEX)
  set(ALLOBJ  ${BLASLIB} ${CBLAS1} ${CB1AUX} 
	${ALLBLAS} ${CBLAS2})
endif()
if(BLAS_COMPLEX16)
  set(ALLOBJ  ${BLASLIB} ${ZBLAS1} ${ZB1AUX} 
	${ALLBLAS} ${ZBLAS2} ${ZBLAS3})
endif()
  
  
add_library(blas ${ALLOBJ})
#if(UNIX)
#  target_link_libraries(blas m)
#endif()
set_target_properties(
  blas PROPERTIES
  VERSION ${LAPACK_VERSION}
  SOVERSION ${LAPACK_MAJOR_VERSION}
  )
target_link_libraries(blas)
lapack_install_library(blas)
