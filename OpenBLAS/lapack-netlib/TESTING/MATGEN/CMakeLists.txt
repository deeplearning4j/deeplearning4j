#######################################################################
#  This is the makefile to create a library of the test matrix
#  generators used in LAPACK.  The files are organized as follows:
#
#     SCATGEN  -- Auxiliary routines called from both REAL and COMPLEX
#     DZATGEN  -- Auxiliary routines called from both DOUBLE PRECISION
#                 and COMPLEX*16
#     SMATGEN  -- Single precision real matrix generation routines
#     CMATGEN  -- Single precision complex matrix generation routines
#     DMATGEN  -- Double precision real matrix generation routines
#     ZMATGEN  -- Double precision complex matrix generation routines
#
#  The library can be set up to include routines for any combination
#  of the four precisions.  To create or add to the library, enter make
#  followed by one or more of the precisions desired.  Some examples:
#       make single
#       make single complex
#       make single double complex complex16
#  Alternatively, the command
#       make
#  without any arguments creates a library of all four precisions.
#  The library is called
#       tmglib.a
#  and is created at the LAPACK directory level.
#
#  To remove the object files after the library is created, enter
#       make clean
#  On some systems, you can force the source files to be recompiled by
#  entering (for example)
#       make single FRC=FRC
#
#######################################################################
 
set(SCATGEN  slatm1.f slaran.f slarnd.f)

set(SMATGEN  slatms.f slatme.f slatmr.f slatmt.f 
   slagge.f slagsy.f slakf2.f slarge.f slaror.f slarot.f slatm2.f 
   slatm3.f slatm5.f slatm6.f slatm7.f slahilb.f)

set(CMATGEN  clatms.f clatme.f clatmr.f clatmt.f 
   clagge.f claghe.f clagsy.f clakf2.f clarge.f claror.f clarot.f 
   clatm1.f clarnd.f clatm2.f clatm3.f clatm5.f clatm6.f clahilb.f slatm7.f)

set(DZATGEN  dlatm1.f dlaran.f dlarnd.f)

set(DMATGEN  dlatms.f dlatme.f dlatmr.f dlatmt.f 
   dlagge.f dlagsy.f dlakf2.f dlarge.f dlaror.f dlarot.f dlatm2.f 
   dlatm3.f dlatm5.f dlatm6.f dlatm7.f dlahilb.f)

set(ZMATGEN  zlatms.f zlatme.f zlatmr.f zlatmt.f 
  zlagge.f zlaghe.f zlagsy.f zlakf2.f zlarge.f zlaror.f zlarot.f 
  zlatm1.f zlarnd.f zlatm2.f zlatm3.f zlatm5.f zlatm6.f zlahilb.f dlatm7.f)

if(BUILD_SINGLE)
  set(ALLOBJ ${SMATGEN} ${SCATGEN})
endif()
if(BUILD_DOUBLE)
  set(ALLOBJ ${ALLOBJ} ${DMATGEN} ${DZATGEN})
endif()
if(BUILD_COMPLEX)
  set(ALLOBJ  ${ALLOBJ} ${CMATGEN} ${SCATGEN})
endif()
if(BUILD_COMPLEX16)
  set(ALLOBJ ${ALLOBJ} ${ZMATGEN} ${DZATGEN})
endif()

if (NOT ALLOBJ)
set(ALLOBJ ${SMATGEN} ${CMATGEN} ${SCATGEN} ${DMATGEN} ${ZMATGEN}
  ${DZATGEN})
else()
  list(REMOVE_DUPLICATES ALLOBJ)
endif()
add_library(tmglib ${ALLOBJ} )
target_link_libraries(tmglib ${LAPACK_LIBRARIES})
lapack_install_library(tmglib)
