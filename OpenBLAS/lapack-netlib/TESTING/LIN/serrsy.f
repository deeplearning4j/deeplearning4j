*> \brief \b SERRSY
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       SUBROUTINE SERRSY( PATH, NUNIT )
* 
*       .. Scalar Arguments ..
*       CHARACTER*3        PATH
*       INTEGER            NUNIT
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> SERRSY tests the error exits for the REAL routines
*> for symmetric indefinite matrices.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] PATH
*> \verbatim
*>          PATH is CHARACTER*3
*>          The LAPACK path name for the routines to be tested.
*> \endverbatim
*>
*> \param[in] NUNIT
*> \verbatim
*>          NUNIT is INTEGER
*>          The unit number for output.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date November 2015
*
*> \ingroup single_lin
*
*  =====================================================================
      SUBROUTINE SERRSY( PATH, NUNIT )
*
*  -- LAPACK test routine (version 3.6.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2015
*
*     .. Scalar Arguments ..
      CHARACTER*3        PATH
      INTEGER            NUNIT
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            NMAX
      PARAMETER          ( NMAX = 4 )
*     ..
*     .. Local Scalars ..
      CHARACTER*2        C2
      INTEGER            I, INFO, J
      REAL               ANRM, RCOND
*     ..
*     .. Local Arrays ..
      INTEGER            IP( NMAX ), IW( NMAX )
      REAL               A( NMAX, NMAX ), AF( NMAX, NMAX ), B( NMAX ),
     $                   R1( NMAX ), R2( NMAX ), W( 3*NMAX ), X( NMAX )
*     ..
*     .. External Functions ..
      LOGICAL            LSAMEN
      EXTERNAL           LSAMEN
*     ..
*     .. External Subroutines ..
      EXTERNAL           ALAESM, CHKXER, SSPCON, SSYCON_ROOK, SSPRFS,
     $                   SSPTRF, SSPTRI, SSPTRS, SSYCON, SSYRFS, SSYTF2,
     $                   SSYTF2_ROOK, SSYTRF, SSYTRF_ROOK, SSYTRI,
     $                   SSYTRI_ROOK, SSYTRI2, SSYTRS, SSYTRS_ROOK
*     ..
*     .. Scalars in Common ..
      LOGICAL            LERR, OK
      CHARACTER*32       SRNAMT
      INTEGER            INFOT, NOUT
*     ..
*     .. Common blocks ..
      COMMON             / INFOC / INFOT, NOUT, OK, LERR
      COMMON             / SRNAMC / SRNAMT
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          REAL
*     ..
*     .. Executable Statements ..
*
      NOUT = NUNIT
      WRITE( NOUT, FMT = * )
      C2 = PATH( 2: 3 )
*
*     Set the variables to innocuous values.
*
      DO 20 J = 1, NMAX
         DO 10 I = 1, NMAX
            A( I, J ) = 1. / REAL( I+J )
            AF( I, J ) = 1. / REAL( I+J )
   10    CONTINUE
         B( J ) = 0.
         R1( J ) = 0.
         R2( J ) = 0.
         W( J ) = 0.
         X( J ) = 0.
         IP( J ) = J
         IW( J ) = J
   20 CONTINUE
      ANRM = 1.0
      RCOND = 1.0
      OK = .TRUE.
*
      IF( LSAMEN( 2, C2, 'SY' ) ) THEN
*
*        Test error exits of the routines that use factorization
*        of a symmetric indefinite matrix with patrial
*        (Bunch-Kaufman) pivoting.
*
*        SSYTRF
*
         SRNAMT = 'SSYTRF'
         INFOT = 1
         CALL SSYTRF( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'SSYTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTRF( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'SSYTRF', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYTRF( 'U', 2, A, 1, IP, W, 4, INFO )
         CALL CHKXER( 'SSYTRF', INFOT, NOUT, LERR, OK )
*
*        SSYTF2
*
         SRNAMT = 'SSYTF2'
         INFOT = 1
         CALL SSYTF2( '/', 0, A, 1, IP, INFO )
         CALL CHKXER( 'SSYTF2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTF2( 'U', -1, A, 1, IP, INFO )
         CALL CHKXER( 'SSYTF2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYTF2( 'U', 2, A, 1, IP, INFO )
         CALL CHKXER( 'SSYTF2', INFOT, NOUT, LERR, OK )
*
*        SSYTRI
*
         SRNAMT = 'SSYTRI'
         INFOT = 1
         CALL SSYTRI( '/', 0, A, 1, IP, W, INFO )
         CALL CHKXER( 'SSYTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTRI( 'U', -1, A, 1, IP, W, INFO )
         CALL CHKXER( 'SSYTRI', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYTRI( 'U', 2, A, 1, IP, W, INFO )
         CALL CHKXER( 'SSYTRI', INFOT, NOUT, LERR, OK )
*
*        SSYTRI2
*
         SRNAMT = 'SSYTRI2'
         INFOT = 1
         CALL SSYTRI2( '/', 0, A, 1, IP, W, IW(1), INFO )
         CALL CHKXER( 'SSYTRI2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTRI2( 'U', -1, A, 1, IP, W, IW(1), INFO )
         CALL CHKXER( 'SSYTRI2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYTRI2( 'U', 2, A, 1, IP, W, IW(1), INFO )
         CALL CHKXER( 'SSYTRI2', INFOT, NOUT, LERR, OK )
*
*        SSYTRS
*
         SRNAMT = 'SSYTRS'
         INFOT = 1
         CALL SSYTRS( '/', 0, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'SSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTRS( 'U', -1, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'SSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL SSYTRS( 'U', 0, -1, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'SSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL SSYTRS( 'U', 2, 1, A, 1, IP, B, 2, INFO )
         CALL CHKXER( 'SSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL SSYTRS( 'U', 2, 1, A, 2, IP, B, 1, INFO )
         CALL CHKXER( 'SSYTRS', INFOT, NOUT, LERR, OK )
*
*        SSYRFS
*
         SRNAMT = 'SSYRFS'
         INFOT = 1
         CALL SSYRFS( '/', 0, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'SSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYRFS( 'U', -1, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, IW, INFO )
         CALL CHKXER( 'SSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL SSYRFS( 'U', 0, -1, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, IW, INFO )
         CALL CHKXER( 'SSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL SSYRFS( 'U', 2, 1, A, 1, AF, 2, IP, B, 2, X, 2, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'SSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL SSYRFS( 'U', 2, 1, A, 2, AF, 1, IP, B, 2, X, 2, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'SSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL SSYRFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 1, X, 2, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'SSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 12
         CALL SSYRFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 2, X, 1, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'SSYRFS', INFOT, NOUT, LERR, OK )
*
*        SSYCON
*
         SRNAMT = 'SSYCON'
         INFOT = 1
         CALL SSYCON( '/', 0, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYCON( 'U', -1, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYCON( 'U', 2, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 6
         CALL SSYCON( 'U', 1, A, 1, IP, -1.0, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSYCON', INFOT, NOUT, LERR, OK )
*
      ELSE IF( LSAMEN( 2, C2, 'SR' ) ) THEN
*
*        Test error exits of the routines that use factorization
*        of a symmetric indefinite matrix with rook
*        (bounded Bunch-Kaufman) pivoting.
*
*        SSYTRF_ROOK
*
         SRNAMT = 'SSYTRF_ROOK'
         INFOT = 1
         CALL SSYTRF_ROOK( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'SSYTRF_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTRF_ROOK( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'SSYTRF_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYTRF_ROOK( 'U', 2, A, 1, IP, W, 4, INFO )
         CALL CHKXER( 'SSYTRF_ROOK', INFOT, NOUT, LERR, OK )
*
*        SSYTF2_ROOK
*
         SRNAMT = 'SSYTF2_ROOK'
         INFOT = 1
         CALL SSYTF2_ROOK( '/', 0, A, 1, IP, INFO )
         CALL CHKXER( 'SSYTF2_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTF2_ROOK( 'U', -1, A, 1, IP, INFO )
         CALL CHKXER( 'SSYTF2_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYTF2_ROOK( 'U', 2, A, 1, IP, INFO )
         CALL CHKXER( 'SSYTF2_ROOK', INFOT, NOUT, LERR, OK )
*
*        SSYTRI_ROOK
*
         SRNAMT = 'SSYTRI_ROOK'
         INFOT = 1
         CALL SSYTRI_ROOK( '/', 0, A, 1, IP, W, INFO )
         CALL CHKXER( 'SSYTRI_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTRI_ROOK( 'U', -1, A, 1, IP, W, INFO )
         CALL CHKXER( 'SSYTRI_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYTRI_ROOK( 'U', 2, A, 1, IP, W, INFO )
         CALL CHKXER( 'SSYTRI_ROOK', INFOT, NOUT, LERR, OK )
*
*        SSYTRS_ROOK
*
         SRNAMT = 'SSYTRS_ROOK'
         INFOT = 1
         CALL SSYTRS_ROOK( '/', 0, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'SSYTRS_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYTRS_ROOK( 'U', -1, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'SSYTRS_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL SSYTRS_ROOK( 'U', 0, -1, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'SSYTRS_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL SSYTRS_ROOK( 'U', 2, 1, A, 1, IP, B, 2, INFO )
         CALL CHKXER( 'SSYTRS_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL SSYTRS_ROOK( 'U', 2, 1, A, 2, IP, B, 1, INFO )
         CALL CHKXER( 'SSYTRS_ROOK', INFOT, NOUT, LERR, OK )
*
*        SSYCON_ROOK
*
         SRNAMT = 'SSYCON_ROOK'
         INFOT = 1
         CALL SSYCON_ROOK( '/', 0, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSYCON_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSYCON_ROOK( 'U', -1, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSYCON_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL SSYCON_ROOK( 'U', 2, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSYCON_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 6
         CALL SSYCON_ROOK( 'U', 1, A, 1, IP, -1.0, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSYCON_ROOK', INFOT, NOUT, LERR, OK )
*
*        Test error exits of the routines that use factorization
*        of a symmetric indefinite packed matrix with patrial
*        (Bunch-Kaufman) pivoting.
*
      ELSE IF( LSAMEN( 2, C2, 'SP' ) ) THEN
*
*        SSPTRF
*
         SRNAMT = 'SSPTRF'
         INFOT = 1
         CALL SSPTRF( '/', 0, A, IP, INFO )
         CALL CHKXER( 'SSPTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSPTRF( 'U', -1, A, IP, INFO )
         CALL CHKXER( 'SSPTRF', INFOT, NOUT, LERR, OK )
*
*        SSPTRI
*
         SRNAMT = 'SSPTRI'
         INFOT = 1
         CALL SSPTRI( '/', 0, A, IP, W, INFO )
         CALL CHKXER( 'SSPTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSPTRI( 'U', -1, A, IP, W, INFO )
         CALL CHKXER( 'SSPTRI', INFOT, NOUT, LERR, OK )
*
*        SSPTRS
*
         SRNAMT = 'SSPTRS'
         INFOT = 1
         CALL SSPTRS( '/', 0, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'SSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSPTRS( 'U', -1, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'SSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL SSPTRS( 'U', 0, -1, A, IP, B, 1, INFO )
         CALL CHKXER( 'SSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL SSPTRS( 'U', 2, 1, A, IP, B, 1, INFO )
         CALL CHKXER( 'SSPTRS', INFOT, NOUT, LERR, OK )
*
*        SSPRFS
*
         SRNAMT = 'SSPRFS'
         INFOT = 1
         CALL SSPRFS( '/', 0, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'SSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSPRFS( 'U', -1, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'SSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL SSPRFS( 'U', 0, -1, A, AF, IP, B, 1, X, 1, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'SSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL SSPRFS( 'U', 2, 1, A, AF, IP, B, 1, X, 2, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'SSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL SSPRFS( 'U', 2, 1, A, AF, IP, B, 2, X, 1, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'SSPRFS', INFOT, NOUT, LERR, OK )
*
*        SSPCON
*
         SRNAMT = 'SSPCON'
         INFOT = 1
         CALL SSPCON( '/', 0, A, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSPCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL SSPCON( 'U', -1, A, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSPCON', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL SSPCON( 'U', 1, A, IP, -1.0, RCOND, W, IW, INFO )
         CALL CHKXER( 'SSPCON', INFOT, NOUT, LERR, OK )
      END IF
*
*     Print a summary line.
*
      CALL ALAESM( PATH, OK, NOUT )
*
      RETURN
*
*     End of SERRSY
*
      END
