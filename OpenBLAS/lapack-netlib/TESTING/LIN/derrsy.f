*> \brief \b DERRSY
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       SUBROUTINE DERRSY( PATH, NUNIT )
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
*> DERRSY tests the error exits for the DOUBLE PRECISION routines
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
*> \ingroup double_lin
*
*  =====================================================================
      SUBROUTINE DERRSY( PATH, NUNIT )
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
      DOUBLE PRECISION   ANRM, RCOND
*     ..
*     .. Local Arrays ..
      INTEGER            IP( NMAX ), IW( NMAX )
      DOUBLE PRECISION   A( NMAX, NMAX ), AF( NMAX, NMAX ), B( NMAX ),
     $                   R1( NMAX ), R2( NMAX ), W( 3*NMAX ), X( NMAX )
*     ..
*     .. External Functions ..
      LOGICAL            LSAMEN
      EXTERNAL           LSAMEN
*     ..
*     .. External Subroutines ..
      EXTERNAL           ALAESM, CHKXER, DSPCON, DSPRFS, DSPTRF, DSPTRI,
     $                   DSPTRS, DSYCON, DSYCON_ROOK, DSYRFS, DSYTF2,
     $                   DSYTF2_ROOK, DSYTRF, DSYTRF_ROOK, DSYTRI,
     $                   DSYTRI_ROOK, DSYTRI2, DSYTRS, DSYTRS_ROOK
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
      INTRINSIC          DBLE
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
            A( I, J ) = 1.D0 / DBLE( I+J )
            AF( I, J ) = 1.D0 / DBLE( I+J )
   10    CONTINUE
         B( J ) = 0.D0
         R1( J ) = 0.D0
         R2( J ) = 0.D0
         W( J ) = 0.D0
         X( J ) = 0.D0
         IP( J ) = J
         IW( J ) = J
   20 CONTINUE
      ANRM = 1.0D0
      RCOND = 1.0D0
      OK = .TRUE.
*
      IF( LSAMEN( 2, C2, 'SY' ) ) THEN
*
*        Test error exits of the routines that use factorization
*        of a symmetric indefinite matrix with patrial
*        (Bunch-Kaufman) pivoting.
*
*        DSYTRF
*
         SRNAMT = 'DSYTRF'
         INFOT = 1
         CALL DSYTRF( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'DSYTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTRF( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'DSYTRF', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYTRF( 'U', 2, A, 1, IP, W, 4, INFO )
         CALL CHKXER( 'DSYTRF', INFOT, NOUT, LERR, OK )
*
*        DSYTF2
*
         SRNAMT = 'DSYTF2'
         INFOT = 1
         CALL DSYTF2( '/', 0, A, 1, IP, INFO )
         CALL CHKXER( 'DSYTF2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTF2( 'U', -1, A, 1, IP, INFO )
         CALL CHKXER( 'DSYTF2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYTF2( 'U', 2, A, 1, IP, INFO )
         CALL CHKXER( 'DSYTF2', INFOT, NOUT, LERR, OK )
*
*        DSYTRI
*
         SRNAMT = 'DSYTRI'
         INFOT = 1
         CALL DSYTRI( '/', 0, A, 1, IP, W, INFO )
         CALL CHKXER( 'DSYTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTRI( 'U', -1, A, 1, IP, W, INFO )
         CALL CHKXER( 'DSYTRI', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYTRI( 'U', 2, A, 1, IP, W, INFO )
         CALL CHKXER( 'DSYTRI', INFOT, NOUT, LERR, OK )
*
*        DSYTRI2
*
         SRNAMT = 'DSYTRI2'
         INFOT = 1
         CALL DSYTRI2( '/', 0, A, 1, IP, W, IW(1), INFO )
         CALL CHKXER( 'DSYTRI2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTRI2( 'U', -1, A, 1, IP, W, IW(1), INFO )
         CALL CHKXER( 'DSYTRI2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYTRI2( 'U', 2, A, 1, IP, W, IW(1), INFO )
         CALL CHKXER( 'DSYTRI2', INFOT, NOUT, LERR, OK )
*
*        DSYTRS
*
         SRNAMT = 'DSYTRS'
         INFOT = 1
         CALL DSYTRS( '/', 0, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'DSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTRS( 'U', -1, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'DSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL DSYTRS( 'U', 0, -1, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'DSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL DSYTRS( 'U', 2, 1, A, 1, IP, B, 2, INFO )
         CALL CHKXER( 'DSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL DSYTRS( 'U', 2, 1, A, 2, IP, B, 1, INFO )
         CALL CHKXER( 'DSYTRS', INFOT, NOUT, LERR, OK )
*
*        DSYRFS
*
         SRNAMT = 'DSYRFS'
         INFOT = 1
         CALL DSYRFS( '/', 0, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'DSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYRFS( 'U', -1, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, IW, INFO )
         CALL CHKXER( 'DSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL DSYRFS( 'U', 0, -1, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, IW, INFO )
         CALL CHKXER( 'DSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL DSYRFS( 'U', 2, 1, A, 1, AF, 2, IP, B, 2, X, 2, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'DSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL DSYRFS( 'U', 2, 1, A, 2, AF, 1, IP, B, 2, X, 2, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'DSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL DSYRFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 1, X, 2, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'DSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 12
         CALL DSYRFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 2, X, 1, R1, R2, W,
     $                IW, INFO )
         CALL CHKXER( 'DSYRFS', INFOT, NOUT, LERR, OK )
*
*        DSYCON
*
         SRNAMT = 'DSYCON'
         INFOT = 1
         CALL DSYCON( '/', 0, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYCON( 'U', -1, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYCON( 'U', 2, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 6
         CALL DSYCON( 'U', 1, A, 1, IP, -1.0D0, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSYCON', INFOT, NOUT, LERR, OK )
*
      ELSE IF( LSAMEN( 2, C2, 'SR' ) ) THEN
*
*        Test error exits of the routines that use factorization
*        of a symmetric indefinite matrix with rook
*        (bounded Bunch-Kaufman) pivoting.
*
*        DSYTRF_ROOK
*
         SRNAMT = 'DSYTRF_ROOK'
         INFOT = 1
         CALL DSYTRF_ROOK( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'DSYTRF_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTRF_ROOK( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'DSYTRF_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYTRF_ROOK( 'U', 2, A, 1, IP, W, 4, INFO )
         CALL CHKXER( 'DSYTRF_ROOK', INFOT, NOUT, LERR, OK )
*
*        DSYTF2_ROOK
*
         SRNAMT = 'DSYTF2_ROOK'
         INFOT = 1
         CALL DSYTF2_ROOK( '/', 0, A, 1, IP, INFO )
         CALL CHKXER( 'DSYTF2_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTF2_ROOK( 'U', -1, A, 1, IP, INFO )
         CALL CHKXER( 'DSYTF2_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYTF2_ROOK( 'U', 2, A, 1, IP, INFO )
         CALL CHKXER( 'DSYTF2_ROOK', INFOT, NOUT, LERR, OK )
*
*        DSYTRI_ROOK
*
         SRNAMT = 'DSYTRI_ROOK'
         INFOT = 1
         CALL DSYTRI_ROOK( '/', 0, A, 1, IP, W, INFO )
         CALL CHKXER( 'DSYTRI_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTRI_ROOK( 'U', -1, A, 1, IP, W, INFO )
         CALL CHKXER( 'DSYTRI_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYTRI_ROOK( 'U', 2, A, 1, IP, W, INFO )
         CALL CHKXER( 'DSYTRI_ROOK', INFOT, NOUT, LERR, OK )
*
*        DSYTRS_ROOK
*
         SRNAMT = 'DSYTRS_ROOK'
         INFOT = 1
         CALL DSYTRS_ROOK( '/', 0, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'DSYTRS_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYTRS_ROOK( 'U', -1, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'DSYTRS_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL DSYTRS_ROOK( 'U', 0, -1, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'DSYTRS_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL DSYTRS_ROOK( 'U', 2, 1, A, 1, IP, B, 2, INFO )
         CALL CHKXER( 'DSYTRS_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL DSYTRS_ROOK( 'U', 2, 1, A, 2, IP, B, 1, INFO )
         CALL CHKXER( 'DSYTRS_ROOK', INFOT, NOUT, LERR, OK )
*
*        DSYCON_ROOK
*
         SRNAMT = 'DSYCON_ROOK'
         INFOT = 1
         CALL DSYCON_ROOK( '/', 0, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSYCON_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSYCON_ROOK( 'U', -1, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSYCON_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL DSYCON_ROOK( 'U', 2, A, 1, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSYCON_ROOK', INFOT, NOUT, LERR, OK )
         INFOT = 6
         CALL DSYCON_ROOK( 'U', 1, A, 1, IP, -1.0D0, RCOND, W, IW, INFO)
         CALL CHKXER( 'DSYCON_ROOK', INFOT, NOUT, LERR, OK )
*
      ELSE IF( LSAMEN( 2, C2, 'SP' ) ) THEN
*
*        Test error exits of the routines that use factorization
*        of a symmetric indefinite packed matrix with patrial
*        (Bunch-Kaufman) pivoting.
*
*        DSPTRF
*
         SRNAMT = 'DSPTRF'
         INFOT = 1
         CALL DSPTRF( '/', 0, A, IP, INFO )
         CALL CHKXER( 'DSPTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSPTRF( 'U', -1, A, IP, INFO )
         CALL CHKXER( 'DSPTRF', INFOT, NOUT, LERR, OK )
*
*        DSPTRI
*
         SRNAMT = 'DSPTRI'
         INFOT = 1
         CALL DSPTRI( '/', 0, A, IP, W, INFO )
         CALL CHKXER( 'DSPTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSPTRI( 'U', -1, A, IP, W, INFO )
         CALL CHKXER( 'DSPTRI', INFOT, NOUT, LERR, OK )
*
*        DSPTRS
*
         SRNAMT = 'DSPTRS'
         INFOT = 1
         CALL DSPTRS( '/', 0, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'DSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSPTRS( 'U', -1, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'DSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL DSPTRS( 'U', 0, -1, A, IP, B, 1, INFO )
         CALL CHKXER( 'DSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL DSPTRS( 'U', 2, 1, A, IP, B, 1, INFO )
         CALL CHKXER( 'DSPTRS', INFOT, NOUT, LERR, OK )
*
*        DSPRFS
*
         SRNAMT = 'DSPRFS'
         INFOT = 1
         CALL DSPRFS( '/', 0, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'DSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSPRFS( 'U', -1, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'DSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL DSPRFS( 'U', 0, -1, A, AF, IP, B, 1, X, 1, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'DSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL DSPRFS( 'U', 2, 1, A, AF, IP, B, 1, X, 2, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'DSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL DSPRFS( 'U', 2, 1, A, AF, IP, B, 2, X, 1, R1, R2, W, IW,
     $                INFO )
         CALL CHKXER( 'DSPRFS', INFOT, NOUT, LERR, OK )
*
*        DSPCON
*
         SRNAMT = 'DSPCON'
         INFOT = 1
         CALL DSPCON( '/', 0, A, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSPCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL DSPCON( 'U', -1, A, IP, ANRM, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSPCON', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL DSPCON( 'U', 1, A, IP, -1.0D0, RCOND, W, IW, INFO )
         CALL CHKXER( 'DSPCON', INFOT, NOUT, LERR, OK )
      END IF
*
*     Print a summary line.
*
      CALL ALAESM( PATH, OK, NOUT )
*
      RETURN
*
*     End of DERRSY
*
      END
