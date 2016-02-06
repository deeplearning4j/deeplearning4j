*> \brief \b ZGESDD
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*> \htmlonly
*> Download ZGESDD + dependencies 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgesdd.f"> 
*> [TGZ]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgesdd.f"> 
*> [ZIP]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgesdd.f"> 
*> [TXT]</a>
*> \endhtmlonly 
*
*  Definition:
*  ===========
*
*       SUBROUTINE ZGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK,
*                          LWORK, RWORK, IWORK, INFO )
* 
*       .. Scalar Arguments ..
*       CHARACTER          JOBZ
*       INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
*       ..
*       .. Array Arguments ..
*       INTEGER            IWORK( * )
*       DOUBLE PRECISION   RWORK( * ), S( * )
*       COMPLEX*16         A( LDA, * ), U( LDU, * ), VT( LDVT, * ),
*      $                   WORK( * )
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> ZGESDD computes the singular value decomposition (SVD) of a complex
*> M-by-N matrix A, optionally computing the left and/or right singular
*> vectors, by using divide-and-conquer method. The SVD is written
*>
*>      A = U * SIGMA * conjugate-transpose(V)
*>
*> where SIGMA is an M-by-N matrix which is zero except for its
*> min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
*> V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
*> are the singular values of A; they are real and non-negative, and
*> are returned in descending order.  The first min(m,n) columns of
*> U and V are the left and right singular vectors of A.
*>
*> Note that the routine returns VT = V**H, not V.
*>
*> The divide and conquer algorithm makes very mild assumptions about
*> floating point arithmetic. It will work on machines with a guard
*> digit in add/subtract, or on those binary machines without guard
*> digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
*> Cray-2. It could conceivably fail on hexadecimal or decimal machines
*> without guard digits, but we know of none.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] JOBZ
*> \verbatim
*>          JOBZ is CHARACTER*1
*>          Specifies options for computing all or part of the matrix U:
*>          = 'A':  all M columns of U and all N rows of V**H are
*>                  returned in the arrays U and VT;
*>          = 'S':  the first min(M,N) columns of U and the first
*>                  min(M,N) rows of V**H are returned in the arrays U
*>                  and VT;
*>          = 'O':  If M >= N, the first N columns of U are overwritten
*>                  in the array A and all rows of V**H are returned in
*>                  the array VT;
*>                  otherwise, all columns of U are returned in the
*>                  array U and the first M rows of V**H are overwritten
*>                  in the array A;
*>          = 'N':  no columns of U or rows of V**H are computed.
*> \endverbatim
*>
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>          The number of rows of the input matrix A.  M >= 0.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The number of columns of the input matrix A.  N >= 0.
*> \endverbatim
*>
*> \param[in,out] A
*> \verbatim
*>          A is COMPLEX*16 array, dimension (LDA,N)
*>          On entry, the M-by-N matrix A.
*>          On exit,
*>          if JOBZ = 'O',  A is overwritten with the first N columns
*>                          of U (the left singular vectors, stored
*>                          columnwise) if M >= N;
*>                          A is overwritten with the first M rows
*>                          of V**H (the right singular vectors, stored
*>                          rowwise) otherwise.
*>          if JOBZ .ne. 'O', the contents of A are destroyed.
*> \endverbatim
*>
*> \param[in] LDA
*> \verbatim
*>          LDA is INTEGER
*>          The leading dimension of the array A.  LDA >= max(1,M).
*> \endverbatim
*>
*> \param[out] S
*> \verbatim
*>          S is DOUBLE PRECISION array, dimension (min(M,N))
*>          The singular values of A, sorted so that S(i) >= S(i+1).
*> \endverbatim
*>
*> \param[out] U
*> \verbatim
*>          U is COMPLEX*16 array, dimension (LDU,UCOL)
*>          UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N;
*>          UCOL = min(M,N) if JOBZ = 'S'.
*>          If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M
*>          unitary matrix U;
*>          if JOBZ = 'S', U contains the first min(M,N) columns of U
*>          (the left singular vectors, stored columnwise);
*>          if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced.
*> \endverbatim
*>
*> \param[in] LDU
*> \verbatim
*>          LDU is INTEGER
*>          The leading dimension of the array U.  LDU >= 1; if
*>          JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M.
*> \endverbatim
*>
*> \param[out] VT
*> \verbatim
*>          VT is COMPLEX*16 array, dimension (LDVT,N)
*>          If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the
*>          N-by-N unitary matrix V**H;
*>          if JOBZ = 'S', VT contains the first min(M,N) rows of
*>          V**H (the right singular vectors, stored rowwise);
*>          if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced.
*> \endverbatim
*>
*> \param[in] LDVT
*> \verbatim
*>          LDVT is INTEGER
*>          The leading dimension of the array VT.  LDVT >= 1; if
*>          JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N;
*>          if JOBZ = 'S', LDVT >= min(M,N).
*> \endverbatim
*>
*> \param[out] WORK
*> \verbatim
*>          WORK is COMPLEX*16 array, dimension (MAX(1,LWORK))
*>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
*> \endverbatim
*>
*> \param[in] LWORK
*> \verbatim
*>          LWORK is INTEGER
*>          The dimension of the array WORK. LWORK >= 1.
*>          if JOBZ = 'N', LWORK >= 2*min(M,N)+max(M,N).
*>          if JOBZ = 'O',
*>                LWORK >= 2*min(M,N)*min(M,N)+2*min(M,N)+max(M,N).
*>          if JOBZ = 'S' or 'A',
*>                LWORK >= min(M,N)*min(M,N)+2*min(M,N)+max(M,N).
*>          For good performance, LWORK should generally be larger.
*>
*>          If LWORK = -1, a workspace query is assumed.  The optimal
*>          size for the WORK array is calculated and stored in WORK(1),
*>          and no other work except argument checking is performed.
*> \endverbatim
*>
*> \param[out] RWORK
*> \verbatim
*>          RWORK is DOUBLE PRECISION array, dimension (MAX(1,LRWORK))
*>          If JOBZ = 'N', LRWORK >= 7*min(M,N).
*>          Otherwise,
*>          LRWORK >= min(M,N)*max(5*min(M,N)+7,2*max(M,N)+2*min(M,N)+1)
*> \endverbatim
*>
*> \param[out] IWORK
*> \verbatim
*>          IWORK is INTEGER array, dimension (8*min(M,N))
*> \endverbatim
*>
*> \param[out] INFO
*> \verbatim
*>          INFO is INTEGER
*>          = 0:  successful exit.
*>          < 0:  if INFO = -i, the i-th argument had an illegal value.
*>          > 0:  The updating process of DBDSDC did not converge.
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
*> \ingroup complex16GEsing
*
*> \par Contributors:
*  ==================
*>
*>     Ming Gu and Huan Ren, Computer Science Division, University of
*>     California at Berkeley, USA
*>
*  =====================================================================
      SUBROUTINE ZGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK,
     $                   LWORK, RWORK, IWORK, INFO )
*
*  -- LAPACK driver routine (version 3.6.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2015
*
*     .. Scalar Arguments ..
      CHARACTER          JOBZ
      INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( * )
      DOUBLE PRECISION   RWORK( * ), S( * )
      COMPLEX*16         A( LDA, * ), U( LDU, * ), VT( LDVT, * ),
     $                   WORK( * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            LQUERV
      PARAMETER          ( LQUERV = -1 )
      COMPLEX*16         CZERO, CONE
      PARAMETER          ( CZERO = ( 0.0D+0, 0.0D+0 ),
     $                   CONE = ( 1.0D+0, 0.0D+0 ) )
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            WNTQA, WNTQAS, WNTQN, WNTQO, WNTQS
      INTEGER            BLK, CHUNK, I, IE, IERR, IL, IR, IRU, IRVT,
     $                   ISCL, ITAU, ITAUP, ITAUQ, IU, IVT, LDWKVT,
     $                   LDWRKL, LDWRKR, LDWRKU, MAXWRK, MINMN, MINWRK,
     $                   MNTHR1, MNTHR2, NRWORK, NWORK, WRKBL
      DOUBLE PRECISION   ANRM, BIGNUM, EPS, SMLNUM
*     ..
*     .. Local Arrays ..
      INTEGER            IDUM( 1 )
      DOUBLE PRECISION   DUM( 1 )
*     ..
*     .. External Subroutines ..
      EXTERNAL           DBDSDC, DLASCL, XERBLA, ZGEBRD, ZGELQF, ZGEMM,
     $                   ZGEQRF, ZLACP2, ZLACPY, ZLACRM, ZLARCM, ZLASCL,
     $                   ZLASET, ZUNGBR, ZUNGLQ, ZUNGQR, ZUNMBR
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            ILAENV
      DOUBLE PRECISION   DLAMCH, ZLANGE
      EXTERNAL           LSAME, ILAENV, DLAMCH, ZLANGE
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          INT, MAX, MIN, SQRT
*     ..
*     .. Executable Statements ..
*
*     Test the input arguments
*
      INFO = 0
      MINMN = MIN( M, N )
      MNTHR1 = INT( MINMN*17.0D0 / 9.0D0 )
      MNTHR2 = INT( MINMN*5.0D0 / 3.0D0 )
      WNTQA = LSAME( JOBZ, 'A' )
      WNTQS = LSAME( JOBZ, 'S' )
      WNTQAS = WNTQA .OR. WNTQS
      WNTQO = LSAME( JOBZ, 'O' )
      WNTQN = LSAME( JOBZ, 'N' )
      MINWRK = 1
      MAXWRK = 1
*
      IF( .NOT.( WNTQA .OR. WNTQS .OR. WNTQO .OR. WNTQN ) ) THEN
         INFO = -1
      ELSE IF( M.LT.0 ) THEN
         INFO = -2
      ELSE IF( N.LT.0 ) THEN
         INFO = -3
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -5
      ELSE IF( LDU.LT.1 .OR. ( WNTQAS .AND. LDU.LT.M ) .OR.
     $         ( WNTQO .AND. M.LT.N .AND. LDU.LT.M ) ) THEN
         INFO = -8
      ELSE IF( LDVT.LT.1 .OR. ( WNTQA .AND. LDVT.LT.N ) .OR.
     $         ( WNTQS .AND. LDVT.LT.MINMN ) .OR.
     $         ( WNTQO .AND. M.GE.N .AND. LDVT.LT.N ) ) THEN
         INFO = -10
      END IF
*
*     Compute workspace
*      (Note: Comments in the code beginning "Workspace:" describe the
*       minimal amount of workspace needed at that point in the code,
*       as well as the preferred amount for good performance.
*       CWorkspace refers to complex workspace, and RWorkspace to
*       real workspace. NB refers to the optimal block size for the
*       immediately following subroutine, as returned by ILAENV.)
*
      IF( INFO.EQ.0 .AND. M.GT.0 .AND. N.GT.0 ) THEN
         IF( M.GE.N ) THEN
*
*           There is no complex work space needed for bidiagonal SVD
*           The real work space needed for bidiagonal SVD is BDSPAC
*           for computing singular values and singular vectors; BDSPAN
*           for computing singular values only.
*           BDSPAC = 5*N*N + 7*N
*           BDSPAN = MAX(7*N+4, 3*N+2+SMLSIZ*(SMLSIZ+8))
*
            IF( M.GE.MNTHR1 ) THEN
               IF( WNTQN ) THEN
*
*                 Path 1 (M much larger than N, JOBZ='N')
*
                  MAXWRK = N + N*ILAENV( 1, 'ZGEQRF', ' ', M, N, -1,
     $                     -1 )
                  MAXWRK = MAX( MAXWRK, 2*N+2*N*
     $                     ILAENV( 1, 'ZGEBRD', ' ', N, N, -1, -1 ) )
                  MINWRK = 3*N
               ELSE IF( WNTQO ) THEN
*
*                 Path 2 (M much larger than N, JOBZ='O')
*
                  WRKBL = N + N*ILAENV( 1, 'ZGEQRF', ' ', M, N, -1, -1 )
                  WRKBL = MAX( WRKBL, N+N*ILAENV( 1, 'ZUNGQR', ' ', M,
     $                    N, N, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+2*N*
     $                    ILAENV( 1, 'ZGEBRD', ' ', N, N, -1, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+N*
     $                    ILAENV( 1, 'ZUNMBR', 'QLN', N, N, N, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+N*
     $                    ILAENV( 1, 'ZUNMBR', 'PRC', N, N, N, -1 ) )
                  MAXWRK = M*N + N*N + WRKBL
                  MINWRK = 2*N*N + 3*N
               ELSE IF( WNTQS ) THEN
*
*                 Path 3 (M much larger than N, JOBZ='S')
*
                  WRKBL = N + N*ILAENV( 1, 'ZGEQRF', ' ', M, N, -1, -1 )
                  WRKBL = MAX( WRKBL, N+N*ILAENV( 1, 'ZUNGQR', ' ', M,
     $                    N, N, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+2*N*
     $                    ILAENV( 1, 'ZGEBRD', ' ', N, N, -1, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+N*
     $                    ILAENV( 1, 'ZUNMBR', 'QLN', N, N, N, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+N*
     $                    ILAENV( 1, 'ZUNMBR', 'PRC', N, N, N, -1 ) )
                  MAXWRK = N*N + WRKBL
                  MINWRK = N*N + 3*N
               ELSE IF( WNTQA ) THEN
*
*                 Path 4 (M much larger than N, JOBZ='A')
*
                  WRKBL = N + N*ILAENV( 1, 'ZGEQRF', ' ', M, N, -1, -1 )
                  WRKBL = MAX( WRKBL, N+M*ILAENV( 1, 'ZUNGQR', ' ', M,
     $                    M, N, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+2*N*
     $                    ILAENV( 1, 'ZGEBRD', ' ', N, N, -1, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+N*
     $                    ILAENV( 1, 'ZUNMBR', 'QLN', N, N, N, -1 ) )
                  WRKBL = MAX( WRKBL, 2*N+N*
     $                    ILAENV( 1, 'ZUNMBR', 'PRC', N, N, N, -1 ) )
                  MAXWRK = N*N + WRKBL
                  MINWRK = N*N + 2*N + M
               END IF
            ELSE IF( M.GE.MNTHR2 ) THEN
*
*              Path 5 (M much larger than N, but not as much as MNTHR1)
*
               MAXWRK = 2*N + ( M+N )*ILAENV( 1, 'ZGEBRD', ' ', M, N,
     $                  -1, -1 )
               MINWRK = 2*N + M
               IF( WNTQO ) THEN
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNGBR', 'P', N, N, N, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNGBR', 'Q', M, N, N, -1 ) )
                  MAXWRK = MAXWRK + M*N
                  MINWRK = MINWRK + N*N
               ELSE IF( WNTQS ) THEN
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNGBR', 'P', N, N, N, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNGBR', 'Q', M, N, N, -1 ) )
               ELSE IF( WNTQA ) THEN
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNGBR', 'P', N, N, N, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*N+M*
     $                     ILAENV( 1, 'ZUNGBR', 'Q', M, M, N, -1 ) )
               END IF
            ELSE
*
*              Path 6 (M at least N, but not much larger)
*
               MAXWRK = 2*N + ( M+N )*ILAENV( 1, 'ZGEBRD', ' ', M, N,
     $                  -1, -1 )
               MINWRK = 2*N + M
               IF( WNTQO ) THEN
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNMBR', 'PRC', N, N, N, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNMBR', 'QLN', M, N, N, -1 ) )
                  MAXWRK = MAXWRK + M*N
                  MINWRK = MINWRK + N*N
               ELSE IF( WNTQS ) THEN
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNMBR', 'PRC', N, N, N, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNMBR', 'QLN', M, N, N, -1 ) )
               ELSE IF( WNTQA ) THEN
                  MAXWRK = MAX( MAXWRK, 2*N+N*
     $                     ILAENV( 1, 'ZUNGBR', 'PRC', N, N, N, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*N+M*
     $                     ILAENV( 1, 'ZUNGBR', 'QLN', M, M, N, -1 ) )
               END IF
            END IF
         ELSE
*
*           There is no complex work space needed for bidiagonal SVD
*           The real work space needed for bidiagonal SVD is BDSPAC
*           for computing singular values and singular vectors; BDSPAN
*           for computing singular values only.
*           BDSPAC = 5*M*M + 7*M
*           BDSPAN = MAX(7*M+4, 3*M+2+SMLSIZ*(SMLSIZ+8))
*
            IF( N.GE.MNTHR1 ) THEN
               IF( WNTQN ) THEN
*
*                 Path 1t (N much larger than M, JOBZ='N')
*
                  MAXWRK = M + M*ILAENV( 1, 'ZGELQF', ' ', M, N, -1,
     $                     -1 )
                  MAXWRK = MAX( MAXWRK, 2*M+2*M*
     $                     ILAENV( 1, 'ZGEBRD', ' ', M, M, -1, -1 ) )
                  MINWRK = 3*M
               ELSE IF( WNTQO ) THEN
*
*                 Path 2t (N much larger than M, JOBZ='O')
*
                  WRKBL = M + M*ILAENV( 1, 'ZGELQF', ' ', M, N, -1, -1 )
                  WRKBL = MAX( WRKBL, M+M*ILAENV( 1, 'ZUNGLQ', ' ', M,
     $                    N, M, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+2*M*
     $                    ILAENV( 1, 'ZGEBRD', ' ', M, M, -1, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+M*
     $                    ILAENV( 1, 'ZUNMBR', 'PRC', M, M, M, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+M*
     $                    ILAENV( 1, 'ZUNMBR', 'QLN', M, M, M, -1 ) )
                  MAXWRK = M*N + M*M + WRKBL
                  MINWRK = 2*M*M + 3*M
               ELSE IF( WNTQS ) THEN
*
*                 Path 3t (N much larger than M, JOBZ='S')
*
                  WRKBL = M + M*ILAENV( 1, 'ZGELQF', ' ', M, N, -1, -1 )
                  WRKBL = MAX( WRKBL, M+M*ILAENV( 1, 'ZUNGLQ', ' ', M,
     $                    N, M, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+2*M*
     $                    ILAENV( 1, 'ZGEBRD', ' ', M, M, -1, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+M*
     $                    ILAENV( 1, 'ZUNMBR', 'PRC', M, M, M, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+M*
     $                    ILAENV( 1, 'ZUNMBR', 'QLN', M, M, M, -1 ) )
                  MAXWRK = M*M + WRKBL
                  MINWRK = M*M + 3*M
               ELSE IF( WNTQA ) THEN
*
*                 Path 4t (N much larger than M, JOBZ='A')
*
                  WRKBL = M + M*ILAENV( 1, 'ZGELQF', ' ', M, N, -1, -1 )
                  WRKBL = MAX( WRKBL, M+N*ILAENV( 1, 'ZUNGLQ', ' ', N,
     $                    N, M, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+2*M*
     $                    ILAENV( 1, 'ZGEBRD', ' ', M, M, -1, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+M*
     $                    ILAENV( 1, 'ZUNMBR', 'PRC', M, M, M, -1 ) )
                  WRKBL = MAX( WRKBL, 2*M+M*
     $                    ILAENV( 1, 'ZUNMBR', 'QLN', M, M, M, -1 ) )
                  MAXWRK = M*M + WRKBL
                  MINWRK = M*M + 2*M + N
               END IF
            ELSE IF( N.GE.MNTHR2 ) THEN
*
*              Path 5t (N much larger than M, but not as much as MNTHR1)
*
               MAXWRK = 2*M + ( M+N )*ILAENV( 1, 'ZGEBRD', ' ', M, N,
     $                  -1, -1 )
               MINWRK = 2*M + N
               IF( WNTQO ) THEN
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNGBR', 'P', M, N, M, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNGBR', 'Q', M, M, N, -1 ) )
                  MAXWRK = MAXWRK + M*N
                  MINWRK = MINWRK + M*M
               ELSE IF( WNTQS ) THEN
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNGBR', 'P', M, N, M, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNGBR', 'Q', M, M, N, -1 ) )
               ELSE IF( WNTQA ) THEN
                  MAXWRK = MAX( MAXWRK, 2*M+N*
     $                     ILAENV( 1, 'ZUNGBR', 'P', N, N, M, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNGBR', 'Q', M, M, N, -1 ) )
               END IF
            ELSE
*
*              Path 6t (N greater than M, but not much larger)
*
               MAXWRK = 2*M + ( M+N )*ILAENV( 1, 'ZGEBRD', ' ', M, N,
     $                  -1, -1 )
               MINWRK = 2*M + N
               IF( WNTQO ) THEN
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNMBR', 'PRC', M, N, M, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNMBR', 'QLN', M, M, N, -1 ) )
                  MAXWRK = MAXWRK + M*N
                  MINWRK = MINWRK + M*M
               ELSE IF( WNTQS ) THEN
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNGBR', 'PRC', M, N, M, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNGBR', 'QLN', M, M, N, -1 ) )
               ELSE IF( WNTQA ) THEN
                  MAXWRK = MAX( MAXWRK, 2*M+N*
     $                     ILAENV( 1, 'ZUNGBR', 'PRC', N, N, M, -1 ) )
                  MAXWRK = MAX( MAXWRK, 2*M+M*
     $                     ILAENV( 1, 'ZUNGBR', 'QLN', M, M, N, -1 ) )
               END IF
            END IF
         END IF
         MAXWRK = MAX( MAXWRK, MINWRK )
      END IF
      IF( INFO.EQ.0 ) THEN
         WORK( 1 ) = MAXWRK
         IF( LWORK.LT.MINWRK .AND. LWORK.NE.LQUERV )
     $      INFO = -13
      END IF
*
*     Quick returns
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'ZGESDD', -INFO )
         RETURN
      END IF
      IF( LWORK.EQ.LQUERV )
     $   RETURN
      IF( M.EQ.0 .OR. N.EQ.0 ) THEN
         RETURN
      END IF
*
*     Get machine constants
*
      EPS = DLAMCH( 'P' )
      SMLNUM = SQRT( DLAMCH( 'S' ) ) / EPS
      BIGNUM = ONE / SMLNUM
*
*     Scale A if max element outside range [SMLNUM,BIGNUM]
*
      ANRM = ZLANGE( 'M', M, N, A, LDA, DUM )
      ISCL = 0
      IF( ANRM.GT.ZERO .AND. ANRM.LT.SMLNUM ) THEN
         ISCL = 1
         CALL ZLASCL( 'G', 0, 0, ANRM, SMLNUM, M, N, A, LDA, IERR )
      ELSE IF( ANRM.GT.BIGNUM ) THEN
         ISCL = 1
         CALL ZLASCL( 'G', 0, 0, ANRM, BIGNUM, M, N, A, LDA, IERR )
      END IF
*
      IF( M.GE.N ) THEN
*
*        A has at least as many rows as columns. If A has sufficiently
*        more rows than columns, first reduce using the QR
*        decomposition (if sufficient workspace available)
*
         IF( M.GE.MNTHR1 ) THEN
*
            IF( WNTQN ) THEN
*
*              Path 1 (M much larger than N, JOBZ='N')
*              No singular vectors to be computed
*
               ITAU = 1
               NWORK = ITAU + N
*
*              Compute A=Q*R
*              (CWorkspace: need 2*N, prefer N+N*NB)
*              (RWorkspace: need 0)
*
               CALL ZGEQRF( M, N, A, LDA, WORK( ITAU ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Zero out below R
*
               CALL ZLASET( 'L', N-1, N-1, CZERO, CZERO, A( 2, 1 ),
     $                      LDA )
               IE = 1
               ITAUQ = 1
               ITAUP = ITAUQ + N
               NWORK = ITAUP + N
*
*              Bidiagonalize R in A
*              (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
*              (RWorkspace: need N)
*
               CALL ZGEBRD( N, N, A, LDA, S, RWORK( IE ), WORK( ITAUQ ),
     $                      WORK( ITAUP ), WORK( NWORK ), LWORK-NWORK+1,
     $                      IERR )
               NRWORK = IE + N
*
*              Perform bidiagonal SVD, compute singular values only
*              (CWorkspace: 0)
*              (RWorkspace: need BDSPAN)
*
               CALL DBDSDC( 'U', 'N', N, S, RWORK( IE ), DUM, 1, DUM, 1,
     $                      DUM, IDUM, RWORK( NRWORK ), IWORK, INFO )
*
            ELSE IF( WNTQO ) THEN
*
*              Path 2 (M much larger than N, JOBZ='O')
*              N left singular vectors to be overwritten on A and
*              N right singular vectors to be computed in VT
*
               IU = 1
*
*              WORK(IU) is N by N
*
               LDWRKU = N
               IR = IU + LDWRKU*N
               IF( LWORK.GE.M*N+N*N+3*N ) THEN
*
*                 WORK(IR) is M by N
*
                  LDWRKR = M
               ELSE
                  LDWRKR = ( LWORK-N*N-3*N ) / N
               END IF
               ITAU = IR + LDWRKR*N
               NWORK = ITAU + N
*
*              Compute A=Q*R
*              (CWorkspace: need N*N+2*N, prefer M*N+N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZGEQRF( M, N, A, LDA, WORK( ITAU ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy R to WORK( IR ), zeroing out below it
*
               CALL ZLACPY( 'U', N, N, A, LDA, WORK( IR ), LDWRKR )
               CALL ZLASET( 'L', N-1, N-1, CZERO, CZERO, WORK( IR+1 ),
     $                      LDWRKR )
*
*              Generate Q in A
*              (CWorkspace: need 2*N, prefer N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZUNGQR( M, N, N, A, LDA, WORK( ITAU ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
               IE = 1
               ITAUQ = ITAU
               ITAUP = ITAUQ + N
               NWORK = ITAUP + N
*
*              Bidiagonalize R in WORK(IR)
*              (CWorkspace: need N*N+3*N, prefer M*N+2*N+2*N*NB)
*              (RWorkspace: need N)
*
               CALL ZGEBRD( N, N, WORK( IR ), LDWRKR, S, RWORK( IE ),
     $                      WORK( ITAUQ ), WORK( ITAUP ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of R in WORK(IRU) and computing right singular vectors
*              of R in WORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = IE + N
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
*              Overwrite WORK(IU) by the left singular vectors of R
*              (CWorkspace: need 2*N*N+3*N, prefer M*N+N*N+2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRU ), N, WORK( IU ),
     $                      LDWRKU )
               CALL ZUNMBR( 'Q', 'L', 'N', N, N, N, WORK( IR ), LDWRKR,
     $                      WORK( ITAUQ ), WORK( IU ), LDWRKU,
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by the right singular vectors of R
*              (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRVT ), N, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', N, N, N, WORK( IR ), LDWRKR,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Multiply Q in A by left singular vectors of R in
*              WORK(IU), storing result in WORK(IR) and copying to A
*              (CWorkspace: need 2*N*N, prefer N*N+M*N)
*              (RWorkspace: 0)
*
               DO 10 I = 1, M, LDWRKR
                  CHUNK = MIN( M-I+1, LDWRKR )
                  CALL ZGEMM( 'N', 'N', CHUNK, N, N, CONE, A( I, 1 ),
     $                        LDA, WORK( IU ), LDWRKU, CZERO,
     $                        WORK( IR ), LDWRKR )
                  CALL ZLACPY( 'F', CHUNK, N, WORK( IR ), LDWRKR,
     $                         A( I, 1 ), LDA )
   10          CONTINUE
*
            ELSE IF( WNTQS ) THEN
*
*              Path 3 (M much larger than N, JOBZ='S')
*              N left singular vectors to be computed in U and
*              N right singular vectors to be computed in VT
*
               IR = 1
*
*              WORK(IR) is N by N
*
               LDWRKR = N
               ITAU = IR + LDWRKR*N
               NWORK = ITAU + N
*
*              Compute A=Q*R
*              (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZGEQRF( M, N, A, LDA, WORK( ITAU ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy R to WORK(IR), zeroing out below it
*
               CALL ZLACPY( 'U', N, N, A, LDA, WORK( IR ), LDWRKR )
               CALL ZLASET( 'L', N-1, N-1, CZERO, CZERO, WORK( IR+1 ),
     $                      LDWRKR )
*
*              Generate Q in A
*              (CWorkspace: need 2*N, prefer N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZUNGQR( M, N, N, A, LDA, WORK( ITAU ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
               IE = 1
               ITAUQ = ITAU
               ITAUP = ITAUQ + N
               NWORK = ITAUP + N
*
*              Bidiagonalize R in WORK(IR)
*              (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
*              (RWorkspace: need N)
*
               CALL ZGEBRD( N, N, WORK( IR ), LDWRKR, S, RWORK( IE ),
     $                      WORK( ITAUQ ), WORK( ITAUP ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = IE + N
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix U
*              Overwrite U by left singular vectors of R
*              (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRU ), N, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', N, N, N, WORK( IR ), LDWRKR,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by right singular vectors of R
*              (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRVT ), N, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', N, N, N, WORK( IR ), LDWRKR,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Multiply Q in A by left singular vectors of R in
*              WORK(IR), storing result in U
*              (CWorkspace: need N*N)
*              (RWorkspace: 0)
*
               CALL ZLACPY( 'F', N, N, U, LDU, WORK( IR ), LDWRKR )
               CALL ZGEMM( 'N', 'N', M, N, N, CONE, A, LDA, WORK( IR ),
     $                     LDWRKR, CZERO, U, LDU )
*
            ELSE IF( WNTQA ) THEN
*
*              Path 4 (M much larger than N, JOBZ='A')
*              M left singular vectors to be computed in U and
*              N right singular vectors to be computed in VT
*
               IU = 1
*
*              WORK(IU) is N by N
*
               LDWRKU = N
               ITAU = IU + LDWRKU*N
               NWORK = ITAU + N
*
*              Compute A=Q*R, copying result to U
*              (CWorkspace: need 2*N, prefer N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZGEQRF( M, N, A, LDA, WORK( ITAU ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
               CALL ZLACPY( 'L', M, N, A, LDA, U, LDU )
*
*              Generate Q in U
*              (CWorkspace: need N+M, prefer N+M*NB)
*              (RWorkspace: 0)
*
               CALL ZUNGQR( M, M, N, U, LDU, WORK( ITAU ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Produce R in A, zeroing out below it
*
               CALL ZLASET( 'L', N-1, N-1, CZERO, CZERO, A( 2, 1 ),
     $                      LDA )
               IE = 1
               ITAUQ = ITAU
               ITAUP = ITAUQ + N
               NWORK = ITAUP + N
*
*              Bidiagonalize R in A
*              (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
*              (RWorkspace: need N)
*
               CALL ZGEBRD( N, N, A, LDA, S, RWORK( IE ), WORK( ITAUQ ),
     $                      WORK( ITAUP ), WORK( NWORK ), LWORK-NWORK+1,
     $                      IERR )
               IRU = IE + N
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
*              Overwrite WORK(IU) by left singular vectors of R
*              (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRU ), N, WORK( IU ),
     $                      LDWRKU )
               CALL ZUNMBR( 'Q', 'L', 'N', N, N, N, A, LDA,
     $                      WORK( ITAUQ ), WORK( IU ), LDWRKU,
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by right singular vectors of R
*              (CWorkspace: need 3*N, prefer 2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRVT ), N, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', N, N, N, A, LDA,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Multiply Q in U by left singular vectors of R in
*              WORK(IU), storing result in A
*              (CWorkspace: need N*N)
*              (RWorkspace: 0)
*
               CALL ZGEMM( 'N', 'N', M, N, N, CONE, U, LDU, WORK( IU ),
     $                     LDWRKU, CZERO, A, LDA )
*
*              Copy left singular vectors of A from A to U
*
               CALL ZLACPY( 'F', M, N, A, LDA, U, LDU )
*
            END IF
*
         ELSE IF( M.GE.MNTHR2 ) THEN
*
*           MNTHR2 <= M < MNTHR1
*
*           Path 5 (M much larger than N, but not as much as MNTHR1)
*           Reduce to bidiagonal form without QR decomposition, use
*           ZUNGBR and matrix multiplication to compute singular vectors
*
            IE = 1
            NRWORK = IE + N
            ITAUQ = 1
            ITAUP = ITAUQ + N
            NWORK = ITAUP + N
*
*           Bidiagonalize A
*           (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
*           (RWorkspace: need N)
*
            CALL ZGEBRD( M, N, A, LDA, S, RWORK( IE ), WORK( ITAUQ ),
     $                   WORK( ITAUP ), WORK( NWORK ), LWORK-NWORK+1,
     $                   IERR )
            IF( WNTQN ) THEN
*
*              Compute singular values only
*              (Cworkspace: 0)
*              (Rworkspace: need BDSPAN)
*
               CALL DBDSDC( 'U', 'N', N, S, RWORK( IE ), DUM, 1, DUM, 1,
     $                      DUM, IDUM, RWORK( NRWORK ), IWORK, INFO )
            ELSE IF( WNTQO ) THEN
               IU = NWORK
               IRU = NRWORK
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
*
*              Copy A to VT, generate P**H
*              (Cworkspace: need 2*N, prefer N+N*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'U', N, N, A, LDA, VT, LDVT )
               CALL ZUNGBR( 'P', N, N, N, VT, LDVT, WORK( ITAUP ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Generate Q in A
*              (CWorkspace: need 2*N, prefer N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZUNGBR( 'Q', M, N, N, A, LDA, WORK( ITAUQ ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
               IF( LWORK.GE.M*N+3*N ) THEN
*
*                 WORK( IU ) is M by N
*
                  LDWRKU = M
               ELSE
*
*                 WORK(IU) is LDWRKU by N
*
                  LDWRKU = ( LWORK-3*N ) / N
               END IF
               NWORK = IU + LDWRKU*N
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Multiply real matrix RWORK(IRVT) by P**H in VT,
*              storing the result in WORK(IU), copying to VT
*              (Cworkspace: need 0)
*              (Rworkspace: need 3*N*N)
*
               CALL ZLARCM( N, N, RWORK( IRVT ), N, VT, LDVT,
     $                      WORK( IU ), LDWRKU, RWORK( NRWORK ) )
               CALL ZLACPY( 'F', N, N, WORK( IU ), LDWRKU, VT, LDVT )
*
*              Multiply Q in A by real matrix RWORK(IRU), storing the
*              result in WORK(IU), copying to A
*              (CWorkspace: need N*N, prefer M*N)
*              (Rworkspace: need 3*N*N, prefer N*N+2*M*N)
*
               NRWORK = IRVT
               DO 20 I = 1, M, LDWRKU
                  CHUNK = MIN( M-I+1, LDWRKU )
                  CALL ZLACRM( CHUNK, N, A( I, 1 ), LDA, RWORK( IRU ),
     $                         N, WORK( IU ), LDWRKU, RWORK( NRWORK ) )
                  CALL ZLACPY( 'F', CHUNK, N, WORK( IU ), LDWRKU,
     $                         A( I, 1 ), LDA )
   20          CONTINUE
*
            ELSE IF( WNTQS ) THEN
*
*              Copy A to VT, generate P**H
*              (Cworkspace: need 2*N, prefer N+N*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'U', N, N, A, LDA, VT, LDVT )
               CALL ZUNGBR( 'P', N, N, N, VT, LDVT, WORK( ITAUP ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Copy A to U, generate Q
*              (Cworkspace: need 2*N, prefer N+N*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'L', M, N, A, LDA, U, LDU )
               CALL ZUNGBR( 'Q', M, N, N, U, LDU, WORK( ITAUQ ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = NRWORK
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Multiply real matrix RWORK(IRVT) by P**H in VT,
*              storing the result in A, copying to VT
*              (Cworkspace: need 0)
*              (Rworkspace: need 3*N*N)
*
               CALL ZLARCM( N, N, RWORK( IRVT ), N, VT, LDVT, A, LDA,
     $                      RWORK( NRWORK ) )
               CALL ZLACPY( 'F', N, N, A, LDA, VT, LDVT )
*
*              Multiply Q in U by real matrix RWORK(IRU), storing the
*              result in A, copying to U
*              (CWorkspace: need 0)
*              (Rworkspace: need N*N+2*M*N)
*
               NRWORK = IRVT
               CALL ZLACRM( M, N, U, LDU, RWORK( IRU ), N, A, LDA,
     $                      RWORK( NRWORK ) )
               CALL ZLACPY( 'F', M, N, A, LDA, U, LDU )
            ELSE
*
*              Copy A to VT, generate P**H
*              (Cworkspace: need 2*N, prefer N+N*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'U', N, N, A, LDA, VT, LDVT )
               CALL ZUNGBR( 'P', N, N, N, VT, LDVT, WORK( ITAUP ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Copy A to U, generate Q
*              (Cworkspace: need 2*N, prefer N+N*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'L', M, N, A, LDA, U, LDU )
               CALL ZUNGBR( 'Q', M, M, N, U, LDU, WORK( ITAUQ ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = NRWORK
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Multiply real matrix RWORK(IRVT) by P**H in VT,
*              storing the result in A, copying to VT
*              (Cworkspace: need 0)
*              (Rworkspace: need 3*N*N)
*
               CALL ZLARCM( N, N, RWORK( IRVT ), N, VT, LDVT, A, LDA,
     $                      RWORK( NRWORK ) )
               CALL ZLACPY( 'F', N, N, A, LDA, VT, LDVT )
*
*              Multiply Q in U by real matrix RWORK(IRU), storing the
*              result in A, copying to U
*              (CWorkspace: 0)
*              (Rworkspace: need 3*N*N)
*
               NRWORK = IRVT
               CALL ZLACRM( M, N, U, LDU, RWORK( IRU ), N, A, LDA,
     $                      RWORK( NRWORK ) )
               CALL ZLACPY( 'F', M, N, A, LDA, U, LDU )
            END IF
*
         ELSE
*
*           M .LT. MNTHR2
*
*           Path 6 (M at least N, but not much larger)
*           Reduce to bidiagonal form without QR decomposition
*           Use ZUNMBR to compute singular vectors
*
            IE = 1
            NRWORK = IE + N
            ITAUQ = 1
            ITAUP = ITAUQ + N
            NWORK = ITAUP + N
*
*           Bidiagonalize A
*           (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
*           (RWorkspace: need N)
*
            CALL ZGEBRD( M, N, A, LDA, S, RWORK( IE ), WORK( ITAUQ ),
     $                   WORK( ITAUP ), WORK( NWORK ), LWORK-NWORK+1,
     $                   IERR )
            IF( WNTQN ) THEN
*
*              Compute singular values only
*              (Cworkspace: 0)
*              (Rworkspace: need BDSPAN)
*
               CALL DBDSDC( 'U', 'N', N, S, RWORK( IE ), DUM, 1, DUM, 1,
     $                      DUM, IDUM, RWORK( NRWORK ), IWORK, INFO )
            ELSE IF( WNTQO ) THEN
               IU = NWORK
               IRU = NRWORK
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
               IF( LWORK.GE.M*N+3*N ) THEN
*
*                 WORK( IU ) is M by N
*
                  LDWRKU = M
               ELSE
*
*                 WORK( IU ) is LDWRKU by N
*
                  LDWRKU = ( LWORK-3*N ) / N
               END IF
               NWORK = IU + LDWRKU*N
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by right singular vectors of A
*              (Cworkspace: need 2*N, prefer N+N*NB)
*              (Rworkspace: need 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRVT ), N, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', N, N, N, A, LDA,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
               IF( LWORK.GE.M*N+3*N ) THEN
*
*              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
*              Overwrite WORK(IU) by left singular vectors of A, copying
*              to A
*              (Cworkspace: need M*N+2*N, prefer M*N+N+N*NB)
*              (Rworkspace: need 0)
*
                  CALL ZLASET( 'F', M, N, CZERO, CZERO, WORK( IU ),
     $                         LDWRKU )
                  CALL ZLACP2( 'F', N, N, RWORK( IRU ), N, WORK( IU ),
     $                         LDWRKU )
                  CALL ZUNMBR( 'Q', 'L', 'N', M, N, N, A, LDA,
     $                         WORK( ITAUQ ), WORK( IU ), LDWRKU,
     $                         WORK( NWORK ), LWORK-NWORK+1, IERR )
                  CALL ZLACPY( 'F', M, N, WORK( IU ), LDWRKU, A, LDA )
               ELSE
*
*                 Generate Q in A
*                 (Cworkspace: need 2*N, prefer N+N*NB)
*                 (Rworkspace: need 0)
*
                  CALL ZUNGBR( 'Q', M, N, N, A, LDA, WORK( ITAUQ ),
     $                         WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*                 Multiply Q in A by real matrix RWORK(IRU), storing the
*                 result in WORK(IU), copying to A
*                 (CWorkspace: need N*N, prefer M*N)
*                 (Rworkspace: need 3*N*N, prefer N*N+2*M*N)
*
                  NRWORK = IRVT
                  DO 30 I = 1, M, LDWRKU
                     CHUNK = MIN( M-I+1, LDWRKU )
                     CALL ZLACRM( CHUNK, N, A( I, 1 ), LDA,
     $                            RWORK( IRU ), N, WORK( IU ), LDWRKU,
     $                            RWORK( NRWORK ) )
                     CALL ZLACPY( 'F', CHUNK, N, WORK( IU ), LDWRKU,
     $                            A( I, 1 ), LDA )
   30             CONTINUE
               END IF
*
            ELSE IF( WNTQS ) THEN
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = NRWORK
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix U
*              Overwrite U by left singular vectors of A
*              (CWorkspace: need 3*N, prefer 2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLASET( 'F', M, N, CZERO, CZERO, U, LDU )
               CALL ZLACP2( 'F', N, N, RWORK( IRU ), N, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', M, N, N, A, LDA,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by right singular vectors of A
*              (CWorkspace: need 3*N, prefer 2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRVT ), N, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', N, N, N, A, LDA,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
            ELSE
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = NRWORK
               IRVT = IRU + N*N
               NRWORK = IRVT + N*N
               CALL DBDSDC( 'U', 'I', N, S, RWORK( IE ), RWORK( IRU ),
     $                      N, RWORK( IRVT ), N, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Set the right corner of U to identity matrix
*
               CALL ZLASET( 'F', M, M, CZERO, CZERO, U, LDU )
               IF( M.GT.N ) THEN
                  CALL ZLASET( 'F', M-N, M-N, CZERO, CONE,
     $                         U( N+1, N+1 ), LDU )
               END IF
*
*              Copy real matrix RWORK(IRU) to complex matrix U
*              Overwrite U by left singular vectors of A
*              (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRU ), N, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', M, M, N, A, LDA,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by right singular vectors of A
*              (CWorkspace: need 3*N, prefer 2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', N, N, RWORK( IRVT ), N, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', N, N, N, A, LDA,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
            END IF
*
         END IF
*
      ELSE
*
*        A has more columns than rows. If A has sufficiently more
*        columns than rows, first reduce using the LQ decomposition (if
*        sufficient workspace available)
*
         IF( N.GE.MNTHR1 ) THEN
*
            IF( WNTQN ) THEN
*
*              Path 1t (N much larger than M, JOBZ='N')
*              No singular vectors to be computed
*
               ITAU = 1
               NWORK = ITAU + M
*
*              Compute A=L*Q
*              (CWorkspace: need 2*M, prefer M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZGELQF( M, N, A, LDA, WORK( ITAU ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Zero out above L
*
               CALL ZLASET( 'U', M-1, M-1, CZERO, CZERO, A( 1, 2 ),
     $                      LDA )
               IE = 1
               ITAUQ = 1
               ITAUP = ITAUQ + M
               NWORK = ITAUP + M
*
*              Bidiagonalize L in A
*              (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
*              (RWorkspace: need M)
*
               CALL ZGEBRD( M, M, A, LDA, S, RWORK( IE ), WORK( ITAUQ ),
     $                      WORK( ITAUP ), WORK( NWORK ), LWORK-NWORK+1,
     $                      IERR )
               NRWORK = IE + M
*
*              Perform bidiagonal SVD, compute singular values only
*              (CWorkspace: 0)
*              (RWorkspace: need BDSPAN)
*
               CALL DBDSDC( 'U', 'N', M, S, RWORK( IE ), DUM, 1, DUM, 1,
     $                      DUM, IDUM, RWORK( NRWORK ), IWORK, INFO )
*
            ELSE IF( WNTQO ) THEN
*
*              Path 2t (N much larger than M, JOBZ='O')
*              M right singular vectors to be overwritten on A and
*              M left singular vectors to be computed in U
*
               IVT = 1
               LDWKVT = M
*
*              WORK(IVT) is M by M
*
               IL = IVT + LDWKVT*M
               IF( LWORK.GE.M*N+M*M+3*M ) THEN
*
*                 WORK(IL) M by N
*
                  LDWRKL = M
                  CHUNK = N
               ELSE
*
*                 WORK(IL) is M by CHUNK
*
                  LDWRKL = M
                  CHUNK = ( LWORK-M*M-3*M ) / M
               END IF
               ITAU = IL + LDWRKL*CHUNK
               NWORK = ITAU + M
*
*              Compute A=L*Q
*              (CWorkspace: need 2*M, prefer M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZGELQF( M, N, A, LDA, WORK( ITAU ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy L to WORK(IL), zeroing about above it
*
               CALL ZLACPY( 'L', M, M, A, LDA, WORK( IL ), LDWRKL )
               CALL ZLASET( 'U', M-1, M-1, CZERO, CZERO,
     $                      WORK( IL+LDWRKL ), LDWRKL )
*
*              Generate Q in A
*              (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZUNGLQ( M, N, M, A, LDA, WORK( ITAU ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
               IE = 1
               ITAUQ = ITAU
               ITAUP = ITAUQ + M
               NWORK = ITAUP + M
*
*              Bidiagonalize L in WORK(IL)
*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
*              (RWorkspace: need M)
*
               CALL ZGEBRD( M, M, WORK( IL ), LDWRKL, S, RWORK( IE ),
     $                      WORK( ITAUQ ), WORK( ITAUP ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = IE + M
               IRVT = IRU + M*M
               NRWORK = IRVT + M*M
               CALL DBDSDC( 'U', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
*              Overwrite WORK(IU) by the left singular vectors of L
*              (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRU ), M, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', M, M, M, WORK( IL ), LDWRKL,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
*              Overwrite WORK(IVT) by the right singular vectors of L
*              (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRVT ), M, WORK( IVT ),
     $                      LDWKVT )
               CALL ZUNMBR( 'P', 'R', 'C', M, M, M, WORK( IL ), LDWRKL,
     $                      WORK( ITAUP ), WORK( IVT ), LDWKVT,
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Multiply right singular vectors of L in WORK(IL) by Q
*              in A, storing result in WORK(IL) and copying to A
*              (CWorkspace: need 2*M*M, prefer M*M+M*N))
*              (RWorkspace: 0)
*
               DO 40 I = 1, N, CHUNK
                  BLK = MIN( N-I+1, CHUNK )
                  CALL ZGEMM( 'N', 'N', M, BLK, M, CONE, WORK( IVT ), M,
     $                        A( 1, I ), LDA, CZERO, WORK( IL ),
     $                        LDWRKL )
                  CALL ZLACPY( 'F', M, BLK, WORK( IL ), LDWRKL,
     $                         A( 1, I ), LDA )
   40          CONTINUE
*
            ELSE IF( WNTQS ) THEN
*
*             Path 3t (N much larger than M, JOBZ='S')
*             M right singular vectors to be computed in VT and
*             M left singular vectors to be computed in U
*
               IL = 1
*
*              WORK(IL) is M by M
*
               LDWRKL = M
               ITAU = IL + LDWRKL*M
               NWORK = ITAU + M
*
*              Compute A=L*Q
*              (CWorkspace: need 2*M, prefer M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZGELQF( M, N, A, LDA, WORK( ITAU ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy L to WORK(IL), zeroing out above it
*
               CALL ZLACPY( 'L', M, M, A, LDA, WORK( IL ), LDWRKL )
               CALL ZLASET( 'U', M-1, M-1, CZERO, CZERO,
     $                      WORK( IL+LDWRKL ), LDWRKL )
*
*              Generate Q in A
*              (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZUNGLQ( M, N, M, A, LDA, WORK( ITAU ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
               IE = 1
               ITAUQ = ITAU
               ITAUP = ITAUQ + M
               NWORK = ITAUP + M
*
*              Bidiagonalize L in WORK(IL)
*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
*              (RWorkspace: need M)
*
               CALL ZGEBRD( M, M, WORK( IL ), LDWRKL, S, RWORK( IE ),
     $                      WORK( ITAUQ ), WORK( ITAUP ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = IE + M
               IRVT = IRU + M*M
               NRWORK = IRVT + M*M
               CALL DBDSDC( 'U', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix U
*              Overwrite U by left singular vectors of L
*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRU ), M, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', M, M, M, WORK( IL ), LDWRKL,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by left singular vectors of L
*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRVT ), M, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', M, M, M, WORK( IL ), LDWRKL,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy VT to WORK(IL), multiply right singular vectors of L
*              in WORK(IL) by Q in A, storing result in VT
*              (CWorkspace: need M*M)
*              (RWorkspace: 0)
*
               CALL ZLACPY( 'F', M, M, VT, LDVT, WORK( IL ), LDWRKL )
               CALL ZGEMM( 'N', 'N', M, N, M, CONE, WORK( IL ), LDWRKL,
     $                     A, LDA, CZERO, VT, LDVT )
*
            ELSE IF( WNTQA ) THEN
*
*              Path 9t (N much larger than M, JOBZ='A')
*              N right singular vectors to be computed in VT and
*              M left singular vectors to be computed in U
*
               IVT = 1
*
*              WORK(IVT) is M by M
*
               LDWKVT = M
               ITAU = IVT + LDWKVT*M
               NWORK = ITAU + M
*
*              Compute A=L*Q, copying result to VT
*              (CWorkspace: need 2*M, prefer M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZGELQF( M, N, A, LDA, WORK( ITAU ), WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
               CALL ZLACPY( 'U', M, N, A, LDA, VT, LDVT )
*
*              Generate Q in VT
*              (CWorkspace: need M+N, prefer M+N*NB)
*              (RWorkspace: 0)
*
               CALL ZUNGLQ( N, N, M, VT, LDVT, WORK( ITAU ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Produce L in A, zeroing out above it
*
               CALL ZLASET( 'U', M-1, M-1, CZERO, CZERO, A( 1, 2 ),
     $                      LDA )
               IE = 1
               ITAUQ = ITAU
               ITAUP = ITAUQ + M
               NWORK = ITAUP + M
*
*              Bidiagonalize L in A
*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
*              (RWorkspace: need M)
*
               CALL ZGEBRD( M, M, A, LDA, S, RWORK( IE ), WORK( ITAUQ ),
     $                      WORK( ITAUP ), WORK( NWORK ), LWORK-NWORK+1,
     $                      IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRU = IE + M
               IRVT = IRU + M*M
               NRWORK = IRVT + M*M
               CALL DBDSDC( 'U', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix U
*              Overwrite U by left singular vectors of L
*              (CWorkspace: need 3*M, prefer 2*M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRU ), M, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', M, M, M, A, LDA,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
*              Overwrite WORK(IVT) by right singular vectors of L
*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
*              (RWorkspace: 0)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRVT ), M, WORK( IVT ),
     $                      LDWKVT )
               CALL ZUNMBR( 'P', 'R', 'C', M, M, M, A, LDA,
     $                      WORK( ITAUP ), WORK( IVT ), LDWKVT,
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Multiply right singular vectors of L in WORK(IVT) by
*              Q in VT, storing result in A
*              (CWorkspace: need M*M)
*              (RWorkspace: 0)
*
               CALL ZGEMM( 'N', 'N', M, N, M, CONE, WORK( IVT ), LDWKVT,
     $                     VT, LDVT, CZERO, A, LDA )
*
*              Copy right singular vectors of A from A to VT
*
               CALL ZLACPY( 'F', M, N, A, LDA, VT, LDVT )
*
            END IF
*
         ELSE IF( N.GE.MNTHR2 ) THEN
*
*           MNTHR2 <= N < MNTHR1
*
*           Path 5t (N much larger than M, but not as much as MNTHR1)
*           Reduce to bidiagonal form without QR decomposition, use
*           ZUNGBR and matrix multiplication to compute singular vectors
*
*
            IE = 1
            NRWORK = IE + M
            ITAUQ = 1
            ITAUP = ITAUQ + M
            NWORK = ITAUP + M
*
*           Bidiagonalize A
*           (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
*           (RWorkspace: M)
*
            CALL ZGEBRD( M, N, A, LDA, S, RWORK( IE ), WORK( ITAUQ ),
     $                   WORK( ITAUP ), WORK( NWORK ), LWORK-NWORK+1,
     $                   IERR )
*
            IF( WNTQN ) THEN
*
*              Compute singular values only
*              (Cworkspace: 0)
*              (Rworkspace: need BDSPAN)
*
               CALL DBDSDC( 'L', 'N', M, S, RWORK( IE ), DUM, 1, DUM, 1,
     $                      DUM, IDUM, RWORK( NRWORK ), IWORK, INFO )
            ELSE IF( WNTQO ) THEN
               IRVT = NRWORK
               IRU = IRVT + M*M
               NRWORK = IRU + M*M
               IVT = NWORK
*
*              Copy A to U, generate Q
*              (Cworkspace: need 2*M, prefer M+M*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'L', M, M, A, LDA, U, LDU )
               CALL ZUNGBR( 'Q', M, M, N, U, LDU, WORK( ITAUQ ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Generate P**H in A
*              (Cworkspace: need 2*M, prefer M+M*NB)
*              (Rworkspace: 0)
*
               CALL ZUNGBR( 'P', M, N, M, A, LDA, WORK( ITAUP ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
               LDWKVT = M
               IF( LWORK.GE.M*N+3*M ) THEN
*
*                 WORK( IVT ) is M by N
*
                  NWORK = IVT + LDWKVT*N
                  CHUNK = N
               ELSE
*
*                 WORK( IVT ) is M by CHUNK
*
                  CHUNK = ( LWORK-3*M ) / M
                  NWORK = IVT + LDWKVT*CHUNK
               END IF
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               CALL DBDSDC( 'L', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Multiply Q in U by real matrix RWORK(IRVT)
*              storing the result in WORK(IVT), copying to U
*              (Cworkspace: need 0)
*              (Rworkspace: need 2*M*M)
*
               CALL ZLACRM( M, M, U, LDU, RWORK( IRU ), M, WORK( IVT ),
     $                      LDWKVT, RWORK( NRWORK ) )
               CALL ZLACPY( 'F', M, M, WORK( IVT ), LDWKVT, U, LDU )
*
*              Multiply RWORK(IRVT) by P**H in A, storing the
*              result in WORK(IVT), copying to A
*              (CWorkspace: need M*M, prefer M*N)
*              (Rworkspace: need 2*M*M, prefer 2*M*N)
*
               NRWORK = IRU
               DO 50 I = 1, N, CHUNK
                  BLK = MIN( N-I+1, CHUNK )
                  CALL ZLARCM( M, BLK, RWORK( IRVT ), M, A( 1, I ), LDA,
     $                         WORK( IVT ), LDWKVT, RWORK( NRWORK ) )
                  CALL ZLACPY( 'F', M, BLK, WORK( IVT ), LDWKVT,
     $                         A( 1, I ), LDA )
   50          CONTINUE
            ELSE IF( WNTQS ) THEN
*
*              Copy A to U, generate Q
*              (Cworkspace: need 2*M, prefer M+M*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'L', M, M, A, LDA, U, LDU )
               CALL ZUNGBR( 'Q', M, M, N, U, LDU, WORK( ITAUQ ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Copy A to VT, generate P**H
*              (Cworkspace: need 2*M, prefer M+M*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'U', M, N, A, LDA, VT, LDVT )
               CALL ZUNGBR( 'P', M, N, M, VT, LDVT, WORK( ITAUP ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRVT = NRWORK
               IRU = IRVT + M*M
               NRWORK = IRU + M*M
               CALL DBDSDC( 'L', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Multiply Q in U by real matrix RWORK(IRU), storing the
*              result in A, copying to U
*              (CWorkspace: need 0)
*              (Rworkspace: need 3*M*M)
*
               CALL ZLACRM( M, M, U, LDU, RWORK( IRU ), M, A, LDA,
     $                      RWORK( NRWORK ) )
               CALL ZLACPY( 'F', M, M, A, LDA, U, LDU )
*
*              Multiply real matrix RWORK(IRVT) by P**H in VT,
*              storing the result in A, copying to VT
*              (Cworkspace: need 0)
*              (Rworkspace: need M*M+2*M*N)
*
               NRWORK = IRU
               CALL ZLARCM( M, N, RWORK( IRVT ), M, VT, LDVT, A, LDA,
     $                      RWORK( NRWORK ) )
               CALL ZLACPY( 'F', M, N, A, LDA, VT, LDVT )
            ELSE
*
*              Copy A to U, generate Q
*              (Cworkspace: need 2*M, prefer M+M*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'L', M, M, A, LDA, U, LDU )
               CALL ZUNGBR( 'Q', M, M, N, U, LDU, WORK( ITAUQ ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Copy A to VT, generate P**H
*              (Cworkspace: need 2*M, prefer M+M*NB)
*              (Rworkspace: 0)
*
               CALL ZLACPY( 'U', M, N, A, LDA, VT, LDVT )
               CALL ZUNGBR( 'P', N, N, M, VT, LDVT, WORK( ITAUP ),
     $                      WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRVT = NRWORK
               IRU = IRVT + M*M
               NRWORK = IRU + M*M
               CALL DBDSDC( 'L', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Multiply Q in U by real matrix RWORK(IRU), storing the
*              result in A, copying to U
*              (CWorkspace: need 0)
*              (Rworkspace: need 3*M*M)
*
               CALL ZLACRM( M, M, U, LDU, RWORK( IRU ), M, A, LDA,
     $                      RWORK( NRWORK ) )
               CALL ZLACPY( 'F', M, M, A, LDA, U, LDU )
*
*              Multiply real matrix RWORK(IRVT) by P**H in VT,
*              storing the result in A, copying to VT
*              (Cworkspace: need 0)
*              (Rworkspace: need M*M+2*M*N)
*
               CALL ZLARCM( M, N, RWORK( IRVT ), M, VT, LDVT, A, LDA,
     $                      RWORK( NRWORK ) )
               CALL ZLACPY( 'F', M, N, A, LDA, VT, LDVT )
            END IF
*
         ELSE
*
*           N .LT. MNTHR2
*
*           Path 6t (N greater than M, but not much larger)
*           Reduce to bidiagonal form without LQ decomposition
*           Use ZUNMBR to compute singular vectors
*
            IE = 1
            NRWORK = IE + M
            ITAUQ = 1
            ITAUP = ITAUQ + M
            NWORK = ITAUP + M
*
*           Bidiagonalize A
*           (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
*           (RWorkspace: M)
*
            CALL ZGEBRD( M, N, A, LDA, S, RWORK( IE ), WORK( ITAUQ ),
     $                   WORK( ITAUP ), WORK( NWORK ), LWORK-NWORK+1,
     $                   IERR )
            IF( WNTQN ) THEN
*
*              Compute singular values only
*              (Cworkspace: 0)
*              (Rworkspace: need BDSPAN)
*
               CALL DBDSDC( 'L', 'N', M, S, RWORK( IE ), DUM, 1, DUM, 1,
     $                      DUM, IDUM, RWORK( NRWORK ), IWORK, INFO )
            ELSE IF( WNTQO ) THEN
               LDWKVT = M
               IVT = NWORK
               IF( LWORK.GE.M*N+3*M ) THEN
*
*                 WORK( IVT ) is M by N
*
                  CALL ZLASET( 'F', M, N, CZERO, CZERO, WORK( IVT ),
     $                         LDWKVT )
                  NWORK = IVT + LDWKVT*N
               ELSE
*
*                 WORK( IVT ) is M by CHUNK
*
                  CHUNK = ( LWORK-3*M ) / M
                  NWORK = IVT + LDWKVT*CHUNK
               END IF
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRVT = NRWORK
               IRU = IRVT + M*M
               NRWORK = IRU + M*M
               CALL DBDSDC( 'L', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix U
*              Overwrite U by left singular vectors of A
*              (Cworkspace: need 2*M, prefer M+M*NB)
*              (Rworkspace: need 0)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRU ), M, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', M, M, N, A, LDA,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
               IF( LWORK.GE.M*N+3*M ) THEN
*
*              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
*              Overwrite WORK(IVT) by right singular vectors of A,
*              copying to A
*              (Cworkspace: need M*N+2*M, prefer M*N+M+M*NB)
*              (Rworkspace: need 0)
*
                  CALL ZLACP2( 'F', M, M, RWORK( IRVT ), M, WORK( IVT ),
     $                         LDWKVT )
                  CALL ZUNMBR( 'P', 'R', 'C', M, N, M, A, LDA,
     $                         WORK( ITAUP ), WORK( IVT ), LDWKVT,
     $                         WORK( NWORK ), LWORK-NWORK+1, IERR )
                  CALL ZLACPY( 'F', M, N, WORK( IVT ), LDWKVT, A, LDA )
               ELSE
*
*                 Generate P**H in A
*                 (Cworkspace: need 2*M, prefer M+M*NB)
*                 (Rworkspace: need 0)
*
                  CALL ZUNGBR( 'P', M, N, M, A, LDA, WORK( ITAUP ),
     $                         WORK( NWORK ), LWORK-NWORK+1, IERR )
*
*                 Multiply Q in A by real matrix RWORK(IRU), storing the
*                 result in WORK(IU), copying to A
*                 (CWorkspace: need M*M, prefer M*N)
*                 (Rworkspace: need 3*M*M, prefer M*M+2*M*N)
*
                  NRWORK = IRU
                  DO 60 I = 1, N, CHUNK
                     BLK = MIN( N-I+1, CHUNK )
                     CALL ZLARCM( M, BLK, RWORK( IRVT ), M, A( 1, I ),
     $                            LDA, WORK( IVT ), LDWKVT,
     $                            RWORK( NRWORK ) )
                     CALL ZLACPY( 'F', M, BLK, WORK( IVT ), LDWKVT,
     $                            A( 1, I ), LDA )
   60             CONTINUE
               END IF
            ELSE IF( WNTQS ) THEN
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRVT = NRWORK
               IRU = IRVT + M*M
               NRWORK = IRU + M*M
               CALL DBDSDC( 'L', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix U
*              Overwrite U by left singular vectors of A
*              (CWorkspace: need 3*M, prefer 2*M+M*NB)
*              (RWorkspace: M*M)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRU ), M, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', M, M, N, A, LDA,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by right singular vectors of A
*              (CWorkspace: need 3*M, prefer 2*M+M*NB)
*              (RWorkspace: M*M)
*
               CALL ZLASET( 'F', M, N, CZERO, CZERO, VT, LDVT )
               CALL ZLACP2( 'F', M, M, RWORK( IRVT ), M, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', M, N, M, A, LDA,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
            ELSE
*
*              Perform bidiagonal SVD, computing left singular vectors
*              of bidiagonal matrix in RWORK(IRU) and computing right
*              singular vectors of bidiagonal matrix in RWORK(IRVT)
*              (CWorkspace: need 0)
*              (RWorkspace: need BDSPAC)
*
               IRVT = NRWORK
               IRU = IRVT + M*M
               NRWORK = IRU + M*M
*
               CALL DBDSDC( 'L', 'I', M, S, RWORK( IE ), RWORK( IRU ),
     $                      M, RWORK( IRVT ), M, DUM, IDUM,
     $                      RWORK( NRWORK ), IWORK, INFO )
*
*              Copy real matrix RWORK(IRU) to complex matrix U
*              Overwrite U by left singular vectors of A
*              (CWorkspace: need 3*M, prefer 2*M+M*NB)
*              (RWorkspace: M*M)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRU ), M, U, LDU )
               CALL ZUNMBR( 'Q', 'L', 'N', M, M, N, A, LDA,
     $                      WORK( ITAUQ ), U, LDU, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
*
*              Set all of VT to identity matrix
*
               CALL ZLASET( 'F', N, N, CZERO, CONE, VT, LDVT )
*
*              Copy real matrix RWORK(IRVT) to complex matrix VT
*              Overwrite VT by right singular vectors of A
*              (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
*              (RWorkspace: M*M)
*
               CALL ZLACP2( 'F', M, M, RWORK( IRVT ), M, VT, LDVT )
               CALL ZUNMBR( 'P', 'R', 'C', N, N, M, A, LDA,
     $                      WORK( ITAUP ), VT, LDVT, WORK( NWORK ),
     $                      LWORK-NWORK+1, IERR )
            END IF
*
         END IF
*
      END IF
*
*     Undo scaling if necessary
*
      IF( ISCL.EQ.1 ) THEN
         IF( ANRM.GT.BIGNUM )
     $      CALL DLASCL( 'G', 0, 0, BIGNUM, ANRM, MINMN, 1, S, MINMN,
     $                   IERR )
         IF( INFO.NE.0 .AND. ANRM.GT.BIGNUM )
     $      CALL DLASCL( 'G', 0, 0, BIGNUM, ANRM, MINMN-1, 1,
     $                   RWORK( IE ), MINMN, IERR )
         IF( ANRM.LT.SMLNUM )
     $      CALL DLASCL( 'G', 0, 0, SMLNUM, ANRM, MINMN, 1, S, MINMN,
     $                   IERR )
         IF( INFO.NE.0 .AND. ANRM.LT.SMLNUM )
     $      CALL DLASCL( 'G', 0, 0, SMLNUM, ANRM, MINMN-1, 1,
     $                   RWORK( IE ), MINMN, IERR )
      END IF
*
*     Return optimal workspace in WORK(1)
*
      WORK( 1 ) = MAXWRK
*
      RETURN
*
*     End of ZGESDD
*
      END
