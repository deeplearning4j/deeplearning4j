/*****************************************************************************
  Copyright (c) 2014, Intel Corp.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
  THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************
* Contents: Native middle-level C interface to LAPACK function zbbcsd
* Author: Intel Corporation
* Generated November 2015
*****************************************************************************/

#include "lapacke_utils.h"

lapack_int LAPACKE_zbbcsd_work( int matrix_layout, char jobu1, char jobu2,
                                char jobv1t, char jobv2t, char trans,
                                lapack_int m, lapack_int p, lapack_int q,
                                double* theta, double* phi,
                                lapack_complex_double* u1, lapack_int ldu1,
                                lapack_complex_double* u2, lapack_int ldu2,
                                lapack_complex_double* v1t, lapack_int ldv1t,
                                lapack_complex_double* v2t, lapack_int ldv2t,
                                double* b11d, double* b11e, double* b12d,
                                double* b12e, double* b21d, double* b21e,
                                double* b22d, double* b22e, double* rwork,
                                lapack_int lrwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        /* Call LAPACK function and adjust info */
        LAPACK_zbbcsd( &jobu1, &jobu2, &jobv1t, &jobv2t, &trans, &m, &p, &q,
                       theta, phi, u1, &ldu1, u2, &ldu2, v1t, &ldv1t, v2t,
                       &ldv2t, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e,
                       rwork, &lrwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int nrows_u1 = ( LAPACKE_lsame( jobu1, 'y' ) ? p : 1);
        lapack_int nrows_u2 = ( LAPACKE_lsame( jobu2, 'y' ) ? m-p : 1);
        lapack_int nrows_v1t = ( LAPACKE_lsame( jobv1t, 'y' ) ? q : 1);
        lapack_int nrows_v2t = ( LAPACKE_lsame( jobv2t, 'y' ) ? m-q : 1);
        lapack_int ldu1_t = MAX(1,nrows_u1);
        lapack_int ldu2_t = MAX(1,nrows_u2);
        lapack_int ldv1t_t = MAX(1,nrows_v1t);
        lapack_int ldv2t_t = MAX(1,nrows_v2t);
        lapack_complex_double* u1_t = NULL;
        lapack_complex_double* u2_t = NULL;
        lapack_complex_double* v1t_t = NULL;
        lapack_complex_double* v2t_t = NULL;
        /* Check leading dimension(s) */
        if( ldu1 < p ) {
            info = -13;
            LAPACKE_xerbla( "LAPACKE_zbbcsd_work", info );
            return info;
        }
        if( ldu2 < m-p ) {
            info = -15;
            LAPACKE_xerbla( "LAPACKE_zbbcsd_work", info );
            return info;
        }
        if( ldv1t < q ) {
            info = -17;
            LAPACKE_xerbla( "LAPACKE_zbbcsd_work", info );
            return info;
        }
        if( ldv2t < m-q ) {
            info = -19;
            LAPACKE_xerbla( "LAPACKE_zbbcsd_work", info );
            return info;
        }
        /* Query optimal working array(s) size if requested */
        if( lrwork == -1 ) {
            LAPACK_zbbcsd( &jobu1, &jobu2, &jobv1t, &jobv2t, &trans, &m, &p, &q,
                           theta, phi, u1, &ldu1_t, u2, &ldu2_t, v1t, &ldv1t_t,
                           v2t, &ldv2t_t, b11d, b11e, b12d, b12e, b21d, b21e,
                           b22d, b22e, rwork, &lrwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        /* Allocate memory for temporary array(s) */
        if( LAPACKE_lsame( jobu1, 'y' ) ) {
            u1_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldu1_t * MAX(1,p) );
            if( u1_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_0;
            }
        }
        if( LAPACKE_lsame( jobu2, 'y' ) ) {
            u2_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldu2_t * MAX(1,m-p) );
            if( u2_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobv1t, 'y' ) ) {
            v1t_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldv1t_t * MAX(1,q) );
            if( v1t_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        if( LAPACKE_lsame( jobv2t, 'y' ) ) {
            v2t_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldv2t_t * MAX(1,m-q) );
            if( v2t_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_3;
            }
        }
        /* Transpose input matrices */
        if( LAPACKE_lsame( jobu1, 'y' ) ) {
            LAPACKE_zge_trans( matrix_layout, nrows_u1, p, u1, ldu1, u1_t,
                               ldu1_t );
        }
        if( LAPACKE_lsame( jobu2, 'y' ) ) {
            LAPACKE_zge_trans( matrix_layout, nrows_u2, m-p, u2, ldu2, u2_t,
                               ldu2_t );
        }
        if( LAPACKE_lsame( jobv1t, 'y' ) ) {
            LAPACKE_zge_trans( matrix_layout, nrows_v1t, q, v1t, ldv1t, v1t_t,
                               ldv1t_t );
        }
        if( LAPACKE_lsame( jobv2t, 'y' ) ) {
            LAPACKE_zge_trans( matrix_layout, nrows_v2t, m-q, v2t, ldv2t, v2t_t,
                               ldv2t_t );
        }
        /* Call LAPACK function and adjust info */
        LAPACK_zbbcsd( &jobu1, &jobu2, &jobv1t, &jobv2t, &trans, &m, &p, &q,
                       theta, phi, u1_t, &ldu1_t, u2_t, &ldu2_t, v1t_t,
                       &ldv1t_t, v2t_t, &ldv2t_t, b11d, b11e, b12d, b12e, b21d,
                       b21e, b22d, b22e, rwork, &lrwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        /* Transpose output matrices */
        if( LAPACKE_lsame( jobu1, 'y' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, nrows_u1, p, u1_t, ldu1_t, u1,
                               ldu1 );
        }
        if( LAPACKE_lsame( jobu2, 'y' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, nrows_u2, m-p, u2_t, ldu2_t,
                               u2, ldu2 );
        }
        if( LAPACKE_lsame( jobv1t, 'y' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, nrows_v1t, q, v1t_t, ldv1t_t,
                               v1t, ldv1t );
        }
        if( LAPACKE_lsame( jobv2t, 'y' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, nrows_v2t, m-q, v2t_t, ldv2t_t,
                               v2t, ldv2t );
        }
        /* Release memory and exit */
        if( LAPACKE_lsame( jobv2t, 'y' ) ) {
            LAPACKE_free( v2t_t );
        }
exit_level_3:
        if( LAPACKE_lsame( jobv1t, 'y' ) ) {
            LAPACKE_free( v1t_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobu2, 'y' ) ) {
            LAPACKE_free( u2_t );
        }
exit_level_1:
        if( LAPACKE_lsame( jobu1, 'y' ) ) {
            LAPACKE_free( u1_t );
        }
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zbbcsd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zbbcsd_work", info );
    }
    return info;
}
