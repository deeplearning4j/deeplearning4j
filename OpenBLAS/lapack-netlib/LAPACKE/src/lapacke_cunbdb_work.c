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
* Contents: Native middle-level C interface to LAPACK function cunbdb
* Author: Intel Corporation
* Generated November 2015
*****************************************************************************/

#include "lapacke_utils.h"

lapack_int LAPACKE_cunbdb_work( int matrix_layout, char trans, char signs,
                                lapack_int m, lapack_int p, lapack_int q,
                                lapack_complex_float* x11, lapack_int ldx11,
                                lapack_complex_float* x12, lapack_int ldx12,
                                lapack_complex_float* x21, lapack_int ldx21,
                                lapack_complex_float* x22, lapack_int ldx22,
                                float* theta, float* phi,
                                lapack_complex_float* taup1,
                                lapack_complex_float* taup2,
                                lapack_complex_float* tauq1,
                                lapack_complex_float* tauq2,
                                lapack_complex_float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        /* Call LAPACK function and adjust info */
        LAPACK_cunbdb( &trans, &signs, &m, &p, &q, x11, &ldx11, x12, &ldx12,
                       x21, &ldx21, x22, &ldx22, theta, phi, taup1, taup2,
                       tauq1, tauq2, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int nrows_x11 = ( LAPACKE_lsame( trans, 'n' ) ? p : q);
        lapack_int nrows_x12 = ( LAPACKE_lsame( trans, 'n' ) ? p : m-q);
        lapack_int nrows_x21 = ( LAPACKE_lsame( trans, 'n' ) ? m-p : q);
        lapack_int nrows_x22 = ( LAPACKE_lsame( trans, 'n' ) ? m-p : m-q);
        lapack_int ldx11_t = MAX(1,nrows_x11);
        lapack_int ldx12_t = MAX(1,nrows_x12);
        lapack_int ldx21_t = MAX(1,nrows_x21);
        lapack_int ldx22_t = MAX(1,nrows_x22);
        lapack_complex_float* x11_t = NULL;
        lapack_complex_float* x12_t = NULL;
        lapack_complex_float* x21_t = NULL;
        lapack_complex_float* x22_t = NULL;
        /* Check leading dimension(s) */
        if( ldx11 < q ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_cunbdb_work", info );
            return info;
        }
        if( ldx12 < m-q ) {
            info = -10;
            LAPACKE_xerbla( "LAPACKE_cunbdb_work", info );
            return info;
        }
        if( ldx21 < q ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_cunbdb_work", info );
            return info;
        }
        if( ldx22 < m-q ) {
            info = -14;
            LAPACKE_xerbla( "LAPACKE_cunbdb_work", info );
            return info;
        }
        /* Query optimal working array(s) size if requested */
        if( lwork == -1 ) {
            LAPACK_cunbdb( &trans, &signs, &m, &p, &q, x11, &ldx11_t, x12,
                           &ldx12_t, x21, &ldx21_t, x22, &ldx22_t, theta, phi,
                           taup1, taup2, tauq1, tauq2, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        /* Allocate memory for temporary array(s) */
        x11_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * ldx11_t * MAX(1,q) );
        if( x11_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        x12_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) *
                            ldx12_t * MAX(1,m-q) );
        if( x12_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        x21_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * ldx21_t * MAX(1,q) );
        if( x21_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_2;
        }
        x22_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) *
                            ldx22_t * MAX(1,m-q) );
        if( x22_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_3;
        }
        /* Transpose input matrices */
        LAPACKE_cge_trans( matrix_layout, nrows_x11, q, x11, ldx11, x11_t,
                           ldx11_t );
        LAPACKE_cge_trans( matrix_layout, nrows_x12, m-q, x12, ldx12, x12_t,
                           ldx12_t );
        LAPACKE_cge_trans( matrix_layout, nrows_x21, q, x21, ldx21, x21_t,
                           ldx21_t );
        LAPACKE_cge_trans( matrix_layout, nrows_x22, m-q, x22, ldx22, x22_t,
                           ldx22_t );
        /* Call LAPACK function and adjust info */
        LAPACK_cunbdb( &trans, &signs, &m, &p, &q, x11_t, &ldx11_t, x12_t,
                       &ldx12_t, x21_t, &ldx21_t, x22_t, &ldx22_t, theta, phi,
                       taup1, taup2, tauq1, tauq2, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        /* Transpose output matrices */
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, nrows_x11, q, x11_t, ldx11_t, x11,
                           ldx11 );
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, nrows_x12, m-q, x12_t, ldx12_t,
                           x12, ldx12 );
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, nrows_x21, q, x21_t, ldx21_t, x21,
                           ldx21 );
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, nrows_x22, m-q, x22_t, ldx22_t,
                           x22, ldx22 );
        /* Release memory and exit */
        LAPACKE_free( x22_t );
exit_level_3:
        LAPACKE_free( x21_t );
exit_level_2:
        LAPACKE_free( x12_t );
exit_level_1:
        LAPACKE_free( x11_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cunbdb_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cunbdb_work", info );
    }
    return info;
}
