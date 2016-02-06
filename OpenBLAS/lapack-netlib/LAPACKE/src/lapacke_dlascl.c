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
* Contents: Native high-level C interface to LAPACK function dlaswp
* Author: Intel Corporation
* Generated November, 2011
*****************************************************************************/

#include "lapacke_utils.h"

lapack_int LAPACKE_dlascl( int matrix_layout, char type, lapack_int kl,
                           lapack_int ku, double cfrom, double cto, 
                           lapack_int m, lapack_int n, double* a, 
                           lapack_int lda )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dlascl", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    /* Optionally check input matrices for NaNs */
    switch (type) {
    case 'G':
       if( LAPACKE_dge_nancheck( matrix_layout, lda, n, a, lda ) ) {
           return -9;
           }
        break;
    case 'L':
       // TYPE = 'L' - lower triangular matrix.
       if( LAPACKE_dtr_nancheck( matrix_layout, 'L', 'N', n, a, lda ) ) {
           return -9;
          }
        break;
    case 'U':
       // TYPE = 'U' - upper triangular matrix
       if( LAPACKE_dtr_nancheck( matrix_layout, 'U', 'N', n, a, lda ) ) {
           return -9;
           } 
        break;
    case 'H':
       // TYPE = 'H' - upper Hessenberg matrix   
       if( LAPACKE_dhs_nancheck( matrix_layout, n, a, lda ) ) {
           return -9;
           }    
        break;
    case 'B':
       // TYPE = 'B' - A is a symmetric band matrix with lower bandwidth KL
       //             and upper bandwidth KU and with the only the lower
       //             half stored.   
       if( LAPACKE_dsb_nancheck( matrix_layout, 'L', n, kl, a, lda ) ) {
           return -9;
           }
         break;
   case 'Q':
       // TYPE = 'Q' - A is a symmetric band matrix with lower bandwidth KL
       //             and upper bandwidth KU and with the only the upper
       //             half stored.   
       if( LAPACKE_dsb_nancheck( matrix_layout, 'U', n, ku, a, lda ) ) {
           return -9;
           }
        break;
    case 'Z':
       // TYPE = 'Z' -  A is a band matrix with lower bandwidth KL and upper
       //             bandwidth KU. See DGBTRF for storage details.        
       if( LAPACKE_dgb_nancheck( matrix_layout, n, n, kl, kl+ku, a, lda ) ) {
           return -6;
           }
        break;
    }
#endif
    return LAPACKE_dlascl_work( matrix_layout, type, kl, ku, cfrom, cto, m,  n, a, lda );
}
