/***************************************************************************
Copyright (c) 2014, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/


#include "common.h"

#if defined(BULLDOZER) 
#include "sdot_microk_bulldozer-2.c"
#elif defined(STEAMROLLER) || defined(PILEDRIVER)
#include "sdot_microk_steamroller-2.c"
#elif defined(NEHALEM)
#include "sdot_microk_nehalem-2.c"
#elif defined(HASWELL)
#include "sdot_microk_haswell-2.c"
#elif defined(SANDYBRIDGE)
#include "sdot_microk_sandy-2.c"
#endif


#ifndef HAVE_KERNEL_16

static void sdot_kernel_16(BLASLONG n, FLOAT *x, FLOAT *y, FLOAT *d)
{
	BLASLONG register i = 0;
	FLOAT dot = 0.0;

	while(i < n)
        {
              dot += y[i]  * x[i]
                  + y[i+1] * x[i+1]
                  + y[i+2] * x[i+2]
                  + y[i+3] * x[i+3]
                  + y[i+4] * x[i+4]
                  + y[i+5] * x[i+5]
                  + y[i+6] * x[i+6]
                  + y[i+7] * x[i+7] ;

              i+=8 ;

       }
       *d += dot;

}

#endif

FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y)
{
	BLASLONG i=0;
	BLASLONG ix=0,iy=0;

	FLOAT  dot = 0.0 ;

	if ( n <= 0 )  return(dot);

	if ( (inc_x == 1) && (inc_y == 1) )
	{

		BLASLONG n1 = n & -32;

		if ( n1 )
			sdot_kernel_16(n1, x, y , &dot );


		i = n1;
		while(i < n)
		{

			dot += y[i] * x[i] ;
			i++ ;

		}
		return(dot);


	}

	BLASLONG n1 = n & -2;

	while(i < n1)
	{

		dot += y[iy] * x[ix] + y[iy+inc_y] * x[ix+inc_x];
		ix  += inc_x*2 ;
		iy  += inc_y*2 ;
		i+=2 ;

	}

	while(i < n)
	{

		dot += y[iy] * x[ix] ;
		ix  += inc_x ;
		iy  += inc_y ;
		i++ ;

	}
	return(dot);

}


