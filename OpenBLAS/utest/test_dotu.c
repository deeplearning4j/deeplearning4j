/*****************************************************************************
Copyright (c) 2011-2014, The OpenBLAS Project
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
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

#include "common_utest.h"
#include <complex.h>

void test_zdotu_n_1(void)
{
	int N=1,incX=1,incY=1;
	double x1[]={1.0,1.0};
	double y1[]={1.0,2.0};
	double x2[]={1.0,1.0};
	double y2[]={1.0,2.0};
	double _Complex result1=0.0;
	double _Complex result2=0.0;
	//OpenBLAS
	result1=BLASFUNC(zdotu)(&N,x1,&incX,y1,&incY);
	//reference
	result2=BLASFUNC_REF(zdotu)(&N,x2,&incX,y2,&incY);

	CU_ASSERT_DOUBLE_EQUAL(creal(result1), creal(result2), CHECK_EPS);
	CU_ASSERT_DOUBLE_EQUAL(cimag(result1), cimag(result2), CHECK_EPS);
//	printf("\%lf,%lf\n",creal(result1),cimag(result1));

}

void test_zdotu_offset_1(void)
{
	int N=1,incX=1,incY=1;
	double x1[]={1.0,2.0,3.0,4.0};
	double y1[]={5.0,6.0,7.0,8.0};
	double x2[]={1.0,2.0,3.0,4.0};
	double y2[]={5.0,6.0,7.0,8.0};
	double _Complex result1=0.0;
	double _Complex result2=0.0;
	//OpenBLAS
	result1=BLASFUNC(zdotu)(&N,x1+1,&incX,y1+1,&incY);
	//reference
	result2=BLASFUNC_REF(zdotu)(&N,x2+1,&incX,y2+1,&incY);

	CU_ASSERT_DOUBLE_EQUAL(creal(result1), creal(result2), CHECK_EPS);
	CU_ASSERT_DOUBLE_EQUAL(cimag(result1), cimag(result2), CHECK_EPS);
//	printf("\%lf,%lf\n",creal(result1),cimag(result1));

}

