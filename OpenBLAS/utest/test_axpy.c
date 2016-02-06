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

void test_daxpy_inc_0(void)
{
	int i;
	int N=8,incX=0,incY=0;
	double a=0.25;
	double x1[]={1.0,3.0,5.0,7.0,1.0,3.0,5.0,7.0};
	double y1[]={2.0,4.0,6.0,8.0,2.0,4.0,6.0,8.0};
	double x2[]={1.0,3.0,5.0,7.0,1.0,3.0,5.0,7.0};
	double y2[]={2.0,4.0,6.0,8.0,2.0,4.0,6.0,8.0};

	//OpenBLAS
	BLASFUNC(daxpy)(&N,&a,x1,&incX,y1,&incY);
	//reference
	BLASFUNC_REF(daxpy)(&N,&a,x2,&incX,y2,&incY);

	for(i=0; i<N; i++){
		CU_ASSERT_DOUBLE_EQUAL(x1[i], x2[i], CHECK_EPS);
		CU_ASSERT_DOUBLE_EQUAL(y1[i], y2[i], CHECK_EPS);
	}
}

void test_zaxpy_inc_0(void)
{
	int i;
	int N=4,incX=0,incY=0;
	double a[2]={0.25,0.5};
	double x1[]={1.0,3.0,5.0,7.0,1.0,3.0,5.0,7.0};
	double y1[]={2.0,4.0,6.0,8.0,2.0,4.0,6.0,8.0};
	double x2[]={1.0,3.0,5.0,7.0,1.0,3.0,5.0,7.0};
	double y2[]={2.0,4.0,6.0,8.0,2.0,4.0,6.0,8.0};

	//OpenBLAS
	BLASFUNC(zaxpy)(&N,a,x1,&incX,y1,&incY);
	//reference
	BLASFUNC_REF(zaxpy)(&N,a,x2,&incX,y2,&incY);

	for(i=0; i<2*N; i++){
		CU_ASSERT_DOUBLE_EQUAL(x1[i], x2[i], CHECK_EPS);
		CU_ASSERT_DOUBLE_EQUAL(y1[i], y2[i], CHECK_EPS);
	}
}

void test_saxpy_inc_0(void)
{
	int i;
	int N=8,incX=0,incY=0;
	float a=0.25;
	float x1[]={1.0,3.0,5.0,7.0,1.0,3.0,5.0,7.0};
	float y1[]={2.0,4.0,6.0,8.0,2.0,4.0,6.0,8.0};
	float x2[]={1.0,3.0,5.0,7.0,1.0,3.0,5.0,7.0};
	float y2[]={2.0,4.0,6.0,8.0,2.0,4.0,6.0,8.0};

	//OpenBLAS
	BLASFUNC(saxpy)(&N,&a,x1,&incX,y1,&incY);
	//reference
	BLASFUNC_REF(saxpy)(&N,&a,x2,&incX,y2,&incY);

	for(i=0; i<N; i++){
		CU_ASSERT_DOUBLE_EQUAL(x1[i], x2[i], CHECK_EPS);
		CU_ASSERT_DOUBLE_EQUAL(y1[i], y2[i], CHECK_EPS);
	}
}

void test_caxpy_inc_0(void)
{
	int i;
	int N=4,incX=0,incY=0;
	float a[2]={0.25,0.5};
	float x1[]={1.0,3.0,5.0,7.0,1.0,3.0,5.0,7.0};
	float y1[]={2.0,4.0,6.0,8.0,2.0,4.0,6.0,8.0};
	float x2[]={1.0,3.0,5.0,7.0,1.0,3.0,5.0,7.0};
	float y2[]={2.0,4.0,6.0,8.0,2.0,4.0,6.0,8.0};

	//OpenBLAS
	BLASFUNC(caxpy)(&N,a,x1,&incX,y1,&incY);
	//reference
	BLASFUNC_REF(caxpy)(&N,a,x2,&incX,y2,&incY);

	for(i=0; i<2*N; i++){
		CU_ASSERT_DOUBLE_EQUAL(x1[i], x2[i], CHECK_EPS);
		CU_ASSERT_DOUBLE_EQUAL(y1[i], y2[i], CHECK_EPS);
	}
}
