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

void test_drotmg()
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i=0;
	te_d1= tr_d1=0.21149573940783739;
	te_d2= tr_d2=0.046892057172954082;
	te_x1= tr_x1=-0.42272687517106533;
	te_y1= tr_y1=0.42211309121921659;

	for(i=0; i<5; i++){
	  te_param[i]=tr_param[i]=0.0;
	}

	//OpenBLAS
	BLASFUNC(drotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);
	//reference
	BLASFUNC_REF(drotmg)(&tr_d1, &tr_d2, &tr_x1, &tr_y1, tr_param);

	CU_ASSERT_DOUBLE_EQUAL(te_d1, tr_d1, CHECK_EPS);
	CU_ASSERT_DOUBLE_EQUAL(te_d2, tr_d2, CHECK_EPS);
	CU_ASSERT_DOUBLE_EQUAL(te_x1, tr_x1, CHECK_EPS);
	CU_ASSERT_DOUBLE_EQUAL(te_y1, tr_y1, CHECK_EPS);

	for(i=0; i<5; i++){
		CU_ASSERT_DOUBLE_EQUAL(te_param[i], tr_param[i], CHECK_EPS);
	}
}

void test_drotmg_D1eqD2_X1eqX2()
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i=0;
	te_d1= tr_d1=2.;
	te_d2= tr_d2=2.;
	te_x1= tr_x1=8.;
	te_y1= tr_y1=8.;

	for(i=0; i<5; i++){
	  te_param[i]=tr_param[i]=0.0;
	}

	//OpenBLAS
	BLASFUNC(drotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);
	//reference
	BLASFUNC_REF(drotmg)(&tr_d1, &tr_d2, &tr_x1, &tr_y1, tr_param);

	CU_ASSERT_DOUBLE_EQUAL(te_d1, tr_d1, CHECK_EPS);
	CU_ASSERT_DOUBLE_EQUAL(te_d2, tr_d2, CHECK_EPS);
	CU_ASSERT_DOUBLE_EQUAL(te_x1, tr_x1, CHECK_EPS);
	CU_ASSERT_DOUBLE_EQUAL(te_y1, tr_y1, CHECK_EPS);

	for(i=0; i<5; i++){
		CU_ASSERT_DOUBLE_EQUAL(te_param[i], tr_param[i], CHECK_EPS);
	}
}
