/***************************************************************************
Copyright (c) 2013, The OpenBLAS Project
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

/**************************************************************************************
* 2013/11/25 Saar
* 	 BLASTEST 		: OK
* 	 CTEST			: OK
* 	 TEST			: OK
*
**************************************************************************************/

#define ASSEMBLER
#include "common.h"

#define STACKSIZE 256

#define	OLD_LDA		[fp, #0 ]
#define	X		[fp, #4 ]
#define	OLD_INC_X	[fp, #8 ]
#define	Y		[fp, #12 ]
#define	OLD_INC_Y	[fp, #16 ]
#define OLD_A		r3
#define	OLD_N		r1

#define M	r0
#define AO1	r1
#define J	r2

#define AO2	r4
#define XO	r5
#define YO	r6
#define LDA	r7
#define INC_X	r8
#define INC_Y	r9

#define I	r12

#define FP_ZERO [fp, #-228]
#define FP_ZERO_0 [fp, #-228]
#define FP_ZERO_1 [fp, #-224]

#define N	[fp, #-252 ]
#define A	[fp, #-256 ]


#define X_PRE	512
#define A_PRE	512

/**************************************************************************************
* Macro definitions
**************************************************************************************/


#if	defined(DOUBLE)

.macro INIT_F2

	fldd		d2, FP_ZERO
	vmov.f64	d3 , d2 

.endm

.macro KERNEL_F2X4

	pld	[ XO  , #X_PRE ]
	fldmiad	XO! ,  { d12 - d15 }
	pld	[ AO1 , #A_PRE ]
	fldmiad	AO1!,  { d8 - d9   }
	pld	[ AO2 , #A_PRE ]
	fldmiad	AO2!,  { d4 - d5 }
	fldmiad	AO1!,  { d10 - d11 }
	fldmiad	AO2!,  { d6 - d7 }

	vmla.f64	d2 , d12 , d8
	vmla.f64	d3 , d12 , d4
	vmla.f64	d2 , d13 , d9
	vmla.f64	d3 , d13 , d5
	vmla.f64	d2 , d14, d10
	vmla.f64	d3 , d14, d6
	vmla.f64	d2 , d15, d11
	vmla.f64	d3 , d15, d7

.endm

.macro KERNEL_F2X1

	fldmiad	XO! ,  { d1 }
	fldmiad	AO1!,  { d8 }
	fldmiad	AO2!,  { d4 }
	vmla.f64	d2 , d1 , d8
	vmla.f64	d3 , d1 , d4

.endm

.macro	SAVE_F2

	fldmiad	YO,  { d4 - d5 }
	vmla.f64	d4, d0, d2
	vmla.f64	d5, d0, d3
	fstmiad	YO!, { d4 - d5 }

.endm

.macro INIT_F1

	fldd		d2, FP_ZERO
	vmov.f64	d3 , d2 

.endm

.macro KERNEL_F1X4

	pld	[ XO  , #X_PRE ]
	fldmiad	XO! ,  { d12 - d15 }
	pld	[ AO1 , #A_PRE ]
	fldmiad	AO1!,  { d8 - d9   }
	fldmiad	AO1!,  { d10 - d11 }
	vmla.f64	d2 , d12 , d8
	vmla.f64	d2 , d13 , d9
	vmla.f64	d2 , d14, d10
	vmla.f64	d2 , d15, d11

.endm

.macro KERNEL_F1X1

	fldmiad	XO! ,  { d1 }
	fldmiad	AO1!,  { d8 }
	vmla.f64	d2 , d1 , d8

.endm

.macro	SAVE_F1

	fldmiad	YO,  { d4 }
	vmla.f64	d4, d0, d2
	fstmiad	YO!, { d4 }

.endm


.macro INIT_S2

	fldd		d2, FP_ZERO
	vmov.f64	d3 , d2 

.endm

.macro KERNEL_S2X4

	fldmiad	XO ,  { d12 }
	add	XO, XO, INC_X

	pld	[ AO1 , #A_PRE ]
	fldmiad	AO1!,  { d8 - d9   }
	pld	[ AO2 , #A_PRE ]
	fldmiad	AO2!,  { d4 - d5 }

	fldmiad	XO ,  { d13 }
	add	XO, XO, INC_X
	fldmiad	AO1!,  { d10 - d11 }
	fldmiad	AO2!,  { d6 - d7 }

	fldmiad	XO ,  { d14 }
	add	XO, XO, INC_X

	fldmiad	XO ,  { d15 }
	add	XO, XO, INC_X

	vmla.f64	d2 , d12 , d8
	vmla.f64	d3 , d12 , d4
	vmla.f64	d2 , d13 , d9
	vmla.f64	d3 , d13 , d5
	vmla.f64	d2 , d14, d10
	vmla.f64	d3 , d14, d6
	vmla.f64	d2 , d15, d11
	vmla.f64	d3 , d15, d7

.endm

.macro KERNEL_S2X1

	fldmiad	XO ,  { d1 }
	fldmiad	AO1!,  { d8 }
	fldmiad	AO2!,  { d4 }
	vmla.f64	d2 , d1 , d8
	add	XO, XO, INC_X
	vmla.f64	d3 , d1 , d4

.endm

.macro	SAVE_S2

	fldmiad	YO,  { d4 }
	vmla.f64	d4, d0, d2
	fstmiad	YO, { d4  }
	add	YO, YO, INC_Y

	fldmiad	YO,  { d5 }
	vmla.f64	d5, d0, d3
	fstmiad	YO, { d5  }
	add	YO, YO, INC_Y

.endm

.macro INIT_S1

	fldd		d2, FP_ZERO
	vmov.f64	d3 , d2 

.endm

.macro KERNEL_S1X4

	fldmiad	XO ,  { d12 }
	add	XO, XO, INC_X

	pld	[ AO1 , #A_PRE ]
	fldmiad	AO1!,  { d8 - d9   }

	fldmiad	XO ,  { d13 }
	add	XO, XO, INC_X
	fldmiad	AO1!,  { d10 - d11 }

	fldmiad	XO ,  { d14 }
	add	XO, XO, INC_X

	fldmiad	XO ,  { d15 }
	add	XO, XO, INC_X

	vmla.f64	d2 , d12 , d8
	vmla.f64	d2 , d13 , d9
	vmla.f64	d2 , d14, d10
	vmla.f64	d2 , d15, d11

.endm

.macro KERNEL_S1X1

	fldmiad	XO ,  { d1 }
	fldmiad	AO1!,  { d8 }
	vmla.f64	d2 , d1 , d8
	add	XO, XO, INC_X

.endm

.macro	SAVE_S1

	fldmiad	YO,  { d4 }
	vmla.f64	d4, d0, d2
	fstmiad	YO, { d4  }
	add	YO, YO, INC_Y

.endm


#else	/************************* SINGLE PRECISION *****************************************/

.macro INIT_F2

	flds		s2 , FP_ZERO
	vmov.f32	s3 , s2 


.endm

.macro KERNEL_F2X4

	fldmias	XO! ,  { s12 - s15 }
	fldmias	AO1!,  { s8 - s9   }
	fldmias	AO2!,  { s4 - s5 }
	fldmias	AO1!,  { s10 - s11 }
	fldmias	AO2!,  { s6 - s7 }

	vmla.f32	s2 , s12 , s8
	vmla.f32	s3 , s12 , s4
	vmla.f32	s2 , s13 , s9
	vmla.f32	s3 , s13 , s5
	vmla.f32	s2 , s14, s10
	vmla.f32	s3 , s14, s6
	vmla.f32	s2 , s15, s11
	vmla.f32	s3 , s15, s7

.endm

.macro KERNEL_F2X1

	fldmias	XO! ,  { s1 }
	fldmias	AO1!,  { s8 }
	fldmias	AO2!,  { s4 }
	vmla.f32	s2 , s1 , s8
	vmla.f32	s3 , s1 , s4

.endm

.macro	SAVE_F2

	fldmias	YO,  { s4 - s5 }
	vmla.f32	s4, s0, s2
	vmla.f32	s5, s0, s3
	fstmias	YO!, { s4 - s5 }

.endm

.macro INIT_F1

	flds		s2 , FP_ZERO

.endm

.macro KERNEL_F1X4

	fldmias	XO! ,  { s12 - s15 }
	fldmias	AO1!,  { s8 - s9   }
	fldmias	AO1!,  { s10 - s11 }
	vmla.f32	s2 , s12 , s8
	vmla.f32	s2 , s13 , s9
	vmla.f32	s2 , s14, s10
	vmla.f32	s2 , s15, s11

.endm

.macro KERNEL_F1X1

	fldmias	XO! ,  { s1 }
	fldmias	AO1!,  { s8 }
	vmla.f32	s2 , s1 , s8

.endm

.macro	SAVE_F1

	fldmias	YO,  { s4 }
	vmla.f32	s4, s0, s2
	fstmias	YO!, { s4 }

.endm


.macro INIT_S2

	flds		s2 , FP_ZERO
	vmov.f32	s3 , s2 

.endm

.macro KERNEL_S2X4

	fldmias	XO ,  { s12 }
	add	XO, XO, INC_X

	fldmias	AO1!,  { s8 - s9   }
	fldmias	AO2!,  { s4 - s5 }

	fldmias	XO ,  { s13 }
	add	XO, XO, INC_X
	fldmias	AO1!,  { s10 - s11 }
	fldmias	AO2!,  { s6 - s7 }

	fldmias	XO ,  { s14 }
	add	XO, XO, INC_X

	fldmias	XO ,  { s15 }
	add	XO, XO, INC_X

	vmla.f32	s2 , s12 , s8
	vmla.f32	s3 , s12 , s4
	vmla.f32	s2 , s13 , s9
	vmla.f32	s3 , s13 , s5
	vmla.f32	s2 , s14, s10
	vmla.f32	s3 , s14, s6
	vmla.f32	s2 , s15, s11
	vmla.f32	s3 , s15, s7

.endm

.macro KERNEL_S2X1

	fldmias	XO ,  { s1 }
	fldmias	AO1!,  { s8 }
	fldmias	AO2!,  { s4 }
	vmla.f32	s2 , s1 , s8
	add	XO, XO, INC_X
	vmla.f32	s3 , s1 , s4

.endm

.macro	SAVE_S2

	fldmias	YO,  { s4 }
	vmla.f32	s4, s0, s2
	fstmias	YO, { s4  }
	add	YO, YO, INC_Y

	fldmias	YO,  { s5 }
	vmla.f32	s5, s0, s3
	fstmias	YO, { s5  }
	add	YO, YO, INC_Y

.endm

.macro INIT_S1

	flds		s2 , FP_ZERO

.endm

.macro KERNEL_S1X4

	fldmias	XO ,  { s12 }
	add	XO, XO, INC_X

	pld	[ AO1 , #A_PRE ]
	fldmias	AO1!,  { s8 - s9   }

	fldmias	XO ,  { s13 }
	add	XO, XO, INC_X
	fldmias	AO1!,  { s10 - s11 }

	fldmias	XO ,  { s14 }
	add	XO, XO, INC_X

	fldmias	XO ,  { s15 }
	add	XO, XO, INC_X

	vmla.f32	s2 , s12 , s8
	vmla.f32	s2 , s13 , s9
	vmla.f32	s2 , s14, s10
	vmla.f32	s2 , s15, s11

.endm

.macro KERNEL_S1X1

	fldmias	XO ,  { s1 }
	fldmias	AO1!,  { s8 }
	vmla.f32	s2 , s1 , s8
	add	XO, XO, INC_X

.endm

.macro	SAVE_S1

	fldmias	YO,  { s4 }
	vmla.f32	s4, s0, s2
	fstmias	YO, { s4  }
	add	YO, YO, INC_Y

.endm



#endif

/**************************************************************************************
* End of macro definitions
**************************************************************************************/

	PROLOGUE

	.align 5
	push    {r4 - r9 , fp}
        add     fp, sp, #28
	sub     sp, sp, #STACKSIZE                              // reserve stack

        sub     r12, fp, #192

#if	defined(DOUBLE)
        vstm    r12, { d8 - d15 }                                 // store floating point registers
#else
        vstm    r12, { s8 - s15 }                                 // store floating point registers
#endif

        movs    r12, #0
        str     r12, FP_ZERO
        str     r12, FP_ZERO_1

	cmp	M, #0
	ble	gemvt_kernel_L999

	cmp	OLD_N, #0
	ble	gemvt_kernel_L999

	str	OLD_A, A
	str	OLD_N, N

	ldr    INC_X , OLD_INC_X
	ldr    INC_Y , OLD_INC_Y

	cmp	INC_X, #0
	beq	gemvt_kernel_L999

	cmp	INC_Y, #0
	beq	gemvt_kernel_L999

	ldr	LDA, OLD_LDA


#if defined(DOUBLE)
	lsl	LDA, LDA, #3				// LDA * SIZE
#else
	lsl	LDA, LDA, #2				// LDA * SIZE
#endif

	cmp	INC_X, #1
	bne	gemvt_kernel_S2_BEGIN

	cmp	INC_Y, #1
	bne	gemvt_kernel_S2_BEGIN


gemvt_kernel_F2_BEGIN:

	ldr	YO , Y

	ldr	J, N
	asrs	J, J, #1					// J = N / 2
	ble	gemvt_kernel_F1_BEGIN

gemvt_kernel_F2X4:

	ldr	AO1, A
	add	AO2, AO1, LDA
	add	r3 , AO2, LDA
	str	r3 , A

	ldr	XO , X

	INIT_F2

	asrs	I, M, #2					// I = M / 4
	ble	gemvt_kernel_F2X1


gemvt_kernel_F2X4_10:

	KERNEL_F2X4

	subs	I, I, #1
	bne	gemvt_kernel_F2X4_10


gemvt_kernel_F2X1:

	ands	I, M , #3
	ble	gemvt_kernel_F2_END

gemvt_kernel_F2X1_10:

	KERNEL_F2X1

	subs	I, I, #1
	bne	gemvt_kernel_F2X1_10


gemvt_kernel_F2_END:

	SAVE_F2

	subs	J , J , #1
	bne	gemvt_kernel_F2X4


gemvt_kernel_F1_BEGIN:

	ldr	J, N
	ands	J, J, #1
	ble	gemvt_kernel_L999

gemvt_kernel_F1X4:

	ldr	AO1, A

	ldr	XO , X

	INIT_F1

	asrs	I, M, #2					// I = M / 4
	ble	gemvt_kernel_F1X1


gemvt_kernel_F1X4_10:

	KERNEL_F1X4

	subs	I, I, #1
	bne	gemvt_kernel_F1X4_10


gemvt_kernel_F1X1:

	ands	I, M , #3
	ble	gemvt_kernel_F1_END

gemvt_kernel_F1X1_10:

	KERNEL_F1X1

	subs	I, I, #1
	bne	gemvt_kernel_F1X1_10


gemvt_kernel_F1_END:

	SAVE_F1

	b	gemvt_kernel_L999



/*************************************************************************************************************/

gemvt_kernel_S2_BEGIN:

#if defined(DOUBLE)
	lsl	INC_X, INC_X, #3				// INC_X * SIZE
	lsl	INC_Y, INC_Y, #3				// INC_Y * SIZE
#else
	lsl	INC_X, INC_X, #2				// INC_X * SIZE
	lsl	INC_Y, INC_Y, #2				// INC_Y * SIZE
#endif

	ldr	YO , Y

	ldr	J, N
	asrs	J, J, #1					// J = N / 2
	ble	gemvt_kernel_S1_BEGIN

gemvt_kernel_S2X4:

	ldr	AO1, A
	add	AO2, AO1, LDA
	add	r3 , AO2, LDA
	str	r3 , A

	ldr	XO , X

	INIT_S2

	asrs	I, M, #2					// I = M / 4
	ble	gemvt_kernel_S2X1


gemvt_kernel_S2X4_10:

	KERNEL_S2X4

	subs	I, I, #1
	bne	gemvt_kernel_S2X4_10


gemvt_kernel_S2X1:

	ands	I, M , #3
	ble	gemvt_kernel_S2_END

gemvt_kernel_S2X1_10:

	KERNEL_S2X1

	subs	I, I, #1
	bne	gemvt_kernel_S2X1_10


gemvt_kernel_S2_END:

	SAVE_S2

	subs	J , J , #1
	bne	gemvt_kernel_S2X4


gemvt_kernel_S1_BEGIN:

	ldr	J, N
	ands	J, J, #1
	ble	gemvt_kernel_L999

gemvt_kernel_S1X4:

	ldr	AO1, A

	ldr	XO , X

	INIT_S1

	asrs	I, M, #2					// I = M / 4
	ble	gemvt_kernel_S1X1


gemvt_kernel_S1X4_10:

	KERNEL_S1X4

	subs	I, I, #1
	bne	gemvt_kernel_S1X4_10


gemvt_kernel_S1X1:

	ands	I, M , #3
	ble	gemvt_kernel_S1_END

gemvt_kernel_S1X1_10:

	KERNEL_S1X1

	subs	I, I, #1
	bne	gemvt_kernel_S1X1_10


gemvt_kernel_S1_END:

	SAVE_S1



/*************************************************************************************************************/

gemvt_kernel_L999:

        sub     r3, fp, #192

#if	defined(DOUBLE)
        vldm    r3, { d8 - d15 }                                 // restore floating point registers
#else
        vldm    r3, { s8 - s15 }                                 // restore floating point registers
#endif

	mov	r0, #0		// set return value

	sub     sp, fp, #28
	pop     {r4 -r9 ,fp}
	bx	lr

	EPILOGUE

