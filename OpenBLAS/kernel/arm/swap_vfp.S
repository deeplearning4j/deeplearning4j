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
* 2013/11/14 Saar
* 	 BLASTEST 		: OK
* 	 CTEST			: OK
* 	 TEST			: OK
*
**************************************************************************************/

#define ASSEMBLER
#include "common.h"

#define STACKSIZE 256

#define	OLD_INC_X	[fp, #0 ]
#define	OLD_Y		[fp, #4 ]
#define	OLD_INC_Y	[fp, #8 ]


#define	N	r0
#define Y	r1
#define	INC_X	r2
#define	X	r3
#define INC_Y	r4

#define I	r12

#define X_PRE	512

/**************************************************************************************
* Macro definitions
**************************************************************************************/

/*****************************************************************************************/



#if	!defined(COMPLEX)

#if	defined(DOUBLE)

.macro KERNEL_F4

	pld	[ X, #X_PRE ]
	pld	[ Y, #X_PRE ]
	fldmiad	X,  { d0 - d3 }
	fldmiad	Y,  { d4 - d7 }
	fstmiad	Y!, { d0 - d3 }
	fstmiad	X!, { d4 - d7}

.endm


.macro KERNEL_F1

	fldmiad	X,  { d0 }
	fldmiad	Y,  { d4 }
	fstmiad	Y!, { d0 }
	fstmiad	X!, { d4 }

.endm

.macro KERNEL_S1

	fldmiad	X, { d0 }
	fldmiad	Y, { d4 }
	fstmiad	Y, { d0 }
	fstmiad	X, { d4 }
	add	X, X, INC_X
	add	Y, Y, INC_Y

.endm

#else

.macro KERNEL_F4

	fldmias	X,  { s0 - s3 }
	fldmias	Y,  { s4 - s7 }
	fstmias	Y!, { s0 - s3 }
	fstmias	X!, { s4 - s7}

.endm


.macro KERNEL_F1

	fldmias	X,  { s0 }
	fldmias	Y,  { s4 }
	fstmias	Y!, { s0 }
	fstmias	X!, { s4 }

.endm

.macro KERNEL_S1

	fldmias	X, { s0 }
	fldmias	Y, { s4 }
	fstmias	Y, { s0 }
	fstmias	X, { s4 }
	add	X, X, INC_X
	add	Y, Y, INC_Y

.endm


#endif

#else

#if	defined(DOUBLE)

.macro KERNEL_F4

	pld	[ X, #X_PRE ]
	pld	[ Y, #X_PRE ]
	fldmiad	X,  { d0 - d3 }
	fldmiad	Y,  { d4 - d7 }
	fstmiad	Y!, { d0 - d3 }
	fstmiad	X!, { d4 - d7}

	pld	[ X, #X_PRE ]
	pld	[ Y, #X_PRE ]
	fldmiad	X,  { d0 - d3 }
	fldmiad	Y,  { d4 - d7 }
	fstmiad	Y!, { d0 - d3 }
	fstmiad	X!, { d4 - d7}

.endm

.macro KERNEL_F1

	fldmiad	X,  { d0 - d1 }
	fldmiad	Y,  { d4 - d5 }
	fstmiad	Y!, { d0 - d1 }
	fstmiad	X!, { d4 - d5 }

.endm

.macro KERNEL_S1

	fldmiad	X,  { d0 - d1 }
	fldmiad	Y,  { d4 - d5 }
	fstmiad	Y,  { d0 - d1 }
	fstmiad	X,  { d4 - d5 }
	add	X, X, INC_X
	add	Y, Y, INC_Y

.endm


#else

.macro KERNEL_F4

	pld	[ X, #X_PRE ]
	pld	[ Y, #X_PRE ]
	fldmias	X,  { s0 - s3 }
	fldmias	Y,  { s4 - s7 }
	fstmias	Y!, { s0 - s3 }
	fstmias	X!, { s4 - s7}

	fldmias	X,  { s0 - s3 }
	fldmias	Y,  { s4 - s7 }
	fstmias	Y!, { s0 - s3 }
	fstmias	X!, { s4 - s7}

.endm

.macro KERNEL_F1

	fldmias	X,  { s0 - s1 }
	fldmias	Y,  { s4 - s5 }
	fstmias	Y!, { s0 - s1 }
	fstmias	X!, { s4 - s5 }

.endm

.macro KERNEL_S1

	fldmias	X,  { s0 - s1 }
	fldmias	Y,  { s4 - s5 }
	fstmias	Y,  { s0 - s1 }
	fstmias	X,  { s4 - s5 }
	add	X, X, INC_X
	add	Y, Y, INC_Y

.endm



#endif

#endif

/**************************************************************************************
* End of macro definitions
**************************************************************************************/

	PROLOGUE

	.align 5
	push    {r4 , fp}
        add     fp, sp, #8

	ldr    INC_X , OLD_INC_X
	ldr         Y, OLD_Y
	ldr    INC_Y , OLD_INC_Y


	cmp	N, #0
	ble	swap_kernel_L999

	cmp	INC_X, #0
	beq	swap_kernel_L999

	cmp	INC_Y, #0
	beq	swap_kernel_L999

	cmp	INC_X, #1
	bne	swap_kernel_S_BEGIN

	cmp	INC_Y, #1
	bne	swap_kernel_S_BEGIN


swap_kernel_F_BEGIN:


	asrs	I, N, #2					// I = N / 4
	ble	swap_kernel_F1

	.align 5

swap_kernel_F4:

#if !defined(COMPLEX) && !defined(DOUBLE)
	pld	[ X, #X_PRE ]
	pld	[ Y, #X_PRE ]
#endif

	KERNEL_F4

	subs	I, I, #1
	ble	swap_kernel_F1

	KERNEL_F4

	subs	I, I, #1
	bne	swap_kernel_F4

swap_kernel_F1:

	ands	I, N, #3
	ble	swap_kernel_L999

swap_kernel_F10:

	KERNEL_F1

	subs    I, I, #1
        bne     swap_kernel_F10

	b	swap_kernel_L999

swap_kernel_S_BEGIN:

#if defined(COMPLEX)

#if defined(DOUBLE)
	lsl	INC_X, INC_X, #4				// INC_X * SIZE * 2
	lsl	INC_Y, INC_Y, #4				// INC_Y * SIZE * 2
#else
	lsl	INC_X, INC_X, #3				// INC_X * SIZE * 2
	lsl	INC_Y, INC_Y, #3				// INC_Y * SIZE * 2
#endif

#else

#if defined(DOUBLE)
	lsl	INC_X, INC_X, #3				// INC_X * SIZE
	lsl	INC_Y, INC_Y, #3				// INC_Y * SIZE
#else
	lsl	INC_X, INC_X, #2				// INC_X * SIZE
	lsl	INC_Y, INC_Y, #2				// INC_Y * SIZE
#endif

#endif


	asrs	I, N, #2					// I = N / 4
	ble	swap_kernel_S1

	.align 5

swap_kernel_S4:

	KERNEL_S1
	KERNEL_S1
	KERNEL_S1
	KERNEL_S1

	subs	I, I, #1
	bne	swap_kernel_S4

swap_kernel_S1:

	ands	I, N, #3
	ble	swap_kernel_L999

swap_kernel_S10:

	KERNEL_S1

	subs    I, I, #1
        bne     swap_kernel_S10


swap_kernel_L999:

	mov	r0, #0		// set return value

	sub     sp, fp, #8
	pop     {r4,fp}
	bx	lr

	EPILOGUE

