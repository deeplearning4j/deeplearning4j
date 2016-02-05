/*******************************************************************************
Copyright (c) 2015, The OpenBLAS Project
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
*******************************************************************************/

#define ASSEMBLER
#include "common.h"

#define	N	x0	/* vector length */
#define	X	x1	/* X vector address */
#define	INC_X	x2	/* X stride */
#define I	x5	/* loop variable */

/*******************************************************************************
* Macro definitions
*******************************************************************************/

#define TMPF	d6
#define SSQ	d0
#define TMPVF	{v6.d}[0]
#define SZ	8

/******************************************************************************/

.macro KERNEL_F1
	ldr	TMPF, [X], #SZ
	fmul	TMPF, TMPF, TMPF
	fadd	SSQ, SSQ, TMPF
.endm

.macro KERNEL_F8
	ld1	{v1.2d, v2.2d}, [X], #32
	fmla	v0.2d, v1.2d, v1.2d
	fmla	v5.2d, v2.2d, v2.2d
	ld1	{v3.2d, v4.2d}, [X], #32
	fmla	v0.2d, v3.2d, v3.2d
	fmla	v5.2d, v4.2d, v4.2d
	PRFM	PLDL1KEEP, [X, #1024]
.endm

.macro nrm2_kernel_F8_FINALIZE
	fadd	v0.2d, v0.2d, v5.2d
	faddp	SSQ, v0.2d
.endm

.macro INIT_S
	lsl	INC_X, INC_X, #3
	ld1	TMPVF, [X], INC_X
	fmul	SSQ, TMPF, TMPF
.endm

.macro KERNEL_S1
	ld1	TMPVF, [X], INC_X
	fmul	TMPF, TMPF, TMPF
	fadd	SSQ, SSQ, TMPF
.endm

/*******************************************************************************
* End of macro definitions
*******************************************************************************/

	PROLOGUE

	fmov	SSQ, xzr
	fmov	d5, SSQ

	cmp	N, xzr
	ble	nrm2_kernel_zero
	cmp	INC_X, xzr
	ble	nrm2_kernel_zero
	cmp	INC_X, #1
	bne	nrm2_kernel_S_BEGIN

nrm2_kernel_F_BEGIN:

	asr	I, N, #3
	cmp	I, xzr
	beq	nrm2_kernel_F1_INIT

nrm2_kernel_F8:

	KERNEL_F8

	subs	I, I, #1
	bne	nrm2_kernel_F8

	nrm2_kernel_F8_FINALIZE

nrm2_kernel_F1:

	ands	I, N, #7
	ble	nrm2_kernel_L999

nrm2_kernel_F10:

	KERNEL_F1

	subs    I, I, #1
        bne     nrm2_kernel_F10

	b	nrm2_kernel_L999

nrm2_kernel_F1_INIT:

	b	nrm2_kernel_F1

nrm2_kernel_S_BEGIN:

	INIT_S

	subs	N, N, #1
	ble	nrm2_kernel_L999

	asr	I, N, #2
	cmp	I, xzr
	ble	nrm2_kernel_S1

nrm2_kernel_S4:

	KERNEL_S1
	KERNEL_S1
	KERNEL_S1
	KERNEL_S1

	subs	I, I, #1
	bne	nrm2_kernel_S4

nrm2_kernel_S1:

	ands	I, N, #3
	ble	nrm2_kernel_L999

nrm2_kernel_S10:

	KERNEL_S1

	subs    I, I, #1
	bne     nrm2_kernel_S10

nrm2_kernel_L999:
	fsqrt	SSQ, SSQ
	ret

nrm2_kernel_zero:
	ret

	EPILOGUE
