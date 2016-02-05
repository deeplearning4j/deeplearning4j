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
#define INDEX	x3	/* index of max/min value */
#define Z	x4	/* vector index */
#define I	x5	/* loop variable */

/*******************************************************************************
* Macro definitions
*******************************************************************************/

#if defined(USE_MIN)
#define COND	le
#else
#define COND	ge
#endif

#if !defined(DOUBLE)
#define MAXF	s0
#define TMPF	s1
#define TMPVF	{v1.s}[0]
#define SZ	4
#else
#define MAXF	d0
#define TMPF	d1
#define TMPVF	{v1.d}[0]
#define SZ	8
#endif

/******************************************************************************/

.macro INIT_S
#if !defined(DOUBLE)
	lsl	INC_X, INC_X, #2
	ld1	{v0.s}[0], [X], INC_X
#else
	lsl	INC_X, INC_X, #3
	ld1	{v0.d}[0], [X], INC_X
#endif
	mov	Z, #1
	mov	INDEX, Z
	fabs	MAXF, MAXF
.endm

.macro KERNEL_S1
	ld1	TMPVF, [X], INC_X
	add	Z, Z, #1
	fabs	TMPF, TMPF
	fcmp	MAXF, TMPF
	fcsel	MAXF, MAXF, TMPF, COND
	csel	INDEX, INDEX, Z, COND
.endm

/*******************************************************************************
* End of macro definitions
*******************************************************************************/

	PROLOGUE

	cmp	N, xzr
	ble	iamax_kernel_zero
	cmp	INC_X, xzr
	ble	iamax_kernel_zero

	INIT_S

	subs	N, N, #1
	ble	iamax_kernel_L999

	asr	I, N, #2
	cmp	I, xzr
	ble	iamax_kernel_S1

iamax_kernel_S4:

	KERNEL_S1
	KERNEL_S1
	KERNEL_S1
	KERNEL_S1

	subs	I, I, #1
	bne	iamax_kernel_S4

iamax_kernel_S1:

	ands	I, N, #3
	ble	iamax_kernel_L999

iamax_kernel_S10:

	KERNEL_S1
	subs	I, I, #1
	bne	iamax_kernel_S10

iamax_kernel_L999:

	mov	x0, INDEX
	ret

iamax_kernel_zero:

	mov	x0, xzr
	ret

	EPILOGUE
