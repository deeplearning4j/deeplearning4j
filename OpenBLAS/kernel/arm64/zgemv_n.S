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

#define	M	x0	/* Y vector length */
#define	N	x1	/* X vector length */
#define	A	x3	/* A vector address */
#define	LDA	x4	/* A stride */
#define	X	x5	/* X vector address */
#define	INC_X	x6	/* X stride */
#define	Y	x7	/* Y vector address */
#define	INC_Y	x2	/* Y stride */
#define	A_PTR	x9	/* loop A vector address */
#define	Y_IPTR	x10	/* loop Y vector address */
#define J	x11	/* loop variable */
#define I	x12	/* loop variable */
#define	Y_OPTR	x13	/* loop Y vector address */
#define	X_PTR	x14	/* loop X vector address */

/*******************************************************************************
* Macro definitions
*******************************************************************************/

#if !defined(DOUBLE)
#define ALPHA_R		s0
#define ALPHA_I		s1
#define ALPHA_R_COPY	s7
#define ALPHA_I_COPY	s8
#define SHZ		3
#else
#define ALPHA_R		d0
#define ALPHA_I		d1
#define ALPHA_R_COPY	d7
#define ALPHA_I_COPY	d8
#define SHZ		4
#endif

/******************************************************************************/

.macro SAVE_REGS
	add	sp, sp, #-(11 * 16)
	stp	d8, d9, [sp, #(0 * 16)]
	stp	d10, d11, [sp, #(1 * 16)]
	stp	d12, d13, [sp, #(2 * 16)]
	stp	d14, d15, [sp, #(3 * 16)]
	stp	d16, d17, [sp, #(4 * 16)]
	stp	x18, x19, [sp, #(5 * 16)]
	stp	x20, x21, [sp, #(6 * 16)]
	stp	x22, x23, [sp, #(7 * 16)]
	stp	x24, x25, [sp, #(8 * 16)]
	stp	x26, x27, [sp, #(9 * 16)]
	str	x28, [sp, #(10 * 16)]
.endm

.macro RESTORE_REGS
	ldp	d8, d9, [sp, #(0 * 16)]
	ldp	d10, d11, [sp, #(1 * 16)]
	ldp	d12, d13, [sp, #(2 * 16)]
	ldp	d14, d15, [sp, #(3 * 16)]
	ldp	d16, d17, [sp, #(4 * 16)]
	ldp	x18, x19, [sp, #(5 * 16)]
	ldp	x20, x21, [sp, #(6 * 16)]
	ldp	x22, x23, [sp, #(7 * 16)]
	ldp	x24, x25, [sp, #(8 * 16)]
	ldp	x26, x27, [sp, #(9 * 16)]
	ldr	x28, [sp, #(10 * 16)]
	add	sp, sp, #(11*16)
.endm


.macro INIT
	/********** INIT FOR F4 LOOP **********/
	fmov	ALPHA_R_COPY, ALPHA_R
	fmov	ALPHA_I_COPY, ALPHA_I
#if !defined(DOUBLE)
	ins	v7.s[1], v7.s[0]		// R(ALPHA), R(ALPHA) 
	ins	v8.s[1], v8.s[0]		// I(ALPHA), I(ALPHA) 
	ins	v7.d[1], v7.d[0]
	ins	v8.d[1], v8.d[0]
#else
	ins	v7.d[1], v7.d[0]		// R(ALPHA), R(ALPHA)
	ins	v8.d[1], v8.d[0]		// I(ALPHA), I(ALPHA)
#endif

	/******* INIT FOR F1 AND S1 LOOP ******/
#if !defined(DOUBLE)
	ins	v0.s[1], v0.s[0]		// R(ALPHA), R(ALPHA)
	eor	v2.16b, v2.16b, v2.16b
	fsub	s2, s2, ALPHA_I
	ins	v1.s[1], v2.s[0]		// -I(ALPHA), I(ALPHA) 
#if !defined(XCONJ)
	ext	v1.8b, v1.8b, v1.8b, #4		// I(ALPHA), -I(ALPHA)
#endif
#else
	ins	v0.d[1], v0.d[0]		// R(ALPHA), R(ALPHA)
	eor	v2.16b, v2.16b, v2.16b
	fsub	d2, d2, ALPHA_I
	ins	v1.d[1], v2.d[0]		// -I(ALPHA), I(ALPHA) 
#if !defined(XCONJ)
	ext	v1.16b, v1.16b, v1.16b, #8	// I(ALPHA), -I(ALPHA)
#endif
#endif
.endm

.macro INIT_LOOP
	/********** INIT_LOOP FOR F4 LOOP **********/
#if !defined(DOUBLE)
	ld1	{v9.2s}, [X_PTR]		// [I(X), R(X)]
	ins	v10.s[0], v9.s[1]
	ins	v9.s[1], v9.s[0]		// [R(X), R(X)]
	ins	v10.s[1], v10.s[0]		// [I(X), I(X)]
	ins	v9.d[1], v9.d[0]
	ins	v10.d[1], v10.d[0]
#if !defined(CONJ)
#if !defined(XCONJ)
	fmul	v11.4s, v9.4s, v7.4s		// [+ R(X) * R(ALPHA)]
	fmls	v11.4s, v10.4s, v8.4s		// [- I(X) * I(ALPHA)]
	fmul	v12.4s, v9.4s, v8.4s		// [+ R(X) * I(ALPHA)]
	fmla	v12.4s, v10.4s, v7.4s		// [+ I(X) * R(ALPHA)]
#else
	fmul	v11.4s, v9.4s, v7.4s		// [+ R(X) * R(ALPHA)]
	fmla	v11.4s, v10.4s, v8.4s		// [+ I(X) * I(ALPHA)]
	fmul	v12.4s, v9.4s, v8.4s		// [+ R(X) * I(ALPHA)]
	fmls	v12.4s, v10.4s, v7.4s		// [- I(X) * R(ALPHA)]
#endif
#else // CONJ
#if !defined(XCONJ)
	fmul	v11.4s, v9.4s, v7.4s		// [+ R(X) * R(ALPHA)]
	fmls	v11.4s, v10.4s, v8.4s		// [+ I(X) * I(ALPHA)]
	fmul	v12.4s, v10.4s, v7.4s		// [+ I(X) * R(ALPHA)]
	fmls	v12.4s, v9.4s, v8.4s		// [- R(X) * I(ALPHA)]
#else
	fmul	v11.4s, v9.4s, v7.4s		// [+ R(X) * R(ALPHA)]
	fmls	v11.4s, v10.4s, v8.4s		// [- I(X) * I(ALPHA)]
	eor	v12.16b, v12.16b, v12.16b
	fmls	v12.4s, v9.4s, v8.4s		// [- R(X) * I(ALPHA)]
	fmla	v12.4s, v10.4s, v7.4s		// [- I(X) * R(ALPHA)]
#endif
#endif // CONJ

	/****** INIT_LOOP FOR F1 AND S1 LOOP ******/
	ld1	{v2.2s}, [X_PTR]		// [I(X), R(X)]
	ext	v3.8b, v2.8b, v2.8b, #4		// [R(X), I(X)]
	fmul	v2.2s, v0.2s, v2.2s
	fmla	v2.2s, v1.2s, v3.2s		// [I(TEMP), R(TEMP)]
	ins	v3.s[0], v2.s[1]
#if !defined(CONJ)
#if !defined(XCONJ)
	eor	v4.16b, v4.16b, v4.16b
	fsub	s4, s4, s3
	ins	v3.s[1], v4.s[0]
	ext	v3.8b, v3.8b, v3.8b, #4		// [I(TEMP), -I(TEMP)]
	ins	v2.s[1], v2.s[0]		// [R(TEMP), R(TEMP)]
#else
	eor	v4.16b, v4.16b, v4.16b
	fsub	s4, s4, s3
	ins	v3.s[1], v4.s[0]		// [-I(TEMP), I(TEMP)]
	ins	v2.s[1], v2.s[0]		// [R(TEMP), R(TEMP)]
#endif
#else // CONJ
#if !defined(XCONJ)
	ins	v3.s[1], v3.s[0]		// [I(TEMP), I(TEMP)]
	eor	v4.16b, v4.16b, v4.16b
	fsub	s4, s4, s2
	ins	v2.s[1], v4.s[0]		// [-R(TEMP), R(TEMP)]
#else
	eor	v4.16b, v4.16b, v4.16b
	fsub	s3, s4, s3
	ins	v3.s[1], v3.s[0]		// [-I(TEMP), -I(TEMP)]
	eor	v4.16b, v4.16b, v4.16b
	fsub	s4, s4, s2
	ins	v2.s[1], v4.s[0]		// [-R(TEMP), R(TEMP)]
#endif
#endif // CONJ

#else // DOUBLE

	/********** INIT_LOOP FOR F4 LOOP **********/
	ld1	{v9.2d}, [X_PTR]		// [I(X), R(X)]
	ins	v10.d[0], v9.d[1]
	ins	v9.d[1], v9.d[0]		// [R(X), R(X)]
	ins	v10.d[1], v10.d[0]		// [I(X), I(X)]
#if !defined(CONJ)
#if !defined(XCONJ)
	fmul	v11.2d, v9.2d, v7.2d		// [+ R(X) * R(ALPHA)]
	fmls	v11.2d, v10.2d, v8.2d		// [- I(X) * I(ALPHA)]
	fmul	v12.2d, v9.2d, v8.2d		// [+ R(X) * I(ALPHA)]
	fmla	v12.2d, v10.2d, v7.2d		// [+ I(X) * R(ALPHA)]
#else
	fmul	v11.2d, v9.2d, v7.2d		// [+ R(X) * R(ALPHA)]
	fmla	v11.2d, v10.2d, v8.2d		// [+ I(X) * I(ALPHA)]
	fmul	v12.2d, v9.2d, v8.2d		// [+ R(X) * I(ALPHA)]
	fmls	v12.2d, v10.2d, v7.2d		// [- I(X) * R(ALPHA)]
#endif
#else // CONJ
#if !defined(XCONJ)
	fmul	v11.2d, v9.2d, v7.2d		// [+ R(X) * R(ALPHA)]
	fmls	v11.2d, v10.2d, v8.2d		// [+ I(X) * I(ALPHA)]
	fmul	v12.2d, v10.2d, v7.2d		// [+ I(X) * R(ALPHA)]
	fmls	v12.2d, v9.2d, v8.2d		// [- R(X) * I(ALPHA)]
#else
	fmul	v11.2d, v9.2d, v7.2d		// [+ R(X) * R(ALPHA)]
	fmls	v11.2d, v10.2d, v8.2d		// [- I(X) * I(ALPHA)]
	eor	v12.16b, v12.16b, v12.16b
	fmls	v12.2d, v9.2d, v8.2d		// [- R(X) * I(ALPHA)]
	fmla	v12.2d, v10.2d, v7.2d		// [- I(X) * R(ALPHA)]
#endif
#endif // CONJ

	/****** INIT_LOOP FOR F1 AND S1 LOOP ******/
	ld1	{v2.2d}, [X_PTR]		// [I(X), R(X)]
	ext	v3.16b, v2.16b, v2.16b, #8	// [R(X), I(X)]
	fmul	v2.2d, v0.2d, v2.2d
	fmla	v2.2d, v1.2d, v3.2d		// [I(TEMP), R(TEMP)]
	ins	v3.d[0], v2.d[1]		// I(TEMP)
#if !defined(CONJ)
#if !defined(XCONJ)
	eor	v4.16b, v4.16b, v4.16b
	fsub	d4, d4, d3
	ins	v3.d[1], v4.d[0]
	ext	v3.16b, v3.16b, v3.16b, #8	// [I(TEMP), -I(TEMP)]
	ins	v2.d[1], v2.d[0]		// [R(TEMP), R(TEMP)]
#else
	eor	v4.16b, v4.16b, v4.16b
	fsub	d4, d4, d3
	ins	v3.d[1], v4.d[0]		// [-I(TEMP), I(TEMP)]
	ins	v2.d[1], v2.d[0]		// [R(TEMP), R(TEMP)]
#endif
#else // CONJ
#if !defined(XCONJ)
	ins	v3.d[1], v3.d[0]		// [I(TEMP), I(TEMP)]
	eor	v4.16b, v4.16b, v4.16b
	fsub	d4, d4, d2
	ins	v2.d[1], v4.d[0]		// [-R(TEMP), R(TEMP)]
#else
	eor	v4.16b, v4.16b, v4.16b
	fsub	d3, d4, d3
	ins	v3.d[1], v3.d[0]		// [-I(TEMP), -I(TEMP)]
	eor	v4.16b, v4.16b, v4.16b
	fsub	d4, d4, d2
	ins	v2.d[1], v4.d[0]		// [-R(TEMP), R(TEMP)]
#endif
#endif // CONJ

#endif // DOUBLE
.endm

.macro KERNEL_F4
#if !defined(DOUBLE)

	ld2	{v13.4s, v14.4s}, [A_PTR], #32
	ld2	{v15.4s, v16.4s}, [Y_IPTR], #32
#if !defined(CONJ)
#if !defined(XCONJ)
	fmla	v15.4s, v11.4s, v13.4s		// [+ R(ALPHA * X) * A_R]
	fmls	v15.4s, v12.4s, v14.4s		// [- I(ALPHA * X) * A_I]
	fmla	v16.4s, v11.4s, v14.4s		// [+ R(ALPHA * X) * A_I]
	fmla	v16.4s, v12.4s, v13.4s		// [+ I(ALPHA * X) * A_R]
#else
	fmla	v15.4s, v11.4s, v13.4s		// [+ R(ALPHA * X) * A_R]
	fmla	v15.4s, v12.4s, v14.4s		// [+ I(ALPHA * X) * A_I]
	fmla	v16.4s, v11.4s, v14.4s		// [+ R(ALPHA * X) * A_I]
	fmls	v16.4s, v12.4s, v13.4s		// [- I(ALPHA * X) * A_R]
#endif
#else // CONJ
#if !defined(XCONJ)
	fmla	v15.4s, v11.4s, v13.4s		// [+ R(ALPHA * X) * A_R]
	fmla	v15.4s, v12.4s, v14.4s		// [+ I(ALPHA * X) * A_I]
	fmls	v16.4s, v11.4s, v14.4s		// [- R(ALPHA * X) * A_I]
	fmla	v16.4s, v12.4s, v13.4s		// [+ I(ALPHA * X) * A_R]
#else
	fmla	v15.4s, v11.4s, v13.4s		// [+ R(ALPHA * X) * A_R]
	fmls	v15.4s, v12.4s, v14.4s		// [- I(ALPHA * X) * A_I]
	fmls	v16.4s, v11.4s, v14.4s		// [- R(ALPHA * X) * A_I]
	fmls	v16.4s, v12.4s, v13.4s		// [- I(ALPHA * X) * A_R]
#endif
#endif // CONJ
	st2	{v15.4s, v16.4s}, [Y_OPTR], #32

#else // DOUBLE

	ld2	{v13.2d, v14.2d}, [A_PTR], #32
	ld2	{v15.2d, v16.2d}, [Y_IPTR], #32
#if !defined(CONJ)
#if !defined(XCONJ)
	fmla	v15.2d, v11.2d, v13.2d		// [+ R(ALPHA * X) * A_R]
	fmls	v15.2d, v12.2d, v14.2d		// [- I(ALPHA * X) * A_I]
	fmla	v16.2d, v11.2d, v14.2d		// [+ R(ALPHA * X) * A_I]
	fmla	v16.2d, v12.2d, v13.2d		// [+ I(ALPHA * X) * A_R]
#else
	fmla	v15.2d, v11.2d, v13.2d		// [+ R(ALPHA * X) * A_R]
	fmla	v15.2d, v12.2d, v14.2d		// [+ I(ALPHA * X) * A_I]
	fmla	v16.2d, v11.2d, v14.2d		// [+ R(ALPHA * X) * A_I]
	fmls	v16.2d, v12.2d, v13.2d		// [- I(ALPHA * X) * A_R]
#endif
#else // CONJ
#if !defined(XCONJ)
	fmla	v15.2d, v11.2d, v13.2d		// [+ R(ALPHA * X) * A_R]
	fmla	v15.2d, v12.2d, v14.2d		// [+ I(ALPHA * X) * A_I]
	fmls	v16.2d, v11.2d, v14.2d		// [- R(ALPHA * X) * A_I]
	fmla	v16.2d, v12.2d, v13.2d		// [+ I(ALPHA * X) * A_R]
#else
	fmla	v15.2d, v11.2d, v13.2d		// [+ R(ALPHA * X) * A_R]
	fmls	v15.2d, v12.2d, v14.2d		// [- I(ALPHA * X) * A_I]
	fmls	v16.2d, v11.2d, v14.2d		// [- R(ALPHA * X) * A_I]
	fmls	v16.2d, v12.2d, v13.2d		// [- I(ALPHA * X) * A_R]
#endif
#endif // CONJ
	st2	{v15.2d, v16.2d}, [Y_OPTR], #32

	ld2	{v17.2d, v18.2d}, [A_PTR], #32
	ld2	{v19.2d, v20.2d}, [Y_IPTR], #32
#if !defined(CONJ)
#if !defined(XCONJ)
	fmla	v19.2d, v11.2d, v17.2d		// [+ R(ALPHA * X) * A_R]
	fmls	v19.2d, v12.2d, v18.2d		// [- I(ALPHA * X) * A_I]
	fmla	v20.2d, v11.2d, v18.2d		// [+ R(ALPHA * X) * A_I]
	fmla	v20.2d, v12.2d, v17.2d		// [+ I(ALPHA * X) * A_R]
#else
	fmla	v19.2d, v11.2d, v17.2d		// [+ R(ALPHA * X) * A_R]
	fmla	v19.2d, v12.2d, v18.2d		// [- I(ALPHA * X) * A_I]
	fmla	v20.2d, v11.2d, v18.2d		// [+ R(ALPHA * X) * A_I]
	fmls	v20.2d, v12.2d, v17.2d		// [+ I(ALPHA * X) * A_R]
#endif
#else // CONJ
#if !defined(XCONJ)
	fmla	v19.2d, v11.2d, v17.2d		// [+ R(ALPHA * X) * A_R]
	fmla	v19.2d, v12.2d, v18.2d		// [- I(ALPHA * X) * A_I]
	fmls	v20.2d, v11.2d, v18.2d		// [+ R(ALPHA * X) * A_I]
	fmla	v20.2d, v12.2d, v17.2d		// [+ I(ALPHA * X) * A_R]
#else
	fmla	v19.2d, v11.2d, v17.2d		// [+ R(ALPHA * X) * A_R]
	fmls	v19.2d, v12.2d, v18.2d		// [- I(ALPHA * X) * A_I]
	fmls	v20.2d, v11.2d, v18.2d		// [+ R(ALPHA * X) * A_I]
	fmls	v20.2d, v12.2d, v17.2d		// [+ I(ALPHA * X) * A_R]
#endif
#endif // CONJ
	st2	{v19.2d, v20.2d}, [Y_OPTR], #32

#endif

.endm

.macro KERNEL_F1
#if !defined(DOUBLE)
	ld1	{v4.2s}, [A_PTR], #8
	ld1	{v5.2s}, [Y_IPTR], #8
	ext	v6.8b, v4.8b, v4.8b, #4
	fmla	v5.2s, v2.2s, v4.2s
	fmla	v5.2s, v3.2s, v6.2s
	st1	{v5.2s}, [Y_OPTR], #8
#else // DOUBLE
	ld1	{v4.2d}, [A_PTR], #16
	ld1	{v5.2d}, [Y_IPTR], #16
	ext	v6.16b, v4.16b, v4.16b, #8
	fmla	v5.2d, v2.2d, v4.2d
	fmla	v5.2d, v3.2d, v6.2d
	st1	{v5.2d}, [Y_OPTR], #16
#endif
.endm

.macro INIT_S
	lsl	INC_Y, INC_Y, #SHZ
.endm

.macro KERNEL_S1
#if !defined(DOUBLE)
	ld1	{v4.2s}, [A_PTR], #8
	ld1	{v5.2s}, [Y_IPTR], INC_Y
	ext	v6.8b, v4.8b, v4.8b, #4
	fmla	v5.2s, v2.2s, v4.2s
	fmla	v5.2s, v3.2s, v6.2s
	st1	{v5.2s}, [Y_OPTR], INC_Y
#else // DOUBLE
	ld1	{v4.2d}, [A_PTR], #16
	ld1	{v5.2d}, [Y_IPTR], INC_Y
	ext	v6.16b, v4.16b, v4.16b, #8
	fmla	v5.2d, v2.2d, v4.2d
	fmla	v5.2d, v3.2d, v6.2d
	st1	{v5.2d}, [Y_OPTR], INC_Y
#endif
.endm

/*******************************************************************************
* End of macro definitions
*******************************************************************************/

	PROLOGUE

	ldr	INC_Y, [sp]

	SAVE_REGS

	cmp	N, xzr
	ble	zgemv_n_kernel_L999
	cmp	M, xzr
	ble	zgemv_n_kernel_L999

	lsl	LDA, LDA, #SHZ
	lsl	INC_X, INC_X, #SHZ
	mov	J, N

	INIT

	cmp	INC_Y, #1
	bne	zgemv_n_kernel_S_BEGIN

zgemv_n_kernel_F_LOOP:
	mov	A_PTR, A
	mov	Y_IPTR, Y
	mov	Y_OPTR, Y
	mov	X_PTR, X
	add	X, X, INC_X
	INIT_LOOP

	asr	I, M, #2
	cmp	I, xzr
	beq	zgemv_n_kernel_F1

zgemv_n_kernel_F4:

	KERNEL_F1
	KERNEL_F1
	KERNEL_F1
	KERNEL_F1

	subs	I, I, #1
	bne	zgemv_n_kernel_F4

zgemv_n_kernel_F1:

	ands	I, M, #3
	ble	zgemv_n_kernel_F_END

zgemv_n_kernel_F10:

	KERNEL_F1

	subs	I, I, #1
	bne	zgemv_n_kernel_F10

zgemv_n_kernel_F_END:

	add	A, A, LDA
	subs    J, J, #1
        bne     zgemv_n_kernel_F_LOOP

	b	zgemv_n_kernel_L999

zgemv_n_kernel_S_BEGIN:

	INIT_S

zgemv_n_kernel_S_LOOP:
	mov	A_PTR, A
	mov	Y_IPTR, Y
	mov	Y_OPTR, Y
	mov	X_PTR, X
	add	X, X, INC_X
	INIT_LOOP

	asr	I, M, #2
	cmp	I, xzr
	ble	zgemv_n_kernel_S1

zgemv_n_kernel_S4:

	KERNEL_S1
	KERNEL_S1
	KERNEL_S1
	KERNEL_S1

	subs	I, I, #1
	bne	zgemv_n_kernel_S4

zgemv_n_kernel_S1:

	ands	I, M, #3
	ble	zgemv_n_kernel_S_END

zgemv_n_kernel_S10:

	KERNEL_S1

	subs    I, I, #1
        bne     zgemv_n_kernel_S10

zgemv_n_kernel_S_END:

	add	A, A, LDA
	subs    J, J, #1
        bne     zgemv_n_kernel_S_LOOP

zgemv_n_kernel_L999:
	RESTORE_REGS

	mov	w0, wzr
	ret

	EPILOGUE
