/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Common B40C Routines 
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {


/******************************************************************************
 * Macro utilities
 ******************************************************************************/

/**
 * Select maximum
 */
#define B40C_MAX(a, b) ((a > b) ? a : b)


/**
 * Select maximum
 */
#define B40C_MIN(a, b) ((a < b) ? a : b)

/**
 * Return the size in quad-words of a number of bytes
 */
#define B40C_QUADS(bytes) (((bytes + sizeof(uint4) - 1) / sizeof(uint4)))

/******************************************************************************
 * Simple templated utilities
 ******************************************************************************/

/**
 * Supress warnings for unused constants
 */
template <typename T>
__host__ __device__ __forceinline__ void SuppressUnusedConstantWarning(const T) {}


/**
 * Perform a swap
 */
template <typename T> 
void __host__ __device__ __forceinline__ Swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}


template <typename K, int magnitude, bool shift_left> struct MagnitudeShiftOp;

/**
 * MagnitudeShift().  Allows you to shift left for positive magnitude values, 
 * right for negative.   
 * 
 * N.B. This code is a little strange; we are using this meta-programming 
 * pattern of partial template specialization for structures in order to 
 * decide whether to shift left or right.  Normally we would just use a 
 * conditional to decide if something was negative or not and then shift 
 * accordingly, knowing that the compiler will elide the untaken branch, 
 * i.e., the out-of-bounds shift during dead code elimination. However, 
 * the pass for bounds-checking shifts seems to happen before the DCE 
 * phase, which results in a an unsightly number of compiler warnings, so 
 * we force the issue earlier using structural template specialization.
 */
template <typename K, int magnitude> 
__device__ __forceinline__ K MagnitudeShift(K key)
{
	return MagnitudeShiftOp<K, (magnitude > 0) ? magnitude : magnitude * -1, (magnitude > 0)>::Shift(key);
}

template <typename K, int magnitude>
struct MagnitudeShiftOp<K, magnitude, true>
{
	__device__ __forceinline__ static K Shift(K key)
	{
		return key << magnitude;
	}
};

template <typename K, int magnitude>
struct MagnitudeShiftOp<K, magnitude, false>
{
	__device__ __forceinline__ static K Shift(K key)
	{
		return key >> magnitude;
	}
};


/******************************************************************************
 * Metaprogramming Utilities
 ******************************************************************************/

/**
 * Null type
 */
struct NullType {};


/**
 * Int2Type
 */
template <int N>
struct Int2Type
{
	enum {VALUE = N};
};


/**
 * Statically determine log2(N), rounded up, e.g.,
 * 		Log2<8>::VALUE == 3
 * 		Log2<3>::VALUE == 2
 */
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
	// Inductive case
	static const int VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE;
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
	// Base case
	static const int VALUE = (1 << (COUNT - 1) < N) ?
		COUNT :
		COUNT - 1;
};


/**
 * If/Then/Else
 */
template <bool IF, typename ThenType, typename ElseType>
struct If
{
	// true
	typedef ThenType Type;
};

template <typename ThenType, typename ElseType>
struct If<false, ThenType, ElseType>
{
	// false
	typedef ElseType Type;
};


/**
 * Equals 
 */
template <typename A, typename B>
struct Equals
{
	enum {
		VALUE = 0,
		NEGATE = 1
	};
};

template <typename A>
struct Equals <A, A>
{
	enum {
		VALUE = 1,
		NEGATE = 0
	};
};



/**
 * Is volatile
 */
template <typename Tp>
struct IsVolatile
{
	enum { VALUE = 0 };
};
template <typename Tp>
struct IsVolatile<Tp volatile>
{
	enum { VALUE = 1 };
};


/**
 * Removes pointers
 */
template <typename Tp, typename Up>
struct RemovePointersHelper
{
	typedef Tp Type;
};
template <typename Tp, typename Up>
struct RemovePointersHelper<Tp, Up*>
{
	typedef typename RemovePointersHelper<Up, Up>::Type Type;
};
template <typename Tp>
struct RemovePointers : RemovePointersHelper<Tp, Tp> {};



} // namespace util
} // namespace b40c

