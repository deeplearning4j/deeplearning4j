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
 * Simple tuple types for assisting AOS <-> SOA work
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace util {


template <
	typename T0 = NullType,
	typename T1 = NullType,
	typename T2 = NullType,
	typename T3 = NullType>
struct Tuple;

/**
 * 1 element tuple
 */
template <typename _T0>
struct Tuple<_T0, NullType, NullType, NullType>
{
	// Typedefs
	typedef _T0 T0;

	enum {
		NUM_FIELDS 		= 1,
		VOLATILE 		= util::IsVolatile<typename util::RemovePointers<T0>::Type>::VALUE,
	};


	// Fields
	T0 t0;

	// Constructors
	__host__ __device__ __forceinline__ Tuple() {}
	__host__ __device__ __forceinline__ Tuple(T0 t0) : t0(t0) {}

	// Manipulators

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(
		const TupleSlice &tuple,
		const int col)
	{
		t0[col] = tuple.t0;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(
		const TupleSlice &tuple,
		const int row,
		const int col)
	{
		t0[row][col] = tuple.t0;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Get(
		TupleSlice &retval,
		const int col) const
	{
		retval = TupleSlice(t0[col]);
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Get(
		TupleSlice &retval,
		const int row,
		const int col) const
	{
		retval = TupleSlice(t0[row][col]);
	}
};


/**
 * 2 element tuple
 */
template <typename _T0, typename _T1>
struct Tuple<_T0, _T1, NullType, NullType>
{
	// Typedefs
	typedef _T0 T0;
	typedef _T1 T1;

	enum {
		NUM_FIELDS 		= 2,
		VOLATILE 		= (util::IsVolatile<typename util::RemovePointers<T0>::Type>::VALUE &&
							util::IsVolatile<typename util::RemovePointers<T1>::Type>::VALUE),
	};

	// Fields
	T0 t0;
	T1 t1;


	// Constructors
	__host__ __device__ __forceinline__ Tuple() {}
	__host__ __device__ __forceinline__ Tuple(T0 t0) : t0(t0) {}
	__host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1) : t0(t0), t1(t1) {}

	// Manipulators

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(
		const TupleSlice &tuple,
		const int col)
	{
		t0[col] = tuple.t0;
		t1[col] = tuple.t1;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(
		const TupleSlice &tuple,
		const int row,
		const int col)
	{
		t0[row][col] = tuple.t0;
		t1[row][col] = tuple.t1;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Get(
		TupleSlice &retval,
		const int col) const
	{
		retval = TupleSlice(t0[col], t1[col]);
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Get(
		TupleSlice &retval,
		const int row,
		const int col) const
	{
		retval = TupleSlice(t0[row][col], t1[row][col]);
	}
};


/**
 * 3 element tuple
 */
template <typename _T0, typename _T1, typename _T2>
struct Tuple<_T0, _T1, _T2, NullType>
{
	// Typedefs
	typedef _T0 T0;
	typedef _T1 T1;
	typedef _T2 T2;

	enum {
		NUM_FIELDS 		= 3,
		VOLATILE 		= (util::IsVolatile<typename util::RemovePointers<T0>::Type>::VALUE &&
							util::IsVolatile<typename util::RemovePointers<T1>::Type>::VALUE &&
							util::IsVolatile<typename util::RemovePointers<T2>::Type>::VALUE),
	};

	// Fields
	T0 t0;
	T1 t1;
	T2 t2;

	// Constructor
	__host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1, T2 t2) : t0(t0), t1(t1), t2(t2) {}

	// Manipulators

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(
		const TupleSlice &tuple,
		const int col)
	{
		t0[col] = tuple.t0;
		t1[col] = tuple.t1;
		t2[col] = tuple.t2;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(
		const TupleSlice &tuple,
		const int row,
		const int col)
	{
		t0[row][col] = tuple.t0;
		t1[row][col] = tuple.t1;
		t2[row][col] = tuple.t2;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Get(
		TupleSlice &retval,
		const int col) const
	{
		retval = TupleSlice(
			t0[col],
			t1[col],
			t2[col]);
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Get(
		TupleSlice &retval,
		const int row,
		const int col) const
	{
		retval = TupleSlice(
			t0[row][col],
			t1[row][col],
			t2[row][col]);
	}
};


/**
 * 4 element tuple
 */
template <typename _T0, typename _T1, typename _T2, typename _T3>
struct Tuple
{
	// Typedefs
	typedef _T0 T0;
	typedef _T1 T1;
	typedef _T2 T2;
	typedef _T3 T3;

	enum {
		NUM_FIELDS 		= 3,
		VOLATILE 		= (util::IsVolatile<typename util::RemovePointers<T0>::Type>::VALUE &&
							util::IsVolatile<typename util::RemovePointers<T1>::Type>::VALUE &&
							util::IsVolatile<typename util::RemovePointers<T2>::Type>::VALUE &&
							util::IsVolatile<typename util::RemovePointers<T3>::Type>::VALUE),
	};

	// Fields
	T0 t0;
	T1 t1;
	T2 t2;
	T3 t3;

	// Constructor
	__host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1, T2 t2, T3 t3) : t0(t0), t1(t1), t2(t2), t3(t3) {}

	// Manipulators

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(
		const TupleSlice &tuple,
		const int col)
	{
		t0[col] = tuple.t0;
		t1[col] = tuple.t1;
		t2[col] = tuple.t2;
		t3[col] = tuple.t3;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(
		const TupleSlice &tuple,
		const int row,
		const int col)
	{
		t0[row][col] = tuple.t0;
		t1[row][col] = tuple.t1;
		t2[row][col] = tuple.t2;
		t3[row][col] = tuple.t3;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Get(
		TupleSlice &retval,
		const int col) const
	{
		retval = TupleSlice(
			t0[col],
			t1[col],
			t2[col],
			t3[col]);
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Get(
		TupleSlice &retval,
		const int row,
		const int col) const
	{
		retval = TupleSlice(
			t0[row][col],
			t1[row][col],
			t2[row][col],
			t3[row][col]);
	}
};




} // namespace util
} // namespace b40c
