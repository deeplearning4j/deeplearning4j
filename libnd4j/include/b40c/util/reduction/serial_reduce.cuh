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
 * Serial reduction over array types
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace reduction {

/**
 * Have each thread perform a serial reduction over its specified segment
 */
template <int NUM_ELEMENTS>
struct SerialReduce
{
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(T *partials, ReductionOp reduction_op)
		{
			T a = Iterate<COUNT - 2, TOTAL>::Invoke(partials, reduction_op);
			T b = partials[TOTAL - COUNT];
			T c = partials[TOTAL - (COUNT - 1)];

			// TODO: consider specializing with a video 3-op instructions on SM2.0+,
			// e.g., asm("vadd.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(a) : "r"(a), "r"(b), "r"(c));
			return reduction_op(a, reduction_op(b, c));
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<2, TOTAL>
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(T *partials, ReductionOp reduction_op)
		{
			return reduction_op(partials[TOTAL - 2], partials[TOTAL - 1]);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<1, TOTAL>
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(T *partials, ReductionOp reduction_op)
		{
			return partials[TOTAL - 1];
		}
	};
	
	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Serial reduction with the specified operator
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T *partials,
		ReductionOp reduction_op)
	{
		return Iterate<NUM_ELEMENTS, NUM_ELEMENTS>::Invoke(partials, reduction_op);
	}


	/**
	 * Serial reduction with the addition operator
	 */
	template <typename T>
	static __device__ __forceinline__ T Invoke(
		T *partials)
	{
		Sum<T> reduction_op;
		return Invoke(partials, reduction_op);
	}


	/**
	 * Serial reduction with the specified operator, seeded with the
	 * given exclusive partial
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T *partials,
		T exclusive_partial,
		ReductionOp reduction_op)
	{
		return reduction_op(
			exclusive_partial,
			Invoke(partials, reduction_op));
	}

	/**
	 * Serial reduction with the addition operator, seeded with the
	 * given exclusive partial
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T *partials,
		T exclusive_partial)
	{
		Sum<T> reduction_op;
		 return Invoke(partials, exclusive_partial, reduction_op);
	}
};


} // namespace reduction
} // namespace util
} // namespace b40c


