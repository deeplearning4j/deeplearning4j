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
 * Serial scan over array types
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace scan {

/**
 * Have each thread perform a serial scan over its specified segment.
 */
template <
	int NUM_ELEMENTS,				// Length of array segment to scan
	bool EXCLUSIVE = true>			// Whether or not this is an exclusive scan
struct SerialScan
{
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T partials[],
			T results[],
			T exclusive_partial,
			ReductionOp scan_op)
		{
			T inclusive_partial = scan_op(partials[COUNT], exclusive_partial);
			results[COUNT] = (EXCLUSIVE) ? exclusive_partial : inclusive_partial;
			return Iterate<COUNT + 1, TOTAL>::Invoke(
				partials, results, inclusive_partial, scan_op);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T partials[], T results[], T exclusive_partial, ReductionOp scan_op)
		{
			return exclusive_partial;
		}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Serial scan with the specified operator
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T exclusive_partial,			// Exclusive partial to seed with
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(
			partials, partials, exclusive_partial, scan_op);
	}

	/**
	 * Serial scan with the addition operator
	 */
	template <typename T>
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T exclusive_partial)			// Exclusive partial to seed with
	{
		Sum<T> reduction_op;
		return Invoke(partials, exclusive_partial, reduction_op);
	}


	/**
	 * Serial scan with the specified operator
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T results[],
		T exclusive_partial,			// Exclusive partial to seed with
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(
			partials, results, exclusive_partial, scan_op);
	}

	/**
	 * Serial scan with the addition operator
	 */
	template <typename T>
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T results[],
		T exclusive_partial)			// Exclusive partial to seed with
	{
		Sum<T> reduction_op;
		return Invoke(partials, results, exclusive_partial, reduction_op);
	}
};



} // namespace scan
} // namespace util
} // namespace b40c

