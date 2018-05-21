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
 * Cooperative warp-reduction
 *
 * Does not support non-commutative operators.  (Suggested to use a warpscan
 * instead for those scenarios
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace reduction {


/**
 * Perform NUM_ELEMENTS of warp-synchronous reduction.
 *
 * Can be used to perform concurrent, independent warp-reductions if
 * storage pointers and their local-thread indexing id's are set up properly.
 *
 * Requires a 2D "warpscan" structure of smem storage having dimensions [2][NUM_ELEMENTS].
 */
template <int LOG_NUM_ELEMENTS>				// Log of number of elements to warp-reduce
struct WarpReduce
{
	enum {
		NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS,
	};


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate
	template <int OFFSET_LEFT, int __dummy = 0>
	struct Iterate
	{
		template <typename T, typename WarpscanT, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T exclusive_partial,
			WarpscanT warpscan[][NUM_ELEMENTS],
			ReductionOp reduction_op,
			int warpscan_tid)
		{
			// Store exclusive partial
			warpscan[1][warpscan_tid] = exclusive_partial;

			if (!IsVolatile<WarpscanT>::VALUE) __threadfence_block();

			// Load current partial
			T current_partial = warpscan[1][warpscan_tid - OFFSET_LEFT];

			if (!IsVolatile<WarpscanT>::VALUE) __threadfence_block();

			// Compute inclusive partial
			T inclusive_partial = reduction_op(exclusive_partial, current_partial);

			// Recurse
			return Iterate<OFFSET_LEFT / 2>::Invoke(
				inclusive_partial, warpscan, reduction_op, warpscan_tid);
		}
	};
	
	// Termination
	template <int __dummy>
	struct Iterate<0, __dummy>
	{
		template <typename T, typename WarpscanT, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T partial,
			WarpscanT warpscan[][NUM_ELEMENTS],
			ReductionOp reduction_op,
			int warpscan_tid)
		{
			return partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Warp reduction with the specified operator, result is returned in last warpscan thread
	 */
	template <typename T, typename WarpscanT, typename ReductionOp>
	static __device__ __forceinline__ T InvokeSingle(
		T partial,								// Input partial
		WarpscanT warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		ReductionOp reduction_op,				// Reduction operator
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		return Iterate<NUM_ELEMENTS / 2>::Invoke(
			partial, warpscan, reduction_op, warpscan_tid);
	}


	/**
	 * Warp reduction with the addition operator, result is returned in last warpscan thread
	 */
	template <typename T, typename WarpscanT>
	static __device__ __forceinline__ T InvokeSingle(
		T partial,								// Input partial
		WarpscanT warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		Sum<T> reduction_op;
		return InvokeSingle(partial, warpscan, reduction_op, warpscan_tid);
	}


	/**
	 * Warp reduction with the specified operator, result is returned in all warpscan threads)
	 */
	template <typename T, typename WarpscanT, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T current_partial,						// Input partial
		WarpscanT warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		ReductionOp reduction_op,				// Reduction operator
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		T inclusive_partial = InvokeSingle(
			current_partial, warpscan, reduction_op, warpscan_tid);

		// Write our inclusive partial
		warpscan[1][warpscan_tid] = inclusive_partial;

		if (!IsVolatile<WarpscanT>::VALUE) __threadfence_block();

		// Return last thread's inclusive partial
		return warpscan[1][NUM_ELEMENTS - 1];
	}


	/**
	 * Warp reduction with the addition operator, result is returned in all warpscan threads)
	 */
	template <typename T, typename WarpscanT>
	static __device__ __forceinline__ T Invoke(
		T current_partial,						// Input partial
		WarpscanT warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		Sum<T> reduction_op;
		return Invoke(current_partial, warpscan, reduction_op, warpscan_tid);
	}
};


} // namespace reduction
} // namespace util
} // namespace b40c

