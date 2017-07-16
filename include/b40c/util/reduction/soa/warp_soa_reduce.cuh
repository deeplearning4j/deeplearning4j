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
 * Cooperative tuple warp-reduction
 *
 * Does not support non-commutative operators.  (Suggested to use a warpscan
 * instead for those scenarios)
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace reduction {
namespace soa {

/**
 * Perform NUM_ELEMENTS of warp-synchronous reduction.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 *
 * Can be used to perform concurrent, independent warp-reductions if
 * storage pointers and their local-thread indexing id's are set up properly.
 *
 * The type WarpscanSoa is a 2D structure-of-array of smem storage, each SOA array having
 * dimensions [2][NUM_ELEMENTS].
 */
template <int LOG_NUM_ELEMENTS>				// Log of number of elements to warp-reduce
struct WarpSoaReduce
{
	enum {
		NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS,
	};

	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iteration
	template <int OFFSET_LEFT, int __dummy = 0>
	struct Iterate
	{
		// Reduce
		template <
			typename Tuple,
			typename WarpscanSoa,
			typename ReductionOp>
		static __device__ __forceinline__ Tuple Reduce(
			Tuple exclusive_partial,
			WarpscanSoa warpscan_partials,
			ReductionOp reduction_op,
			int warpscan_tid)
		{
			// Store exclusive partial
			warpscan_partials.Set(exclusive_partial, 1, warpscan_tid);

			if (!WarpscanSoa::VOLATILE) __threadfence_block();

			// Load current partial
			Tuple current_partial;
			warpscan_partials.Get(current_partial, 1, warpscan_tid - OFFSET_LEFT);

			if (!WarpscanSoa::VOLATILE) __threadfence_block();

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = reduction_op(current_partial, exclusive_partial);

			// Recurse
			return Iterate<OFFSET_LEFT / 2>::Reduce(
				inclusive_partial, warpscan_partials, reduction_op, warpscan_tid);
		}
	};

	// Termination
	template <int __dummy>
	struct Iterate<0, __dummy>
	{
		// Reduce
		template <
			typename Tuple,
			typename WarpscanSoa,
			typename ReductionOp>
		static __device__ __forceinline__ Tuple Reduce(
			Tuple exclusive_partial,
			WarpscanSoa warpscan_partials,
			ReductionOp reduction_op,
			int warpscan_tid)
		{
			return exclusive_partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Warp-reduction.  Result is returned in all warpscan threads.
	 */
	template <
		typename Tuple,
		typename WarpscanSoa,
		typename ReductionOp>
	static __device__ __forceinline__ Tuple Reduce(
		Tuple current_partial,					// Input partial
		WarpscanSoa warpscan_partials,			// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		ReductionOp reduction_op,
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		Tuple inclusive_partial = Iterate<NUM_ELEMENTS / 2>::Reduce(
			current_partial, warpscan_partials, reduction_op, warpscan_tid);

		// Write our inclusive partial
		warpscan_partials.Set(inclusive_partial, 1, warpscan_tid);

		if (!WarpscanSoa::VOLATILE) __threadfence_block();

		// Return last thread's inclusive partial
		Tuple retval;
		return warpscan_partials.Get(retval, 1, NUM_ELEMENTS - 1);
	}


	/**
	 * Warp-reduction.  Result is returned in last warpscan thread.
	 */
	template <
		typename Tuple,
		typename WarpscanSoa,
		typename ReductionOp>
	static __device__ __forceinline__ Tuple ReduceInLast(
		Tuple exclusive_partial,				// Input partial
		WarpscanSoa warpscan_partials,			// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		ReductionOp reduction_op,
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		return Iterate<NUM_ELEMENTS / 2>::Reduce(
			exclusive_partial, warpscan_partials, reduction_op, warpscan_tid);
	}

};



} // namespace soa
} // namespace reduction
} // namespace util
} // namespace b40c

