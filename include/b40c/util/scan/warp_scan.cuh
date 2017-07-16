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
 * Cooperative warp-scan
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace scan {



/**
 * Performs STEPS steps of a Kogge-Stone style prefix scan.
 *
 * Requires a 2D "warpscan" structure of smem storage having dimensions [2][NUM_ELEMENTS].
 */
template <
	int LOG_NUM_ELEMENTS,					// Log of number of elements to warp-reduce
	bool EXCLUSIVE = true,					// Whether or not this is an exclusive scan
	int STEPS = LOG_NUM_ELEMENTS>			// Number of steps to run, i.e., produce scanned segments of (1 << STEPS) elements
struct WarpScan
{
	enum {
		NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS,
	};

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// General iteration
	template <int OFFSET_LEFT, int WIDTH>
	struct Iterate
	{
		template <typename T, typename WarpscanT, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T exclusive_partial,
			WarpscanT warpscan[][NUM_ELEMENTS],
			ReductionOp scan_op,
			int warpscan_tid)
		{
			warpscan[1][warpscan_tid] = exclusive_partial;

			if (!IsVolatile<WarpscanT>::VALUE) __threadfence_block();

			T offset_partial = warpscan[1][warpscan_tid - OFFSET_LEFT];

			if (!IsVolatile<WarpscanT>::VALUE) __threadfence_block();

			T inclusive_partial = scan_op(offset_partial, exclusive_partial);

			return Iterate<OFFSET_LEFT * 2, WIDTH>::Invoke(
				inclusive_partial,
				warpscan,
				scan_op,
				warpscan_tid);
		}
	};

	// Termination
	template <int WIDTH>
	struct Iterate<WIDTH, WIDTH>
	{
		template <typename T, typename WarpscanT, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T exclusive_partial,
			WarpscanT warpscan[][NUM_ELEMENTS],
			ReductionOp scan_op,
			int warpscan_tid)
		{
			return exclusive_partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Warpscan with the specified operator
	 */
	template <typename T, typename WarpscanT, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T current_partial,							// Input partial
		WarpscanT warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		ReductionOp scan_op,						// Scan operator
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		const int WIDTH = 1 << STEPS;
		T inclusive_partial = Iterate<1, WIDTH>::Invoke(
			current_partial,
			warpscan,
			scan_op,
			warpscan_tid);

		if (EXCLUSIVE) {
			// Write out our inclusive partial
			warpscan[1][warpscan_tid] = inclusive_partial;

			if (!IsVolatile<WarpscanT>::VALUE) __threadfence_block();

			// Return exclusive partial
			return warpscan[1][warpscan_tid - 1];

		} else {
			return inclusive_partial;
		}
	}


	/**
	 * Warpscan with the addition operator
	 */
	template <typename T, typename WarpscanT>
	static __device__ __forceinline__ T Invoke(
		T current_partial,							// Input partial
		WarpscanT warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		Sum<T> scan_op;
		return Invoke(
			current_partial,
			warpscan,
			scan_op,
			warpscan_tid);
	}


	/**
	 * Warpscan with the specified operator, returning the cumulative reduction
	 */
	template <typename T, typename WarpscanT, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T current_partial,							// Input partial
		T &total_reduction,							// Total reduction (out param)
		WarpscanT warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		ReductionOp scan_op,						// Scan operator
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		const int WIDTH = 1 << STEPS;
		T inclusive_partial = Iterate<1, WIDTH>::Invoke(
			current_partial,
			warpscan,
			scan_op,
			warpscan_tid);

		// Write our inclusive partial and then set total to the last thread's inclusive partial
		warpscan[1][warpscan_tid] = inclusive_partial;

		if (!IsVolatile<WarpscanT>::VALUE) __threadfence_block();

		// Get total
		total_reduction = warpscan[1][NUM_ELEMENTS - 1];

		if (EXCLUSIVE) {

			// Return exclusive partial
			return warpscan[1][warpscan_tid - 1];

		} else {
			return inclusive_partial;
		}
	}

	/**
	 * Warpscan with the addition operator, returning the cumulative reduction
	 */
	template <typename T, typename WarpscanT>
	static __device__ __forceinline__ T Invoke(
		T current_partial,							// Input partial
		T &total_reduction,							// Total reduction (out param)
		WarpscanT warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		Sum<T> scan_op;
		return Invoke(
			current_partial,
			total_reduction,
			warpscan,
			scan_op,
			warpscan_tid);
	}
};



} // namespace scan
} // namespace util
} // namespace b40c

