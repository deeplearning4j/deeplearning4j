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
 * Cooperative tuple warp-scan
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace scan {
namespace soa {

/**
 * Structure-of-arrays tuple warpscan.  Performs STEPS steps of a
 * Kogge-Stone style prefix scan.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 *
 * The type WarpscanSoa is a 2D structure-of-array of smem storage, each SOA array having
 * dimensions [2][NUM_ELEMENTS].
 */
template <
	int LOG_NUM_ELEMENTS,					// Log of number of elements to warp-reduce
	bool EXCLUSIVE = true,					// Whether or not this is an exclusive scan
	int STEPS = LOG_NUM_ELEMENTS>			// Number of steps to run, i.e., produce scanned segments of (1 << STEPS) elements
struct WarpSoaScan
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iteration
	template <int OFFSET_LEFT, int WIDTH>
	struct Iterate
	{
		// Scan
		template <
			typename Tuple,
			typename WarpscanSoa,
			typename ReductionOp>
		static __device__ __forceinline__ Tuple Scan(
			Tuple partial,
			WarpscanSoa warpscan_partials,
			ReductionOp scan_op,
			int warpscan_tid)
		{
			// Store exclusive partial
			warpscan_partials.Set(partial, 1, warpscan_tid);

			if (!WarpscanSoa::VOLATILE) __threadfence_block();

			if (ReductionOp::IDENTITY_STRIDES || (warpscan_tid >= OFFSET_LEFT)) {

				// Load current partial
				Tuple current_partial;
				warpscan_partials.Get(current_partial, 1, warpscan_tid - OFFSET_LEFT);

				// Compute inclusive partial from exclusive and current partials
				partial = scan_op(current_partial, partial);
			}

			if (!WarpscanSoa::VOLATILE) __threadfence_block();

			// Recurse
			return Iterate<OFFSET_LEFT * 2, WIDTH>::Scan(
				partial, warpscan_partials, scan_op, warpscan_tid);
		}
	};

	// Termination
	template <int WIDTH>
	struct Iterate<WIDTH, WIDTH>
	{
		// Scan
		template <
			typename Tuple,
			typename WarpscanSoa,
			typename ReductionOp>
		static __device__ __forceinline__ Tuple Scan(
			Tuple exclusive_partial,
			WarpscanSoa warpscan_partials,
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
	 * Cooperative warp-scan.  Returns the calling thread's scanned partial.
	 */
	template <
		typename Tuple,
		typename WarpscanSoa,
		typename ReductionOp>
	static __device__ __forceinline__ Tuple Scan(
		Tuple current_partial,						// Input partial
		WarpscanSoa warpscan_partials,				// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		ReductionOp scan_op,						// Scan operator
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		Tuple inclusive_partial = Iterate<1, (1 << STEPS)>::Scan(
			current_partial,
			warpscan_partials,
			scan_op,
			warpscan_tid);

		if (EXCLUSIVE) {

			// Write our inclusive partial
			warpscan_partials.Set(inclusive_partial, 1, warpscan_tid);

			if (!WarpscanSoa::VOLATILE) __threadfence_block();

			// Return exclusive partial
			Tuple exclusive_partial;
			warpscan_partials.Get(exclusive_partial, 1, warpscan_tid - 1);
			return exclusive_partial;

		} else {
			return inclusive_partial;
		}
	}

	/**
	 * Cooperative warp-scan.  Returns the calling thread's scanned partial.
	 */
	template <
		typename Tuple,
		typename WarpscanSoa,
		typename ReductionOp>
	static __device__ __forceinline__ Tuple Scan(
		Tuple current_partial,						// Input partial
		Tuple &total_reduction,						// Total reduction (out param)
		WarpscanSoa warpscan_partials,				// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		ReductionOp scan_op,						// Scan operator
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		Tuple inclusive_partial = Iterate<1, (1 << STEPS)>::Scan(
			current_partial,
			warpscan_partials,
			scan_op,
			warpscan_tid);

		// Write our inclusive partial
		warpscan_partials.Set(inclusive_partial, 1, warpscan_tid);

		if (!WarpscanSoa::VOLATILE) __threadfence_block();

		// Set total to the last thread's inclusive partial
		warpscan_partials.Get(total_reduction, 1, NUM_ELEMENTS - 1);

		if (EXCLUSIVE) {

			// Return exclusive partial
			Tuple exclusive_partial;
			warpscan_partials.Get(exclusive_partial, 1, warpscan_tid - 1);
			return exclusive_partial;

		} else {
			return inclusive_partial;
		}
	}
};


} // namespace soa
} // namespace scan
} // namespace util
} // namespace b40c

