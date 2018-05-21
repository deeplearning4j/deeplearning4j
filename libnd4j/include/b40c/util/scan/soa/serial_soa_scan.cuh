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
 * Serial tuple scan over structure-of-array types.
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace scan {
namespace soa {


/**
 * Have each thread perform a serial scan over its specified SOA segment
 */
template <
	int NUM_ELEMENTS,				// Length of SOA array segment to scan
	bool EXCLUSIVE = true>			// Whether or not this is an exclusive scan
struct SerialSoaScan
{
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		// Scan
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Scan(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial,
			ReductionOp scan_op)
		{
			// Load current partial
			Tuple current_partial;
			raking_partials.Get(current_partial, COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = scan_op(exclusive_partial, current_partial);

			if (EXCLUSIVE) {
				// Store exclusive partial
				raking_results.Set(exclusive_partial, COUNT);
			} else {
				// Store inclusive partial
				raking_results.Set(inclusive_partial, COUNT);
			}

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Scan(
				raking_partials, raking_results, inclusive_partial, scan_op);
		}

		// Scan 2D
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Scan(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial,
			int row,
			ReductionOp scan_op)
		{
			// Load current partial
			Tuple current_partial;
			raking_partials.Get(current_partial, row, COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = scan_op(exclusive_partial, current_partial);

			if (EXCLUSIVE) {
				// Store exclusive partial
				raking_results.Set(exclusive_partial, row, COUNT);
			} else {
				// Store inclusive partial
				raking_results.Set(inclusive_partial, row, COUNT);
			}

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Scan(
				raking_partials, raking_results, inclusive_partial, row, scan_op);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		// Scan
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Scan(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial,
			ReductionOp scan_op)
		{
			return exclusive_partial;
		}

		// Scan 2D
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Scan(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial,
			int row,
			ReductionOp scan_op)
		{
			return exclusive_partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Scan a 2D structure-of-array RakingSoa, seeded  with exclusive_partial.
	 * The tuple returned is the inclusive total.
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Scan(
		RakingSoa raking_partials,			// Scan input/output
		Tuple exclusive_partial,			// Exclusive partial to seed with
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Scan(
			raking_partials, raking_partials, exclusive_partial, scan_op);
	}


	/**
	 * Scan a 2D structure-of-array RakingSoa, seeded  with exclusive_partial.
	 * The tuple returned is the inclusive total.
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Scan(
		RakingSoa raking_partials,			// Scan input
		RakingSoa raking_results,			// Scan output
		Tuple exclusive_partial,			// Exclusive partial to seed with
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Scan(
			raking_partials, raking_results, exclusive_partial, scan_op);
	}


	/**
	 * Scan one row of a 2D structure-of-array RakingSoa, seeded
	 * with exclusive_partial.  The tuple returned is the inclusive total.
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Scan(
		RakingSoa raking_partials,			// Scan input/output
		Tuple exclusive_partial,			// Exclusive partial to seed with
		int row,
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Scan(
			raking_partials, raking_partials, exclusive_partial, row, scan_op);
	}


	/**
	 * Scan one row of a 2D structure-of-array RakingSoa, seeded
	 * with exclusive_partial.  The tuple returned is the inclusive total.
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Scan(
		RakingSoa raking_partials,			// Scan input
		RakingSoa raking_results,			// Scan output
		Tuple exclusive_partial,			// Exclusive partial to seed with
		int row,
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Scan(
			raking_partials, raking_results, exclusive_partial, row, scan_op);
	}
};

} // namespace soa
} // namespace scan
} // namespace util
} // namespace b40c

