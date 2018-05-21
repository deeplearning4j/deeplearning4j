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
 * Serial tuple reduction over structure-of-array types.
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace reduction {
namespace soa {


/**
 * Have each thread perform a serial reduction over its specified SOA segment
 */
template <int NUM_ELEMENTS>					// Length of SOA array segment to reduce
struct SerialSoaReduce
{
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Next SOA tuple
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		// Reduce
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Reduce(
			RakingSoa raking_partials,
			Tuple exclusive_partial,
			ReductionOp reduction_op)
		{
			// Load current partial
			Tuple current_partial;
			raking_partials.Get(current_partial, COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = reduction_op(exclusive_partial, current_partial);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Reduce(
				raking_partials, inclusive_partial, reduction_op);
		}

		// Reduce 2D
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Reduce(
			RakingSoa raking_partials,
			Tuple exclusive_partial,
			int row,
			ReductionOp reduction_op)
		{
			// Load current partial
			Tuple current_partial;
			raking_partials.Get(current_partial, row, COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = reduction_op(exclusive_partial, current_partial);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Reduce(
				raking_partials, inclusive_partial, row, reduction_op);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		// Reduce
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Reduce(
			RakingSoa raking_partials,
			Tuple exclusive_partial,
			ReductionOp reduction_op)
		{
			return exclusive_partial;
		}

		// Reduce 2D
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Reduce(
			RakingSoa raking_partials,
			Tuple exclusive_partial,
			int row,
			ReductionOp reduction_op)
		{
			return exclusive_partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Reduce a structure-of-array RakingSoa into a single Tuple "slice"
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ void Reduce(
		Tuple &retval,
		RakingSoa raking_partials,
		ReductionOp reduction_op)
	{
		// Get first partial
		Tuple current_partial;
		raking_partials.Get(current_partial, 0);

		retval = Iterate<1, NUM_ELEMENTS>::Reduce(
			raking_partials, current_partial, reduction_op);
	}

	/**
	 * Reduce a structure-of-array RakingSoa into a single Tuple "slice", seeded
	 * with exclusive_partial
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple SeedReduce(
		RakingSoa raking_partials,
		Tuple exclusive_partial,
		ReductionOp reduction_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Reduce(
			raking_partials, exclusive_partial, reduction_op);
	}


	/**
	 * Reduce one row of a 2D structure-of-array RakingSoa into a single Tuple "slice"
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ void Reduce(
		Tuple &retval,
		RakingSoa raking_partials,
		int row,
		ReductionOp reduction_op)
	{
		// Get first partial
		Tuple current_partial;
		raking_partials.Get(current_partial, row, 0);

		retval = Iterate<1, NUM_ELEMENTS>::Reduce(
			raking_partials, current_partial, row, reduction_op);
	}


	/**
	 * Reduce one row of a 2D structure-of-array RakingSoa into a single Tuple "slice", seeded
	 * with exclusive_partial
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple SeedReduce(
		RakingSoa raking_partials,
		Tuple exclusive_partial,
		int row,
		ReductionOp reduction_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Reduce(
			raking_partials, exclusive_partial, row, reduction_op);
	}
};


} // namespace soa
} // namespace reduction
} // namespace util
} // namespace b40c

