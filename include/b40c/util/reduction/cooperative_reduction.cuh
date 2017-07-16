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
 * Cooperative tile reduction within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/warp_reduce.cuh>

namespace b40c {
namespace util {
namespace reduction {


/**
 * Cooperative reduction in raking smem grid hierarchies
 */
template <
	typename RakingDetails,
	typename SecondaryRakingDetails = typename RakingDetails::SecondaryRakingDetails>
struct CooperativeGridReduction;


/**
 * Cooperative tile reduction
 */
template <int VEC_SIZE>
struct CooperativeTileReduction
{
	//---------------------------------------------------------------------
	// Iteration structures for reducing tile vectors into their
	// corresponding raking lanes
	//---------------------------------------------------------------------

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ReduceLane
	{
		template <typename RakingDetails, typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			RakingDetails raking_details,
			typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
			ReductionOp reduction_op)
		{
			// Reduce the partials in this lane/load
			typename RakingDetails::T partial_reduction = SerialReduce<VEC_SIZE>::Invoke(
				data[LANE], reduction_op);

			// Store partial reduction into raking grid
			raking_details.lane_partial[LANE][0] = partial_reduction;

			// Next load
			ReduceLane<LANE + 1, TOTAL_LANES>::Invoke(
				raking_details, data, reduction_op);
		}
	};


	// Terminate
	template <int TOTAL_LANES>
	struct ReduceLane<TOTAL_LANES, TOTAL_LANES>
	{
		template <typename RakingDetails, typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			RakingDetails raking_details,
			typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
			ReductionOp reduction_op) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Reduce a single tile.  Carry is computed (or updated if REDUCE_INTO_CARRY is set)
	 * only in last raking thread
	 *
	 * Caution: Post-synchronization is needed before grid reuse.
	 */
	template <
		bool REDUCE_INTO_CARRY, 				// Whether or not to assign carry or reduce into it
		typename RakingDetails,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		RakingDetails raking_details,
		typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
		typename RakingDetails::T &carry,
		ReductionOp reduction_op)
	{
		// Reduce partials in each vector-load, placing resulting partials in raking smem grid lanes (one lane per load)
		ReduceLane<0, RakingDetails::SCAN_LANES>::Invoke(raking_details, data, reduction_op);

		__syncthreads();

		CooperativeGridReduction<RakingDetails>::template ReduceTileWithCarry<REDUCE_INTO_CARRY>(
			raking_details, carry, reduction_op);
	}

	/**
	 * Reduce a single tile.  Result is computed and returned in all threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename RakingDetails, typename ReductionOp>
	static __device__ __forceinline__ typename RakingDetails::T ReduceTile(
		RakingDetails raking_details,
		typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
		ReductionOp reduction_op)
	{
		// Reduce partials in each vector-load, placing resulting partials in raking smem grid lanes (one lane per load)
		ReduceLane<0, RakingDetails::SCAN_LANES>::Invoke(raking_details, data, reduction_op);

		__syncthreads();

		return CooperativeGridReduction<RakingDetails>::ReduceTile(
			raking_details, reduction_op);
	}
};




/******************************************************************************
 * CooperativeGridReduction
 ******************************************************************************/

/**
 * Cooperative raking grid reduction (specialized for last-level of raking grid)
 */
template <typename RakingDetails>
struct CooperativeGridReduction<RakingDetails, NullType>
{
	typedef typename RakingDetails::T T;

	/**
	 * Reduction in last-level raking grid.  Carry is assigned (or reduced into
	 * if REDUCE_INTO_CARRY is set), but only in last raking thread
	 */
	template <
		bool REDUCE_INTO_CARRY,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		RakingDetails raking_details,
		T &carry,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, reduction_op);

			// Warp reduction
			T warpscan_total = WarpReduce<RakingDetails::LOG_RAKING_THREADS>::InvokeSingle(
				partial, raking_details.warpscan, reduction_op);

			carry = (REDUCE_INTO_CARRY) ?
				reduction_op(carry, warpscan_total) : 	// Reduce into carry
				warpscan_total;							// Assign carry
		}
	}


	/**
	 * Reduction in last-level raking grid.  Result is computed in all threads.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T ReduceTile(
		RakingDetails raking_details,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, reduction_op);

			// Warp reduction
			WarpReduce<RakingDetails::LOG_RAKING_THREADS>::InvokeSingle(
				partial, raking_details.warpscan, reduction_op);
		}

		__syncthreads();

		// Return last thread's inclusive partial
		return raking_details.CumulativePartial();
	}
};


/**
 * Cooperative raking grid reduction for multi-level raking grids
 */
template <typename RakingDetails, typename SecondaryRakingDetails>
struct CooperativeGridReduction
{
	typedef typename RakingDetails::T T;

	/**
	 * Reduction in raking grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	template <bool REDUCE_INTO_CARRY, typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		RakingDetails raking_details,
		T &carry,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, reduction_op);

			// Place partial in next grid
			raking_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively reduce in next grid
		CooperativeGridReduction<SecondaryRakingDetails>::template ReduceTileWithCarry<REDUCE_INTO_CARRY>(
			raking_details.secondary_details, carry, reduction_op);
	}


	/**
	 * Reduction in raking grid.  Result is computed in all threads.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T ReduceTile(
		RakingDetails raking_details,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, reduction_op);

			// Place partial in next grid
			raking_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively reduce in next grid
		return CooperativeGridReduction<SecondaryRakingDetails>::ReduceTile(
			raking_details.secondary_details, reduction_op);
	}
};



} // namespace reduction
} // namespace util
} // namespace b40c

