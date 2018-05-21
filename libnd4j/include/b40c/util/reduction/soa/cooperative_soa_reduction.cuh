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
 * Cooperative tile SOA (structure-of-arrays) reduction within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/soa/serial_soa_reduce.cuh>
#include <b40c/util/reduction/soa/warp_soa_reduce.cuh>
#include <b40c/util/scan/soa/warp_soa_scan.cuh>

namespace b40c {
namespace util {
namespace reduction {
namespace soa {


/**
 * Cooperative SOA reduction in raking smem grid hierarchies
 */
template <
	typename RakingSoaDetails,
	typename SecondaryRakingSoaDetails = typename RakingSoaDetails::SecondaryRakingSoaDetails>
struct CooperativeSoaGridReduction;


/**
 * Cooperative SOA tile reduction
 */
template <int VEC_SIZE>
struct CooperativeSoaTileReduction
{
	//---------------------------------------------------------------------
	// Iteration structures for reducing tile SOA vectors into their
	// corresponding raking lanes
	//---------------------------------------------------------------------

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ReduceLane
	{
		template <
			typename RakingSoaDetails,
			typename TileSoa,
			typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			RakingSoaDetails raking_soa_details,
			TileSoa tile_soa,
			ReductionOp reduction_op)
		{
			// Reduce the partials in this lane/load
			typename RakingSoaDetails::TileTuple partial_reduction;
			SerialSoaReduce<VEC_SIZE>::Reduce(
				partial_reduction, tile_soa, LANE, reduction_op);

			// Store partial reduction into raking grid
			raking_soa_details.lane_partials.Set(partial_reduction, LANE, 0);

			// Next load
			ReduceLane<LANE + 1, TOTAL_LANES>::Invoke(
				raking_soa_details, tile_soa, reduction_op);
		}
	};

	// Terminate
	template <int TOTAL_LANES>
	struct ReduceLane<TOTAL_LANES, TOTAL_LANES>
	{
		template <
			typename RakingSoaDetails,
			typename TileSoa,
			typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			RakingSoaDetails raking_soa_details,
			TileSoa tile_soa,
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
		bool REDUCE_INTO_CARRY,
		typename RakingSoaDetails,
		typename TileSoa,
		typename TileTuple,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		RakingSoaDetails raking_soa_details,
		TileSoa tile_soa,
		TileTuple &carry,
		ReductionOp reduction_op)
	{
		// Reduce vectors in tile, placing resulting partial into corresponding raking grid lanes
		ReduceLane<0, RakingSoaDetails::SCAN_LANES>::Invoke(
			raking_soa_details, tile_soa, reduction_op);

		__syncthreads();

		CooperativeSoaGridReduction<RakingSoaDetails>::template ReduceTileWithCarry<REDUCE_INTO_CARRY>(
			raking_soa_details, carry, reduction_op);
	}

	/**
	 * Reduce a single tile.  Result is computed and returned in all threads.
	 *
	 * No post-synchronization needed before raking_details reuse.
	 */
	template <
		typename TileTuple,
		typename RakingSoaDetails,
		typename TileSoa,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTile(
		TileTuple &retval,
		RakingSoaDetails raking_soa_details,
		TileSoa tile_soa,
		ReductionOp reduction_op)
	{
		// Reduce vectors in tile, placing resulting partial into corresponding raking grid lanes
		ReduceLane<0, RakingSoaDetails::SCAN_LANES>::Invoke(
			raking_soa_details, tile_soa, reduction_op);

		__syncthreads();

		return CooperativeSoaGridReduction<RakingSoaDetails>::ReduceTile(
			raking_soa_details, reduction_op);
	}
};




/******************************************************************************
 * CooperativeSoaGridReduction
 ******************************************************************************/

/**
 * Cooperative SOA raking grid reduction (specialized for last-level of raking grid)
 */
template <typename RakingSoaDetails>
struct CooperativeSoaGridReduction<RakingSoaDetails, NullType>
{
	typedef typename RakingSoaDetails::TileTuple TileTuple;

	/**
	 * Reduction in last-level raking grid.  Carry is assigned (or reduced into
	 * if REDUCE_INTO_CARRY is set), but only in last raking thread
	 */
	template <
		bool REDUCE_INTO_CARRY,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		RakingSoaDetails raking_soa_details,
		TileTuple &carry,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {

			// Raking reduction
			TileTuple inclusive_partial;
			SerialSoaReduce<RakingSoaDetails::PARTIALS_PER_SEG>::Reduce(
				inclusive_partial,
				raking_soa_details.raking_segments,
				reduction_op);

			// Inclusive warp scan that sets warpscan total in all
			// Raking threads. (Use warp scan instead of warp reduction
			// because the latter supports non-commutative reduction
			// operators)
			TileTuple warpscan_total;
			scan::soa::WarpSoaScan<
				RakingSoaDetails::LOG_RAKING_THREADS,
				false>::Scan(
					inclusive_partial,
					warpscan_total,
					raking_soa_details.warpscan_partials,
					reduction_op);

			// Update/set carry
			carry = (REDUCE_INTO_CARRY) ?
				reduction_op(carry, warpscan_total) :
				warpscan_total;
		}
	}


	/**
	 * Reduction in last-level raking grid.  Result is computed in all threads.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ TileTuple ReduceTile(
		RakingSoaDetails raking_soa_details,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {

			// Raking reduction
			TileTuple inclusive_partial = SerialSoaReduce<RakingSoaDetails::PARTIALS_PER_SEG>::Reduce(
				raking_soa_details.raking_segments, reduction_op);

			// Warp reduction
			TileTuple warpscan_total = WarpSoaReduce<RakingSoaDetails::LOG_RAKING_THREADS>::ReduceInLast(
				inclusive_partial,
				raking_soa_details.warpscan_partials,
				reduction_op);
		}

		__syncthreads();

		// Return last thread's inclusive partial
		return raking_soa_details.CumulativePartial();
	}
};


} // namespace soa
} // namespace reduction
} // namespace util
} // namespace b40c

