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
 * Operational details for threads working in an SOA (structure of arrays)
 * raking grid
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace util {


/**
 * Operational details for threads working in an raking grid
 */
template <
	typename TileTuple,
	typename RakingGridTuple,
	int Grids = RakingGridTuple::NUM_FIELDS,
	typename SecondaryRakingGridTuple = typename If<
		Equals<NullType, typename RakingGridTuple::T0::SecondaryGrid>::VALUE,
		NullType,
		Tuple<
			typename RakingGridTuple::T0::SecondaryGrid,
			typename RakingGridTuple::T1::SecondaryGrid> >::Type>
struct RakingSoaDetails;


/**
 * Two-field raking details
 */
template <
	typename _TileTuple,
	typename RakingGridTuple>
struct RakingSoaDetails<
	_TileTuple,
	RakingGridTuple,
	2,
	NullType> : RakingGridTuple::T0
{
	enum {
		CUMULATIVE_THREAD 	= RakingSoaDetails::RAKING_THREADS - 1,
		WARP_THREADS 		= B40C_WARP_THREADS(RakingSoaDetails::CUDA_ARCH)
	};

	// Simple SOA tuple "slice" type
	typedef _TileTuple TileTuple;

	// SOA type of raking lanes
	typedef Tuple<
		typename TileTuple::T0*,
		typename TileTuple::T1*> GridStorageSoa;

	// SOA type of warpscan storage
	typedef Tuple<
		typename RakingGridTuple::T0::WarpscanT (*)[WARP_THREADS],
		typename RakingGridTuple::T1::WarpscanT (*)[WARP_THREADS]> WarpscanSoa;

	// SOA type of partial-insertion pointers
	typedef Tuple<
		typename RakingGridTuple::T0::LanePartial,
		typename RakingGridTuple::T1::LanePartial> LaneSoa;

	// SOA type of raking segments
	typedef Tuple<
		typename RakingGridTuple::T0::RakingSegment,
		typename RakingGridTuple::T1::RakingSegment> RakingSoa;

	typedef NullType SecondaryRakingSoaDetails;

	/**
	 * Warpscan storages
	 */
	WarpscanSoa warpscan_partials;

	/**
	 * Lane insertion/extraction pointers.
	 */
	LaneSoa lane_partials;

	/**
	 * Raking pointers
	 */
	RakingSoa raking_segments;


	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ RakingSoaDetails(
		GridStorageSoa smem_pools,
		WarpscanSoa warpscan_partials) :

			warpscan_partials(warpscan_partials),
			lane_partials(												// set lane partial pointer
				RakingGridTuple::T0::MyLanePartial(smem_pools.t0),
				RakingGridTuple::T1::MyLanePartial(smem_pools.t1))
	{
		if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {

			// Set raking segment pointers
			raking_segments = RakingSoa(
				RakingGridTuple::T0::MyRakingSegment(smem_pools.t0),
				RakingGridTuple::T1::MyRakingSegment(smem_pools.t1));
		}
	}


	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ RakingSoaDetails(
		GridStorageSoa smem_pools,
		WarpscanSoa warpscan_partials,
		TileTuple soa_tuple_identity) :

			warpscan_partials(warpscan_partials),
			lane_partials(												// set lane partial pointer
				RakingGridTuple::T0::MyLanePartial(smem_pools.t0),
				RakingGridTuple::T1::MyLanePartial(smem_pools.t1))
	{
		if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {

			// Set raking segment pointers
			raking_segments = RakingSoa(
				RakingGridTuple::T0::MyRakingSegment(smem_pools.t0),
				RakingGridTuple::T1::MyRakingSegment(smem_pools.t1));

			// Initialize first half of warpscan storages to identity
			warpscan_partials.Set(soa_tuple_identity, 0, threadIdx.x);
		}
	}


	/**
	 * Return the cumulative partial left in the final warpscan cell
	 */
	__device__ __forceinline__ TileTuple CumulativePartial()
	{
		TileTuple retval;
		warpscan_partials.Get(retval, 1, CUMULATIVE_THREAD);
		return retval;
	}
};



/**
 * Two-field raking details
 */
template <
	typename _TileTuple,
	typename RakingGridTuple,
	typename SecondaryRakingGridTuple>
struct RakingSoaDetails<
	_TileTuple,
	RakingGridTuple,
	2,
	SecondaryRakingGridTuple> : RakingGridTuple::T0
{
	enum {
		CUMULATIVE_THREAD 	= RakingSoaDetails::RAKING_THREADS - 1,
		WARP_THREADS 		= B40C_WARP_THREADS(RakingSoaDetails::CUDA_ARCH)
	};

	// Simple SOA tuple "slice" type
	typedef _TileTuple TileTuple;

	// SOA type of raking lanes
	typedef Tuple<
		typename TileTuple::T0*,
		typename TileTuple::T1*> GridStorageSoa;

	// SOA type of warpscan storage
	typedef Tuple<
		typename RakingGridTuple::T0::WarpscanT (*)[WARP_THREADS],
		typename RakingGridTuple::T1::WarpscanT (*)[WARP_THREADS]> WarpscanSoa;

	// SOA type of partial-insertion pointers
	typedef Tuple<
		typename RakingGridTuple::T0::LanePartial,
		typename RakingGridTuple::T1::LanePartial> LaneSoa;

	// SOA type of raking segments
	typedef Tuple<
		typename RakingGridTuple::T0::RakingSegment,
		typename RakingGridTuple::T1::RakingSegment> RakingSoa;

	// SOA type of secondary details
	typedef RakingSoaDetails<TileTuple, SecondaryRakingGridTuple> SecondaryRakingSoaDetails;

	/**
	 * Lane insertion/extraction pointers.
	 */
	LaneSoa lane_partials;

	/**
	 * Raking pointers
	 */
	RakingSoa raking_segments;

	/**
	 * Secondary-level grid details
	 */
	SecondaryRakingSoaDetails secondary_details;


	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ RakingSoaDetails(
		GridStorageSoa smem_pools,
		WarpscanSoa warpscan_partials) :

			lane_partials(												// set lane partial pointer
				RakingGridTuple::T0::MyLanePartial(smem_pools.t0),
				RakingGridTuple::T1::MyLanePartial(smem_pools.t1)),
			secondary_details(
				GridStorageSoa(
					smem_pools.t0 + RakingGridTuple::T0::RAKING_ELEMENTS,
					smem_pools.t1 + RakingGridTuple::T1::RAKING_ELEMENTS),
				warpscan_partials)
	{
		if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {

			// Set raking segment pointers
			raking_segments = RakingSoa(
				RakingGridTuple::T0::MyRakingSegment(smem_pools.t0),
				RakingGridTuple::T1::MyRakingSegment(smem_pools.t1));
		}
	}


	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ RakingSoaDetails(
		GridStorageSoa smem_pools,
		WarpscanSoa warpscan_partials,
		TileTuple soa_tuple_identity) :

			lane_partials(												// set lane partial pointer
				RakingGridTuple::T0::MyLanePartial(smem_pools.t0),
				RakingGridTuple::T1::MyLanePartial(smem_pools.t1)),
			secondary_details(
				GridStorageSoa(
					smem_pools.t0 + RakingGridTuple::T0::RAKING_ELEMENTS,
					smem_pools.t1 + RakingGridTuple::T1::RAKING_ELEMENTS),
				warpscan_partials)
	{
		if (threadIdx.x < RakingSoaDetails::RAKING_THREADS) {

			// Set raking segment pointers
			raking_segments = RakingSoa(
				RakingGridTuple::T0::MyRakingSegment(smem_pools.t0),
				RakingGridTuple::T1::MyRakingSegment(smem_pools.t1));
		}
	}


	/**
	 * Return the cumulative partial left in the final warpscan cell
	 */
	__device__ __forceinline__ TileTuple CumulativePartial()
	{
		return secondary_details.CumulativePartial();
	}
};







} // namespace util
} // namespace b40c

