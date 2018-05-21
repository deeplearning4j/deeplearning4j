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
 * Raking grid abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace util {


/**
 * Raking grid abstraction.
 *
 * A "raking lane" is a region of shared memory into which a group of N active
 * threads (e.g., a CTA) can place N data items (e.g., partial reductions) in
 * thread-rank-order.  The lane logically partitions these shared partials
 * into S contiguous segments of N/S elements each.   These segments can be
 * mapped to a corresponding a set of S "raking threads" in thread-rank-order.
 *
 * The lane is arranged with regular padding cells so that raking threads will
 * not incur bank conflicts when accessing identical segment offsets.  E.g.,
 * raking threads are conflict-free when linearly sweeping through their segments.
 *
 * A "raking grid" is a set of L raking lanes and R raking threads (R >= L).
 * Each lane comprises N data items (one item for each of the N active threads)
 * and is alloted S = R/L raking threads.
 */
template <
	int _CUDA_ARCH,					// CUDA SM architecture to generate code for
	typename _T,					// Type of items we will be raking
	int _LOG_ACTIVE_THREADS, 		// Number of threads placing a lane partial (i.e., the number of partials per lane) (log)
	int _LOG_RAKING_THREADS, 		// Number of threads used for raking (typically 1 warp) (log)
	int _LOG_LANES = 0>				// Number of raking lanes, default = 1 lane (log)
struct RakingGrid
{

	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	// Type of items to be placed in the grid
	typedef _T T;
	
	/**
	 * Enumerated constants
	 *
	 * A "row" is a continuous sequence of raking segments without padding cells
	 */
	enum {
		CUDA_ARCH						= _CUDA_ARCH,

		// Number of scan lanes
		LOG_LANES						= _LOG_LANES,
		LANES							= 1 <<LOG_LANES,

		// Number number of partials per lane
		LOG_PARTIALS_PER_LANE 			= _LOG_ACTIVE_THREADS,
		PARTIALS_PER_LANE				= 1 << LOG_PARTIALS_PER_LANE,

		// Number of raking threads
		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		// Number of raking threads per lane
		LOG_RAKING_THREADS_PER_LANE		= LOG_RAKING_THREADS - LOG_LANES,
		RAKING_THREADS_PER_LANE			= 1 << LOG_RAKING_THREADS_PER_LANE,

		// Partials to be raked per raking thread
		LOG_PARTIALS_PER_SEG 			= LOG_PARTIALS_PER_LANE - LOG_RAKING_THREADS_PER_LANE,
		PARTIALS_PER_SEG 				= 1 << LOG_PARTIALS_PER_SEG,

		// Number of partials that we can put in one stripe across the shared memory banks
		LOG_PARTIALS_PER_BANK_ARRAY		= B40C_LOG_MEM_BANKS(CUDA_ARCH) +
											B40C_LOG_BANK_STRIDE_BYTES(CUDA_ARCH) -
											Log2<sizeof(T)>::VALUE,
		PARTIALS_PER_BANK_ARRAY			= 1 << LOG_PARTIALS_PER_BANK_ARRAY,

		LOG_SEGS_PER_BANK_ARRAY 		= B40C_MAX(0, LOG_PARTIALS_PER_BANK_ARRAY - LOG_PARTIALS_PER_SEG),
		SEGS_PER_BANK_ARRAY				= 1 << LOG_SEGS_PER_BANK_ARRAY,

		// Whether or not one warp of raking threads can rake entirely in one stripe across the shared memory banks
		NO_PADDING = (LOG_SEGS_PER_BANK_ARRAY >= B40C_LOG_WARP_THREADS(CUDA_ARCH)),

		// Number of raking segments we can have without padding (i.e., a "row")
		LOG_SEGS_PER_ROW 				= (NO_PADDING) ?
											LOG_RAKING_THREADS :												// All raking threads (segments)
											B40C_MIN(LOG_RAKING_THREADS_PER_LANE, LOG_SEGS_PER_BANK_ARRAY),		// Up to as many segments per lane (all lanes must have same amount of padding to have constant lane stride)
		SEGS_PER_ROW					= 1 << LOG_SEGS_PER_ROW,

		// Number of partials per row
		LOG_PARTIALS_PER_ROW			= LOG_SEGS_PER_ROW + LOG_PARTIALS_PER_SEG,
		PARTIALS_PER_ROW				= 1 << LOG_PARTIALS_PER_ROW,

		// Number of partials that we must use to "pad out" one memory bank
		LOG_BANK_PADDING_PARTIALS		= B40C_MAX(0, B40C_LOG_BANK_STRIDE_BYTES(CUDA_ARCH) - Log2<sizeof(T)>::VALUE),
		BANK_PADDING_PARTIALS			= 1 << LOG_BANK_PADDING_PARTIALS,

		// Number of partials that we must use to "pad out" a lane to one memory bank
		LANE_PADDING_PARTIALS			= B40C_MAX(0, PARTIALS_PER_BANK_ARRAY - PARTIALS_PER_LANE),

		// Number of partials (including padding) per "row"
		PADDED_PARTIALS_PER_ROW			= (NO_PADDING) ?
											PARTIALS_PER_ROW :
											PARTIALS_PER_ROW + LANE_PADDING_PARTIALS + BANK_PADDING_PARTIALS,

		// Number of rows in the grid
		LOG_ROWS						= LOG_RAKING_THREADS - LOG_SEGS_PER_ROW,
		ROWS 							= 1 << LOG_ROWS,

		// Number of rows per lane (always at least one)
		LOG_ROWS_PER_LANE				= B40C_MAX(0, LOG_RAKING_THREADS_PER_LANE - LOG_SEGS_PER_ROW),
		ROWS_PER_LANE					= 1 << LOG_ROWS_PER_LANE,

		// Padded stride between lanes (in partials)
		LANE_STRIDE						= (NO_PADDING) ?
											PARTIALS_PER_LANE :
											ROWS_PER_LANE * PADDED_PARTIALS_PER_ROW,

		// Number of elements needed to back this level of the raking grid
		RAKING_ELEMENTS					= ROWS * PADDED_PARTIALS_PER_ROW,
	};

	// Type of pointer for inserting partials into lanes, e.g., lane_partial[LANE][0] = ...
	typedef T (*LanePartial)[LANE_STRIDE];


	// Type of pointer for raking across lane segments
	typedef T* RakingSegment;


	//---------------------------------------------------------------------
	// Shared storage types
	//---------------------------------------------------------------------

	// Buffer type for lane storage
	typedef T LaneStorage[RAKING_ELEMENTS];


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	/**
	 * Shared lane storage
	 */
	LaneStorage &lane_storage;

	/**
	 * The location in the smem grid where the calling thread can insert/extract
	 * its partial for raking reduction/scan into the first lane.
	 */
	typename RakingGrid::LanePartial my_lane_partial;

	/**
	 * The location in the smem grid where the calling thread can begin serial
	 * raking/scanning (if it is a raking thread)
	 */
	typename RakingGrid::RakingSegment my_raking_segment;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Initialize lane_partial
	 */
	__device__ __forceinline__  void InitLanePartial(int tid = threadIdx.x)
	{
		lane_partial = (LanePartial) (
			lane_storage + 													// base
			tid + 															// logical thread offset
			((tid >> LOG_PARTIALS_PER_ROW) * BANK_PADDING_PARTIALS));		// padding
	}


	/**
	 * Initialize raking_segment
	 */
	__device__ __forceinline__  void InitRakingSegment(int tid = threadIdx.x)
	{
		raking_segment = (RakingSegment) (
			lane_storage +													// base
			(tid << LOG_PARTIALS_PER_SEG) +									// logical segment offset
			((threadIdx.x >> LOG_SEGS_PER_ROW) * BANK_PADDING_PARTIALS));	// padding
	}


	/**
	 * Constructor
	 */
	__device__ __forceinline__ RakingGrid(LaneStorage &lane_storage) :
		lane_storage(lane_storage)
	{
		InitLanePartial();
		if (threadIdx.x < RakingGrid::RAKING_THREADS) {
			InitRakingSegment();
		}
	}


	/**
	 * Displays configuration to standard out
	 */
	static __host__ __device__ __forceinline__ void Print()
	{
		printf("LANES: %d\n"
				"PARTIALS_PER_LANE: %d\n"
				"RAKING_THREADS: %d\n"
				"RAKING_THREADS_PER_LANE: %d\n"
				"PARTIALS_PER_SEG: %d\n"
				"PARTIALS_PER_BANK_ARRAY: %d\n"
				"SEGS_PER_BANK_ARRAY: %d\n"
				"NO_PADDING: %d\n"
				"SEGS_PER_ROW: %d\n"
				"PARTIALS_PER_ROW: %d\n"
				"BANK_PADDING_PARTIALS: %d\n"
				"LANE_PADDING_PARTIALS: %d\n"
				"PADDED_PARTIALS_PER_ROW: %d\n"
				"ROWS: %d\n"
				"ROWS_PER_LANE: %d\n"
				"LANE_STRIDE: %d\n"
				"RAKING_ELEMENTS: %d\n",
			LANES,
			PARTIALS_PER_LANE,
			RAKING_THREADS,
			RAKING_THREADS_PER_LANE,
			PARTIALS_PER_SEG,
			PARTIALS_PER_BANK_ARRAY,
			SEGS_PER_BANK_ARRAY,
			NO_PADDING,
			SEGS_PER_ROW,
			PARTIALS_PER_ROW,
			BANK_PADDING_PARTIALS,
			LANE_PADDING_PARTIALS,
			PADDED_PARTIALS_PER_ROW,
			ROWS,
			ROWS_PER_LANE,
			LANE_STRIDE,
			RAKING_ELEMENTS);
	}
};



} // namespace util
} // namespace b40c

