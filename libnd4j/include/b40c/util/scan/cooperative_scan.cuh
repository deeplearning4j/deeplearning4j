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
 * Cooperative tile reduction and scanning within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/cooperative_reduction.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>

namespace b40c {
namespace util {
namespace scan {

/**
 * Cooperative reduction in raking smem grid hierarchies
 */
template <
	typename RakingDetails,
	typename SecondaryRakingDetails = typename RakingDetails::SecondaryRakingDetails>
struct CooperativeGridScan;



/**
 * Cooperative tile scan
 */
template <
	int VEC_SIZE,							// Length of vector-loads (e.g, vec-1, vec-2, vec-4)
	bool EXCLUSIVE = true>					// Whether or not this is an exclusive scan
struct CooperativeTileScan
{
	//---------------------------------------------------------------------
	// Iteration structures for extracting partials from raking lanes and
	// using them to seed scans of tile vectors
	//---------------------------------------------------------------------

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ScanLane
	{
		template <typename RakingDetails, typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			RakingDetails raking_details,
			typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
			ReductionOp scan_op)
		{
			// Retrieve partial reduction from raking grid
			typename RakingDetails::T exclusive_partial = raking_details.lane_partial[LANE][0];

			// Scan the partials in this lane/load
			SerialScan<VEC_SIZE, EXCLUSIVE>::Invoke(
				data[LANE], exclusive_partial, scan_op);

			// Next load
			ScanLane<LANE + 1, TOTAL_LANES>::Invoke(
				raking_details, data, scan_op);
		}
	};

	// Terminate
	template <int TOTAL_LANES>
	struct ScanLane<TOTAL_LANES, TOTAL_LANES>
	{
		template <typename RakingDetails, typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			RakingDetails raking_details,
			typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
			ReductionOp scan_op) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Scan a single tile.  Total aggregate is computed and returned in all threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename RakingDetails, typename ReductionOp>
	static __device__ __forceinline__ typename RakingDetails::T ScanTile(
		RakingDetails raking_details,
		typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
		ReductionOp scan_op)
	{
		// Reduce partials in each vector-load, placing resulting partial in raking smem grid lanes (one lane per load)
		reduction::CooperativeTileReduction<VEC_SIZE>::template ReduceLane<0, RakingDetails::SCAN_LANES>::Invoke(
			raking_details, data, scan_op);

		__syncthreads();

		CooperativeGridScan<RakingDetails>::ScanTile(
			raking_details, scan_op);

		__syncthreads();

		// Scan each vector-load, seeded with the resulting partial from its raking grid lane,
		ScanLane<0, RakingDetails::SCAN_LANES>::Invoke(
			raking_details, data, scan_op);

		// Return last thread's inclusive partial
		return raking_details.CumulativePartial();
	}

	/**
	 * Scan a single tile where carry is updated with the total aggregate only
	 * in raking threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename RakingDetails, typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithCarry(
		RakingDetails raking_details,
		typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
		typename RakingDetails::T &carry,
		ReductionOp scan_op)
	{
		// Reduce partials in each vector-load, placing resulting partials in raking smem grid lanes (one lane per load)
		reduction::CooperativeTileReduction<VEC_SIZE>::template ReduceLane<0, RakingDetails::SCAN_LANES>::Invoke(
			raking_details, data, scan_op);

		__syncthreads();

		CooperativeGridScan<RakingDetails>::ScanTileWithCarry(
			raking_details, carry, scan_op);

		__syncthreads();

		// Scan each vector-load, seeded with the resulting partial from its raking grid lane,
		ScanLane<0, RakingDetails::SCAN_LANES>::Invoke(
			raking_details, data, scan_op);
	}


	/**
	 * Scan a single tile with atomic enqueue.  Returns updated queue offset.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename RakingDetails, typename ReductionOp>
	static __device__ __forceinline__ typename RakingDetails::T ScanTileWithEnqueue(
		RakingDetails raking_details,
		typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
		typename RakingDetails::T* d_enqueue_counter,
		ReductionOp scan_op)
	{
		// Reduce partials in each vector-load, placing resulting partial in raking smem grid lanes (one lane per load)
		reduction::CooperativeTileReduction<VEC_SIZE>::template ReduceLane<0, RakingDetails::SCAN_LANES>::Invoke(
			raking_details, data, scan_op);

		__syncthreads();

		CooperativeGridScan<RakingDetails>::ScanTileWithEnqueue(
			raking_details, d_enqueue_counter, scan_op);

		__syncthreads();

		// Scan each vector-load, seeded with the resulting partial from its raking grid lane,
		ScanLane<0, RakingDetails::SCAN_LANES>::Invoke(
			raking_details, data, scan_op);

		return scan_op(raking_details.QueueReservation(), raking_details.CumulativePartial());
	}


	/**
	 * Scan a single tile with atomic enqueue.  Local aggregate is computed and
	 * returned in all threads.  Enqueue offset is returned in all threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename RakingDetails, typename ReductionOp>
	static __device__ __forceinline__ typename RakingDetails::T ScanTileWithEnqueue(
		RakingDetails raking_details,
		typename RakingDetails::T data[RakingDetails::SCAN_LANES][VEC_SIZE],
		typename RakingDetails::T *d_enqueue_counter,
		typename RakingDetails::T &enqueue_offset,
		ReductionOp scan_op)
	{
		// Reduce partials in each vector-load, placing resulting partial in raking smem grid lanes (one lane per load)
		reduction::CooperativeTileReduction<VEC_SIZE>::template ReduceLane<0, RakingDetails::SCAN_LANES>::Invoke(
			raking_details, data, scan_op);

		__syncthreads();

		CooperativeGridScan<RakingDetails>::ScanTileWithEnqueue(
			raking_details, d_enqueue_counter, enqueue_offset, scan_op);

		__syncthreads();

		// Scan each vector-load, seeded with the resulting partial from its raking grid lane,
		ScanLane<0, RakingDetails::SCAN_LANES>::Invoke(
			raking_details, data, scan_op);

		// Return last thread's inclusive partial
		return raking_details.CumulativePartial();
	}
};




/******************************************************************************
 * CooperativeGridScan
 ******************************************************************************/

/**
 * Cooperative raking grid reduction (specialized for last-level of raking grid)
 */
template <typename RakingDetails>
struct CooperativeGridScan<RakingDetails, NullType>
{
	typedef typename RakingDetails::T T;

	/**
	 * Scan in last-level raking grid.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTile(
		RakingDetails raking_details,
		ReductionOp scan_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, scan_op);

			// Exclusive warp scan
			T exclusive_partial = WarpScan<RakingDetails::LOG_RAKING_THREADS>::Invoke(
				inclusive_partial, raking_details.warpscan, scan_op);

			// Exclusive raking scan
			SerialScan<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, exclusive_partial, scan_op);

		}
	}


	/**
	 * Scan in last-level raking grid.  Carry-in/out is updated only in raking threads
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithCarry(
		RakingDetails raking_details,
		T &carry,
		ReductionOp scan_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<RakingDetails::LOG_RAKING_THREADS>::Invoke(
				inclusive_partial, warpscan_total, raking_details.warpscan, scan_op);

			// Seed exclusive partial with carry-in
			exclusive_partial = scan_op(carry, exclusive_partial);

			// Exclusive raking scan
			SerialScan<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, exclusive_partial, scan_op);

			// Update carry
			carry = scan_op(carry, warpscan_total);			// Increment the CTA's running total by the full tile reduction
		}
	}


	/**
	 * Scan in last-level raking grid with atomic enqueue
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithEnqueue(
		RakingDetails raking_details,
		T *d_enqueue_counter,
		ReductionOp scan_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<RakingDetails::LOG_RAKING_THREADS>::Invoke(
					inclusive_partial, warpscan_total, raking_details.warpscan, scan_op);

			// Atomic-increment the global counter with the total allocation
			T reservation_offset;
			if (threadIdx.x == 0) {
				reservation_offset = util::AtomicInt<T>::Add(
					d_enqueue_counter,
					warpscan_total);
				raking_details.warpscan[1][0] = reservation_offset;
			}

			// Seed exclusive partial with queue reservation offset
			reservation_offset = raking_details.warpscan[1][0];
			exclusive_partial = scan_op(reservation_offset, exclusive_partial);

			// Exclusive raking scan
			SerialScan<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, exclusive_partial, scan_op);
		}
	}

	/**
	 * Scan in last-level raking grid with atomic enqueue
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithEnqueue(
		RakingDetails raking_details,
		T *d_enqueue_counter,
		T &enqueue_offset,
		ReductionOp scan_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<RakingDetails::LOG_RAKING_THREADS>::Invoke(
				inclusive_partial, warpscan_total, raking_details.warpscan, scan_op);

			// Atomic-increment the global counter with the total allocation
			if (threadIdx.x == 0) {
				enqueue_offset = util::AtomicInt<T>::Add(
					d_enqueue_counter,
					warpscan_total);
			}

			// Exclusive raking scan
			SerialScan<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, exclusive_partial, scan_op);
		}
	}
};


/**
 * Cooperative raking grid reduction for multi-level raking grids
 */
template <
	typename RakingDetails,
	typename SecondaryRakingDetails>
struct CooperativeGridScan
{
	typedef typename RakingDetails::T T;

	/**
	 * Scan in raking grid.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTile(
		RakingDetails raking_details,
		ReductionOp scan_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = reduction::SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, scan_op);

			// Place partial in next grid
			raking_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeGridScan<SecondaryRakingDetails>::ScanTile(
			raking_details.secondary_details, scan_op);

		__syncthreads();

		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			T exclusive_partial = raking_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialScan<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, exclusive_partial, scan_op);
		}
	}

	/**
	 * Scan in raking grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithCarry(
		RakingDetails raking_details,
		T &carry,
		ReductionOp scan_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = reduction::SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, scan_op);

			// Place partial in next grid
			raking_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeGridScan<SecondaryRakingDetails>::ScanTileWithCarry(
			raking_details.secondary_details, carry, scan_op);

		__syncthreads();

		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			T exclusive_partial = raking_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialScan<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, exclusive_partial, scan_op);
		}
	}

	/**
	 * Scan in raking grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithEnqueue(
		RakingDetails raking_details,
		T* d_enqueue_counter,
		ReductionOp scan_op)
	{
		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = reduction::SerialReduce<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, scan_op);

			// Place partial in next grid
			raking_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeGridScan<SecondaryRakingDetails>::ScanTileWithEnqueue(
			raking_details.secondary_details, d_enqueue_counter, scan_op);

		__syncthreads();

		if (threadIdx.x < RakingDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			T exclusive_partial = raking_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialScan<RakingDetails::PARTIALS_PER_SEG>::Invoke(
				raking_details.raking_segment, exclusive_partial, scan_op);
		}
	}
};



} // namespace scan
} // namespace util
} // namespace b40c

