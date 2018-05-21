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
 * Abstract CTA-processing functionality for partitioning upsweep
 * reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>

#include <b40c/partition/upsweep/aggregate_counters.cuh>
#include <b40c/partition/upsweep/composite_counters.cuh>
#include <b40c/partition/upsweep/tile.cuh>

#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace partition {
namespace upsweep {



/**
 * Partitioning upsweep reduction CTA
 *
 * Abstract class
 */
template <
	typename KernelPolicy,
	typename DerivedCta,						// Derived CTA class
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE,
		typename Policy> class Tile>			// Derived Tile class to use
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;
	typedef DerivedCta 										Dispatch;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	typename KernelPolicy::SmemStorage 	&smem_storage;

	// Shared-memory lanes of composite-counters
	CompostiteCounters<KernelPolicy> 	composite_counters;

	// Thread-local counters for periodically aggregating composite-counter lanes
	AggregateCounters<KernelPolicy>		aggregate_counters;

	// Input and output device pointers
	KeyType								*d_in_keys;
	SizeT								*d_spine;

	int 								warp_id;
	int 								warp_idx;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Unrolled tile processing
	 */
	struct UnrollTiles
	{
		// Recurse over counts
		template <int UNROLL_COUNT, int __dummy = 0>
		struct Iterate
		{
			static const int HALF = UNROLL_COUNT / 2;

			template <typename Cta>
			static __device__ __forceinline__ void ProcessTiles(
				Cta *cta, SizeT cta_offset)
			{
				Iterate<HALF>::ProcessTiles(cta, cta_offset);
				Iterate<HALF>::ProcessTiles(cta, cta_offset + (KernelPolicy::TILE_ELEMENTS * HALF));
			}
		};

		// Terminate (process one tile)
		template <int __dummy>
		struct Iterate<1, __dummy>
		{
			template <typename Cta>
			static __device__ __forceinline__ void ProcessTiles(
				Cta *cta, SizeT cta_offset)
			{
				cta->ProcessFullTile(cta_offset);
			}
		};
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_spine(d_spine),
			warp_id(threadIdx.x >> B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__)),
			warp_idx(util::LaneId())
	{}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
	{
		Dispatch *dispatch = (Dispatch*) this;

		Tile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy> tile;

		// Load keys
		tile.LoadKeys(dispatch, cta_offset);

		// Prevent bucketing from being hoisted (otherwise we don't get the desired outstanding loads)
		if (KernelPolicy::LOADS_PER_TILE > 1) __syncthreads();

		// Bucket tile of keys
		tile.Bucket(dispatch);

		// Store keys (if necessary)
		tile.StoreKeys(dispatch, cta_offset);
	}


	/**
	 * Processes a single load (may have some threads masked off)
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		const SizeT &out_of_bounds)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Process partial tile if necessary using single loads
		while (cta_offset + threadIdx.x < out_of_bounds) {

			Tile<0, 0, KernelPolicy> tile;

			// Load keys
			tile.LoadKeys(dispatch, cta_offset);

			// Bucket tile of keys
			tile.Bucket(dispatch);

			// Store keys (if necessary)
			tile.StoreKeys(dispatch, cta_offset);

			cta_offset += KernelPolicy::THREADS;
		}
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT cta_offset = work_limits.offset;

		aggregate_counters.ResetCounters(dispatch);
		composite_counters.ResetCompositeCounters(dispatch);


#if 1	// Use deep unrolling for better instruction efficiency

		// Unroll batches of full tiles
		const int UNROLLED_ELEMENTS = KernelPolicy::UNROLL_COUNT * KernelPolicy::TILE_ELEMENTS;
		while (cta_offset  + UNROLLED_ELEMENTS < work_limits.out_of_bounds) {

			UnrollTiles::template Iterate<KernelPolicy::UNROLL_COUNT>::ProcessTiles(
				dispatch,
				cta_offset);
			cta_offset += UNROLLED_ELEMENTS;

			__syncthreads();

			// Aggregate back into local_count registers to prevent overflow
			aggregate_counters.ExtractComposites(dispatch);

			__syncthreads();

			// Reset composite counters in lanes
			composite_counters.ResetCompositeCounters(dispatch);
		}

		// Unroll single full tiles
		while (cta_offset + KernelPolicy::TILE_ELEMENTS < work_limits.out_of_bounds) {

			UnrollTiles::template Iterate<1>::ProcessTiles(
				dispatch,
				cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;
		}

#else 	// Use shallow unrolling for faster compilation tiles

		// Unroll single full tiles
		while (cta_offset < work_limits.guarded_offset) {

			ProcessFullTile(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;

			const SizeT UNROLL_MASK = (KernelPolicy::UNROLL_COUNT - 1) << KernelPolicy::LOG_TILE_ELEMENTS;
			if ((cta_offset & UNROLL_MASK) == 0) {

				__syncthreads();

				// Aggregate back into local_count registers to prevent overflow
				aggregate_counters.ExtractComposites(dispatch);

				__syncthreads();

				// Reset composite counters in lanes
				composite_counters.ResetCompositeCounters(dispatch);
			}
		}
#endif

		// Process partial tile if necessary
		ProcessPartialTile(cta_offset, work_limits.out_of_bounds);

		__syncthreads();

		// Aggregate back into local_count registers
		aggregate_counters.ExtractComposites(dispatch);

		__syncthreads();

		//Final raking reduction of counts by bin, output to spine.

		aggregate_counters.ShareCounters(dispatch);

		__syncthreads();

		// Rake-reduce and write out the bin_count reductions
		if (threadIdx.x < KernelPolicy::BINS) {

			SizeT bin_count = util::reduction::SerialReduce<KernelPolicy::AGGREGATED_PARTIALS_PER_ROW>::Invoke(
				smem_storage.aggregate[threadIdx.x]);

			int spine_bin_offset = util::FastMul(gridDim.x, threadIdx.x) + blockIdx.x;

			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					bin_count, d_spine + spine_bin_offset);
		}
	}
};



} // namespace upsweep
} // namespace partition
} // namespace b40c

