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
 * Abstract tile-processing functionality for partitioning upsweep reduction
 * kernels
 ******************************************************************************/

#pragma once

namespace b40c {
namespace partition {
namespace upsweep {


/**
 * Tile
 *
 * Abstract class
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	typename KernelPolicy,
	typename DerivedTile>
struct Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 			KeyType;
	typedef typename KernelPolicy::SizeT			SizeT;
	typedef DerivedTile 							Dispatch;

	enum {
		LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Dequeued vertex ids
	KeyType 	keys[LOADS_PER_TILE][LOAD_VEC_SIZE];


	//---------------------------------------------------------------------
	// Abstract Interface
	//---------------------------------------------------------------------

	/**
	 * Returns the bin into which the specified key is to be placed
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ int DecodeBin(KeyType key, Cta *cta);


	/**
	 * Returns whether or not the key is valid.
	 *
	 * To be overloaded.
	 */
	template <int LOAD, int VEC>
	__device__ __forceinline__ bool IsValid();


	/**
	 * Loads keys into the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(Cta *cta, SizeT cta_offset);


	/**
	 * Stores keys from the tile (if necessary)
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void StoreKeys(Cta *cta, SizeT cta_offset);


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int LOAD, int VEC, int dummy = 0>
	struct Iterate
	{
		// Bucket
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Bucket(Cta *cta, Tile *tile)
		{
			if (tile->template IsValid<LOAD, VEC>()) {

				const KeyType COUNTER_BYTE_MASK = (KernelPolicy::LOG_BINS < 2) ? 0x1 : 0x3;

				// Decode the bin for this key
				int bin = tile->DecodeBin(tile->keys[LOAD][VEC], cta);

				// Decode composite-counter lane and sub-counter from bin
				int lane = bin >> 2;										// extract composite counter lane
				int sub_counter = bin & COUNTER_BYTE_MASK;					// extract 8-bit counter offset

				if (__B40C_CUDA_ARCH__ >= 200) {

					// Increment sub-field in composite counter
					cta->smem_storage.composite_counters.counters[lane][threadIdx.x][sub_counter]++;

				} else {

					// Increment sub-field in composite counter
					cta->smem_storage.composite_counters.words[lane][threadIdx.x] += (1 << (sub_counter << 0x3));
				}
			}

			Iterate<LOAD, VEC + 1>::Bucket(cta, tile);
		}
	};


	/**
	 * Iterate next load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// Bucket
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Bucket(Cta *cta, Tile *tile)
		{
			Iterate<LOAD + 1, 0>::Bucket(cta, tile);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// Bucket
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Bucket(Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Decode keys in this tile and updates the cta's corresponding composite counters
	 */
	template <typename Cta>
	__device__ __forceinline__ void Bucket(Cta *cta)
	{
		Iterate<0, 0>::Bucket(cta, (Dispatch *) this);
	}
};



} // namespace upsweep
} // namespace partition
} // namespace b40c
