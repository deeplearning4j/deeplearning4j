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
 * Kernel utilities for two-phase tile scattering
 ******************************************************************************/

#pragma once

#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace util {
namespace io {


/**
 * Performs a two-phase tile scattering in order to improve global-store write
 * coalescing: first to smem, then to global.
 *
 * Does not barrier after usage: a subsequent sync is needed to make shared memory
 * coherent for re-use
 */
template <
	int LOG_LOADS_PER_TILE, 									// Number of vector loads (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector load (log)
	int ACTIVE_THREADS,
	st::CacheModifier CACHE_MODIFIER,
	bool CHECK_ALIGNMENT>
struct TwoPhaseScatterTile
{
	enum {
		LOG_ELEMENTS_PER_THREAD		= LOG_LOADS_PER_TILE + LOG_LOAD_VEC_SIZE,
		ELEMENTS_PER_THREAD			= 1 << LOG_ELEMENTS_PER_THREAD,
		LOADS_PER_TILE 				= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 				= 1 << LOG_LOAD_VEC_SIZE,
		TILE_SIZE					= ELEMENTS_PER_THREAD * ACTIVE_THREADS,
	};

	template <
		typename T,
		typename Flag,
		typename Rank,
		typename SizeT>
	__device__ __forceinline__ void Scatter(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE],								// Elements of data to scatter
		Flag flags[LOADS_PER_TILE][LOAD_VEC_SIZE],							// Valid predicates for data elements
		Rank ranks[LOADS_PER_TILE][LOAD_VEC_SIZE],							// Local ranks of data to scatter
		Rank valid_elements,												// Number of valid elements
		T smem_exchange[TILE_SIZE],											// Smem swap exchange storage
		T *d_out,															// Global output to scatter to
		SizeT cta_offset)													// CTA offset into d_out at which to scatter to
	{
		// Scatter valid data into smem exchange, predicated on head_flags
		ScatterTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			ACTIVE_THREADS,
			st::NONE>::Scatter(
				smem_exchange,
				data,
				flags,
				ranks);

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Tile of compacted elements
		T compacted_data[ELEMENTS_PER_THREAD][1];

		// Gather compacted data from smem exchange (in 1-element stride loads)
		LoadTile<
			LOG_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			ACTIVE_THREADS,
			ld::NONE,
			false>::LoadValid(							// No need to check alignment
				compacted_data,
				smem_exchange,
				0,
				valid_elements);

		// Scatter compacted data to global output
		util::io::StoreTile<
			LOG_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			ACTIVE_THREADS,
			CACHE_MODIFIER,
			CHECK_ALIGNMENT>::Store(
				compacted_data,
				d_out,
				cta_offset,
				valid_elements);
	}
};



} // namespace io
} // namespace util
} // namespace b40c

