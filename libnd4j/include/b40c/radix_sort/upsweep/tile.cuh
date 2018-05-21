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
 * Tile-processing functionality for radix sort upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/partition/upsweep/tile.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {


/**
 * Tile
 *
 * Derives from partition::upsweep::Tile
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	typename KernelPolicy>
struct Tile :
	partition::upsweep::Tile<
		LOG_LOADS_PER_TILE,
		LOG_LOAD_VEC_SIZE,
		KernelPolicy,
		Tile<LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, KernelPolicy> >					// This class
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 		KeyType;
	typedef typename KernelPolicy::SizeT 		SizeT;


	//---------------------------------------------------------------------
	// Derived Interface
	//---------------------------------------------------------------------

	/**
	 * Returns the bin into which the specified key is to be placed
	 */
	template <typename Cta>
	__device__ __forceinline__ int DecodeBin(KeyType key, Cta *cta)
	{
		int bin;
		ExtractKeyBits<
			KeyType,
			KernelPolicy::CURRENT_BIT,
			KernelPolicy::LOG_BINS>::Extract(bin, key);
		return bin;
	}


	/**
	 * Returns whether or not the key is valid.
	 *
	 * Can be overloaded.
	 */
	template <int LOAD, int VEC>
	__device__ __forceinline__ bool IsValid()
	{
		return true;
	}


	/**
	 * Loads keys into the tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset)
	{
		// Read tile of keys
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::template LoadValid<
				KeyType,
				KernelPolicy::PreprocessTraits::Preprocess>(
					(KeyType (*)[Tile::LOAD_VEC_SIZE]) this->keys,
					cta->d_in_keys,
					cta_offset);
	}

	/**
	 * Stores keys from the tile (not necessary)
	 */
	template <typename Cta>
	__device__ __forceinline__ void StoreKeys(
		Cta *cta,
		SizeT cta_offset) {}
};



} // namespace upsweep
} // namespace radix_sort
} // namespace b40c
