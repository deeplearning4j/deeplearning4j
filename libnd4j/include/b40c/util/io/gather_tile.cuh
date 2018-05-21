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
 * Kernel utilities for gathering data
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace io {




/**
 * Gather a tile of data items using the corresponding tile of gather_offsets
 *
 * Uses vec-1 loads.
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	int ACTIVE_THREADS,								// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER>				// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
struct GatherTile
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// Iterate next vec
	template <int LOAD, int VEC, int dummy = 0>
	struct Iterate
	{
		// predicated on valid
		template <typename T, void Transform(T&, bool), typename Flag>
		static __device__ __forceinline__ void Invoke(
			T *src,
			T dest[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE])
		{
			T *d_src = src + (LOAD * LOAD_VEC_SIZE * ACTIVE_THREADS) + (threadIdx.x << LOG_LOAD_VEC_SIZE) + VEC;

			if (valid_flags[LOAD][VEC]) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(dest[LOAD][VEC], d_src);
				Transform(dest[LOAD][VEC], true);
			} else {
				Transform(dest[LOAD][VEC], false);
			}

			Iterate<LOAD, VEC + 1>::template Invoke<T, Transform, Flag>(
				src, dest, valid_flags);
		}
	};

	// Iterate next load
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// predicated on valid
		template <typename T, void Transform(T&, bool), typename Flag>
		static __device__ __forceinline__ void Invoke(
			T *src,
			T dest[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE])
		{
			Iterate<LOAD + 1, 0>::template Invoke<T, Transform, Flag>(
				src, dest, valid_flags);
		}
	};

	// Terminate
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// predicated on valid
		template <typename T, void Transform(T&, bool), typename Flag>
		static __device__ __forceinline__ void Invoke(
			T *src,
			T dest[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE]) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Gather to destination with transform, predicated on the valid flag
	 */
	template <
		typename T,
		void Transform(T&, bool), 							// Assignment function to transform the loaded value
		typename Flag>
	static __device__ __forceinline__ void Gather(
		T *src,
		T dest[][LOAD_VEC_SIZE],
		Flag valid_flags[][LOAD_VEC_SIZE])
	{
		Iterate<0, 0>::template Invoke<T, Transform>(
			src, dest, valid_flags);
	}

	/**
	 * Gather to destination predicated on the valid flag
	 */
	template <typename T, typename Flag>
	static __device__ __forceinline__ void Gather(
		T *src,
		T dest[][LOAD_VEC_SIZE],
		Flag valid_flags[][LOAD_VEC_SIZE])
	{
		Gather<T, NopLdTransform<T>, Flag>(
			src, dest, valid_flags);
	}

};



} // namespace io
} // namespace util
} // namespace b40c

