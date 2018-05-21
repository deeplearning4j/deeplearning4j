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
 * Kernel utilities for scattering data
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace io {




/**
 * Scatter a tile of data items using the corresponding tile of scatter_offsets
 */
template <
	int LOG_LOADS_PER_TILE, 									// Number of vector loads (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector load (log)
	int ACTIVE_THREADS,											// Active threads that will be loading
	st::CacheModifier CACHE_MODIFIER>							// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
struct ScatterTile
{
	enum {
		LOADS_PER_TILE 				= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 				= 1 << LOG_LOAD_VEC_SIZE,
		LOG_ELEMENTS_PER_THREAD		= LOG_LOADS_PER_TILE + LOG_LOAD_VEC_SIZE,
		ELEMENTS_PER_THREAD			= 1 << LOG_ELEMENTS_PER_THREAD,
		TILE_SIZE 					= ACTIVE_THREADS * ELEMENTS_PER_THREAD,
	};

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// Iterate next vector item
	template <int LOAD, int VEC, int dummy = 0>
	struct Iterate
	{
		// Unguarded
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE])
		{
			Transform(data[LOAD][VEC]);
			ModifiedStore<CACHE_MODIFIER>::St(data[LOAD][VEC], dest + scatter_offsets[LOAD][VEC]);

			Iterate<LOAD, VEC + 1>::template Invoke<T, Transform>(
				dest, data, scatter_offsets);
		}

		// Guarded by flags
		template <typename T, void Transform(T&), typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE])
		{
			if (valid_flags[LOAD][VEC]) {

				Transform(data[LOAD][VEC]);
				ModifiedStore<CACHE_MODIFIER>::St(data[LOAD][VEC], dest + scatter_offsets[LOAD][VEC]);
			}

			Iterate<LOAD, VEC + 1>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, valid_flags);
		}

		// Guarded by partial tile size
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			const SizeT &partial_tile_size)
		{
			int tile_rank = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (tile_rank < partial_tile_size) {

				Transform(data[LOAD][VEC]);
				ModifiedStore<CACHE_MODIFIER>::St(data[LOAD][VEC], dest + scatter_offsets[LOAD][VEC]);
			}

			Iterate<LOAD, VEC + 1>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, partial_tile_size);
		}

		// Guarded by flags and partial tile size
		template <typename T, void Transform(T&), typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE],
			const SizeT &partial_tile_size)
		{
			int tile_rank = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (valid_flags[LOAD][VEC] && (tile_rank < partial_tile_size)) {

				Transform(data[LOAD][VEC]);
				ModifiedStore<CACHE_MODIFIER>::St(data[LOAD][VEC], dest + scatter_offsets[LOAD][VEC]);
			}

			Iterate<LOAD, VEC + 1>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, valid_flags, partial_tile_size);
		}
	};


	// Next Load
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// Unguarded
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE])
		{
			Iterate<LOAD + 1, 0>::template Invoke<T, Transform>(
				dest, data, scatter_offsets);
		}

		// Guarded by flags
		template <typename T, void Transform(T&), typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE])
		{
			Iterate<LOAD + 1, 0>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, valid_flags);
		}

		// Guarded by partial tile size
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			const SizeT &partial_tile_size)
		{
			Iterate<LOAD + 1, 0>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, partial_tile_size);
		}

		// Guarded by flags and partial tile size
		template <typename T, void Transform(T&), typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE],
			const SizeT &partial_tile_size)
		{
			Iterate<LOAD + 1, 0>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, valid_flags, partial_tile_size);
		}
	};


	// Terminate
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// Unguarded
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE]) {}

		// Guarded by flags
		template <typename T, void Transform(T&), typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE]) {}

		// Guarded by partial tile size
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			const SizeT &partial_tile_size) {}

		// Guarded by flags and partial tile size
		template <typename T, void Transform(T&), typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T data[][LOAD_VEC_SIZE],
			SizeT scatter_offsets[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE],
			const SizeT &partial_tile_size) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Scatter to destination with transform.  The write is
	 * predicated on the element's index in
	 * the tile is not exceeding the partial tile size
	 */
	template <
		typename T,
		void Transform(T&), 							// Assignment function to transform the stored value
		typename SizeT>
	static __device__ __forceinline__ void Scatter(
		T *dest,
		T data[][LOAD_VEC_SIZE],
		SizeT scatter_offsets[][LOAD_VEC_SIZE],
		const SizeT &partial_tile_size = TILE_SIZE)
	{
		if (partial_tile_size < TILE_SIZE) {

			// guarded IO
			Iterate<0, 0>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, partial_tile_size);

		} else {

			// unguarded IO
			Iterate<0, 0>::template Invoke<T, Transform>(
				dest, data, scatter_offsets);
		}
	}

	/**
	 * Scatter to destination.  The write is predicated on the element's index in
	 * the tile is not exceeding the partial tile size
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void Scatter(
		T *dest,
		T data[][LOAD_VEC_SIZE],
		SizeT scatter_offsets[][LOAD_VEC_SIZE],
		const SizeT &partial_tile_size = TILE_SIZE)
	{
		Scatter<T, Operators<T>::NopTransform>(
			dest, data, scatter_offsets, partial_tile_size);
	}

	/**
	 * Scatter to destination with transform.  The write is
	 * predicated on valid flags and that the element's index in
	 * the tile is not exceeding the partial tile size
	 */
	template <
		typename T,
		void Transform(T&), 							// Assignment function to transform the stored value
		typename Flag,
		typename SizeT>
	static __device__ __forceinline__ void Scatter(
		T *dest,
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		SizeT scatter_offsets[][LOAD_VEC_SIZE],
		const SizeT &partial_tile_size = TILE_SIZE)
	{
		if (partial_tile_size < TILE_SIZE) {

			// guarded by flags and partial tile size
			Iterate<0, 0>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, flags, partial_tile_size);

		} else {

			// guarded by flags
			Iterate<0, 0>::template Invoke<T, Transform>(
				dest, data, scatter_offsets, flags);
		}
	}

	/**
	 * Scatter to destination.  The write is
	 * predicated on valid flags and that the element's index in
	 * the tile is not exceeding the partial tile size
	 */
	template <typename T, typename Flag, typename SizeT>
	static __device__ __forceinline__ void Scatter(
		T *dest,
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		SizeT scatter_offsets[][LOAD_VEC_SIZE],
		const SizeT &partial_tile_size = TILE_SIZE)
	{
		Scatter<T, Operators<T>::NopTransform>(
			dest, data, flags, scatter_offsets, partial_tile_size);
	}

};



} // namespace io
} // namespace util
} // namespace b40c

