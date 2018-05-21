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
 * Kernel utilities for loading tiles of data through global memory
 * with cache modifiers
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>
#include <b40c/util/vector_types.cuh>
#include <b40c/util/io/modified_load.cuh>

namespace b40c {
namespace util {
namespace io {


/**
 * Load a tile of items
 */
template <
	int LOG_LOADS_PER_TILE, 									// Number of vector loads (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector load (log)
	int ACTIVE_THREADS,											// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER,							// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
	bool CHECK_ALIGNMENT>										// Whether or not to check alignment to see if vector loads can be used
struct LoadTile
{
	enum {
		LOADS_PER_TILE 				= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 				= 1 << LOG_LOAD_VEC_SIZE,
		LOG_ELEMENTS_PER_THREAD		= LOG_LOADS_PER_TILE + LOG_LOAD_VEC_SIZE,
		ELEMENTS_PER_THREAD			= 1 << LOG_ELEMENTS_PER_THREAD,
		TILE_SIZE 					= ACTIVE_THREADS * ELEMENTS_PER_THREAD,
	};
	
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	template <int LOAD, int VEC, int dummy = 0> struct Iterate;

	/**
	 * First vec element of a vector-load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, 0, dummy>
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			ModifiedLoad<CACHE_MODIFIER>::Ld(vectors[LOAD], d_in_vectors);
			Transform(data[LOAD][0]);		// Apply transform function with in_bounds = true

			Iterate<LOAD, 1>::template LoadVector<T, Transform>(
				data, vectors, d_in_vectors);
		}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);
			Transform(data[LOAD][0]);

			Iterate<LOAD, 1>::template LoadValid<T, Transform>(
				data, d_in);
		}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);
				Transform(data[LOAD][0]);
			}

			Iterate<LOAD, 1>::template LoadValid<T, Transform>(
				data, d_in, guarded_elements);
		}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T oob_default,
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);
				Transform(data[LOAD][0]);
			} else {
				data[LOAD][0] = oob_default;
			}

			Iterate<LOAD, 1>::template LoadValid<T, Transform>(
				data, oob_default, d_in, guarded_elements);
		}
	};


	/**
	 * Next vec element of a vector-load
	 */
	template <int LOAD, int VEC, int dummy>
	struct Iterate
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			Transform(data[LOAD][VEC]);

			Iterate<LOAD, VEC + 1>::template LoadVector<T, Transform>(
				data, vectors, d_in_vectors);
		}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
			Transform(data[LOAD][VEC]);

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, d_in);
		}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
				Transform(data[LOAD][VEC]);
			}

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, d_in, guarded_elements);
		}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T oob_default,
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
				Transform(data[LOAD][VEC]);
			} else {
				data[LOAD][VEC] = oob_default;
			}

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, oob_default, d_in, guarded_elements);
		}
	};


	/**
	 * Next load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			Iterate<LOAD + 1, 0>::template LoadVector<T, Transform>(
				data, vectors, d_in_vectors + ACTIVE_THREADS);
		}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, d_in);
		}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, d_in, guarded_elements);
		}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T oob_default,
			T *d_in,
			const SizeT &guarded_elements)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, oob_default, d_in, guarded_elements);
		}
	};
	
	/**
	 * Terminate
	 */
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors) {}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in) {}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements) {}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T oob_default,
			T *d_in,
			const SizeT &guarded_elements) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Load a full tile with transform
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset)
	{
		const size_t MASK = ((sizeof(T) * 8 * LOAD_VEC_SIZE) - 1);

		if ((CHECK_ALIGNMENT) && (LOAD_VEC_SIZE > 1) && (((size_t) d_in) & MASK)) {

			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, d_in + cta_offset);

		} else {

			// Use an aliased pointer to keys array to perform built-in vector loads
			typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType;

			VectorType *vectors = (VectorType *) data;
			VectorType *d_in_vectors = (VectorType *) (d_in + cta_offset + (threadIdx.x << LOG_LOAD_VEC_SIZE));

			Iterate<0, 0>::template LoadVector<T, Transform>(
				data, vectors, d_in_vectors);
		}
	}


	/**
	 * Load a full tile
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset)
	{
		LoadValid<T, Operators<T>::NopTransform>(data, d_in, cta_offset);
	}


	/**
	 * Load guarded_elements of a tile with transform and out-of-bounds default
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		T oob_default)
	{
		if (guarded_elements >= TILE_SIZE) {
			LoadValid<T, Transform>(data, d_in, cta_offset);
		} else {
			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, oob_default, d_in + cta_offset, guarded_elements);
		}
	}


	/**
	 * Load guarded_elements of a tile with transform
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		if (guarded_elements >= TILE_SIZE) {
			LoadValid<T, Transform>(data, d_in, cta_offset);
		} else {
			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, d_in + cta_offset, guarded_elements);
		}
	}


	/**
	 * Load guarded_elements of a tile and out_of_bounds default
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		T oob_default)
	{
		LoadValid<T, Operators<T>::NopTransform>(
			data, d_in, cta_offset, guarded_elements, oob_default);
	}


	/**
	 * Load guarded_elements of a tile
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		LoadValid<T, Operators<T>::NopTransform, int>(
			data, d_in, cta_offset, guarded_elements);
	}
};


} // namespace io
} // namespace util
} // namespace b40c

