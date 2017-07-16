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
 * Kernel utilities for storing tiles of data through global memory
 * with cache modifiers
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/vector_types.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace util {
namespace io {

/**
 * Store of a tile of items using guarded stores 
 */
template <
	int LOG_LOADS_PER_TILE, 									// Number of vector stores (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector store (log)
	int ACTIVE_THREADS,											// Active threads that will be storing
	st::CacheModifier CACHE_MODIFIER,							// Cache modifier (e.g., WB/CG/CS/NONE/etc.)
	bool CHECK_ALIGNMENT>										// Whether or not to check alignment to see if vector stores can be used
struct StoreTile
{
	enum {
		LOADS_PER_TILE 			= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 				= 1 << LOG_LOAD_VEC_SIZE,
		LOG_ELEMENTS_PER_THREAD		= LOG_LOADS_PER_TILE + LOG_LOAD_VEC_SIZE,
		ELEMENTS_PER_THREAD			= 1 << LOG_ELEMENTS_PER_THREAD,
		TILE_SIZE 					= ACTIVE_THREADS * LOADS_PER_TILE * LOAD_VEC_SIZE,
	};

	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate over vec-elements
	template <int LOAD, int VEC>
	struct Iterate
	{
		// Vector
		template <typename VectorType>
		static __device__ __forceinline__ void Invoke(
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			Iterate<LOAD, VEC + 1>::Invoke(vectors, d_in_vectors);
		}

		// Unguarded
		template <typename T>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_out)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			ModifiedStore<CACHE_MODIFIER>::St(data[LOAD][VEC], d_out + thread_offset);

			Iterate<LOAD, VEC + 1>::Invoke(data, d_out);
		}

		// Guarded
		template <typename T, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_out,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedStore<CACHE_MODIFIER>::St(data[LOAD][VEC], d_out + thread_offset);
			}
			Iterate<LOAD, VEC + 1>::Invoke(data, d_out, guarded_elements);
		}
	};

	// Iterate over stores
	template <int LOAD>
	struct Iterate<LOAD, LOAD_VEC_SIZE>
	{
		// Vector
		template <typename VectorType>
		static __device__ __forceinline__ void Invoke(
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			ModifiedStore<CACHE_MODIFIER>::St(
				vectors[LOAD], d_in_vectors);

			Iterate<LOAD + 1, 0>::Invoke(vectors, d_in_vectors + ACTIVE_THREADS);
		}

		// Unguarded
		template <typename T>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_out)
		{
			Iterate<LOAD + 1, 0>::Invoke(data, d_out);
		}

		// Guarded
		template <typename T, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_out,
			const SizeT &guarded_elements)
		{
			Iterate<LOAD + 1, 0>::Invoke(data, d_out, guarded_elements);
		}
	};
	
	// Terminate
	template <int VEC>
	struct Iterate<LOADS_PER_TILE, VEC>
	{
		// Vector
		template <typename VectorType>
		static __device__ __forceinline__ void Invoke(
			VectorType vectors[], VectorType *d_in_vectors) {}

		// Unguarded
		template <typename T>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_out) {}

		// Guarded
		template <typename T, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_out,
			const SizeT &guarded_elements) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Store a full tile
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void Store(
		T data[][LOAD_VEC_SIZE],
		T *d_out,
		SizeT cta_offset)
	{
		const size_t MASK = ((sizeof(T) * 8 * LOAD_VEC_SIZE) - 1);

		if ((CHECK_ALIGNMENT) && (LOAD_VEC_SIZE > 1) && (((size_t) d_out) & MASK)) {

			Iterate<0, 0>::Invoke(
				data, d_out + cta_offset);

		} else {

			// Aliased vector type
			typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType;

			// Use an aliased pointer to keys array to perform built-in vector stores
			VectorType *vectors = (VectorType *) data;
			VectorType *d_in_vectors = (VectorType *) (d_out + cta_offset + (threadIdx.x << LOG_LOAD_VEC_SIZE));

			Iterate<0, 0>::Invoke(vectors, d_in_vectors);
		}
	}

	/**
	 * Store guarded_elements of a tile
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void Store(
		T data[][LOAD_VEC_SIZE],
		T *d_out,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		if (guarded_elements >= TILE_SIZE) {

			Store(data, d_out, cta_offset);

		} else {

			Iterate<0, 0>::Invoke(
				data, d_out + cta_offset, guarded_elements);
		}
	} 
};



} // namespace io
} // namespace util
} // namespace b40c

