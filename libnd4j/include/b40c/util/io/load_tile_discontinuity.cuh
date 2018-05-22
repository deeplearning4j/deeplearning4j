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
 * with cache modifiers, marking discontinuities between consecutive elements
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>
#include <b40c/util/vector_types.cuh>
#include <b40c/util/io/modified_load.cuh>

namespace b40c {
namespace util {
namespace io {


/**
 * Load a tile of items and initialize discontinuity flags
 */
template <
	int LOG_LOADS_PER_TILE, 									// Number of vector loads (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector load (log)
	int ACTIVE_THREADS,											// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER,							// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
	bool CHECK_ALIGNMENT,										// Whether or not to check alignment to see if vector loads can be used
	bool CONSECUTIVE_SMEM_ASSIST,								// Whether nor not to use supplied smem to assist in discontinuity detection
	bool FIRST_TILE,											// Whether or not this is the first tile loaded by the CTA
	bool FLAG_FIRST_OOB>										// Whether or not the first element that is out-of-bounds should also be flagged
struct LoadTileDiscontinuity
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
		// Vector with discontinuity flags (unguarded)
		template <
			typename T,
			typename Flag,
			typename VectorType,
			typename EqualityOp>
		static __device__ __forceinline__ void VectorLoadValid(
			T smem[ACTIVE_THREADS + 1],
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors,
			EqualityOp equality_op)
		{
			// Load the vector
			ModifiedLoad<CACHE_MODIFIER>::Ld(vectors[LOAD], d_in_vectors);


			if (CONSECUTIVE_SMEM_ASSIST) {

				// Place last vec element into shared buffer
				smem[threadIdx.x + 1] = data[LOAD][LOAD_VEC_SIZE - 1];

				__syncthreads();

				// Process first vec element
				if (FIRST_TILE && (LOAD == 0) && (threadIdx.x == 0)) {

					// First thread's first load of first tile
					if (blockIdx.x == 0) {

						// First CTA: start a new discontinuity
						flags[LOAD][0] = 1;

					} else {
						// Get the previous vector element from global
						T *d_ptr = (T*) d_in_vectors;
						T previous;
						ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_ptr - 1);
						flags[LOAD][0] = !equality_op(previous, data[LOAD][0]);
					}
				} else {

					T previous = smem[threadIdx.x];
					flags[LOAD][0] = !equality_op(previous, data[LOAD][0]);
				}

				__syncthreads();

				// Save last vector item for first of next load
				if (threadIdx.x == ACTIVE_THREADS - 1) {
					smem[0] = data[LOAD][LOAD_VEC_SIZE - 1];
				}

			} else {

				// Process first vec element
				if (FIRST_TILE && (LOAD == 0) && (blockIdx.x == 0) && (threadIdx.x == 0)) {

					// First thread's first load of first tile of first CTA: start a new discontinuity
					flags[LOAD][0] = 1;

				} else {
					// Get the previous vector element from global
					T *d_ptr = (T*) d_in_vectors;
					T previous;
					ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_ptr - 1);
					flags[LOAD][0] = !equality_op(previous, data[LOAD][0]);
				}
			}

			Iterate<LOAD, 1>::VectorLoadValid(
				smem, data, flags, vectors, d_in_vectors, equality_op);
		}

		// With discontinuity flags (unguarded)
		template <
			typename T,
			typename Flag,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			EqualityOp equality_op)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);

			if (FIRST_TILE && (LOAD == 0) && (blockIdx.x == 0) && (threadIdx.x == 0)) {

				// First load of first tile of first CTA: discontinuity
				flags[LOAD][0] = 1;

			} else {

				// Get the previous vector element (which is in range b/c this one is in range)
				T previous;
				ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_in + thread_offset - 1);
				flags[LOAD][0] = !equality_op(previous, data[LOAD][0]);
			}

			Iterate<LOAD, 1>::LoadValid(
				data, flags, d_in, equality_op);
		}

		// With discontinuity flags (guarded)
		template <
			typename T,
			typename Flag,
			typename SizeT,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements,
			EqualityOp equality_op)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);

				if (FIRST_TILE && (LOAD == 0) && (blockIdx.x == 0) && (threadIdx.x == 0)) {

					// First load of first tile of first CTA: discontinuity
					flags[LOAD][0] = 1;

				} else {

					// Get the previous vector element (which is in range b/c this one is in range)
					T previous;
					ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_in + thread_offset - 1);
					flags[LOAD][0] = !equality_op(previous, data[LOAD][0]);
				}

			} else {
				flags[LOAD][0] = ((FLAG_FIRST_OOB) && (thread_offset == guarded_elements));
			}

			Iterate<LOAD, 1>::LoadValid(
				data, flags, d_in, guarded_elements, equality_op);
		}
	};


	/**
	 * Next vec element of a vector-load
	 */
	template <int LOAD, int VEC, int dummy>
	struct Iterate
	{
		// Vector with discontinuity flags
		template <
			typename T,
			typename Flag,
			typename VectorType,
			typename EqualityOp>
		static __device__ __forceinline__ void VectorLoadValid(
			T smem[ACTIVE_THREADS + 1],
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors,
			EqualityOp equality_op)
		{
			T current = data[LOAD][VEC];
			T previous = data[LOAD][VEC - 1];
			flags[LOAD][VEC] = !equality_op(previous, current);

			Iterate<LOAD, VEC + 1>::VectorLoadValid(
				smem, data, flags, vectors, d_in_vectors, equality_op);
		}

		// With discontinuity flags (unguarded)
		template <
			typename T,
			typename Flag,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			EqualityOp equality_op)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);

			T previous = data[LOAD][VEC - 1];
			T current = data[LOAD][VEC];
			flags[LOAD][VEC] = !equality_op(previous, current);

			Iterate<LOAD, VEC + 1>::LoadValid(
				data, flags, d_in, equality_op);
		}

		// With discontinuity flags (guarded)
		template <
			typename T,
			typename Flag,
			typename SizeT,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements,
			EqualityOp equality_op)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);

				T previous = data[LOAD][VEC - 1];
				T current = data[LOAD][VEC];
				flags[LOAD][VEC] = !equality_op(previous, current);

			} else {
				flags[LOAD][VEC] = ((FLAG_FIRST_OOB) && (thread_offset == guarded_elements));
			}

			Iterate<LOAD, VEC + 1>::LoadValid(
				data, flags, d_in, guarded_elements, equality_op);
		}
	};


	/**
	 * Next load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// Vector with discontinuity flags (unguarded)
		template <
			typename T,
			typename Flag,
			typename VectorType,
			typename EqualityOp>
		static __device__ __forceinline__ void VectorLoadValid(
			T smem[ACTIVE_THREADS + 1],
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors,
			EqualityOp equality_op)
		{
			Iterate<LOAD + 1, 0>::VectorLoadValid(
				smem, data, flags, vectors, d_in_vectors + ACTIVE_THREADS, equality_op);
		}

		// With discontinuity flags (unguarded)
		template <
			typename T,
			typename Flag,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			EqualityOp equality_op)
		{
			Iterate<LOAD + 1, 0>::LoadValid(
				data, flags, d_in, equality_op);
		}

		// With discontinuity flags (guarded)
		template <
			typename T,
			typename Flag,
			typename SizeT,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements,
			EqualityOp equality_op)
		{
			Iterate<LOAD + 1, 0>::LoadValid(
				data, flags, d_in, guarded_elements, equality_op);
		}
	};
	
	/**
	 * Terminate
	 */
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// Vector with discontinuity flags (unguarded)
		template <
			typename T,
			typename Flag,
			typename VectorType,
			typename EqualityOp>
		static __device__ __forceinline__ void VectorLoadValid(
			T smem[ACTIVE_THREADS + 1],
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors,
			EqualityOp equality_op) {}

		// With discontinuity flags (unguarded)
		template <
			typename T,
			typename Flag,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			EqualityOp equality_op) {}

		// With discontinuity flags (guarded)
		template <
			typename T,
			typename Flag,
			typename SizeT,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements,
			EqualityOp equality_op) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Load a full tile and initialize discontinuity flags when values change
	 * between consecutive elements
	 */
	template <
		typename T,										// Tile type
		typename Flag,									// Discontinuity flag type
		typename SizeT,
		typename EqualityOp>
	static __device__ __forceinline__ void LoadValid(
		T smem[ACTIVE_THREADS + 1],
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		EqualityOp equality_op)
	{
		const size_t MASK = ((sizeof(T) * 8 * LOAD_VEC_SIZE) - 1);


		if ((CHECK_ALIGNMENT) && (LOAD_VEC_SIZE > 1) && (((size_t) d_in) & MASK)) {

			Iterate<0, 0>::LoadValid(
				data, flags, d_in + cta_offset, equality_op);

		} else {

			// Use an aliased pointer to keys array to perform built-in vector loads
			typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType;

			VectorType *vectors = (VectorType *) data;
			VectorType *d_in_vectors = (VectorType *) (d_in + cta_offset + (threadIdx.x << LOG_LOAD_VEC_SIZE));

			Iterate<0, 0>::VectorLoadValid(
				smem, data, flags, vectors, d_in_vectors, equality_op);
		}
	}

	/**
	 * Load guarded_elements of a tile and initialize discontinuity flags when
	 * values change between consecutive elements
	 */
	template <
		typename T,										// Tile type
		typename Flag,									// Discontinuity flag type
		typename SizeT,									// Integer type for indexing into global arrays
		typename EqualityOp>
	static __device__ __forceinline__ void LoadValid(
		T smem[ACTIVE_THREADS + 1],
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		EqualityOp equality_op)
	{
		if (guarded_elements >= TILE_SIZE) {

			LoadValid(smem, data, flags, d_in, cta_offset, equality_op);

		} else {

			Iterate<0, 0>::LoadValid(
				data, flags, d_in + cta_offset, guarded_elements, equality_op);
		}
	} 
};


} // namespace io
} // namespace util
} // namespace b40c

