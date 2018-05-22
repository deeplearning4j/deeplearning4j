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
 * Tree reduction
 *
 * Does not support non-commutative operators.  (Suggested to use a scan
 * instead for those scenarios
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace reduction {


/**
 * Perform LOG_CTA_THREADS steps of binary tree reduction, each thread
 * contributes one reduction partial.
 */
template <
	int LOG_CTA_THREADS,
	bool ALL_RETURN>				// If true, everyone returns the result.  If false, only thread-0.
struct TreeReduce
{
	static const int CTA_THREADS = 1 << LOG_CTA_THREADS;

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// General iteration
	template <
		int OFFSET_RIGHT,
		bool WAS_WARPSCAN,
		bool IS_WARPSCAN = (OFFSET_RIGHT <= B40C_WARP_THREADS(__B40C_CUDA_ARCH__))>
	struct Iterate
	{
		template <
			bool ALL_VALID,
			typename T,
			typename TreeT,
			typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T my_partial,
			TreeT reduction_tree[CTA_THREADS],
			int num_elements,
			ReductionOp reduction_op)
		{
			// Store partial
			reduction_tree[threadIdx.x] = my_partial;

			__syncthreads();

			if ((ALL_VALID || (threadIdx.x + OFFSET_RIGHT < num_elements)) && (threadIdx.x < OFFSET_RIGHT)) {
				// Update my partial
				T current_partial = reduction_tree[threadIdx.x + OFFSET_RIGHT];
				my_partial = reduction_op(my_partial, current_partial);
			}

			// Recurse
			return Iterate<OFFSET_RIGHT / 2, WAS_WARPSCAN>::template Invoke<ALL_VALID>(
				my_partial, reduction_tree, num_elements, reduction_op);
		}
	};

	// Transition into warpscan iteration
	template <int OFFSET_RIGHT>
	struct Iterate<OFFSET_RIGHT, false, true>
	{
		template <
			bool ALL_VALID,
			typename T,
			typename TreeT,
			typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T my_partial,
			TreeT reduction_tree[CTA_THREADS],
			int num_elements,
			ReductionOp reduction_op)
		{
			// Store partial
			reduction_tree[threadIdx.x] = my_partial;

			__syncthreads();

			if (threadIdx.x < OFFSET_RIGHT) {

				if (ALL_VALID || (threadIdx.x + OFFSET_RIGHT < num_elements)) {

					// Update my partial
					T current_partial = reduction_tree[threadIdx.x + OFFSET_RIGHT];

					if (!IsVolatile<TreeT>::VALUE) __threadfence_block();

					my_partial = reduction_op(my_partial, current_partial);

				}

				// Recurse in warpscan mode
				my_partial = Iterate<OFFSET_RIGHT / 2, true>::template Invoke<ALL_VALID>(
						my_partial, reduction_tree, num_elements, reduction_op);
			}

			return my_partial;
		}
	};

	// Warpscan iteration
	template <int OFFSET_RIGHT, bool WAS_WARPSCAN>
	struct Iterate<OFFSET_RIGHT, WAS_WARPSCAN, true>
	{
		template <
			bool ALL_VALID,
			typename T,
			typename TreeT,
			typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T my_partial,
			TreeT reduction_tree[CTA_THREADS],
			int num_elements,
			ReductionOp reduction_op)
		{
			// Store partial
			reduction_tree[threadIdx.x] = my_partial;

			if (!IsVolatile<TreeT>::VALUE) __threadfence_block();

			if (ALL_VALID || (threadIdx.x + OFFSET_RIGHT < num_elements)) {

				// Update my partial
				T current_partial = reduction_tree[threadIdx.x + OFFSET_RIGHT];

				if (!IsVolatile<TreeT>::VALUE) __threadfence_block();

				my_partial = reduction_op(my_partial, current_partial);
			}

			// Recurse in warpscan mode
			return Iterate<OFFSET_RIGHT / 2, true>::template Invoke<ALL_VALID>(
					my_partial, reduction_tree, num_elements, reduction_op);

		}
	};

	// Termination
	template <bool WAS_WARPSCAN>
	struct Iterate<0, WAS_WARPSCAN, true>
	{
		template <
			bool ALL_VALID,
			typename T,
			typename TreeT,
			typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T my_partial,
			TreeT reduction_tree[CTA_THREADS],
			int num_elements,
			ReductionOp reduction_op)
		{
			if (ALL_RETURN) {
				reduction_tree[threadIdx.x] = my_partial;

				if (!IsVolatile<TreeT>::VALUE) __threadfence_block();
			}
			return my_partial;
		}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative tree reduction.  Threads with ranks less than
	 * num_elements contribute one reduction partial.
	 */
	template <
		typename T,
		typename TreeT,
		typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T my_partial,									// Input partial
		TreeT reduction_tree[CTA_THREADS],			// Shared memory for tree scan
		int num_elements,								// Number of valid elements to actually reduce (may be less than number of cta-threads)
		ReductionOp reduction_op)						// Reduction operator

	{
		my_partial = Iterate<CTA_THREADS / 2, false>::template Invoke<false>(
			my_partial,
			reduction_tree,
			num_elements,
			reduction_op);

		if (ALL_RETURN) {

			// Return first thread's my partial
			__syncthreads();
			return reduction_tree[0];

		} else {

			return my_partial;
		}
	}


	/**
	 * Perform a cooperative tree reduction.  Each thread contributes one
	 * reduction partial.
	 *
	 * Assumes all threads contribute a valid element (no checks on num_elements)
	 */
	template <
		typename T,
		typename TreeT,
		typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T my_partial,								// Input partial
		TreeT reduction_tree[CTA_THREADS],		// Shared memory for tree scan
		ReductionOp reduction_op)					// Reduction operator
	{
		my_partial = Iterate<CTA_THREADS / 2, false>::template Invoke<true>(
			my_partial,
			reduction_tree,
			0,
			reduction_op);

		if (ALL_RETURN) {

			// Return first thread's my partial
			__syncthreads();
			return reduction_tree[0];

		} else {

			return my_partial;
		}
	}


	/**
	 * Perform a cooperative tree reduction using the addition operator.
	 * Threads with ranks less than num_elements contribute one reduction partial.
	 */
	template <
		typename T,
		typename TreeT>
	static __device__ __forceinline__ T Invoke(
		T my_partial,									// Input partial
		TreeT reduction_tree[CTA_THREADS],			// Shared memory for tree scan
		int num_elements)								// Number of valid elements to actually reduce (may be less than number of cta-threads)
	{
		Sum<T> reduction_op;
		return Invoke(my_partial, reduction_tree, num_elements);
	}


	/**
	 * Perform a cooperative tree reduction using the addition operator.
	 * Each thread contributes one reduction partial.
	 *
	 * Assumes all threads contribute a valid element (no checks on num_elements)
	 */
	template <
		typename T,
		typename TreeT>
	static __device__ __forceinline__ T Invoke(
		T my_partial,								// Input partial
		TreeT reduction_tree[CTA_THREADS])		// Shared memory for tree scan
	{
		Sum<T> reduction_op;
		return Invoke(my_partial, reduction_tree, reduction_op);
	}
};


} // namespace reduction
} // namespace util
} // namespace b40c

