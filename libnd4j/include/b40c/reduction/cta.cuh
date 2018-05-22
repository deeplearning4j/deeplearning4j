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
 * CTA-processing abstraction for reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>

#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>

namespace b40c {
namespace reduction {


/**
 * Reduction CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T 			T;
	typedef typename KernelPolicy::SizeT 		SizeT;
	typedef typename KernelPolicy::SmemStorage	SmemStorage;
	typedef typename KernelPolicy::ReductionOp	ReductionOp;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// The value we will accumulate (in each thread)
	T 				carry;

	// Input and output device pointers
	T* 				d_in;
	T* 				d_out;

	// Shared memory storage for the CTA
	SmemStorage 	&smem_storage;

	// Reduction operator
	ReductionOp		reduction_op;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage &smem_storage,
		T *d_in,
		T *d_out,
		ReductionOp reduction_op) :

			smem_storage(smem_storage),
			d_in(d_in),
			d_out(d_out),
			reduction_op(reduction_op)
	{}


	/**
	 * Process a single, full tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
	{
		// Tile of elements
		T data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				data, d_in, cta_offset);

		// Reduce the data we loaded for this tile
		T tile_partial = util::reduction::SerialReduce<KernelPolicy::TILE_ELEMENTS_PER_THREAD>::Invoke(
			(T*) data,
			reduction_op);

		// Reduce into carry
		if (FIRST_TILE) {
			carry = tile_partial;
		} else {
			carry = reduction_op(carry, tile_partial);
		}
	}


	/**
	 * Process a single, partial tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		SizeT out_of_bounds)
	{
		T datum;
		cta_offset += threadIdx.x;

		if (FIRST_TILE) {
			if (cta_offset < out_of_bounds) {
				util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(carry, d_in + cta_offset);
				cta_offset += KernelPolicy::THREADS;
			}
		}

		// Process loads singly
		while (cta_offset < out_of_bounds) {
			util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(datum, d_in + cta_offset);
			carry = reduction_op(carry, datum);
			cta_offset += KernelPolicy::THREADS;
		}
	}


	/**
	 * Unguarded collective reduction across all threads, stores final reduction
	 * to output.  Used to collectively reduce each thread's aggregate after
	 * striding through the input.
	 *
	 * All threads assumed to have valid carry data.
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		carry = util::reduction::TreeReduce<
			KernelPolicy::LOG_THREADS,
			false>::Invoke(								// No need to return aggregate reduction in all threads
				carry,
				smem_storage.ReductionTree(),
				reduction_op);

		// Write output
		if (threadIdx.x == 0) {
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry, d_out + blockIdx.x);
		}
	}


	/**
	 * Guarded collective reduction across all threads, stores final reduction
	 * to output. Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 *
	 * Only threads with ranks less than num_elements are assumed to have valid
	 * carry data.
	 */
	__device__ __forceinline__ void OutputToSpine(int num_elements)
	{
		carry = util::reduction::TreeReduce<
			KernelPolicy::LOG_THREADS,
			false>::Invoke(								// No need to return aggregate reduction in all threads
				carry,
				smem_storage.ReductionTree(),
				num_elements,
				reduction_op);

		// Write output
		if (threadIdx.x == 0) {
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry, d_out + blockIdx.x);
		}
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT cta_offset = work_limits.offset;

		if (cta_offset < work_limits.guarded_offset) {

			// Process at least one full tile of tile_elements
			ProcessFullTile<true>(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;

			// Process more full tiles (not first tile)
			while (cta_offset < work_limits.guarded_offset) {
				ProcessFullTile<false>(cta_offset);
				cta_offset += KernelPolicy::TILE_ELEMENTS;
			}

			// Clean up last partial tile with guarded-io (not first tile)
			if (work_limits.guarded_elements) {
				ProcessPartialTile<false>(
					cta_offset,
					work_limits.out_of_bounds);
			}

			// Collectively reduce accumulated carry from each thread into output
			// destination (all thread have valid reduction partials)
			OutputToSpine();

		} else {

			// Clean up last partial tile with guarded-io (first tile)
			ProcessPartialTile<true>(
				cta_offset,
				work_limits.out_of_bounds);

			// Collectively reduce accumulated carry from each thread into output
			// destination (not every thread may have a valid reduction partial)
			OutputToSpine(work_limits.elements);
		}
	}

};


} // namespace reduction
} // namespace b40c

