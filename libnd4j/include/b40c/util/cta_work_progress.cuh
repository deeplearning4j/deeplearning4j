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
 * Management of temporary device storage needed for implementing work-stealing
 * progress between CTAs in a single grid
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace util {

/**
 * Manages device storage needed for implementing work-stealing
 * and queuing progress between CTAs in a single grid.
 *
 * Can be used for:
 *
 * (1) Work-stealing. Consists of a pair of counters in
 *     global device memory, optionally, a host-managed selector for
 *     indexing into the pair.
 *
 * (2) Device-managed queue.  Consists of a quadruplet of counters
 *     in global device memory and selection into the counters is made
 *     based upon the supplied iteration count.
 *     		Current iteration: incoming queue length
 *     		Next iteration: outgoing queue length
 *     		Next next iteration: needs to be reset before next iteration
 *     Can be used with work-stealing counters to for work-stealing
 *     queue operation
 *
 * For work-stealing passes, the current counter provides an atomic
 * reference of progress, and the current pass will typically reset
 * the counter for the next.
 *
 */
class CtaWorkProgress
{
protected :

	enum {
		QUEUE_COUNTERS 		= 4,
		STEAL_COUNTERS 		= 2,
		OVERFLOW_COUNTERS 	= 1,
	};

	// Seven pointer-sized counters in global device memory (we may not use
	// all of them, or may only use 32-bit versions of them)
	size_t *d_counters;

	// Host-controlled selector for indexing into d_counters.
	int progress_selector;

public:

	enum {
		COUNTERS = QUEUE_COUNTERS + STEAL_COUNTERS + OVERFLOW_COUNTERS
	};

	/**
	 * Constructor
	 */
	CtaWorkProgress() :
		d_counters(NULL),
		progress_selector(0) {}

	/**
	 * Resets all counters.  Must be called by thread-0 through
	 * thread-(COUNTERS - 1)
	 */
	template <typename SizeT>
	__device__ __forceinline__ void Reset()
	{
		SizeT reset_val = 0;
		util::io::ModifiedStore<util::io::st::cg>::St(
			reset_val, ((SizeT *) d_counters) + threadIdx.x);
	}

	//---------------------------------------------------------------------
	// Work-stealing
	//---------------------------------------------------------------------

	/**
	 * Steals work from the host-indexed progress counter, returning
	 * the offset of that work (from zero) and incrementing it by count.
	 * Typically called by thread-0
	 */
	template <typename SizeT>
	__device__ __forceinline__ SizeT Steal(int count)
	{
		SizeT* d_steal_counters = ((SizeT*) d_counters) + QUEUE_COUNTERS;
		return util::AtomicInt<SizeT>::Add(d_steal_counters + progress_selector, count);
	}

	/**
	 * Steals work from the specified iteration's progress counter, returning the
	 * offset of that work (from zero) and incrementing it by count.
	 * Typically called by thread-0
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ SizeT Steal(int count, IterationT iteration)
	{
		SizeT* d_steal_counters = ((SizeT*) d_counters) + QUEUE_COUNTERS;
		return util::AtomicInt<SizeT>::Add(d_steal_counters + (iteration & 1), count);
	}

	/**
	 * Resets the work progress for the next host-indexed work-stealing
	 * pass.  Typically called by thread-0 in block-0.
	 */
	template <typename SizeT>
	__device__ __forceinline__ void PrepResetSteal()
	{
		SizeT 	reset_val = 0;
		SizeT* 	d_steal_counters = ((SizeT*) d_counters) + QUEUE_COUNTERS;
		util::io::ModifiedStore<util::io::st::cg>::St(
				reset_val, d_steal_counters + (progress_selector ^ 1));
	}

	/**
	 * Resets the work progress for the specified work-stealing iteration.
	 * Typically called by thread-0 in block-0.
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ void PrepResetSteal(IterationT iteration)
	{
		SizeT 	reset_val = 0;
		SizeT* 	d_steal_counters = ((SizeT*) d_counters) + QUEUE_COUNTERS;
		util::io::ModifiedStore<util::io::st::cg>::St(
			reset_val, d_steal_counters + (iteration & 1));
	}


	//---------------------------------------------------------------------
	// Queuing
	//---------------------------------------------------------------------

	/**
	 * Get counter for specified iteration
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ SizeT* GetQueueCounter(IterationT iteration)
	{
		return ((SizeT*) d_counters) + (iteration & 3);
	}

	/**
	 * Load work queue length for specified iteration
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ SizeT LoadQueueLength(IterationT iteration)
	{
		SizeT queue_length;
		util::io::ModifiedLoad<util::io::ld::cg>::Ld(
			queue_length, GetQueueCounter<SizeT>(iteration));
		return queue_length;
	}

	/**
	 * Store work queue length for specified iteration
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ void StoreQueueLength(SizeT queue_length, IterationT iteration)
	{
		util::io::ModifiedStore<util::io::st::cg>::St(
			queue_length, GetQueueCounter<SizeT>(iteration));
	}

	/**
	 * Enqueues work from the specified iteration's queue counter, returning the
	 * offset of that work (from zero) and incrementing it by count.
	 * Typically called by thread-0
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ SizeT Enqueue(SizeT count, IterationT iteration)
	{
		return util::AtomicInt<SizeT>::Add(
			GetQueueCounter<SizeT>(iteration),
			count);
	}

	/**
	 * Sets the overflow counter to non-zero
	 */
	template <typename SizeT>
	__device__ __forceinline__ void SetOverflow ()
	{
		((SizeT*) d_counters)[QUEUE_COUNTERS + STEAL_COUNTERS] = 1;
	}

};


/**
 * Version of work progress with storage lifetime management.
 *
 * We can use this in host enactors, and pass the base CtaWorkProgress
 * as parameters to kernels.
 */
class CtaWorkProgressLifetime : public CtaWorkProgress
{
protected:

	// GPU d_counters was allocated on
	int gpu;

public:

	/**
	 * Constructor
	 */
	CtaWorkProgressLifetime() :
		CtaWorkProgress(),
		gpu(B40C_INVALID_DEVICE) {}


	/**
	 * Deallocates and resets the progress counters
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;

		do {
			if (gpu != B40C_INVALID_DEVICE) {

				// Save current gpu
				int current_gpu;
				if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
					"CtaWorkProgress cudaGetDevice failed: ", __FILE__, __LINE__)) break;

				// Deallocate
				if (retval = util::B40CPerror(cudaSetDevice(gpu),
					"CtaWorkProgress cudaSetDevice failed: ", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFree(d_counters),
					"CtaWorkProgress cudaFree d_counters failed: ", __FILE__, __LINE__)) break;

				d_counters = NULL;
				gpu = B40C_INVALID_DEVICE;

				// Restore current gpu
				if (retval = util::B40CPerror(cudaSetDevice(current_gpu),
					"CtaWorkProgress cudaSetDevice failed: ", __FILE__, __LINE__)) break;
			}

			progress_selector = 0;

		} while (0);

		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~CtaWorkProgressLifetime()
	{
		HostReset();
	}


	/**
	 * Sets up the progress counters for the next kernel launch (lazily
	 * allocating and initializing them if necessary)
	 */
	cudaError_t Setup()
	{
		cudaError_t retval = cudaSuccess;
		do {

			// Make sure that our progress counters are allocated
			if (!d_counters) {

				size_t h_counters[COUNTERS];
				for (int i = 0; i < COUNTERS; i++) {
					h_counters[i] = 0;
				}

				// Allocate and initialize
				if (retval = util::B40CPerror(cudaGetDevice(&gpu),
					"CtaWorkProgress cudaGetDevice failed: ", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_counters, sizeof(h_counters)),
					"CtaWorkProgress cudaMalloc d_counters failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaMemcpy(d_counters, h_counters, sizeof(h_counters), cudaMemcpyHostToDevice),
					"CtaWorkProgress cudaMemcpy d_counters failed", __FILE__, __LINE__)) break;
			}

			// Update our progress counter selector to index the next progress counter
			progress_selector ^= 1;

		} while (0);

		return retval;
	}


	/**
	 * Checks if overflow counter is set
	 */
	template <typename SizeT>
	cudaError_t CheckOverflow(bool &overflow)	// out param
	{
		cudaError_t retval = cudaSuccess;

		do {
			SizeT counter;

			if (retval = util::B40CPerror(cudaMemcpy(
					&counter,
					((SizeT*) d_counters) + QUEUE_COUNTERS + STEAL_COUNTERS,
					1 * sizeof(SizeT),
					cudaMemcpyDeviceToHost),
				"CtaWorkProgress cudaMemcpy d_counters failed", __FILE__, __LINE__)) break;

			overflow = counter;

		} while (0);

		return retval;
	}


	/**
	 * Acquire work queue length
	 */
	template <typename IterationT, typename SizeT>
	cudaError_t GetQueueLength(
		IterationT iteration,
		SizeT &queue_length)		// out param
	{
		cudaError_t retval = cudaSuccess;

		do {
			int queue_length_idx = iteration & 0x3;

			if (retval = util::B40CPerror(cudaMemcpy(
					&queue_length,
					((SizeT*) d_counters) + queue_length_idx,
					1 * sizeof(SizeT),
					cudaMemcpyDeviceToHost),
				"CtaWorkProgress cudaMemcpy d_counters failed", __FILE__, __LINE__)) break;

		} while (0);

		return retval;
	}


	/**
	 * Set work queue length
	 */
	template <typename IterationT, typename SizeT>
	cudaError_t SetQueueLength(
		IterationT iteration,
		SizeT queue_length)
	{
		cudaError_t retval = cudaSuccess;

		do {
			int queue_length_idx = iteration & 0x3;

			if (retval = util::B40CPerror(cudaMemcpy(
					((SizeT*) d_counters) + queue_length_idx,
					&queue_length,
					1 * sizeof(SizeT),
					cudaMemcpyHostToDevice),
				"CtaWorkProgress cudaMemcpy d_counters failed", __FILE__, __LINE__)) break;

		} while (0);

		return retval;
	}
};

} // namespace util
} // namespace b40c

