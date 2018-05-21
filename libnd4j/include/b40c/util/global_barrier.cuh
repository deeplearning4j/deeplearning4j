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
 * Software Global Barrier
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/memset_kernel.cuh>
#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace util {


/**
 * Manages device storage needed for implementing a global software barrier
 * between CTAs in a single grid
 */
class GlobalBarrier
{
public:

	typedef unsigned int SyncFlag;

protected :


	// Counters in global device memory
	SyncFlag* d_sync;

	/**
	 * Simple wrapper for returning a CG-loaded SyncFlag at the specified pointer
	 */
	__device__ __forceinline__ SyncFlag LoadCG(SyncFlag* d_ptr) const
	{
		SyncFlag retval;
		util::io::ModifiedLoad<util::io::ld::cg>::Ld(retval, d_ptr);
		return retval;
	}

public:

	/**
	 * Constructor
	 */
	GlobalBarrier() : d_sync(NULL) {}


	/**
	 * Synchronize
	 */
	__device__ __forceinline__ void Sync() const
	{
        volatile SyncFlag *d_vol_sync = d_sync;

        // Threadfence and syncthreads to make sure global writes are visible before
		// thread-0 reports in with its sync counter
		__threadfence();
		__syncthreads();

		if (blockIdx.x == 0) {

			// Report in ourselves
			if (threadIdx.x == 0) {
			    d_vol_sync[blockIdx.x] = 1;
			}

			__syncthreads();

			// Wait for everyone else to report in
			for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
				while (LoadCG(d_sync + peer_block) == 0) {
					__threadfence_block();
				}
			}

			__syncthreads();

			// Let everyone know it's safe to read their prefix sums
			for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
			    d_vol_sync[peer_block] = 0;
			}

		} else {

			if (threadIdx.x == 0) {
				// Report in
			    d_vol_sync[blockIdx.x] = 1;

				// Wait for acknowledgement
				while (LoadCG(d_sync + blockIdx.x) == 1) {
					__threadfence_block();
				}
			}

			__syncthreads();
		}
	}
};


/**
 * Version of global barrier with storage lifetime management.
 *
 * We can use this in host enactors, and pass the base GlobalBarrier
 * as parameters to kernels.
 */
class GlobalBarrierLifetime : public GlobalBarrier
{
protected:

	// Number of bytes backed by d_sync
	size_t sync_bytes;

public:

	/**
	 * Constructor
	 */
	GlobalBarrierLifetime() : GlobalBarrier(), sync_bytes(0) {}


	/**
	 * Deallocates and resets the progress counters
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;
		if (d_sync) {
			retval = util::B40CPerror(cudaFree(d_sync), "GlobalBarrier cudaFree d_sync failed: ", __FILE__, __LINE__);
			d_sync = NULL;
		}
		sync_bytes = 0;
		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~GlobalBarrierLifetime()
	{
		HostReset();
	}


	/**
	 * Sets up the progress counters for the next kernel launch (lazily
	 * allocating and initializing them if necessary)
	 */
	cudaError_t Setup(int sweep_grid_size)
	{
		cudaError_t retval = cudaSuccess;
		do {
			size_t new_sync_bytes = sweep_grid_size * sizeof(SyncFlag);
			if (new_sync_bytes > sync_bytes) {

				if (d_sync) {
					if (retval = util::B40CPerror(cudaFree(d_sync),
						"GlobalBarrierLifetime cudaFree d_sync failed", __FILE__, __LINE__)) break;
				}

				sync_bytes = new_sync_bytes;

				if (retval = util::B40CPerror(cudaMalloc((void**) &d_sync, sync_bytes),
					"GlobalBarrierLifetime cudaMalloc d_sync failed", __FILE__, __LINE__)) break;

				// Initialize to zero
				util::MemsetKernel<SyncFlag><<<(sweep_grid_size + 128 - 1) / 128, 128>>>(
					d_sync, 0, sweep_grid_size);
				if (retval = util::B40CPerror(cudaThreadSynchronize(),
					"GlobalBarrierLifetime MemsetKernel d_sync failed", __FILE__, __LINE__)) break;
			}
		} while (0);

		return retval;
	}
};




} // namespace util
} // namespace b40c

