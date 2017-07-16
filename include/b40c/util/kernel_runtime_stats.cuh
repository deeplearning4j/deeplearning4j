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
 * Kernel runtime statistics
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>
#include <b40c/util/cuda_properties.cuh>

namespace b40c {
namespace util {


/**
 * Manages device storage needed for conveying kernel runtime stats
 */
class KernelRuntimeStats
{
protected :

	enum {
		CLOCKS		= 0,
		AGGREGATE,

		TOTAL_COUNTERS,
	};

	// Counters in global device memory
	unsigned long long 		*d_stat;


	clock_t 				start;				// Start time
	clock_t 				clocks;				// Accumulated time
	unsigned long long 		aggregate;			// General-purpose aggregate counter

public:

	/**
	 * Constructor
	 */
	KernelRuntimeStats() :
		d_stat(NULL),
		clocks(0),
		aggregate(0) {}

	/**
	 * Marks start time.  Typically called by thread-0.
	 */
	__device__ __forceinline__ void MarkStart()
	{
		start = clock();
	}

	/**
	 * Marks stop time.  Typically called by thread-0.
	 */
	__device__ __forceinline__ void MarkStop()
	{
		clock_t stop = clock();
		clock_t runtime = (stop >= start) ?
			stop - start :
			stop + (((clock_t) -1) - start);
		clocks += runtime;
	}

	/**
	 * Increments the aggregate counter by the specified amount.
	 * Typically called by thread-0.
	 */
	template <typename T>
	__device__ __forceinline__ void Aggregate(T increment)
	{
		aggregate += increment;
	}

	/**
	 * Flushes statistics to global mem
	 */
	__device__ __forceinline__ void Flush()
	{
		if (d_stat != NULL) {
			d_stat[blockIdx.x + (CLOCKS * gridDim.x)] = clocks;
			d_stat[blockIdx.x + (AGGREGATE * gridDim.x)] = aggregate;
		}
	}

	/**
	 * Resets statistics. Typically called by thread-0.
	 */
	__device__ __forceinline__ void Reset() const
	{
		if (d_stat != NULL) {
			d_stat[blockIdx.x + (CLOCKS * gridDim.x)] = 0;
			d_stat[blockIdx.x + (AGGREGATE * gridDim.x)] = 0;
		}
	}

};


/**
 * Version of global barrier with storage lifetime management.
 *
 * We can use this in host enactors, and pass the base GlobalBarrier
 * as parameters to kernels.
 */
class KernelRuntimeStatsLifetime : public KernelRuntimeStats
{
protected:

	// Number of bytes backed by d_stat
	size_t stat_bytes;

	// GPU d_counters was allocated on
	int gpu;

public:

	/**
	 * Constructor
	 */
	KernelRuntimeStatsLifetime() :
		KernelRuntimeStats(),
		stat_bytes(0),
		gpu(B40C_INVALID_DEVICE) {}


	/**
	 * Deallocates and resets the progress counters
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;

		do {

			if (d_stat) {

				// Save current gpu
				int current_gpu;
				if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
					"KernelRuntimeStatsLifetime cudaGetDevice failed: ", __FILE__, __LINE__)) break;

				// Deallocate
				if (retval = util::B40CPerror(cudaSetDevice(gpu),
					"KernelRuntimeStatsLifetime cudaSetDevice failed: ", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFree(d_stat),
					"KernelRuntimeStatsLifetime cudaFree d_stat failed: ", __FILE__, __LINE__)) break;

				d_stat = NULL;
				gpu = B40C_INVALID_DEVICE;

				// Restore current gpu
				if (retval = util::B40CPerror(cudaSetDevice(current_gpu),
					"KernelRuntimeStatsLifetime cudaSetDevice failed: ", __FILE__, __LINE__)) break;
			}

			stat_bytes = 0;

		} while (0);

		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~KernelRuntimeStatsLifetime()
	{
		HostReset();
	}


	/**
	 * Sets up the progress counters for the next kernel launch (lazily
	 * allocating and initializing them if necessary)
	 */
	cudaError_t Setup(int grid_size)
	{
		cudaError_t retval = cudaSuccess;
		do {
			size_t new_stat_bytes = grid_size * sizeof(unsigned long long) * TOTAL_COUNTERS;
			if (new_stat_bytes > stat_bytes) {

				// Deallocate if exists
				if (retval = HostReset()) break;

				// Remember device
				if (retval = util::B40CPerror(cudaGetDevice(&gpu),
					"KernelRuntimeStatsLifetime cudaGetDevice failed: ", __FILE__, __LINE__)) break;

				// Reallocate
				stat_bytes = new_stat_bytes;

				if (retval = util::B40CPerror(cudaMalloc((void**) &d_stat, stat_bytes),
					"KernelRuntimeStatsLifetime cudaMalloc d_stat failed", __FILE__, __LINE__)) break;

				// Initialize to zero
				util::MemsetKernel<unsigned long long><<<(grid_size + 128 - 1) / 128, 128>>>(
					d_stat, 0, grid_size);
				if (retval = util::B40CPerror(cudaThreadSynchronize(),
					"KernelRuntimeStatsLifetime MemsetKernel d_stat failed", __FILE__, __LINE__)) break;
			}
		} while (0);

		return retval;
	}


	/**
	 * Accumulates avg live, max live, and total aggregate
	 */
	cudaError_t Accumulate(
		int grid_size,
		unsigned long long &total_runtimes,
		unsigned long long &total_lifetimes,
		unsigned long long &total_aggregate)
	{
		cudaError_t retval = cudaSuccess;

		do {

			unsigned long long *h_stat = (unsigned long long*) malloc(stat_bytes);

			// Save current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"KernelRuntimeStatsLifetime cudaGetDevice failed: ", __FILE__, __LINE__)) break;

			if (retval = util::B40CPerror(cudaSetDevice(gpu),
				"KernelRuntimeStatsLifetime cudaSetDevice failed: ", __FILE__, __LINE__)) break;

			// Copy out stats
			if (retval = util::B40CPerror(cudaMemcpy(h_stat, d_stat, stat_bytes, cudaMemcpyDeviceToHost),
				"KernelRuntimeStatsLifetime d_stat failed", __FILE__, __LINE__)) break;

			// Restore current gpu
			if (retval = util::B40CPerror(cudaSetDevice(current_gpu),
				"KernelRuntimeStatsLifetime cudaSetDevice failed: ", __FILE__, __LINE__)) break;

			// Compute runtimes, find max
			unsigned long long max_runtime = 0;
			for (int block = 0; block < grid_size; block++) {

				unsigned long long runtime = h_stat[(CLOCKS * grid_size) + block];

				if (runtime > max_runtime) {
					max_runtime = runtime;
				}

				total_runtimes += runtime;
			}

			total_lifetimes += (max_runtime * grid_size);

			// Accumulate aggregates
			for (int block = 0; block < grid_size; block++) {
				total_aggregate += h_stat[(AGGREGATE * grid_size) + block];
			}

			free(h_stat);

		} while (0);

		return retval;
	}


	/**
	 * Accumulates avg live, max live
	 */
	cudaError_t Accumulate(
		int grid_size,
		unsigned long long &total_runtimes,
		unsigned long long &total_lifetimes)
	{
		unsigned long long total_aggregate = 0;
		return Accumulate(grid_size, total_runtimes, total_lifetimes, total_aggregate);
	}
};




} // namespace util
} // namespace b40c

