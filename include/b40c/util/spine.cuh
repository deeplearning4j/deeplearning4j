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
 * Management of temporary device storage needed for maintaining partial
 * reductions between subsequent grids
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace util {

/**
 * Manages device storage needed for communicating partial reductions
 * between CTAs in subsequent grids
 */
struct Spine
{
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Device spine storage
	void *d_spine;

	// Host-mapped spine storage (if so constructed)
	void *h_spine;

	// Number of bytes backed by d_spine
	size_t spine_bytes;

	// GPU d_spine was allocated on
	int gpu;

	// Whether or not the spine has a shadow spine on the host
	bool host_shadow;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor (device-allocated spine)
	 */
	Spine() :
		d_spine(NULL),
		h_spine(NULL),
		spine_bytes(0),
		gpu(B40C_INVALID_DEVICE),
		host_shadow(false) {}


	/**
	 * Constructor
	 *
	 * @param host_shadow
	 * 		Whether or not the spine has a shadow spine on the host
	 */
	Spine(bool host_shadow) :
		d_spine(NULL),
		h_spine(NULL),
		spine_bytes(0),
		gpu(B40C_INVALID_DEVICE),
		host_shadow(host_shadow) {}


	/**
	 * Deallocates and resets the spine
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;
		do {

			if (gpu == B40C_INVALID_DEVICE) return retval;

			// Save current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"Spine cudaGetDevice failed: ", __FILE__, __LINE__)) break;
#if CUDA_VERSION >= 4000
			if (retval = util::B40CPerror(cudaSetDevice(gpu),
				"Spine cudaSetDevice failed: ", __FILE__, __LINE__)) break;
#endif
			if (d_spine) {
				// Deallocate
				if (retval = util::B40CPerror(cudaFree(d_spine),
					"Spine cudaFree d_spine failed: ", __FILE__, __LINE__)) break;
				d_spine = NULL;
			}
			if (h_spine) {
				// Deallocate
				if (retval = util::B40CPerror(cudaFreeHost((void *) h_spine),
					"Spine cudaFreeHost h_spine failed", __FILE__, __LINE__)) break;

				h_spine = NULL;
			}

#if CUDA_VERSION >= 4000
			// Restore current gpu
			if (retval = util::B40CPerror(cudaSetDevice(current_gpu),
				"Spine cudaSetDevice failed: ", __FILE__, __LINE__)) break;
#endif

			gpu 			= B40C_INVALID_DEVICE;
			spine_bytes	 	= 0;

		} while (0);

		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~Spine()
	{
		HostReset();
	}


	/**
	 * Device spine storage accessor
	 */
	void* operator()()
	{
		return d_spine;
	}


	/**
	 * Sets up the spine to accommodate partials of the specified type
	 * produced/consumed by grids of the specified sweep grid size (lazily
	 * allocating it if necessary)
	 *
	 * Grows as necessary.
	 */
	template <typename T>
	cudaError_t Setup(int spine_elements)
	{
		cudaError_t retval = cudaSuccess;
		do {
			size_t problem_spine_bytes = spine_elements * sizeof(T);

			// Get current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"Spine cudaGetDevice failed: ", __FILE__, __LINE__)) break;

			// Check if big enough and if lives on proper GPU
			if ((problem_spine_bytes > spine_bytes) || (gpu != current_gpu)) {

				// Deallocate if exists
				if (retval = HostReset()) break;

				// Remember device
				gpu = current_gpu;

				// Reallocate
				spine_bytes = problem_spine_bytes;

				// Allocate on device
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_spine, spine_bytes),
					"Spine cudaMalloc d_spine failed", __FILE__, __LINE__)) break;

				if (host_shadow) {
					// Allocate pinned memory for h_spine
					int flags = cudaHostAllocMapped;
					if (retval = util::B40CPerror(cudaHostAlloc((void **)&h_spine, problem_spine_bytes, flags),
						"Spine cudaHostAlloc h_spine failed", __FILE__, __LINE__)) break;
				}
			}
		} while (0);

		return retval;
	}


	/**
	 * Syncs the shadow host spine with device spine
	 */
	cudaError_t Sync(cudaStream_t stream)
	{
		return cudaMemcpyAsync(
			h_spine,
			d_spine,
			spine_bytes,
			cudaMemcpyDeviceToHost,
			stream);
	}


	/**
	 * Syncs the shadow host spine with device spine
	 */
	cudaError_t Sync()
	{
		return cudaMemcpy(
			h_spine,
			d_spine,
			spine_bytes,
			cudaMemcpyDeviceToHost);
	}


};

} // namespace util
} // namespace b40c

