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
 * Enactor base class
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace util {



/**
 * Enactor base class
 */
class EnactorBase
{
public:

	//---------------------------------------------------------------------
	// Utility Fields
	//---------------------------------------------------------------------

	// Debug level.  If set, the enactor blocks after kernel calls to check
	// for successful launch/execution
	bool ENACTOR_DEBUG;


	// The arch version of the code for the current device that actually have
	// compiled kernels for
	int PtxVersion()
	{
		return this->cuda_props.kernel_ptx_version;
	}

	// The number of SMs on the current device
	int SmCount()
	{
		return this->cuda_props.device_props.multiProcessorCount;
	}

protected:

	template <typename MyType, typename DerivedType = void>
	struct DispatchType
	{
		typedef DerivedType Type;
	};

	template <typename MyType>
	struct DispatchType<MyType, void>
	{
		typedef MyType Type;
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Device properties
	const util::CudaProperties cuda_props;


	//---------------------------------------------------------------------
	// Tuning Utility Routines
	//---------------------------------------------------------------------

	/**
	 * Computes dynamic smem allocations to ensure all three kernels end up
	 * allocating the same amount of smem per CTA
	 */
	template <
		typename UpsweepKernelPtr,
		typename SpineKernelPtr,
		typename DownsweepKernelPtr>
	cudaError_t PadUniformSmem(
		int dynamic_smem[3],
		UpsweepKernelPtr UpsweepKernel,
		SpineKernelPtr SpineKernel,
		DownsweepKernelPtr DownsweepKernel)
	{
		cudaError_t retval = cudaSuccess;
		do {

			// Get kernel attributes
			cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs, downsweep_kernel_attrs;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepKernel),
				"EnactorBase cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, SpineKernel),
				"EnactorBase cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&downsweep_kernel_attrs, DownsweepKernel),
				"EnactorBase cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

			int max_static_smem = B40C_MAX(
				upsweep_kernel_attrs.sharedSizeBytes,
				B40C_MAX(spine_kernel_attrs.sharedSizeBytes, downsweep_kernel_attrs.sharedSizeBytes));

			dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
			dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
			dynamic_smem[2] = max_static_smem - downsweep_kernel_attrs.sharedSizeBytes;

		} while (0);

		return retval;
	}


	/**
	 * Computes dynamic smem allocations to ensure both kernels end up
	 * allocating the same amount of smem per CTA
	 */
	template <
		typename UpsweepKernelPtr,
		typename SpineKernelPtr>
	cudaError_t PadUniformSmem(
		int dynamic_smem[2],				// out param
		UpsweepKernelPtr UpsweepKernel,
		SpineKernelPtr SpineKernel)
	{
		cudaError_t retval = cudaSuccess;
		do {

			// Get kernel attributes
			cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepKernel),
				"EnactorBase cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, SpineKernel),
				"EnactorBase cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

			int max_static_smem = B40C_MAX(
				upsweep_kernel_attrs.sharedSizeBytes,
				spine_kernel_attrs.sharedSizeBytes);

			dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
			dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;

		} while (0);

		return retval;
	}


	template <typename KernelPtr>
	cudaError_t MaxCtaOccupancy(
		int &max_cta_occupancy,					// out param
		KernelPtr Kernel,
		int threads)
	{
		cudaError_t retval = cudaSuccess;
		do {
			// Get kernel attributes
			cudaFuncAttributes kernel_attrs;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&kernel_attrs, Kernel),
				"EnactorBase cudaFuncGetAttributes kernel_attrs failed", __FILE__, __LINE__)) break;

			max_cta_occupancy = B40C_MIN(
				B40C_SM_CTAS(cuda_props.device_sm_version),
				B40C_MIN(
					B40C_SM_THREADS(cuda_props.device_sm_version) / threads,
					B40C_MIN(
						(kernel_attrs.sharedSizeBytes > 0) ?
							B40C_SMEM_BYTES(cuda_props.device_sm_version) / (kernel_attrs.sharedSizeBytes) :
							B40C_SMEM_BYTES(cuda_props.device_sm_version),
						B40C_SM_REGISTERS(cuda_props.device_sm_version) / (kernel_attrs.numRegs * threads))));

			if (ENACTOR_DEBUG) printf("Occupancy:\t[sweep occupancy: %d]\n", max_cta_occupancy);

		} while (0);

		return retval;

	}

	template <
		typename UpsweepKernelPtr,
		typename DownsweepKernelPtr>
	cudaError_t MaxCtaOccupancy(
		int &max_cta_occupancy,					// out param
		UpsweepKernelPtr UpsweepKernel,
		int upsweep_threads,
		DownsweepKernelPtr DownsweepKernel,
		int downsweep_threads)
	{
		cudaError_t retval = cudaSuccess;
		do {
			int upsweep_cta_occupancy, downsweep_cta_occupancy;

			bool old_debug = ENACTOR_DEBUG;
			ENACTOR_DEBUG = false;

			if (retval = MaxCtaOccupancy(upsweep_cta_occupancy, UpsweepKernel, upsweep_threads)) break;
			if (retval = MaxCtaOccupancy(downsweep_cta_occupancy, DownsweepKernel, downsweep_threads)) break;

			ENACTOR_DEBUG = old_debug;

			if (ENACTOR_DEBUG) printf("Occupancy:\t[upsweep occupancy: %d, downsweep occupancy %d]\n",
				upsweep_cta_occupancy, downsweep_cta_occupancy);

			max_cta_occupancy = B40C_MIN(upsweep_cta_occupancy, downsweep_cta_occupancy);

		} while (0);

		return retval;

	}



	/**
	 * Returns the number of threadblocks to launch for the given problem size.
	 * Does not exceed the full-occupancy on the current device or the
	 * optional max_grid_size limit.
	 *
	 * Useful for kernels that work-steal or use global barriers (where
	 * over-subscription is not ideal or allowed)
	 */
	int OccupiedGridSize(
		int schedule_granularity,
		int max_cta_occupancy,
		int num_elements,
		int max_grid_size = 0)
	{
		int grid_size;

		if (max_grid_size > 0) {
			grid_size = max_grid_size;
		} else {
			grid_size = cuda_props.device_props.multiProcessorCount * max_cta_occupancy;
		}

		// Reduce if we have less work than we can divide up among this
		// many CTAs
		int grains = (num_elements + schedule_granularity - 1) / schedule_granularity;
		if (grid_size > grains) {
			grid_size = grains;
		}


		return grid_size;
	}


	/**
	 * Returns the number of threadblocks to launch for the given problem size.
	 * May over/under subscribe the current device based upon heuristics.  Does not
	 * the optional max_grid_size limit.
	 *
	 * Useful for kernels that evenly divide up the work amongst threadblocks.
	 */
	int OversubscribedGridSize(
		int schedule_granularity,
		int max_cta_occupancy,
		int num_elements,
		int max_grid_size)
	{
		int grid_size;

		if (max_grid_size > 0) {

			grid_size = max_grid_size;

		} else {

			if (cuda_props.device_sm_version < 120) {

				// G80/G90: double CTA occupancy times SM count
				grid_size = cuda_props.device_props.multiProcessorCount * max_cta_occupancy * 2;

			} else if (cuda_props.device_sm_version < 200) {

				// GT200: Special sauce

				// Start with with full downsweep occupancy of all SMs
				grid_size =
					cuda_props.device_props.multiProcessorCount * max_cta_occupancy;

				// Increase by default every 64 million key-values
				int step = 1024 * 1024 * 64;
				grid_size *= (num_elements + step - 1) / step;

				double multiplier1 = 4.0;
				double multiplier2 = 16.0;

				double delta1 = 0.068;
				double delta2 = 0.1285;

				int dividend = (num_elements + 512 - 1) / 512;

				int bumps = 0;
				while(true) {

					if (grid_size <= cuda_props.device_props.multiProcessorCount) {
						break;
					}

					double quotient = ((double) dividend) / (multiplier1 * grid_size);
					quotient -= (int) quotient;

					if ((quotient > delta1) && (quotient < 1 - delta1)) {

						quotient = ((double) dividend) / (multiplier2 * grid_size / 3.0);
						quotient -= (int) quotient;

						if ((quotient > delta2) && (quotient < 1 - delta2)) {
							break;
						}
					}

					if (bumps == 3) {
						// Bump it down by 27
						grid_size -= 27;
						bumps = 0;
					} else {
						// Bump it down by 1
						grid_size--;
						bumps++;
					}
				}

			} else {

				// GF10x
				if (cuda_props.device_sm_version == 210) {
					// GF110
					grid_size = 4 * (cuda_props.device_props.multiProcessorCount * max_cta_occupancy);
				} else {
					// Anything but GF110
					grid_size = 4 * (cuda_props.device_props.multiProcessorCount * max_cta_occupancy) - 2;
				}
			}
		}


		// Reduce if we have less work than we can divide up among this
		// many CTAs
		int grains = (num_elements + schedule_granularity - 1) / schedule_granularity;
		if (grid_size > grains) {
			grid_size = grains;
		}

		return grid_size;
	}


	/**
	 * Returns the number of threadblocks to launch for the given problem size.
	 */
	int GridSize(
		bool oversubscribed,
		int schedule_granularity,
		int max_cta_occupancy,
		int num_elements,
		int max_grid_size)
	{
		return (oversubscribed) ?
			OversubscribedGridSize(
				schedule_granularity,
				max_cta_occupancy,
				num_elements,
				max_grid_size) :
			OccupiedGridSize(
				schedule_granularity,
				max_cta_occupancy,
				num_elements,
				max_grid_size);
	}

	//-----------------------------------------------------------------------------
	// Debug Utility Routines
	//-----------------------------------------------------------------------------

	/**
	 * Utility method to display the contents of a device array
	 */
	template <typename T>
	void DisplayDeviceResults(
		T *d_data,
		size_t num_elements)
	{
		// Allocate array on host and copy back
		T *h_data = (T*) malloc(num_elements * sizeof(T));
		cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

		// Display data
		for (int i = 0; i < num_elements; i++) {
			PrintValue(h_data[i]);
			printf(", ");
		}
		printf("\n\n");

		// Cleanup
		if (h_data) free(h_data);
	}


	/**
	 * Prints key size information
	 */
	template <typename KernelPolicy>
	bool PrintKeySizeInfo(typename KernelPolicy::KeyType *ptr) {
		printf("%lu byte keys, ", (unsigned long) sizeof(typename KernelPolicy::KeyType));
		return true;
	}
	template <typename KernelPolicy>
	bool PrintKeySizeInfo(...) {return false;}

	/**
	 * Prints value size information
	 */
	template <typename KernelPolicy>
	bool PrintValueSizeInfo(typename KernelPolicy::ValueType *ptr) {
		if (!util::Equals<typename KernelPolicy::ValueType, util::NullType>::VALUE) {
			printf("%lu byte values, ", (unsigned long) sizeof(typename KernelPolicy::ValueType));
		}
		return true;
	}
	template <typename KernelPolicy>
	bool PrintValueSizeInfo(...) {return false;}

	/**
	 * Prints T size information
	 */
	template <typename KernelPolicy>
	bool PrintTSizeInfo(typename KernelPolicy::T *ptr) {
		printf("%lu byte data, ", (unsigned long) sizeof(typename KernelPolicy::T));
		return true;
	}
	template <typename KernelPolicy>
	bool PrintTSizeInfo(...) {return false;}

	/**
	 * Prints workstealing information
	 */
	template <typename KernelPolicy>
	bool PrintWorkstealingInfo(int (*data)[KernelPolicy::WORK_STEALING + 1]) {
		printf("%sworkstealing, ", (KernelPolicy::WORK_STEALING) ? "" : "non-");
		return true;
	}
	template <typename KernelPolicy>
	bool PrintWorkstealingInfo(...) {return false;}

	/**
	 * Prints work distribution information
	 */
	template <typename KernelPolicy, typename SizeT>
	void PrintWorkInfo(util::CtaWorkDistribution<SizeT> &work)
	{
		printf("Work: \t\t[");
		if (PrintKeySizeInfo<KernelPolicy>(NULL)) {
			PrintValueSizeInfo<KernelPolicy>(NULL);
		} else {
			PrintTSizeInfo<KernelPolicy>(NULL);
		}
		PrintWorkstealingInfo<KernelPolicy>(NULL);

		unsigned long last_grain_elements =
			(work.num_elements & (KernelPolicy::SCHEDULE_GRANULARITY - 1));
		if (last_grain_elements == 0) last_grain_elements = KernelPolicy::SCHEDULE_GRANULARITY;

		printf("%lu byte SizeT, "
				"%lu elements, "
				"%lu-element granularity, "
				"%lu total grains, "
				"%lu grains per cta, "
				"%lu extra grains, "
				"%lu last-grain elements]\n",
			(unsigned long) sizeof(SizeT),
			(unsigned long) work.num_elements,
			(unsigned long) KernelPolicy::SCHEDULE_GRANULARITY,
			(unsigned long) work.total_grains,
			(unsigned long) work.grains_per_cta,
			(unsigned long) work.extra_grains,
			(unsigned long) last_grain_elements);
		fflush(stdout);
	}


	/**
	 * Prints pass information
	 */
	template <typename UpsweepPolicy, typename SizeT>
	void PrintPassInfo(
		util::CtaWorkDistribution<SizeT> &work,
		int spine_elements = 0)
	{
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d, SM count: %d]\n",
			cuda_props.device_sm_version,
			cuda_props.kernel_ptx_version,
			cuda_props.device_props.multiProcessorCount);
		PrintWorkInfo<UpsweepPolicy, SizeT>(work);
		printf("Upsweep: \t[sweep_grid_size: %d, threads %d, tile_elements: %d]\n",
			work.grid_size,
			UpsweepPolicy::THREADS,
			UpsweepPolicy::TILE_ELEMENTS);
		fflush(stdout);
	}

	/**
	 * Prints pass information
	 */
	template <typename UpsweepPolicy, typename SpinePolicy, typename SizeT>
	void PrintPassInfo(
		util::CtaWorkDistribution<SizeT> &work,
		int spine_elements = 0)
	{
		PrintPassInfo<UpsweepPolicy>(work);
		printf("Spine: \t\t[threads: %d, spine_elements: %d, tile_elements: %d]\n",
			SpinePolicy::THREADS,
			spine_elements,
			SpinePolicy::TILE_ELEMENTS);
		fflush(stdout);
	}

	/**
	 * Prints pass information
	 */
	template <typename UpsweepPolicy, typename SpinePolicy, typename DownsweepPolicy, typename SizeT>
	void PrintPassInfo(
		util::CtaWorkDistribution<SizeT> &work,
		int spine_elements = 0)
	{
		PrintPassInfo<UpsweepPolicy, SpinePolicy>(work, spine_elements);
		printf("Downsweep: \t[sweep_grid_size: %d, threads %d, tile_elements: %d]\n",
			work.grid_size,
			DownsweepPolicy::THREADS,
			DownsweepPolicy::TILE_ELEMENTS);
		fflush(stdout);
	}




	//---------------------------------------------------------------------
	// Constructors
	//---------------------------------------------------------------------

	EnactorBase() :
#if	defined(__THRUST_SYNCHRONOUS) || defined(DEBUG) || defined(_DEBUG)
			ENACTOR_DEBUG(true)
#else
			ENACTOR_DEBUG(false)
#endif
		{}

};


} // namespace util
} // namespace b40c

