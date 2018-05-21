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
 * Upsweep kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>

#include <b40c/reduction/cta.cuh>
#include <b40c/scan/upsweep/cta.cuh>

namespace b40c {
namespace scan {
namespace upsweep {


/**
 * Upsweep reduction pass (specialized to support non-commutative operators)
 */
template <
	typename KernelPolicy,
	bool COMMUTATIVE = KernelPolicy::COMMUTATIVE>
struct UpsweepPass
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::T 									*d_in,
		typename KernelPolicy::T 									*d_out,
		typename KernelPolicy::ReductionOp 							scan_op,
		typename KernelPolicy::IdentityOp 							identity_op,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
		typename KernelPolicy::SmemStorage							&smem_storage)
	{
		typedef Cta<KernelPolicy>					Cta;
		typedef typename KernelPolicy::SizeT 		SizeT;

		// CTA processing abstraction
		Cta cta(
			smem_storage,
			d_in,
			d_out,
			scan_op,
			identity_op);

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelPolicy::LOG_TILE_ELEMENTS,
			KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

		// Quit if we're the last threadblock (no need for it in upsweep).
		if (work_limits.last_block) {
			return;
		}

		cta.ProcessWorkRange(work_limits);
	}
};


/**
 * Upsweep reduction pass (specialized for commutative operators)
 */
template <typename KernelPolicy>
struct UpsweepPass<KernelPolicy, true>
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::T 									*d_in,
		typename KernelPolicy::T 									*d_out,
		typename KernelPolicy::ReductionOp 							scan_op,
		typename KernelPolicy::IdentityOp 							identity_op,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
		typename KernelPolicy::SmemStorage							&smem_storage)
	{
		typedef reduction::Cta<KernelPolicy>		Cta;
		typedef typename KernelPolicy::SizeT 		SizeT;

		// CTA processing abstraction
		Cta cta(
			smem_storage,
			d_in,
			d_out,
			scan_op);

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelPolicy::LOG_TILE_ELEMENTS,
			KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

		// Quit if we're the last threadblock (no need for it in upsweep).
		if (work_limits.last_block) {
			return;
		}

		cta.ProcessWorkRange(work_limits);
	}
};



/******************************************************************************
 * Upsweep Reduction Kernel Entrypoint
 ******************************************************************************/

/**
 * Upsweep reduction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::T 									*d_in,
	typename KernelPolicy::T 									*d_spine,
	typename KernelPolicy::ReductionOp 							scan_op,
	typename KernelPolicy::IdentityOp 							identity_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	UpsweepPass<KernelPolicy>::Invoke(
		d_in,
		d_spine,
		scan_op,
		identity_op,
		work_decomposition,
		smem_storage);
}


} // namespace upsweep
} // namespace scan
} // namespace b40c

