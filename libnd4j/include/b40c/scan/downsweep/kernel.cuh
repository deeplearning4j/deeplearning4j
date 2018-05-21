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
 * Scan downsweep scan kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/srts_details.cuh>

#include <b40c/scan/downsweep/cta.cuh>

namespace b40c {
namespace scan {
namespace downsweep {


/**
 * Scan downsweep scan pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void DownsweepPass(
	typename KernelPolicy::T 									*d_in,
	typename KernelPolicy::T 									*d_out,
	typename KernelPolicy::T 									*d_spine,
	typename KernelPolicy::ReductionOp 							scan_op,
	typename KernelPolicy::IdentityOp 							identity_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
	typename KernelPolicy::SmemStorage							&smem_storage)
{
	typedef Cta<KernelPolicy> 				Cta;
	typedef typename KernelPolicy::T 		T;
	typedef typename KernelPolicy::SizeT 	SizeT;

	// Obtain exclusive spine partial
	T spine_partial;
	util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(
		spine_partial, d_spine + blockIdx.x);

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_in,
		d_out,
		scan_op,
		identity_op,
		spine_partial);

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelPolicy::LOG_TILE_ELEMENTS,
		KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	cta.ProcessWorkRange(work_limits);
}


/**
 * Scan downsweep scan kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::T 				*d_in,
	typename KernelPolicy::T 				*d_out,
	typename KernelPolicy::T 				*d_spine,
	typename KernelPolicy::ReductionOp 		scan_op,
	typename KernelPolicy::IdentityOp 		identity_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	DownsweepPass<KernelPolicy>(
		d_in,
		d_out,
		d_spine,
		scan_op,
		identity_op,
		work_decomposition,
		smem_storage);
}

} // namespace downsweep
} // namespace scan
} // namespace b40c

