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
 * Radix sort upsweep reduction kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/radix_sort/upsweep/cta.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {


/**
 * Radix sort upsweep reduction pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void UpsweepPass(
	int 									*&d_selectors,
	typename KernelPolicy::SizeT 			*&d_spine,
	typename KernelPolicy::KeyType 			*&d_in_keys,
	typename KernelPolicy::KeyType 			*&d_out_keys,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
	typename KernelPolicy::SmemStorage		&smem_storage)
{
	typedef Cta<KernelPolicy> 						Cta;
	typedef typename KernelPolicy::KeyType 			KeyType;
	typedef typename KernelPolicy::SizeT 			SizeT;
	
	// Determine where to read our input

	bool selector = ((KernelPolicy::EARLY_EXIT) && ((KernelPolicy::CURRENT_PASS != 0) && (d_selectors[KernelPolicy::CURRENT_PASS & 0x1]))) ||
		(KernelPolicy::CURRENT_PASS & 0x1);
	KeyType *d_keys = (selector) ? d_out_keys : d_in_keys;

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelPolicy::LOG_TILE_ELEMENTS,
		KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_keys,
		d_spine);
	
	// Accumulate digit counts for all tiles
	cta.ProcessWorkRange(work_limits);
}


/**
 * Radix sort upsweep reduction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	int 								*d_selectors,
	typename KernelPolicy::SizeT 		*d_spine,
	typename KernelPolicy::KeyType 		*d_in_keys,
	typename KernelPolicy::KeyType 		*d_out_keys,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> work_decomposition)
{
	// Shared memory pool
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	UpsweepPass<KernelPolicy>(
		d_selectors,
		d_spine,
		d_in_keys,
		d_out_keys,
		work_decomposition,
		smem_storage);
}


} // namespace upsweep
} // namespace radix_sort
} // namespace b40c

