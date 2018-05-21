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
 * Composite-counter functionality for partitioning upsweep reduction kernels
 ******************************************************************************/

#pragma once

namespace b40c {
namespace partition {
namespace upsweep {


/**
 * Shared-memory lanes of composite counters.
 *
 * We keep our per-thread composite counters in smem because we simply don't
 * have enough register storage.
 */
template <typename KernelPolicy>
struct CompostiteCounters
{
	enum {
		COMPOSITE_LANES = KernelPolicy::COMPOSITE_LANES,
	};


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate lane
	 */
	template <int LANE, int dummy = 0>
	struct Iterate
	{
		// ResetCompositeCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCompositeCounters(Cta *cta)
		{
			cta->smem_storage.composite_counters.words[LANE][threadIdx.x] = 0;
			Iterate<LANE + 1>::ResetCompositeCounters(cta);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<COMPOSITE_LANES, dummy>
	{
		// ResetCompositeCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCompositeCounters(Cta *cta) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Resets our composite-counter lanes
	 */
	template <typename Cta>
	__device__ __forceinline__ void ResetCompositeCounters(Cta *cta)
	{
		Iterate<0>::ResetCompositeCounters(cta);
	}
};


} // namespace upsweep
} // namespace partition
} // namespace b40c

