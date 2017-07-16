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
 * Tuning policy for radix sort downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/partition/downsweep/tuning_policy.cuh>

namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Radix sort downsweep scan tuning policy.
 * 
 * See constraints in base class.
 */
template <
	typename ProblemType,

	int CUDA_ARCH,
	bool CHECK_ALIGNMENT,
	int LOG_BINS,
	int LOG_SCHEDULE_GRANULARITY,
	int MIN_CTA_OCCUPANCY,
	int LOG_THREADS,
	int LOG_LOAD_VEC_SIZE,
	int LOG_LOADS_PER_CYCLE,
	int LOG_CYCLES_PER_TILE,
	int LOG_RAKING_THREADS,
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	partition::downsweep::ScatterStrategy SCATTER_STRATEGY,
	bool _EARLY_EXIT>

struct TuningPolicy :
	partition::downsweep::TuningPolicy <
		ProblemType,
		CUDA_ARCH,
		CHECK_ALIGNMENT,
		LOG_BINS,
		LOG_SCHEDULE_GRANULARITY,
		MIN_CTA_OCCUPANCY,
		LOG_THREADS,
		LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_CYCLE,
		LOG_CYCLES_PER_TILE,
		LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SCATTER_STRATEGY>
{
	enum {
		EARLY_EXIT								= _EARLY_EXIT,
	};
};

} // namespace downsweep
} // namespace radix_sort
} // namespace b40c

