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
 * Tuning policy for partitioning upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace partition {
namespace upsweep {


/**
 * Partitioning upsweep reduction tuning policy .  This type encapsulates our
 * kernel-tuning parameters (they are reflected via the static fields).
 *
 * Note that the number of bins is considered a tuning parameter as opposed
 * to a problem-type parameter.  (We can simply enact subsequent sub-partitioning
 * passes to effect more bins).
 *  
 * The kernel is specialized for problem-type, SM-version, etc. by declaring
 * it with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in 
 * expanding/unrolling the kernel code for specific architectures and problem 
 * types.    
 */
template <
	// Problem type
	typename ProblemType,

	int _CUDA_ARCH,
	bool _CHECK_ALIGNMENT,
	int _LOG_BINS,
	int _LOG_SCHEDULE_GRANULARITY,
	int _MIN_CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	util::io::ld::CacheModifier _READ_MODIFIER,
	util::io::st::CacheModifier _WRITE_MODIFIER>

struct TuningPolicy : ProblemType
{
	enum {
		CUDA_ARCH									= _CUDA_ARCH,
		CHECK_ALIGNMENT								= _CHECK_ALIGNMENT,
		LOG_BINS									= _LOG_BINS,
		LOG_SCHEDULE_GRANULARITY					= _LOG_SCHEDULE_GRANULARITY,
		MIN_CTA_OCCUPANCY  							= _MIN_CTA_OCCUPANCY,
		LOG_THREADS 								= _LOG_THREADS,
		LOG_LOAD_VEC_SIZE  							= _LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE 							= _LOG_LOADS_PER_TILE,

		SCHEDULE_GRANULARITY						= 1 << LOG_SCHEDULE_GRANULARITY,
		THREADS										= 1 << LOG_THREADS,
		TILE_ELEMENTS								= 1 << (LOG_THREADS + LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE),
	};

	static const util::io::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;
};


} // namespace upsweep
} // namespace partition
} // namespace b40c

