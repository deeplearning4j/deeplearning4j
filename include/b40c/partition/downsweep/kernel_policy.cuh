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
 * Configuration policy for partitioning downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/srts_grid.cuh>

namespace b40c {
namespace partition {
namespace downsweep {


/**
 * A detailed partitioning downsweep kernel configuration policy type that specializes
 * kernel code for a specific pass.  It encapsulates tuning configuration policy
 * details derived from TuningPolicy
 */
template <typename TuningPolicy>
struct KernelPolicy : TuningPolicy
{
	typedef typename TuningPolicy::SizeT 		SizeT;
	typedef typename TuningPolicy::KeyType 		KeyType;
	typedef typename TuningPolicy::ValueType 	ValueType;

	enum {

		BINS 							= 1 << TuningPolicy::LOG_BINS,
		THREADS							= 1 << TuningPolicy::LOG_THREADS,

		LOG_WARPS						= TuningPolicy::LOG_THREADS - B40C_LOG_WARP_THREADS(TuningPolicy::CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOAD_VEC_SIZE					= 1 << TuningPolicy::LOG_LOAD_VEC_SIZE,
		LOADS_PER_CYCLE					= 1 << TuningPolicy::LOG_LOADS_PER_CYCLE,
		CYCLES_PER_TILE					= 1 << TuningPolicy::LOG_CYCLES_PER_TILE,

		LOG_LOADS_PER_TILE				= TuningPolicy::LOG_LOADS_PER_CYCLE +
												TuningPolicy::LOG_CYCLES_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_CYCLE_ELEMENTS				= TuningPolicy::LOG_THREADS +
												TuningPolicy::LOG_LOADS_PER_CYCLE +
												TuningPolicy::LOG_LOAD_VEC_SIZE,
		CYCLE_ELEMENTS					= 1 << LOG_CYCLE_ELEMENTS,

		LOG_TILE_ELEMENTS				= TuningPolicy::LOG_CYCLES_PER_TILE + LOG_CYCLE_ELEMENTS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_TILE_ELEMENTS - TuningPolicy::LOG_THREADS,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,
	
		LOG_SCAN_LANES_PER_LOAD			= B40C_MAX((TuningPolicy::LOG_BINS - 2), 0),		// Always at least one lane per load
		SCAN_LANES_PER_LOAD				= 1 << LOG_SCAN_LANES_PER_LOAD,

		LOG_SCAN_LANES_PER_CYCLE		= TuningPolicy::LOG_LOADS_PER_CYCLE + LOG_SCAN_LANES_PER_LOAD,
		SCAN_LANES_PER_CYCLE			= 1 << LOG_SCAN_LANES_PER_CYCLE,
	};


	// Smem raking grid type for reducing and scanning a cycle of 
	// (bins/4) lanes of composite 8-bit bin counters
	typedef util::RakingGrid<
		TuningPolicy::CUDA_ARCH,
		int,									// Partial type
		TuningPolicy::LOG_THREADS,				// Depositing threads (the CTA size)
		LOG_SCAN_LANES_PER_CYCLE,				// Lanes (the number of loads)
		TuningPolicy::LOG_RAKING_THREADS,		// Raking threads
		false>									// Any prefix dependences between lanes are explicitly managed
			Grid;

	
	/**
	 * Shared storage for partitioning upsweep
	 */
	struct SmemStorage
	{
		volatile int 					lanes_warpscan[SCAN_LANES_PER_CYCLE][3][Grid::RAKING_THREADS_PER_LANE];		// One warpscan per lane
		volatile int 					bin_warpscan[2][BINS];

		SizeT							bin_carry[BINS];
		SizeT 							bin_prefixes[CYCLES_PER_TILE][LOADS_PER_CYCLE][BINS];
		union {
			int 						lane_totals[CYCLES_PER_TILE][SCAN_LANES_PER_CYCLE][2];
			unsigned char				lane_totals_c[CYCLES_PER_TILE][LOADS_PER_CYCLE][SCAN_LANES_PER_LOAD][2][4];
		};

		bool 							non_trivial_pass;
		util::CtaWorkLimits<SizeT> 		work_limits;

		union {
			int 						raking_lanes[Grid::RAKING_ELEMENTS];
			KeyType 					key_exchange[TILE_ELEMENTS + 1];			// Last index is for invalid elements to be culled (if any)
			ValueType 					value_exchange[TILE_ELEMENTS + 1];
		};
	};

	enum {
		THREAD_OCCUPANCY					= B40C_SM_THREADS(TuningPolicy::CUDA_ARCH) >> TuningPolicy::LOG_THREADS,
		SMEM_OCCUPANCY						= B40C_SMEM_BYTES(TuningPolicy::CUDA_ARCH) / sizeof(SmemStorage),
		MAX_CTA_OCCUPANCY					= B40C_MIN(B40C_SM_CTAS(TuningPolicy::CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),

		VALID								= (MAX_CTA_OCCUPANCY > 0),
	};


	__device__ __forceinline__ static void PreprocessKey(KeyType &key) {}

	__device__ __forceinline__ static void PostprocessKey(KeyType &key) {}
};
	


} // namespace downsweep
} // namespace partition
} // namespace b40c

