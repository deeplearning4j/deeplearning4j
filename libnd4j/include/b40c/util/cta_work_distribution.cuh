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
 * Work Management Datastructures
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {


/**
 * Structure for describing the limits of work-processing for a given CTA
 */
template <typename SizeT>
struct CtaWorkLimits
{
	SizeT 	offset;				// Offset at which this CTA begins processing
	SizeT 	elements;			// Total number of elements for this CTA to process
	SizeT 	guarded_offset; 	// Offset of final, partially-full tile (requires guarded loads)
	SizeT 	guarded_elements;	// Number of elements in partially-full tile
	SizeT 	out_of_bounds;		// Offset at which this CTA is out-of-bounds
	bool	last_block;			// If this block is the last block in the grid with any work

	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ CtaWorkLimits() {}

	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ CtaWorkLimits(
		SizeT 	offset,
		SizeT 	elements,
		SizeT 	guarded_offset,
		SizeT 	guarded_elements,
		SizeT 	out_of_bounds,
		bool	last_block) :
			offset(offset),
			elements(elements),
			guarded_offset(guarded_offset),
			guarded_elements(guarded_elements),
			out_of_bounds(out_of_bounds),
			last_block(last_block)
		{}
};


/**
 * Description of work distribution amongst CTAs
 *
 * A given threadblock may receive one of three different amounts of 
 * work: "big", "normal", and "last".  The big workloads are one
 * grain greater than the normal, and the last workload 
 * does the extra work.
 */
template <typename SizeT> 		// Integer type for indexing into problem arrays (e.g., int, long long, etc.)
struct CtaWorkDistribution
{
	SizeT 	num_elements;		// Number of elements in the problem
	SizeT 	total_grains;		// Number of "grain" blocks to break the problem into (round up)
	SizeT 	grains_per_cta;		// Number of "grain" blocks per CTA
	SizeT 	extra_grains;		// Number of CTAs having one extra "grain block"
	int 	grid_size;			// Number of CTAs


	/**
	 * Initializer
	 */
	template <int LOG_SCHEDULE_GRANULARITY>
	__host__ __device__ __forceinline__ void Init(
		SizeT num_elements,
		int grid_size)
	{
		const int SCHEDULE_GRANULARITY = 1 << LOG_SCHEDULE_GRANULARITY;

		this->num_elements 		= num_elements;
		this->total_grains 		= ((num_elements + SCHEDULE_GRANULARITY - 1) >> LOG_SCHEDULE_GRANULARITY);	// round up
		this->grains_per_cta 	= total_grains / grid_size;													// round down for the ks
		this->extra_grains 		= total_grains - (grains_per_cta * grid_size);								// the CTAs with +1 grains
		this->grid_size 		= grid_size;
	}

	/**
	 * Initializer
	 */
	__host__ __device__ __forceinline__ void Init(
		SizeT num_elements,
		int grid_size,
		int log_schedule_granularity)
	{
		const int SCHEDULE_GRANULARITY = 1 << log_schedule_granularity;

		this->num_elements 		= num_elements;
		this->total_grains 		= ((num_elements + SCHEDULE_GRANULARITY - 1) >> log_schedule_granularity);	// round up
		this->grains_per_cta 	= total_grains / grid_size;													// round down for the ks
		this->extra_grains 		= total_grains - (grains_per_cta * grid_size);								// the CTAs with +1 grains
		this->grid_size 		= grid_size;
	}


	/**
	 * Computes work limits for the current CTA
	 */	
	template <
		int LOG_TILE_ELEMENTS,			// CTA tile size, i.e., granularity by which the CTA processes work
		int LOG_SCHEDULE_GRANULARITY>	// Problem granularity by which work is distributed amongst CTA threadblocks
	__host__ __device__ __forceinline__ void GetCtaWorkLimits(
		CtaWorkLimits<SizeT> &work_limits)	// Out param
	{
		const int TILE_ELEMENTS 				= 1 << LOG_TILE_ELEMENTS;
		
		// Compute number of elements and offset at which to start tile processing
		if (blockIdx.x < extra_grains) {

			// This CTA gets grains_per_cta+1 grains
			work_limits.elements = (grains_per_cta + 1) << LOG_SCHEDULE_GRANULARITY;
			work_limits.offset = work_limits.elements * blockIdx.x;

		} else if (blockIdx.x < total_grains) {

			// This CTA gets grains_per_cta grains
			work_limits.elements = grains_per_cta << LOG_SCHEDULE_GRANULARITY;
			work_limits.offset = (work_limits.elements * blockIdx.x) + (extra_grains << LOG_SCHEDULE_GRANULARITY);

		} else {

			// This CTA gets no work (problem small enough that some CTAs don't even a single grain)
			work_limits.elements = 0;
			work_limits.offset = 0;
		}

		// The offset at which this CTA is out-of-bounds
		work_limits.out_of_bounds = work_limits.offset + work_limits.elements;

		// Correct for the case where the last CTA having work has rounded its last grain up past the end
		if (work_limits.last_block = work_limits.out_of_bounds >= num_elements) {
			work_limits.out_of_bounds = num_elements;
			work_limits.elements = num_elements - work_limits.offset;
		}

		// The number of extra guarded-load elements to process afterward (always
		// less than a full tile)
		work_limits.guarded_elements = work_limits.elements & (TILE_ELEMENTS - 1);

		// The tile-aligned limit for full-tile processing
		work_limits.guarded_offset = work_limits.out_of_bounds - work_limits.guarded_elements;
	}


	/**
	 * Print to stdout
	 */
	void Print()
	{
		printf("num_elements: %lu, total_grains: %lu, grains_per_cta: %lu, extra_grains: %lu, grid_size: %lu\n",
			(unsigned long) num_elements,
			(unsigned long) total_grains,
			(unsigned long) grains_per_cta,
			(unsigned long) extra_grains,
			(unsigned long) grid_size);
	}
};


} // namespace util
} // namespace b40c

