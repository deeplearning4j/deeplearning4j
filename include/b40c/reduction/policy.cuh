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
 * Unified reduction policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/reduction/kernel_policy.cuh>
#include <b40c/reduction/upsweep/kernel.cuh>
#include <b40c/reduction/spine/kernel.cuh>

namespace b40c {
namespace reduction {


/**
 * Unified reduction policy type.
 *
 * In addition to kernel tuning parameters that guide the kernel compilation for
 * upsweep and spine kernels, this type includes enactor tuning parameters that
 * define kernel-dispatch policy.  By encapsulating all of the kernel tuning policies,
 * we assure operational consistency across all kernels.
 */
template <
	// ProblemType type parameters
	typename ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Common tunable params
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	bool WORK_STEALING,
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE,
	bool _OVERSUBSCRIBED_GRID_SIZE,

	// Upsweep tunable params
	int UPSWEEP_MIN_CTA_OCCUPANCY,
	int UPSWEEP_LOG_THREADS,
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,

	// Spine tunable params
	int SPINE_LOG_THREADS,
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE>

struct Policy : ProblemType
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename ProblemType::T 				T;
	typedef typename ProblemType::SizeT 			SizeT;
	typedef typename ProblemType::ReductionOp 		ReductionOp;

	typedef void (*UpsweepKernelPtr)(T*, T*, ReductionOp, util::CtaWorkDistribution<SizeT>, util::CtaWorkProgress);
	typedef void (*SpineKernelPtr)(T*, T*, SizeT, ReductionOp);
	typedef void (*SingleKernelPtr)(T*, T*, SizeT, ReductionOp);

	//---------------------------------------------------------------------
	// Kernel Policies
	//---------------------------------------------------------------------

	/**
	 * Kernel config for the upsweep reduction kernel
	 */
	typedef KernelPolicy <
		ProblemType,
		CUDA_ARCH,
		true,								// Check alignment
		UPSWEEP_MIN_CTA_OCCUPANCY,
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		WORK_STEALING,
		UPSWEEP_LOG_LOADS_PER_TILE + UPSWEEP_LOG_LOAD_VEC_SIZE + UPSWEEP_LOG_THREADS >
			Upsweep;

	/**
	 * Kernel config for the spine reduction kernel
	 */
	typedef KernelPolicy <
		ProblemType,
		CUDA_ARCH,
		false,								// Do not check alignment
		1,									// Only a single-CTA grid
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		false,								// Workstealing makes no sense in a single-CTA grid
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS>
			Spine;

	/**
	 * Kernel config for a one-level pass using the spine reduction kernel
	 */
	typedef KernelPolicy <
		ProblemType,
		CUDA_ARCH,
		true,								// Check alignment
		1,									// Only a single-CTA grid
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		false,								// Workstealing makes no sense in a single-CTA grid
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS>
			Single;


	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	static UpsweepKernelPtr UpsweepKernel() {
		return upsweep::Kernel<Upsweep>;
	}

	static SpineKernelPtr SpineKernel() {
		return spine::Kernel<Spine>;
	}

	static SingleKernelPtr SingleKernel() {
		return spine::Kernel<Single>;
	}

	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	enum {
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
		OVERSUBSCRIBED_GRID_SIZE	= _OVERSUBSCRIBED_GRID_SIZE,
		VALID 						= Upsweep::VALID & Spine::VALID
	};


	static void Print()
	{
		// ProblemType type parameters
		printf("%d, ", sizeof(T));
		printf("%d, ", sizeof(SizeT));
		printf("%d, ", CUDA_ARCH);

		// Common tunable params
		printf("%s, ", CacheModifierToString(READ_MODIFIER));
		printf("%s, ", CacheModifierToString(WRITE_MODIFIER));
		printf("%s, ", (WORK_STEALING) ? "true" : "false");
		printf("%s ", (_UNIFORM_SMEM_ALLOCATION) ? "true" : "false");
		printf("%s ", (_UNIFORM_GRID_SIZE) ? "true" : "false");
		printf("%s ", (_OVERSUBSCRIBED_GRID_SIZE) ? "true" : "false");

		// Upsweep tunable params
		printf("%d, ", UPSWEEP_MIN_CTA_OCCUPANCY);
		printf("%d, ", UPSWEEP_LOG_THREADS);
		printf("%d, ", UPSWEEP_LOG_LOAD_VEC_SIZE);
		printf("%d, ", UPSWEEP_LOG_LOADS_PER_TILE);

		// Spine tunable params
		printf("%d, ", SPINE_LOG_THREADS);
		printf("%d, ", SPINE_LOG_LOAD_VEC_SIZE);
		printf("%d, ", SPINE_LOG_LOADS_PER_TILE);
	}
};
		

}// namespace reduction
}// namespace b40c

