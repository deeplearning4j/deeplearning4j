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
 * Autotuned radix sort policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/radix_sort/upsweep/kernel.cuh>
#include <b40c/radix_sort/upsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel.cuh>
#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/policy.cuh>

#include <b40c/scan/spine/kernel.cuh>

namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Genre enumerations to classify problems by
 ******************************************************************************/

/**
 * Enumeration of problem-size genres that we may have tuned for
 */
enum ProbSizeGenre
{
	UNKNOWN_SIZE = -1,			// Not actually specialized on: the enactor should use heuristics to select another size genre
	SMALL_SIZE,					// Tuned @ 128KB input
	LARGE_SIZE					// Tuned @ 128MB input
};


/**
 * Enumeration of architecture-family genres that we have tuned for below
 */
enum ArchGenre
{
	SM20 	= 200,
	SM13	= 130,
	SM10	= 100
};


/**
 * Enumeration of type size genres
 */
enum TypeSizeGenre
{
	TINY_TYPE,
	SMALL_TYPE,
	MEDIUM_TYPE,
	LARGE_TYPE
};


/**
 * Autotuning policy genre, to be specialized
 */
template <
	// Problem and machine types
	typename ProblemType,
	int CUDA_ARCH,

	// Genres to specialize upon
	ProbSizeGenre PROB_SIZE_GENRE,
	ArchGenre ARCH_GENRE,
	TypeSizeGenre TYPE_SIZE_GENRE,
	TypeSizeGenre OFFSET_SIZE_GENRE,
	TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre;


/******************************************************************************
 * Classifiers for identifying classification genres
 ******************************************************************************/

/**
 * Classifies a given CUDA_ARCH into an architecture-family genre
 */
template <int CUDA_ARCH>
struct ArchClassifier
{
	static const ArchGenre GENRE 			=	(CUDA_ARCH < SM13) ? SM10 :
												(CUDA_ARCH < SM20) ? SM13 : SM20;
};


/**
 * Classifies the problem type(s) into a type-size genre
 */
template <typename ProblemType>
struct TypeSizeClassifier
{
	enum {
		KEYS_ROUNDED_SIZE		= 1 << util::Log2<sizeof(typename ProblemType::KeyType)>::VALUE,	// Round up to the nearest arch subword
		VALUES_ROUNDED_SIZE		= 1 << util::Log2<sizeof(typename ProblemType::ValueType)>::VALUE,
		MAX_ROUNDED_SIZE		= B40C_MAX(KEYS_ROUNDED_SIZE, VALUES_ROUNDED_SIZE),
	};

	static const TypeSizeGenre GENRE 		= (MAX_ROUNDED_SIZE < 8) ? MEDIUM_TYPE : LARGE_TYPE;
};


/**
 * Classifies the offset type into a type-size genre
 */
template <typename ProblemType>
struct OffsetSizeClassifier
{
	static const TypeSizeGenre GENRE 		= (sizeof(typename ProblemType::SizeT) < 8) ? MEDIUM_TYPE : LARGE_TYPE;
};


/**
 * Classifies the pointer type into a type-size genre
 */
struct PointerSizeClassifier
{
	static const TypeSizeGenre GENRE 		= (sizeof(size_t) < 8) ? MEDIUM_TYPE : LARGE_TYPE;
};


/**
 * Autotuning policy classifier
 */
template <
	typename ProblemType,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE>
struct AutotunedClassifier :
	AutotunedGenre<
		ProblemType,
		CUDA_ARCH,
		PROB_SIZE_GENRE,
		ArchClassifier<CUDA_ARCH>::GENRE,
		TypeSizeClassifier<ProblemType>::GENRE,
		OffsetSizeClassifier<ProblemType>::GENRE,
		PointerSizeClassifier::GENRE>
{};


/******************************************************************************
 * Autotuned genre specializations
 ******************************************************************************/

//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

// Large problems
template <
	typename ProblemType,
	int CUDA_ARCH,
	TypeSizeGenre TYPE_SIZE_GENRE,
	TypeSizeGenre OFFSET_SIZE_GENRE,
	TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM20, TYPE_SIZE_GENRE, OFFSET_SIZE_GENRE, POINTER_SIZE_GENRE>
	: Policy<
		// Problem Type
		ProblemType,

		// Common
		SM20,
		4,						// RADIX_BITS
		10,						// LOG_SCHEDULE_GRANULARITY
		util::io::ld::NONE,		// CACHE_MODIFIER
		util::io::st::NONE,		// CACHE_MODIFIER
		true,					// EARLY_EXIT
		false,					// UNIFORM_SMEM_ALLOCATION
		true, 					// UNIFORM_GRID_SIZE
		true,					// OVERSUBSCRIBED_GRID_SIZE

		// Upsweep Kernel
		8,						// UPSWEEP_MIN_CTA_OCCUPANCY
		7,						// UPSWEEP_LOG_THREADS
		0,						// UPSWEEP_LOG_LOAD_VEC_SIZE
		2,						// UPSWEEP_LOG_LOADS_PER_TILE

		// Spine-scan Kernel
		7,						// SPINE_LOG_THREADS
		2,						// SPINE_LOG_LOAD_VEC_SIZE
		0,						// SPINE_LOG_LOADS_PER_TILE
		5,						// SPINE_LOG_RAKING_THREADS

		// Downsweep Kernel
		partition::downsweep::SCATTER_TWO_PHASE,			// DOWNSWEEP_SCATTER_POLICY
		8,						// DOWNSWEEP_MIN_CTA_OCCUPANCY
		6,						// DOWNSWEEP_LOG_THREADS
		2,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
		1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
		(((ProblemType::KEYS_ONLY) || (POINTER_SIZE_GENRE < LARGE_TYPE)) && (OFFSET_SIZE_GENRE < LARGE_TYPE) && (TYPE_SIZE_GENRE < LARGE_TYPE)) ?		// DOWNSWEEP_LOG_CYCLES_PER_TILE
			1 :
			0,
		6>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};


// Small problems
template <
	typename ProblemType,
	int CUDA_ARCH,
	TypeSizeGenre TYPE_SIZE_GENRE,
	TypeSizeGenre OFFSET_SIZE_GENRE,
	TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM20, TYPE_SIZE_GENRE, OFFSET_SIZE_GENRE, POINTER_SIZE_GENRE>
	: Policy<
		// Problem Type
		ProblemType,

		// Common
		SM20,
		4,						// RADIX_BITS
		9,						// LOG_SCHEDULE_GRANULARITY
		util::io::ld::NONE,		// CACHE_MODIFIER
		util::io::st::NONE,		// CACHE_MODIFIER
		false,					// EARLY_EXIT
		false,					// UNIFORM_SMEM_ALLOCATION
		false, 					// UNIFORM_GRID_SIZE
		false,					// OVERSUBSCRIBED_GRID_SIZE

		// Upsweep Kernel
		8,						// UPSWEEP_MIN_CTA_OCCUPANCY
		7,						// UPSWEEP_LOG_THREADS
		1,						// UPSWEEP_LOG_LOAD_VEC_SIZE
		0,						// UPSWEEP_LOG_LOADS_PER_TILE

		// Spine-scan Kernel
		8,						// SPINE_LOG_THREADS
		2,						// SPINE_LOG_LOAD_VEC_SIZE
		0,						// SPINE_LOG_LOADS_PER_TILE
		5,						// SPINE_LOG_RAKING_THREADS

		// Downsweep Kernel
		partition::downsweep::SCATTER_TWO_PHASE,			// DOWNSWEEP_SCATTER_POLICY
		7,						// DOWNSWEEP_MIN_CTA_OCCUPANCY
		7,						// DOWNSWEEP_LOG_THREADS
		1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
		1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
		0, 						// DOWNSWEEP_LOG_CYCLES_PER_TILE
		7>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};


//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// Large problems
template <
	typename ProblemType,
	int CUDA_ARCH,
	TypeSizeGenre TYPE_SIZE_GENRE,
	TypeSizeGenre OFFSET_SIZE_GENRE,
	TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM13, TYPE_SIZE_GENRE, OFFSET_SIZE_GENRE, POINTER_SIZE_GENRE>
	: Policy<
		// Problem Type
		ProblemType,

		// Common
		SM13,
		4,						// RADIX_BITS
		9,						// LOG_SCHEDULE_GRANULARITY
		util::io::ld::NONE,		// CACHE_MODIFIER
		util::io::st::NONE,		// CACHE_MODIFIER
		true,					// EARLY_EXIT
		true,					// UNIFORM_SMEM_ALLOCATION
		true, 					// UNIFORM_GRID_SIZE
		true,					// OVERSUBSCRIBED_GRID_SIZE

		// Upsweep Kernel
		5,						// UPSWEEP_MIN_CTA_OCCUPANCY
		7,						// UPSWEEP_LOG_THREADS
		(TYPE_SIZE_GENRE < LARGE_TYPE) ?	// UPSWEEP_LOG_LOAD_VEC_SIZE
			0: //1 :
			0,
		(TYPE_SIZE_GENRE < LARGE_TYPE) ?	// UPSWEEP_LOG_LOADS_PER_TILE
			2: //0 :
			1,

		// Spine-scan Kernel
		7,						// SPINE_LOG_THREADS
		2,						// SPINE_LOG_LOAD_VEC_SIZE
		0,						// SPINE_LOG_LOADS_PER_TILE
		5,						// SPINE_LOG_RAKING_THREADS

		// Downsweep Kernel
		partition::downsweep::SCATTER_TWO_PHASE,			// DOWNSWEEP_SCATTER_POLICY
		5,						// DOWNSWEEP_MIN_CTA_OCCUPANCY
		(TYPE_SIZE_GENRE < LARGE_TYPE) ?	// DOWNSWEEP_LOG_THREADS
			6 :
			7,
		(TYPE_SIZE_GENRE < LARGE_TYPE) ?	// DOWNSWEEP_LOG_LOAD_VEC_SIZE
			2 :
			1,
		(TYPE_SIZE_GENRE < LARGE_TYPE) ?	// DOWNSWEEP_LOG_LOADS_PER_CYCLE
			1 :
			0,
		(TYPE_SIZE_GENRE < LARGE_TYPE) ?	// DOWNSWEEP_LOG_CYCLES_PER_TILE
			0 :
			0,
		5>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};


// Small problems
template <
	typename ProblemType,
	int CUDA_ARCH,
	TypeSizeGenre TYPE_SIZE_GENRE,
	TypeSizeGenre OFFSET_SIZE_GENRE,
	TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM13, TYPE_SIZE_GENRE, OFFSET_SIZE_GENRE, POINTER_SIZE_GENRE>
	: Policy<
		// Problem Type
		ProblemType,

		// Common
		SM13,
		4,						// RADIX_BITS
		9,						// LOG_SCHEDULE_GRANULARITY
		util::io::ld::NONE,		// CACHE_MODIFIER
		util::io::st::NONE,		// CACHE_MODIFIER
		true,					// EARLY_EXIT
		true,					// UNIFORM_SMEM_ALLOCATION
		true, 					// UNIFORM_GRID_SIZE
		true,					// OVERSUBSCRIBED_GRID_SIZE

		// Upsweep Kernel
		5,						// UPSWEEP_MIN_CTA_OCCUPANCY
		7,						// UPSWEEP_LOG_THREADS
		1,						// UPSWEEP_LOG_LOAD_VEC_SIZE
		0,						// UPSWEEP_LOG_LOADS_PER_TILE

		// Spine-scan Kernel
		7,						// SPINE_LOG_THREADS
		2,						// SPINE_LOG_LOAD_VEC_SIZE
		0,						// SPINE_LOG_LOADS_PER_TILE
		5,						// SPINE_LOG_RAKING_THREADS

		// Downsweep Kernel
		partition::downsweep::SCATTER_TWO_PHASE,			// DOWNSWEEP_SCATTER_POLICY
		5,						// DOWNSWEEP_MIN_CTA_OCCUPANCY
		6,						// DOWNSWEEP_LOG_THREADS
		2,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
		1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
		(TYPE_SIZE_GENRE < LARGE_TYPE) ?	// DOWNSWEEP_LOG_CYCLES_PER_TILE
			0 :
			0,
		5>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};




//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------

// Large problems
template <
	typename ProblemType,
	int CUDA_ARCH,
	TypeSizeGenre TYPE_SIZE_GENRE,
	TypeSizeGenre OFFSET_SIZE_GENRE,
	TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM10, TYPE_SIZE_GENRE, OFFSET_SIZE_GENRE, POINTER_SIZE_GENRE>
	: Policy<
		// Problem Type
		ProblemType,

		// Common
		SM10,
		4,						// RADIX_BITS
		9,						// LOG_SCHEDULE_GRANULARITY
		util::io::ld::NONE,		// CACHE_MODIFIER
		util::io::st::NONE,		// CACHE_MODIFIER
		true,					// EARLY_EXIT
		false,					// UNIFORM_SMEM_ALLOCATION
		true, 					// UNIFORM_GRID_SIZE
		true,					// OVERSUBSCRIBED_GRID_SIZE

		// Upsweep Kernel
		4,						// UPSWEEP_MIN_CTA_OCCUPANCY
		7,						// UPSWEEP_LOG_THREADS
		0,						// UPSWEEP_LOG_LOAD_VEC_SIZE
		0,						// UPSWEEP_LOG_LOADS_PER_TILE

		// Spine-scan Kernel
		7,						// SPINE_LOG_THREADS
		2,						// SPINE_LOG_LOAD_VEC_SIZE
		0,						// SPINE_LOG_LOADS_PER_TILE
		5,						// SPINE_LOG_RAKING_THREADS

		// Downsweep Kernel
		partition::downsweep::SCATTER_WARP_TWO_PHASE,			// DOWNSWEEP_SCATTER_POLICY
		2,						// DOWNSWEEP_MIN_CTA_OCCUPANCY
		7,						// DOWNSWEEP_LOG_THREADS
		1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
		1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
		0,						// DOWNSWEEP_LOG_CYCLES_PER_TILE
		7>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};


// Small problems
template <
	typename ProblemType,
	int CUDA_ARCH,
	TypeSizeGenre TYPE_SIZE_GENRE,
	TypeSizeGenre OFFSET_SIZE_GENRE,
	TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM10, TYPE_SIZE_GENRE, OFFSET_SIZE_GENRE, POINTER_SIZE_GENRE>
	: Policy<
		// Problem Type
		ProblemType,

		// Common
		SM10,
		4,						// RADIX_BITS
		9,						// LOG_SCHEDULE_GRANULARITY
		util::io::ld::NONE,		// CACHE_MODIFIER
		util::io::st::NONE,		// CACHE_MODIFIER
		false,					// EARLY_EXIT
		false,					// UNIFORM_SMEM_ALLOCATION
		true, 					// UNIFORM_GRID_SIZE
		true,					// OVERSUBSCRIBED_GRID_SIZE

		// Upsweep Kernel
		4,						// UPSWEEP_MIN_CTA_OCCUPANCY
		7,						// UPSWEEP_LOG_THREADS
		0,						// UPSWEEP_LOG_LOAD_VEC_SIZE
		0,						// UPSWEEP_LOG_LOADS_PER_TILE

		// Spine-scan Kernel
		7,						// SPINE_LOG_THREADS
		2,						// SPINE_LOG_LOAD_VEC_SIZE
		0,						// SPINE_LOG_LOADS_PER_TILE
		5,						// SPINE_LOG_RAKING_THREADS

		// Downsweep Kernel
		partition::downsweep::SCATTER_WARP_TWO_PHASE,			// DOWNSWEEP_SCATTER_POLICY
		2,						// DOWNSWEEP_MIN_CTA_OCCUPANCY
		7,						// DOWNSWEEP_LOG_THREADS
		1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
		1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
		0,						// DOWNSWEEP_LOG_CYCLES_PER_TILE
		7>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};




/******************************************************************************
 * Kernel entry points that can derive a tuned granularity type
 * implicitly from the PROB_SIZE_GENRE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned upsweep reduction kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE, typename PassPolicy>
__launch_bounds__ (
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep::THREADS),
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep::MIN_CTA_OCCUPANCY))
__global__ void TunedUpsweepKernel(
	int 													*d_selectors,
	typename ProblemType::SizeT 							*d_spine,
	typename ProblemType::KeyType 							*d_in_keys,
	typename ProblemType::KeyType 							*d_out_keys,
	util::CtaWorkDistribution<typename ProblemType::SizeT> 	work_decomposition)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep TuningPolicy;
	typedef upsweep::KernelPolicy<TuningPolicy, PassPolicy> KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	upsweep::UpsweepPass<KernelPolicy>(
		d_selectors,
		d_spine,
		d_in_keys,
		d_out_keys,
		work_decomposition,
		smem_storage);
}

/**
 * Tuned spine scan kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine::THREADS),
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine::MIN_CTA_OCCUPANCY))
__global__ void TunedSpineKernel(
	typename ProblemType::SizeT 		*d_spine_in,
	typename ProblemType::SizeT 		*d_spine_out,
	int									spine_elements)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	typename KernelPolicy::ReductionOp scan_op;
	typename KernelPolicy::IdentityOp identity_op;

	// Invoke the wrapped kernel logic
	scan::spine::SpinePass<KernelPolicy>(
		d_spine_in,
		d_spine_out,
		spine_elements,
		scan_op,
		identity_op,
		smem_storage);
}


/**
 * Tuned downsweep scan kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE, typename PassPolicy>
__launch_bounds__ (
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep::THREADS),
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep::MIN_CTA_OCCUPANCY))
__global__ void TunedDownsweepKernel(
	int 													*d_selectors,
	typename ProblemType::SizeT 							*d_spine,
	typename ProblemType::KeyType 							*d_keys0,
	typename ProblemType::KeyType 							*d_keys1,
	typename ProblemType::ValueType 						*d_values0,
	typename ProblemType::ValueType							*d_values1,
	util::CtaWorkDistribution<typename ProblemType::SizeT>	work_decomposition)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep TuningPolicy;
	typedef downsweep::KernelPolicy<TuningPolicy, PassPolicy> KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	// Invoke the wrapped kernel logic
	downsweep::DownsweepPass<KernelPolicy>::Invoke(
		d_selectors,
		d_spine,
		d_keys0,
		d_keys1,
		d_values0,
		d_values1,
		work_decomposition,
		smem_storage);
}


/******************************************************************************
 * Autotuned policy
 *******************************************************************************/



/**
 * Autotuned policy type, derives from autotuned genre
 */
template <
	typename ProblemType,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE>
struct AutotunedPolicy :
	AutotunedClassifier<
		ProblemType,
		CUDA_ARCH,
		PROB_SIZE_GENRE>
{
	typedef typename ProblemType::KeyType 		KeyType;
	typedef typename ProblemType::ValueType 	ValueType;
	typedef typename ProblemType::SizeT 		SizeT;

	typedef void (*UpsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)(SizeT*, SizeT*, int);
	typedef void (*DownsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, ValueType*, ValueType*, util::CtaWorkDistribution<SizeT>);

	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	template <typename PassPolicy>
	static UpsweepKernelPtr UpsweepKernel()
	{
		return TunedUpsweepKernel<ProblemType, PROB_SIZE_GENRE, PassPolicy>;
	}

	static SpineKernelPtr SpineKernel()
	{
		return TunedSpineKernel<ProblemType, PROB_SIZE_GENRE>;
	}

	template <typename PassPolicy>
	static DownsweepKernelPtr DownsweepKernel()
	{
		return TunedDownsweepKernel<ProblemType, PROB_SIZE_GENRE, PassPolicy>;
	}
};



}// namespace radix_sort
}// namespace b40c

