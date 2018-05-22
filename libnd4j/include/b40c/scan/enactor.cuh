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
 * Scan enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/arch_dispatch.cuh>

#include <b40c/scan/problem_type.cuh>
#include <b40c/scan/policy.cuh>
#include <b40c/scan/autotuned_policy.cuh>
#include <b40c/scan/downsweep/kernel.cuh>
#include <b40c/scan/spine/kernel.cuh>
#include <b40c/scan/upsweep/kernel.cuh>

namespace b40c {
namespace scan {


/**
 * Scan enactor class.
 */
class Enactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Spine spine;


	//-----------------------------------------------------------------------------
	// Helper structures
	//-----------------------------------------------------------------------------

	template <typename ProblemType>
	friend class Detail;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

    /**
	 * Performs a scan pass
	 */
	template <typename Policy, typename Detail>
	cudaError_t EnactPass(Detail &detail);


public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enacts a scan operation on the specified device data.  Uses
	 * a heuristic for selecting an autotuning policy based upon problem size.
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param num_elements
	 * 		Number of elements in d_src
	 * @param scan_op
	 * 		The function or functor type for binary scan, i.e., a type instance
	 * 		that implements "T (const T&, const T&)"
	 * @param identity_op
	 * 		The function or functor type for the scan identity, i.e., a type instance
	 * 		that implements "T ()"
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		bool EXCLUSIVE,				// Whether or not to perform an exclusive (vs. inclusive) prefix scan
		bool COMMUTATIVE,		// Whether or not the associative scan operator is non-commuatative (the commutative-only implementation is generally faster)
		typename T,
		typename SizeT,
		typename ReductionOp,
		typename IdentityOp>
	cudaError_t Scan(
		T *d_dest,
		T *d_src,
		SizeT num_elements,
		ReductionOp scan_op,
		IdentityOp identity_op,
		int max_grid_size = 0);


	/**
	 * Enacts a scan operation on the specified device data.  Uses the
	 * specified problem size genre enumeration to select autotuning policy.
	 *
	 * (Using this entrypoint can save compile time by not compiling tuned
	 * kernels for each problem size genre.)
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param num_elements
	 * 		Number of elements in d_src
	 * @param scan_op
	 * 		The function or functor type for binary scan, i.e., a type instance
	 * 		that implements "T (const T&, const T&)"
	 * @param identity_op
	 * 		The function or functor type for the scan identity, i.e., a type instance
	 * 		that implements "T ()"
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		bool EXCLUSIVE,				// Whether or not to perform an exclusive (vs. inclusive) prefix scan
		bool COMMUTATIVE,		// Whether or not the associative scan operator is non-commuatative (the commutative-only implementation is generally faster)
		ProbSizeGenre PROB_SIZE_GENRE,
		typename T,
		typename SizeT,
		typename ReductionOp,
		typename IdentityOp>
	cudaError_t Scan(
		T *d_dest,
		T *d_src,
		SizeT num_elements,
		ReductionOp scan_op,
		IdentityOp identity_op,
		int max_grid_size = 0);


	/**
	 * Enacts a scan on the specified device data.  Uses the specified
	 * kernel configuration policy.  (Useful for auto-tuning.)
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param num_elements
	 * 		Number of elements in d_src
	 * @param scan_op
	 * 		The function or functor type for binary scan, i.e., a type instance
	 * 		that implements "T (const T&, const T&)"
	 * @param identity_op
	 * 		The function or functor type for the scan identity, i.e., a type instance
	 * 		that implements "T ()"
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Policy>
	cudaError_t Scan(
		typename Policy::T *d_dest,
		typename Policy::T *d_src,
		typename Policy::SizeT num_elements,
		typename Policy::ReductionOp scan_op,
		typename Policy::IdentityOp identity_op,
		int max_grid_size = 0);
};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <typename ProblemType>
struct Detail : ProblemType
{
	typedef typename ProblemType::T 			T;
	typedef typename ProblemType::SizeT 		SizeT;
	typedef typename ProblemType::ReductionOp 	ReductionOp;
	typedef typename ProblemType::IdentityOp 	IdentityOp;

	Enactor 		*enactor;
	T 				*d_dest;
	T 				*d_src;
	SizeT 			num_elements;
	ReductionOp		scan_op;
	IdentityOp		identity_op;
	int 			max_grid_size;

	// Constructor
	Detail(
		Enactor 		*enactor,
		T 				*d_dest,
		T 				*d_src,
		SizeT 			num_elements,
		ReductionOp		scan_op,
		IdentityOp		identity_op,
		int 			max_grid_size = 0) :
			enactor(enactor),
			d_dest(d_dest),
			d_src(d_src),
			num_elements(num_elements),
			scan_op(scan_op),
			identity_op(identity_op),
			max_grid_size(max_grid_size)
	{}

	template <typename Policy>
	cudaError_t EnactPass()
	{
		return enactor->template EnactPass<Policy>(*this);
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
 */
template <ProbSizeGenre PROB_SIZE_GENRE>
struct PolicyResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Detail>
	static cudaError_t Enact(Detail &detail)
	{
		// Obtain tuned granularity type
		typedef AutotunedPolicy<
			Detail,
			CUDA_ARCH,
			PROB_SIZE_GENRE> AutotunedPolicy;

		// Invoke enactor with type
		return detail.template EnactPass<AutotunedPolicy>();
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
 */
template <>
struct PolicyResolver <UNKNOWN_SIZE>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Detail>
	static cudaError_t Enact(Detail &detail)
	{
		// Obtain large tuned granularity type
		typedef AutotunedPolicy<
			Detail,
			CUDA_ARCH,
			LARGE_SIZE> LargePolicy;

		// Identify the maximum problem size for which we can saturate loads
		int saturating_load = LargePolicy::Upsweep::TILE_ELEMENTS *
			B40C_SM_CTAS(CUDA_ARCH) *
			detail.enactor->SmCount();

		if (detail.num_elements < saturating_load) {

			// Invoke enactor with small-problem config type
			typedef AutotunedPolicy<
				Detail,
				CUDA_ARCH,
				SMALL_SIZE> SmallPolicy;

			return detail.template EnactPass<SmallPolicy>();
		}

		// Invoke enactor with type
		return detail.template EnactPass<LargePolicy>();
	}
};


/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/


/**
 * Performs a scan pass
 */
template <typename Policy, typename DetailType>
cudaError_t Enactor::EnactPass(DetailType &detail)
{
	typedef typename Policy::T 				T;
	typedef typename Policy::SizeT 			SizeT;
	typedef typename Policy::ReductionOp 	ReductionOp;
	typedef typename Policy::IdentityOp 	IdentityOp;

	typedef typename Policy::Upsweep 	Upsweep;
	typedef typename Policy::Spine 		Spine;
	typedef typename Policy::Downsweep 	Downsweep;
	typedef typename Policy::Single 	Single;

	cudaError_t retval = cudaSuccess;
	do {

		// Make sure we have a valid policy
		if (!Policy::VALID) {
			retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Enactor invalid policy", __FILE__, __LINE__);
			break;
		}

		// Kernels
		typename Policy::UpsweepKernelPtr UpsweepKernel = Policy::UpsweepKernel();
		typename Policy::DownsweepKernelPtr DownsweepKernel = Policy::DownsweepKernel();

		// Max CTA occupancy for the actual target device
		int max_cta_occupancy;
		if (retval = MaxCtaOccupancy(
			max_cta_occupancy,
			UpsweepKernel,
			Upsweep::THREADS,
			DownsweepKernel,
			Downsweep::THREADS)) break;

		// Compute sweep grid size
		int sweep_grid_size = GridSize(
			Policy::OVERSUBSCRIBED_GRID_SIZE,
			Upsweep::SCHEDULE_GRANULARITY,
			max_cta_occupancy,
			detail.num_elements,
			detail.max_grid_size);

		// Use single-CTA kernel instead of multi-pass if problem is small enough
		if (detail.num_elements <= Single::TILE_ELEMENTS * 3) {
			sweep_grid_size = 1;
		}

		// Compute spine elements: one element per CTA, rounded
		// up to nearest spine tile size
		int spine_elements = ((sweep_grid_size + Spine::TILE_ELEMENTS - 1) / Spine::TILE_ELEMENTS) * Spine::TILE_ELEMENTS;

		// Obtain a CTA work distribution
		util::CtaWorkDistribution<SizeT> work;
		work.template Init<Downsweep::LOG_SCHEDULE_GRANULARITY>(detail.num_elements, sweep_grid_size);

		if (ENACTOR_DEBUG) {
			if (sweep_grid_size > 1) {
				PrintPassInfo<Upsweep, Spine, Downsweep>(work, spine_elements);
			} else {
				PrintPassInfo<Single>(work);
			}
		}

		if (work.grid_size == 1) {

			// Single-CTA, single-grid operation
			typename Policy::SingleKernelPtr SingleKernel = Policy::SingleKernel();

			SingleKernel<<<1, Single::THREADS, 0>>>(
				detail.d_src,
				detail.d_dest,
				work.num_elements,
				detail.scan_op,
				detail.identity_op);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SingleKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

		} else {

			// Upsweep-downsweep operation
			typename Policy::SpineKernelPtr SpineKernel = Policy::SpineKernel();

			// Make sure our spine is big enough
			if (retval = spine.Setup<T>(spine_elements)) break;

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option: make sure all kernels have the same overall smem allocation
			if (Policy::UNIFORM_SMEM_ALLOCATION) if (retval = PadUniformSmem(dynamic_smem, UpsweepKernel, SpineKernel, DownsweepKernel)) break;

			// Tuning option: make sure that all kernels launch the same number of CTAs)
			if (Policy::UNIFORM_GRID_SIZE) grid_size[1] = grid_size[0];

			// Upsweep into spine
			UpsweepKernel<<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
				detail.d_src,
				(T*) spine(),
				detail.scan_op,
				detail.identity_op,
				work);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Spine scan
			SpineKernel<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
				(T*) spine(),
				(T*) spine(),
				spine_elements,
				detail.scan_op,
				detail.identity_op);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Downsweep from spine
			DownsweepKernel<<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
				detail.d_src,
				detail.d_dest,
				(T*) spine(),
				detail.scan_op,
				detail.identity_op,
				work);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor DownsweepKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;
		}
	} while (0);

	return retval;
}


/**
 * Enacts a scan on the specified device data.
 */
template <typename Policy>
cudaError_t Enactor::Scan(
	typename Policy::T *d_dest,
	typename Policy::T *d_src,
	typename Policy::SizeT num_elements,
	typename Policy::ReductionOp scan_op,
	typename Policy::IdentityOp identity_op,
	int max_grid_size)
{
	Detail<Policy> detail(
		this,
		d_dest,
		d_src,
		num_elements,
		scan_op,
		identity_op,
		max_grid_size);

	return EnactPass<Policy>(detail);
}


/**
 * Enacts a scan operation on the specified device data.
 */
template <
	bool EXCLUSIVE,
	bool COMMUTATIVE,
	ProbSizeGenre PROB_SIZE_GENRE,
	typename T,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
cudaError_t Enactor::Scan(
	T *d_dest,
	T *d_src,
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op,
	int max_grid_size)
{
	typedef ProblemType<
		T,
		SizeT,
		ReductionOp,
		IdentityOp,
		EXCLUSIVE,
		COMMUTATIVE> ProblemType;

	Detail<ProblemType> detail(
		this,
		d_dest,
		d_src,
		num_elements,
		scan_op,
		identity_op,
		max_grid_size);

	return util::ArchDispatch<
		__B40C_CUDA_ARCH__,
		PolicyResolver<PROB_SIZE_GENRE> >::Enact(detail, PtxVersion());
}


/**
 * Enacts a scan operation on the specified device data.
 */
template <
	bool EXCLUSIVE,
	bool COMMUTATIVE,
	typename T,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
cudaError_t Enactor::Scan(
	T *d_dest,
	T *d_src,
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op,
	int max_grid_size)
{
	return Scan<EXCLUSIVE, COMMUTATIVE, UNKNOWN_SIZE>(
		d_dest,
		d_src,
		num_elements,
		scan_op,
		identity_op,
		max_grid_size);
}




} // namespace scan 
} // namespace b40c

