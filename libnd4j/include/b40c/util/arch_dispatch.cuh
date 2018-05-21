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
 * Base class for dynamic architecture dispatch
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>

namespace b40c {
namespace util {


/**
 * Specialization for the device compilation-path.
 *
 * Dispatches to the static method Dispatch::Enact templated by the static CUDA_ARCH.
 * This path drives the actual compilation of kernels, allowing invocation sites to be
 * specialized in type and number by CUDA_ARCH.
 */
template <int CUDA_ARCH, typename Dispatch>
struct ArchDispatch
{
	template<typename Detail>
	static cudaError_t Enact(Detail &detail, int dummy)
	{
		return Dispatch::template Enact<CUDA_ARCH, Detail>(detail);
	}
};


/**
 * Specialization specialization for the host compilation-path.
 *
 * Dispatches to the static method Dispatch::Enact templated by the dynamic
 * ptx_version.  This path does not drive the compilation of kernels.
 */
template <typename Dispatch>
struct ArchDispatch<0, Dispatch>
{
	template<typename Detail>
	static cudaError_t Enact(Detail &detail, int ptx_version)
	{
		// Dispatch
		switch (ptx_version) {
		case 100:
			return Dispatch::template Enact<100, Detail>(detail);
		case 110:
			return Dispatch::template Enact<110, Detail>(detail);
		case 120:
			return Dispatch::template Enact<120, Detail>(detail);
		case 130:
			return Dispatch::template Enact<130, Detail>(detail);
		case 200:
			return Dispatch::template Enact<200, Detail>(detail);
		case 210:
			return Dispatch::template Enact<210, Detail>(detail);
		default:
			// We were compiled for something new: treat it as we would SM2.0
			return Dispatch::template Enact<200, Detail>(detail);
		};
	}
};



} // namespace util
} // namespace b40c

