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
 * CUDA Properties
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {

/******************************************************************************
 * Macros for guiding compilation paths
 ******************************************************************************/

/**
 * CUDA architecture of the current compilation path
 */
#ifndef __CUDA_ARCH__
	#define __B40C_CUDA_ARCH__ 0						// Host path
#else
	#define __B40C_CUDA_ARCH__ __CUDA_ARCH__			// Device path
#endif



/******************************************************************************
 * Device properties by SM architectural version
 ******************************************************************************/

// Invalid CUDA device ordinal
#define B40C_INVALID_DEVICE				(-1)

// Threads per warp. 
#define B40C_LOG_WARP_THREADS(arch)		(5)			// 32 threads in a warp 
#define B40C_WARP_THREADS(arch)			(1 << B40C_LOG_WARP_THREADS(arch))

// SM memory bank stride (in bytes)
#define B40C_LOG_BANK_STRIDE_BYTES(arch)	(2)		// 4 byte words
#define B40C_BANK_STRIDE_BYTES(arch)		(1 << B40C_LOG_BANK_STRIDE_BYTES)

// Memory banks per SM
#define B40C_SM20_LOG_MEM_BANKS()		(5)			// 32 banks on SM2.0+
#define B40C_SM10_LOG_MEM_BANKS()		(4)			// 16 banks on SM1.0-SM1.3
#define B40C_LOG_MEM_BANKS(arch)		((arch >= 200) ? B40C_SM20_LOG_MEM_BANKS() : 	\
														 B40C_SM10_LOG_MEM_BANKS())		

// Physical shared memory per SM (bytes)
#define B40C_SM20_SMEM_BYTES()			(49152)		// 48KB on SM2.0+
#define B40C_SM10_SMEM_BYTES()			(16384)		// 32KB on SM1.0-SM1.3
#define B40C_SMEM_BYTES(arch)			((arch >= 200) ? B40C_SM20_SMEM_BYTES() : 	\
														 B40C_SM10_SMEM_BYTES())		

// Physical threads per SM
#define B40C_SM20_SM_THREADS()			(1536)		// 1536 threads on SM2.0+
#define B40C_SM12_SM_THREADS()			(1024)		// 1024 threads on SM1.2-SM1.3
#define B40C_SM10_SM_THREADS()			(768)		// 768 threads on SM1.0-SM1.1
#define B40C_SM_THREADS(arch)			((arch >= 200) ? B40C_SM20_SM_THREADS() : 	\
										 (arch >= 130) ? B40C_SM12_SM_THREADS() : 	\
												 	 	 B40C_SM10_SM_THREADS())

// Physical threads per CTA
#define B40C_SM20_LOG_CTA_THREADS()		(10)		// 1024 threads on SM2.0+
#define B40C_SM10_LOG_CTA_THREADS()		(9)			// 512 threads on SM1.0-SM1.3
#define B40C_LOG_CTA_THREADS(arch)		((arch >= 200) ? B40C_SM20_LOG_CTA_THREADS() : 	\
												 	 	 B40C_SM10_LOG_CTA_THREADS())

// Max CTAs per SM
#define B40C_SM20_SM_CTAS()				(8)		// 8 CTAs on SM2.0+
#define B40C_SM12_SM_CTAS()				(8)		// 8 CTAs on SM1.2-SM1.3
#define B40C_SM10_SM_CTAS()				(8)		// 8 CTAs on SM1.0-SM1.1
#define B40C_SM_CTAS(arch)				((arch >= 200) ? B40C_SM20_SM_CTAS() : 	\
										 (arch >= 130) ? B40C_SM12_SM_CTAS() : 	\
												 	 	 B40C_SM10_SM_CTAS())

// Max registers per SM
#define B40C_SM20_SM_REGISTERS()		(32768)		// 32768 registers on SM2.0+
#define B40C_SM12_SM_REGISTERS()		(16384)		// 16384 registers on SM1.2-SM1.3
#define B40C_SM10_SM_REGISTERS()		(8192)		// 8192 registers on SM1.0-SM1.1
#define B40C_SM_REGISTERS(arch)			((arch >= 200) ? B40C_SM20_SM_REGISTERS() : 	\
										 (arch >= 130) ? B40C_SM12_SM_REGISTERS() : 	\
												 	 	 B40C_SM10_SM_REGISTERS())

/******************************************************************************
 * Inlined PTX helper macros
 ******************************************************************************/


// Register modifier for pointer-types (for inlining PTX assembly)
#if defined(_WIN64) || defined(__LP64__)
	#define __B40C_LP64__ 1
	// 64-bit register modifier for inlined asm
	#define _B40C_ASM_PTR_ "l"
#else
	#define __B40C_LP64__ 0
	// 32-bit register modifier for inlined asm
	#define _B40C_ASM_PTR_ "r"
#endif



/******************************************************************************
 * CUDA/GPU inspection utilities
 ******************************************************************************/

/**
 * Empty Kernel
 */
template <typename T>
__global__ void FlushKernel(void) { }


/**
 * Class encapsulating device properties for dynamic host-side inspection
 */
class CudaProperties 
{
public:
	
	// Information about our target device
	cudaDeviceProp 		device_props;
	int 				device_sm_version;
	
	// Information about our kernel assembly
	int 				kernel_ptx_version;
	
public:
	
	/**
	 * Constructor
	 */
	CudaProperties() 
	{
		// Get current device properties 
		int current_device;
		cudaGetDevice(&current_device);
		cudaGetDeviceProperties(&device_props, current_device);
		device_sm_version = device_props.major * 100 + device_props.minor * 10;
	
		// Get SM version of compiled kernel assemblies
		cudaFuncAttributes flush_kernel_attrs;
		cudaFuncGetAttributes(&flush_kernel_attrs, FlushKernel<void>);
		kernel_ptx_version = flush_kernel_attrs.ptxVersion * 10;
	}

	/**
	 * Constructor
	 */
	CudaProperties(int gpu)
	{
		// Get current device properties
		cudaGetDeviceProperties(&device_props, gpu);
		device_sm_version = device_props.major * 100 + device_props.minor * 10;

		// Get SM version of compiled kernel assemblies
		cudaFuncAttributes flush_kernel_attrs;
		cudaFuncGetAttributes(&flush_kernel_attrs, FlushKernel<void>);
		kernel_ptx_version = flush_kernel_attrs.ptxVersion * 10;
	}
};



} // namespace util
} // namespace b40c

