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
 * Types and subroutines utilities that are common across all B40C LSB radix 
 * sorting kernels and host enactors  
 ******************************************************************************/

#pragma once

namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Bit-field extraction kernel subroutines
 ******************************************************************************/

/**
 * Extracts a bit field from source and places the zero or sign-extended result 
 * in extract
 */
template <typename T, int BIT_START, int NUM_BITS> 
struct ExtractKeyBits 
{
	__device__ __forceinline__ static void Extract(int &bits, T source)
	{
#if __CUDA_ARCH__ >= 200
		asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(BIT_START), "r"(NUM_BITS));
#else
		const T MASK = (1 << NUM_BITS) - 1;
		bits = (source >> BIT_START) & MASK;
#endif
	}
};
	
/**
 * Extracts a bit field from source and places the zero or sign-extended result 
 * in extract
 */
template <int BIT_START, int NUM_BITS> 
struct ExtractKeyBits<unsigned long long, BIT_START, NUM_BITS> 
{
	__device__ __forceinline__ static void Extract(int &bits, const unsigned long long &source) 
	{
		if (BIT_START >= 32) {											// For extraction on GT200, the compiler goes nuts and shoves hundreds of bytes to lmem unless we use different extractions for upper/lower
			const unsigned long long MASK = (1 << NUM_BITS) - 1;
			bits = (source >> BIT_START) & MASK;
		} else {
			const unsigned long long MASK = ((1ull << NUM_BITS) - 1) << BIT_START;
			bits = (source & MASK) >> BIT_START;
		}
	}
};
	

/******************************************************************************
 * Traits for converting for converting signed and floating point types
 * to unsigned types suitable for radix sorting
 ******************************************************************************/

struct NopKeyConversion
{
	static const bool MustApply = false;		// We may early-exit this pass

	template <typename T>
	__device__ __host__ __forceinline__ static void Preprocess(T &key) {}

	template <typename T>
	__device__ __host__ __forceinline__ static void Postprocess(T &key) {}
};


template <typename UnsignedBits> 
struct UnsignedIntegerKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;
	
	static const bool MustApply = false;		// We may early-exit this pass

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key) {}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key) {}  
};


template <typename UnsignedBits> 
struct SignedIntegerKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MustApply = true;		// We must not early-exit this pass (conversion necessary)

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key)
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		converted_key ^= HIGH_BIT;
	}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key)  
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		converted_key ^= HIGH_BIT;	
	}
};


template <typename UnsignedBits> 
struct FloatingPointKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MustApply = true;		// We must not early-exit this pass (conversion necessary)

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key)
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		UnsignedBits mask = (converted_key & HIGH_BIT) ? (UnsignedBits) -1 : HIGH_BIT;
		converted_key ^= mask;
	}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key) 
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		UnsignedBits mask = (converted_key & HIGH_BIT) ? HIGH_BIT : (UnsignedBits) -1; 
		converted_key ^= mask;
    }
};




// Default unsigned types
template <typename T> struct KeyTraits : UnsignedIntegerKeyConversion<T> {};

// char
template <> struct KeyTraits<char> : SignedIntegerKeyConversion<unsigned char> {};

// signed char
template <> struct KeyTraits<signed char> : SignedIntegerKeyConversion<unsigned char> {};

// short
template <> struct KeyTraits<short> : SignedIntegerKeyConversion<unsigned short> {};

// int
template <> struct KeyTraits<int> : SignedIntegerKeyConversion<unsigned int> {};

// long
template <> struct KeyTraits<long> : SignedIntegerKeyConversion<unsigned long> {};

// long long
template <> struct KeyTraits<long long> : SignedIntegerKeyConversion<unsigned long long> {};

// float
template <> struct KeyTraits<float> : FloatingPointKeyConversion<unsigned int> {};

// double
template <> struct KeyTraits<double> : FloatingPointKeyConversion<unsigned long long> {};




} // namespace radix_sort
} // namespace b40c

