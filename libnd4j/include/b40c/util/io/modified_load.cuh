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
 * Kernel utilities for loading types through global memory with cache modifiers
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/vector_types.cuh>

namespace b40c {
namespace util {
namespace io {


/**
 * Enumeration of data movement cache modifiers.
 */
namespace ld {

	enum CacheModifier {
		NONE,				// default (currently ca)
		cg,					// cache global
		ca,					// cache all
		cs, 				// cache streaming

		LIMIT
	};

} // namespace ld



/**
 * TODO: replace this with something better
 */
#define CacheModifierToString(modifier)	(	((int) modifier == b40c::util::io::ld::NONE) ? 		"util::io::ld::NONE" :	\
											((int) modifier == b40c::util::io::ld::cg) ? 		"util::io::ld::cg" :		\
											((int) modifier == b40c::util::io::ld::ca) ? 		"util::io::ld::ca" :		\
											((int) modifier == b40c::util::io::ld::cs) ? 		"util::io::ld::cs" :		\
											((int) modifier == b40c::util::io::st::NONE) ? 		"util::io::st::NONE" :	\
											((int) modifier == b40c::util::io::st::cg) ? 		"util::io::st::cg" :		\
											((int) modifier == b40c::util::io::st::wb) ? 		"util::io::st::wb" :		\
											((int) modifier == b40c::util::io::st::cs) ? 		"util::io::st::cs" :		\
																								"<ERROR>")

/**
 * Basic utility for performing modified loads through cache.
 */
template <ld::CacheModifier CACHE_MODIFIER>
struct ModifiedLoad
{
	/**
	 * Load operation we will provide specializations for
	 */
	template <typename T>
	__device__ __forceinline__ static void Ld(T &val, T *ptr);

	/**
	 * Vec-4 loads for 64-bit types are implemented as two vec-2 loads
	 */
	__device__ __forceinline__ static void Ld(double4 &val, double4* ptr)
	{
		ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<double2*>(&val.x), reinterpret_cast<double2*>(ptr));
		ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<double2*>(&val.z), reinterpret_cast<double2*>(ptr) + 1);
	}

	__device__ __forceinline__ static void Ld(ulonglong4 &val, ulonglong4* ptr)
	{
		ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<ulonglong2*>(&val.x), reinterpret_cast<ulonglong2*>(ptr));
		ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<ulonglong2*>(&val.z), reinterpret_cast<ulonglong2*>(ptr) + 1);
	}

	__device__ __forceinline__ static void Ld(longlong4 &val, longlong4* ptr)
	{
		ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<longlong2*>(&val.x), reinterpret_cast<longlong2*>(ptr));
		ModifiedLoad<CACHE_MODIFIER>::Ld(*reinterpret_cast<longlong2*>(&val.z), reinterpret_cast<longlong2*>(ptr) + 1);
	}
};


#if __CUDA_ARCH__ >= 200

	/**
	 * Specialization for NONE modifier
	 */
	template <>
	template <typename T>
	__device__ __forceinline__ void ModifiedLoad<ld::NONE>::Ld(T &val, T *ptr)
	{
		val = *ptr;
	}

	/**
	 * Singleton store op
	 */
	#define B40C_LOAD(base_type, ptx_type, reg_mod, cast_type, modifier)																	\
		template<> template<> void ModifiedLoad<ld::modifier>::Ld(base_type &val, base_type* ptr) {												\
			asm volatile ("ld.global."#modifier"."#ptx_type" %0, [%1];" : "="#reg_mod(reinterpret_cast<cast_type&>(val)) : _B40C_ASM_PTR_(ptr));			\
		}																																		\

	/**
	 * Vector load ops
	 */
	#define B40C_LOAD_VEC1(base_type, ptx_type, reg_mod, cast_type, modifier)																	\
		template<> template<> void ModifiedLoad<ld::modifier>::Ld(base_type &val, base_type* ptr) {												\
			asm volatile ("ld.global."#modifier"."#ptx_type" %0, [%1];" : "="#reg_mod(reinterpret_cast<cast_type&>(val.x)) : _B40C_ASM_PTR_(ptr));			\
		}																																		\

	#define B40C_LOAD_VEC2(base_type, ptx_type, reg_mod, cast_type, modifier)																	\
		template<> template<> void ModifiedLoad<ld::modifier>::Ld(base_type &val, base_type* ptr) {												\
			asm volatile ("ld.global."#modifier".v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(reinterpret_cast<cast_type&>(val.x)), "="#reg_mod(reinterpret_cast<cast_type&>(val.y)) : _B40C_ASM_PTR_(ptr));		\
		}

	#define B40C_LOAD_VEC4(base_type, ptx_type, reg_mod, cast_type, modifier)																	\
		template<> template<> void ModifiedLoad<ld::modifier>::Ld(base_type &val, base_type* ptr) {												\
			asm volatile ("ld.global."#modifier".v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(reinterpret_cast<cast_type&>(val.x)), "="#reg_mod(reinterpret_cast<cast_type&>(val.y)), "="#reg_mod(reinterpret_cast<cast_type&>(val.z)), "="#reg_mod(reinterpret_cast<cast_type&>(val.w)) : _B40C_ASM_PTR_(ptr));		\
		}


	/**
	 * Defines specialized load ops for only the base type
	 */
	#define B40C_LOAD_BASE(base_type, ptx_type, reg_mod, cast_type)		\
		B40C_LOAD(base_type, ptx_type, reg_mod, cast_type, cg)			\
		B40C_LOAD(base_type, ptx_type, reg_mod, cast_type, ca)			\
		B40C_LOAD(base_type, ptx_type, reg_mod, cast_type, cs)


	/**
	 * Defines specialized load ops for the base type and for its derivative vec1 and vec2 types
	 */
	#define B40C_LOAD_BASE_ONE_TWO(base_type, dest_type, short_type, ptx_type, reg_mod, cast_type)	\
		B40C_LOAD_BASE(base_type, ptx_type, reg_mod, cast_type)										\
																									\
		B40C_LOAD_VEC1(short_type##1, ptx_type, reg_mod, cast_type, cg)								\
		B40C_LOAD_VEC1(short_type##1, ptx_type, reg_mod, cast_type, ca)								\
		B40C_LOAD_VEC1(short_type##1, ptx_type, reg_mod, cast_type, cs)								\
																									\
		B40C_LOAD_VEC2(short_type##2, ptx_type, reg_mod, cast_type, cg)								\
		B40C_LOAD_VEC2(short_type##2, ptx_type, reg_mod, cast_type, ca)								\
		B40C_LOAD_VEC2(short_type##2, ptx_type, reg_mod, cast_type, cs)


	/**
	 * Defines specialized load ops for the base type and for its derivative vec1, vec2, and vec4 types
	 */
	#define B40C_LOAD_BASE_ONE_TWO_FOUR(base_type, dest_type, short_type, ptx_type, reg_mod, cast_type)	\
		B40C_LOAD_BASE_ONE_TWO(base_type, dest_type, short_type, ptx_type, reg_mod, cast_type)			\
		B40C_LOAD_VEC4(short_type##4, ptx_type, reg_mod, cast_type, cg)									\
		B40C_LOAD_VEC4(short_type##4, ptx_type, reg_mod, cast_type, ca)									\
		B40C_LOAD_VEC4(short_type##4, ptx_type, reg_mod, cast_type, cs)


#if CUDA_VERSION >= 4000
	#define B40C_REG8		h
	#define B40C_REG16 		h
	#define B40C_CAST8 		short
#else
	#define B40C_REG8		r
	#define B40C_REG16 		r
	#define B40C_CAST8 		char
#endif


	/**
	 * Define cache-modified loads for all 4-byte (and smaller) structures
	 */
	B40C_LOAD_BASE_ONE_TWO_FOUR(char, 			char, 			char, 	s8, 	B40C_REG8, 		B40C_CAST8)
	B40C_LOAD_BASE_ONE_TWO_FOUR(short, 			short, 			short, 	s16, 	B40C_REG16, 	short)
	B40C_LOAD_BASE_ONE_TWO_FOUR(int, 			int, 			int, 	s32, 	r, 				int)
	B40C_LOAD_BASE_ONE_TWO_FOUR(unsigned char, 	unsigned char, 	uchar,	u8, 	B40C_REG8, 		unsigned B40C_CAST8)
	B40C_LOAD_BASE_ONE_TWO_FOUR(unsigned short,	unsigned short,	ushort,	u16, 	B40C_REG16, 	unsigned short)
	B40C_LOAD_BASE_ONE_TWO_FOUR(unsigned int, 	unsigned int, 	uint,	u32, 	r, 				unsigned int)
	B40C_LOAD_BASE_ONE_TWO_FOUR(float, 			float, 			float, 	f32, 	f, 				float)

	#if !defined(__LP64__) || (__LP64__ == 0)
	// longs are 64-bit on non-Windows 64-bit compilers
	B40C_LOAD_BASE_ONE_TWO_FOUR(long, 			long, 			long, 	s32, 	r, long)
	B40C_LOAD_BASE_ONE_TWO_FOUR(unsigned long, 	unsigned long, 	ulong, 	u32, 	r, unsigned long)
	#endif

	B40C_LOAD_BASE(signed char, s8, r, unsigned int)		// Only need to define base: char2,char4, etc already defined from char


	/**
	 * Define cache-modified loads for all 8-byte structures
	 */
	B40C_LOAD_BASE_ONE_TWO(unsigned long long, 	unsigned long long, 	ulonglong, 	u64, l, unsigned long long)
	B40C_LOAD_BASE_ONE_TWO(long long, 			long long, 				longlong, 	s64, l, long long)
	B40C_LOAD_BASE_ONE_TWO(double, 				double, 				double, 	s64, l, long long)				// Cast to 64-bit long long a workaround for the fact that the 3.x assembler has no register constraint for doubles

	#if defined(__LP64__)
	// longs are 64-bit on non-Windows 64-bit compilers
	B40C_LOAD_BASE_ONE_TWO(long, 				long, 					long, 		s64, l, long)
	B40C_LOAD_BASE_ONE_TWO(unsigned long, 		unsigned long, 			ulong, 		u64, l, unsigned long)
	#endif


	/**
	 * Undefine macros
	 */
	#undef B40C_LOAD_VEC1
	#undef B40C_LOAD_VEC2
	#undef B40C_LOAD_VEC4
	#undef B40C_LOAD_BASE
	#undef B40C_LOAD_BASE_ONE_TWO
	#undef B40C_LOAD_BASE_ONE_TWO_FOUR
	#undef B40C_CAST8
	#undef B40C_REG8
	#undef B40C_REG16



#else  //__CUDA_ARCH__


	template <ld::CacheModifier READ_MODIFIER>
	template <typename T>
	__device__ __forceinline__ void ModifiedLoad<READ_MODIFIER>::Ld(T &val, T *ptr)
	{
		val = *ptr;
	}


#endif //__CUDA_ARCH__




} // namespace io
} // namespace util
} // namespace b40c

