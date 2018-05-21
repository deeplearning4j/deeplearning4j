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
 * Utility code for working with vector types of arbitary typenames
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {


/**
 * Specializations of this vector-type template can be used to indicate the 
 * proper vector type for a given typename and vector size. We can use the ::Type
 * typedef to declare and work with the appropriate vectorized type for a given 
 * typename T.
 * 
 * For example, consider the following copy kernel that uses vec-2 loads 
 * and stores:
 * 
 * 		template <typename T>
 * 		__global__ void CopyKernel(T *d_in, T *d_out) 
 * 		{
 * 			typedef typename VecType<T, 2>::Type Vector;
 *
 * 			Vector datum;
 * 
 * 			Vector *d_in_v2 = (Vector *) d_in;
 * 			Vector *d_out_v2 = (Vector *) d_out;
 * 
 * 			datum = d_in_v2[threadIdx.x];
 * 			d_out_v2[threadIdx.x] = datum;
 * 		} 
 * 
 */
template <typename T, int vec_elements> struct VecType;

/**
 * Partially-specialized generic vec1 type 
 */
template <typename T> 
struct VecType<T, 1> {
	T x;
	typedef VecType<T, 1> Type;
};

/**
 * Partially-specialized generic vec2 type 
 */
template <typename T> 
struct VecType<T, 2> {
	T x;
	T y;
	typedef VecType<T, 2> Type;
};

/**
 * Partially-specialized generic vec4 type 
 */
template <typename T> 
struct VecType<T, 4> {
	T x;
	T y;
	T z;
	T w;
	typedef VecType<T, 4> Type;
};


/**
 * Macro for expanding partially-specialized built-in vector types
 */
#define B40C_DEFINE_VECTOR_TYPE(base_type,short_type)                           \
  template<> struct VecType<base_type, 1> { typedef short_type##1 Type; };      \
  template<> struct VecType<base_type, 2> { typedef short_type##2 Type; };      \
  template<> struct VecType<base_type, 4> { typedef short_type##4 Type; };     

B40C_DEFINE_VECTOR_TYPE(char,               char)
B40C_DEFINE_VECTOR_TYPE(signed char,        char)
B40C_DEFINE_VECTOR_TYPE(short,              short)
B40C_DEFINE_VECTOR_TYPE(int,                int)
B40C_DEFINE_VECTOR_TYPE(long,               long)
B40C_DEFINE_VECTOR_TYPE(long long,          longlong)
B40C_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
B40C_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
B40C_DEFINE_VECTOR_TYPE(unsigned int,       uint)
B40C_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
B40C_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
B40C_DEFINE_VECTOR_TYPE(float,              float)
B40C_DEFINE_VECTOR_TYPE(double,             double)

#undef B40C_DEFINE_VECTOR_TYPE


} // namespace util
} // namespace b40c

