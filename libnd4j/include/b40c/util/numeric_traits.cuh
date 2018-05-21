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
 * Type traits for numeric types
 ******************************************************************************/

#pragma once


namespace b40c {
namespace util {


enum Representation
{
	NOT_A_NUMBER,
	SIGNED_INTEGER,
	UNSIGNED_INTEGER,
	FLOATING_POINT
};


template <Representation R>
struct BaseTraits
{
	static const Representation REPRESENTATION = R;
};


// Default, non-numeric types
template <typename T> struct NumericTraits : 				BaseTraits<NOT_A_NUMBER> {};

template <> struct NumericTraits<char> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<signed char> : 			BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<short> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<int> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<long> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<long long> : 				BaseTraits<SIGNED_INTEGER> {};

template <> struct NumericTraits<unsigned char> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct NumericTraits<unsigned short> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct NumericTraits<unsigned int> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct NumericTraits<unsigned long> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct NumericTraits<unsigned long long> : 		BaseTraits<UNSIGNED_INTEGER> {};

template <> struct NumericTraits<float> : 					BaseTraits<FLOATING_POINT> {};
template <> struct NumericTraits<double> : 					BaseTraits<FLOATING_POINT> {};


} // namespace util
} // namespace b40c

