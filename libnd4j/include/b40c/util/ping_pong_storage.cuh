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
 *  Storage wrapper for double-buffered vectors (deprecated).
 ******************************************************************************/

#pragma once

#include <b40c/util/multiple_buffering.cuh>

namespace b40c {
namespace util {

/**
 * Ping-pong buffer (a.k.a. page-flip, double-buffer, etc.).
 * Deprecated: see b40c::util::DoubleBuffer instead.
 */
template <
	typename KeyType,
	typename ValueType = util::NullType>
struct PingPongStorage : DoubleBuffer<KeyType, ValueType>
{
	typedef DoubleBuffer<KeyType, ValueType> ParentType;

	// Constructor
	PingPongStorage() : ParentType() {}

	// Constructor
	PingPongStorage(
		KeyType* keys) : ParentType(keys) {}

	// Constructor
	PingPongStorage(
		KeyType* keys,
		ValueType* values) : ParentType(keys, values) {}

	// Constructor
	PingPongStorage(
		KeyType* keys0,
		KeyType* keys1,
		ValueType* values0,
		ValueType* values1) : ParentType(keys0, keys1, values0, values1) {}
};


} // namespace util
} // namespace b40c

