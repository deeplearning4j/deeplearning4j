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
 *  Storage wrapper for multi-pass stream transformations that require a
 *  secondary problem storage array to stream results back and forth from.
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace util {


/**
 * Storage wrapper for multi-pass stream transformations that require a
 * more than one problem storage array to stream results back and forth from.
 * 
 * This wrapper provides maximum flexibility for re-using device allocations
 * for subsequent transformations.  As such, it is the caller's responsibility
 * to free any non-NULL storage arrays when no longer needed.
 * 
 * Many multi-pass stream computations require at least two problem storage
 * arrays, e.g., one for reading in from, the other for writing out to.
 * (And their roles can be reversed for each subsequent pass.) This structure
 * tracks two sets of device vectors (a keys and a values sets), and a "selector"
 * member to index which vector in each set is "currently valid".  I.e., the
 * valid data within "MultipleBuffer<2, int, int> b" is accessible by:
 * 
 * 		b.d_keys[b.selector];
 * 
 */
template <
	int BUFFER_COUNT,
	typename _KeyType,
	typename _ValueType = util::NullType>
struct MultipleBuffer
{
	typedef _KeyType	KeyType;
	typedef _ValueType 	ValueType;

	// Set of device vector pointers for keys
	KeyType* d_keys[BUFFER_COUNT];
	
	// Set of device vector pointers for values
	ValueType* d_values[BUFFER_COUNT];

	// Selector into the set of device vector pointers (i.e., where the results are)
	int selector;

	// Constructor
	MultipleBuffer()
	{
		selector = 0;
		for (int i = 0; i < BUFFER_COUNT; i++) {
			d_keys[i] = NULL;
			d_values[i] = NULL;
		}
	}
};



/**
 * Double buffer (a.k.a. page-flip, ping-pong, etc.) version of the
 * multi-buffer storage abstraction above.
 *
 * Many of the B40C primitives are templated upon the DoubleBuffer type: they
 * are compiled differently depending upon whether the declared type contains
 * keys-only versus key-value pairs (i.e., whether ValueType is util::NullType
 * or some real type).
 *
 * Declaring keys-only storage wrapper:
 *
 * 		DoubleBuffer<KeyType> key_storage;
 *
 * Declaring key-value storage wrapper:
 *
 * 		DoubleBuffer<KeyType, ValueType> key_value_storage;
 *
 */
template <
	typename KeyType,
	typename ValueType = util::NullType>
struct DoubleBuffer : MultipleBuffer<2, KeyType, ValueType>
{
	typedef MultipleBuffer<2, KeyType, ValueType> ParentType;

	// Constructor
	DoubleBuffer() : ParentType() {}

	// Constructor
	DoubleBuffer(
		KeyType* keys) : ParentType()

	{
		this->d_keys[0] = keys;
	}

	// Constructor
	DoubleBuffer(
		KeyType* keys,
		ValueType* values) : ParentType()
	{
		this->d_keys[0] = keys;
		this->d_values[0] = values;
	}

	// Constructor
	DoubleBuffer(
		KeyType* keys0,
		KeyType* keys1,
		ValueType* values0,
		ValueType* values1) : ParentType()
	{
		this->d_keys[0] = keys0;
		this->d_keys[1] = keys1;
		this->d_values[0] = values0;
		this->d_values[1] = values1;
	}
};


/**
 * Triple buffer version of the multi-buffer storage abstraction above.
 */
template <
	typename KeyType,
	typename ValueType = util::NullType>
struct TripleBuffer : MultipleBuffer<3, KeyType, ValueType>
{
	typedef MultipleBuffer<3, KeyType, ValueType> ParentType;

	// Constructor
	TripleBuffer() : ParentType() {}
};



} // namespace util
} // namespace b40c

