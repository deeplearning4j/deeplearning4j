/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author raver119@gmail.com
//
#include <array/ConstantShapeBuffer.h>
#include <sstream>

namespace sd {
ConstantShapeBuffer::ConstantShapeBuffer( PointerWrapper* primary)
    : ConstantShapeBuffer(primary, nullptr) {
#if defined(SD_GCC_FUNCTRACE)
  st = backward::StackTrace();
  st.load_here(32);
#endif

}
ConstantShapeBuffer::ConstantShapeBuffer() : _magic(MAGIC_VALID), _refCount(1) {
  _primaryShapeInfo = nullptr;
  _specialShapeInfo = nullptr;
}
ConstantShapeBuffer::~ConstantShapeBuffer() {
  // Clear magic number first to mark as invalid during destruction
  _magic = 0;

  if(_primaryShapeInfo != nullptr)
    delete _primaryShapeInfo;
  _primaryShapeInfo = nullptr;

  if(_specialShapeInfo != nullptr)
    delete _specialShapeInfo;
  _specialShapeInfo = nullptr;
}

ConstantShapeBuffer::ConstantShapeBuffer( PointerWrapper* primary,
                                          PointerWrapper* special) : _magic(MAGIC_VALID), _refCount(1) {
  _primaryShapeInfo = primary;
  _specialShapeInfo = special;
#if defined(SD_GCC_FUNCTRACE)
  st = backward::StackTrace();
  st.load_here(32);
#endif
}

LongType *ConstantShapeBuffer::primary()  {
  if(_primaryShapeInfo != nullptr) {
    return reinterpret_cast<LongType *>(_primaryShapeInfo->pointer());
  }

  return nullptr;
}

LongType *ConstantShapeBuffer::special()  {
  if(_specialShapeInfo != nullptr) {
    return reinterpret_cast<LongType *>(_specialShapeInfo->pointer());
  }

  return nullptr;
}

LongType *ConstantShapeBuffer::platform()  {
#ifdef SD_CUDA
  return special();
#else
  return primary();
#endif  // CUDABLAS
}

std::string ConstantShapeBuffer::getStackTraceAsString() const {
#if defined(SD_GCC_FUNCTRACE) && !defined(__JAVACPP_HACK__)
  // Use backward::Printer to format the stack trace into a string
  std::ostringstream oss;
  backward::Printer p;
  p.snippet = false;  // Don't include source code snippets
  p.address = true;   // Include addresses
  p.object = false;   // Don't include object file info
  p.color_mode = backward::ColorMode::never;  // No ANSI colors in string

  // Print to our string stream (we need to cast away const to use st)
  // This is safe since print doesn't modify the StackTrace
  backward::StackTrace& mutable_st = const_cast<backward::StackTrace&>(st);
  p.print(mutable_st, oss);

  return oss.str();
#else
  return "";  // Return empty string when functrace is not enabled
#endif
}

void ConstantShapeBuffer::addRef() {
  _refCount.fetch_add(1, std::memory_order_relaxed);
}

void ConstantShapeBuffer::release() {
  // Decrement refcount - buffer stays in cache even when refcount reaches baseline
  // The cache owns these buffers and is responsible for their lifecycle
  // Don't delete when refcount reaches 1 - that means cache is the only owner
  _refCount.fetch_sub(1, std::memory_order_acq_rel);

  // NOTE: Buffers are deleted only when the cache itself is cleared/destroyed,
  // not when temporary users release their references
}

int ConstantShapeBuffer::getRefCount() const {
  return _refCount.load(std::memory_order_relaxed);
}

}  // namespace sd
