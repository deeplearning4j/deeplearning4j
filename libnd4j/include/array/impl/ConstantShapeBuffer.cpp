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
ConstantShapeBuffer::ConstantShapeBuffer() : _magic(MAGIC_VALID), _refCount(1), _isOwner(true), _neverDelete(false) {
  _primaryShapeInfo = nullptr;
  _specialShapeInfo = nullptr;
}
ConstantShapeBuffer::~ConstantShapeBuffer() {
  // Clear magic number first to mark as invalid during destruction
  _magic = 0;

  // Only delete the underlying pointers if this instance owns them.
  // Copies created via copy constructor have _isOwner = false and share pointers.
  if (_isOwner) {
    if(_primaryShapeInfo != nullptr)
      delete _primaryShapeInfo;
    if(_specialShapeInfo != nullptr)
      delete _specialShapeInfo;
  }
  _primaryShapeInfo = nullptr;
  _specialShapeInfo = nullptr;
}

// ============================================================================
// COPY OPERATIONS - DELETED
// ============================================================================
// Copy constructor and copy assignment are DELETED in the header.
// See ConstantShapeBuffer.h for explanation of why this is necessary.
//
// The previous implementation created non-owning copies that shared pointers,
// which led to dangling pointers when the original was deleted.
// ============================================================================

// ============================================================================
// MOVE OPERATIONS - ENABLED
// ============================================================================
// Move semantics are safe because the source is invalidated after the move.
// The destination takes full ownership of the pointers.

ConstantShapeBuffer::ConstantShapeBuffer(ConstantShapeBuffer&& other) noexcept
    : _magic(other._magic),
      _primaryShapeInfo(other._primaryShapeInfo),
      _specialShapeInfo(other._specialShapeInfo),
      _refCount(other._refCount.load(std::memory_order_relaxed)),
      _isOwner(other._isOwner),
      _neverDelete(other._neverDelete) {
  // Invalidate source to prevent double-delete
  other._magic = 0;
  other._primaryShapeInfo = nullptr;
  other._specialShapeInfo = nullptr;
  other._refCount.store(0, std::memory_order_relaxed);
  other._isOwner = false;
  other._neverDelete = false;

#if defined(SD_GCC_FUNCTRACE) && !defined(__JAVACPP_HACK__)
  st = std::move(other.st);
#endif
}

ConstantShapeBuffer& ConstantShapeBuffer::operator=(ConstantShapeBuffer&& other) noexcept {
  if (this != &other) {
    // Clean up our current resources if we own them
    if (_isOwner) {
      if (_primaryShapeInfo != nullptr) delete _primaryShapeInfo;
      if (_specialShapeInfo != nullptr) delete _specialShapeInfo;
    }

    // Take ownership from other
    _magic = other._magic;
    _primaryShapeInfo = other._primaryShapeInfo;
    _specialShapeInfo = other._specialShapeInfo;
    _refCount.store(other._refCount.load(std::memory_order_relaxed), std::memory_order_relaxed);
    _isOwner = other._isOwner;
    _neverDelete = other._neverDelete;

    // Invalidate source to prevent double-delete
    other._magic = 0;
    other._primaryShapeInfo = nullptr;
    other._specialShapeInfo = nullptr;
    other._refCount.store(0, std::memory_order_relaxed);
    other._isOwner = false;
    other._neverDelete = false;

#if defined(SD_GCC_FUNCTRACE) && !defined(__JAVACPP_HACK__)
    st = std::move(other.st);
#endif
  }
  return *this;
}

ConstantShapeBuffer::ConstantShapeBuffer( PointerWrapper* primary,
                                          PointerWrapper* special) : _magic(MAGIC_VALID), _refCount(1), _isOwner(true), _neverDelete(false) {
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
  // If _neverDelete is set, never delete this buffer
  // Used for view buffers that shouldn't be deleted by cache clear operations
  if (_neverDelete) {
    _refCount.fetch_sub(1, std::memory_order_relaxed);
    return;
  }

  int oldCount = _refCount.fetch_sub(1, std::memory_order_acq_rel);

  if (oldCount == 1) {
    // We were the last holder (now refCount == 0), delete ourselves.
    // This is safe because no other thread can be accessing this buffer
    // since refCount was 1 (only we held a reference).
    delete this;
  }
  // If oldCount > 1, other holders still exist, don't delete
}

void ConstantShapeBuffer::setNeverDelete(bool neverDelete) {
  _neverDelete = neverDelete;
}

int ConstantShapeBuffer::getRefCount() const {
  return _refCount.load(std::memory_order_relaxed);
}

}  // namespace sd
