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

#ifndef SD_ARRAY_CONSTANTSHAPEBUFFER_H_
#define SD_ARRAY_CONSTANTSHAPEBUFFER_H_

#include <array/PointerWrapper.h>
#include <system/common.h>

#include <atomic>
#include <memory>
#include <string>
#ifndef  __JAVACPP_HACK__
#if defined(SD_GCC_FUNCTRACE)
#include <exceptions/backward.hpp>
#endif
#endif
namespace sd {

class SD_LIB_EXPORT ConstantShapeBuffer {
 private:
  static constexpr uint32_t MAGIC_VALID = 0x58A9EB0F;  // Magic number for validity check
  uint32_t _magic;  // Validity marker to detect use-after-free/garbage pointers
  PointerWrapper* _primaryShapeInfo;
  PointerWrapper*  _specialShapeInfo;
  std::atomic<int> _refCount;
  bool _isOwner;
  bool _neverDelete;  // If true, release() will never delete this buffer


 public:
  ConstantShapeBuffer( PointerWrapper* primary);
  ConstantShapeBuffer( PointerWrapper* primary, PointerWrapper* special);
  ConstantShapeBuffer();
  ~ConstantShapeBuffer();

  // Padded operator new/delete to protect adjacent glibc chunks from
  // overruns. ConstantShapeBuffer objects are small (~48 bytes) and
  // heavily allocated during shape trie population — any adjacent
  // overrun corrupts the next chunk metadata → SIGABRT on free().
  static void* operator new(size_t size) {
    return std::malloc(size + 4096);
  }
#ifndef __JAVACPP_HACK__
  static void* operator new(size_t size, const std::nothrow_t& tag) noexcept {
    return std::malloc(size + 4096);
  }
#endif
  static void operator delete(void* ptr) noexcept {
    std::free(ptr);
  }
#ifndef __JAVACPP_HACK__
  static void operator delete(void* ptr, const std::nothrow_t& tag) noexcept {
    std::free(ptr);
  }
#endif

  // =========================================================================
  // COPY PREVENTION
  // =========================================================================
  //
  // Copy construction and assignment are DELETED to prevent heap corruption.
  //
  // The previous implementation allowed copying, creating non-owning copies
  // that shared pointers with the original. This was dangerous because:
  //
  // 1. DANGLING POINTERS: If the original buffer is deleted while copies exist,
  //    the copies have dangling pointers to freed memory.
  //
  // 2. REFERENCE COUNT CONFUSION: Each copy had its own refCount, so the
  //    original could be deleted (refCount=0) while copies still held pointers.
  //
  // 3. JNI/JavaCPP ISSUES: The copy constructor was used for @ByVal returns,
  //    creating temporaries with shared pointers that could outlive the original.
  //
  // HOW TO ADAPT CODE THAT PREVIOUSLY COPIED ConstantShapeBuffer:
  //
  // - Use POINTERS instead of values:
  //     ConstantShapeBuffer* buf = helper.bufferForShapeInfo(...);
  //     // Use buf directly, don't copy
  //
  // - Use REFERENCES for function parameters:
  //     void process(const ConstantShapeBuffer& buffer) { ... }
  //
  // - For JavaCPP/JNI: Use pointer returns (@ByRef or Pointer) instead of @ByVal
  //
  // - For STL containers: Use std::vector<ConstantShapeBuffer*> instead of
  //     std::vector<ConstantShapeBuffer>
  //
  // =========================================================================
  ConstantShapeBuffer(const ConstantShapeBuffer& other) = delete;
  ConstantShapeBuffer& operator=(const ConstantShapeBuffer& other) = delete;

  // Move semantics ARE allowed - source is invalidated after move
  ConstantShapeBuffer(ConstantShapeBuffer&& other) noexcept;
  ConstantShapeBuffer& operator=(ConstantShapeBuffer&& other) noexcept;

  /**
   * Check if this buffer is valid (not garbage/use-after-free).
   * Uses magic number validation.
   */
  inline bool isValid() const { return _magic == MAGIC_VALID; }
#ifndef  __JAVACPP_HACK__
#if defined(SD_GCC_FUNCTRACE)
  backward::StackTrace st;
#endif
#endif
  LongType *primary() ;
  LongType *special() ;
  LongType *platform() ;

  /**
   * Get the stack trace as a formatted string.
   * Returns empty string if functrace is not enabled.
   */
  std::string getStackTraceAsString() const;

  /**
   * Manual reference counting for safe cross-JNI ownership.
   * Increments reference count - call when handing out a pointer.
   */
  void addRef();

   /**
    * Manual reference counting for safe cross-JNI ownership.
    * Decrements reference count and deletes when reaching zero.
    * Call when done with a pointer (e.g., in deleteConstantShapeBuffer).
    */
   void release();

   /**
    * Set whether this buffer should never be deleted.
    * Used for view buffers that need to live as long as any NDArray references them,
    * even if the shape cache is cleared.
    */
   void setNeverDelete(bool neverDelete);

   /**
    * Get current reference count (for debugging).
    */
   int getRefCount() const;
 };


}  // namespace sd

#endif  // SD_ARRAY_CONSTANTSHAPEBUFFER_H_
