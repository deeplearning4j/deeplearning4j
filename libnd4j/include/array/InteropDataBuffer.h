/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#include <array/DataBuffer.h>
#include <array/DataType.h>
#include <system/common.h>

#include <memory>

#ifndef LIBND4J_INTEROPDATABUFFER_H
#define LIBND4J_INTEROPDATABUFFER_H

namespace sd {
/**
 * This class is a wrapper for DataBuffer, suitable for sharing DataBuffer between front-end and back-end languages
 */
class SD_LIB_EXPORT InteropDataBuffer {
 private:
  DataBuffer *_dataBuffer = nullptr;
  bool owner;
  DataType _dataType = DataType::UNKNOWN;
 public:
  size_t _cachedLenInBytes = 0;
  bool _closed = false;
  bool isConstant = false;

  InteropDataBuffer(InteropDataBuffer *dataBuffer, uint64_t length);
  InteropDataBuffer(DataBuffer * databuffer);
  InteropDataBuffer(size_t lenInBytes, DataType dtype, bool allocateBoth);
  ~InteropDataBuffer() {
    // Mark as closed FIRST to prevent any concurrent access
    _closed = true;
    // If we own the DataBuffer and it's not constant, clean it up
    if (owner && !isConstant && _dataBuffer != nullptr) {
      delete _dataBuffer;
    }
    // Always null out the pointer to prevent use-after-free
    _dataBuffer = nullptr;
  }

#ifndef __JAVACPP_HACK__
  DataBuffer * getDataBuffer() const;
  DataBuffer * dataBuffer();
  // In InteropDataBuffer class, add these public methods:
  // Check if the underlying DataBuffer pointer is still valid
  // by checking if it's non-null
  bool hasValidDataBuffer() const { return _dataBuffer != nullptr; }

  // Get the DataBuffer pointer directly without validation
  DataBuffer* getDataBufferDirect() const { return _dataBuffer; }

  // Mark the DataBuffer as invalid (set to null)
  void invalidateDataBuffer() { _dataBuffer = nullptr; }
  bool isOwner() const { return owner; }

  // Safe accessor that doesn't touch other fields
  DataBuffer* getDataBufferUnsafe() const {
    return _dataBuffer;
  }
#endif

  void printDbAllocationTrace();

  void *primary() const;
  void *special() const;

  void markConstant(bool reallyConstant) {
    isConstant = reallyConstant;
    dataBuffer()->markConstant(reallyConstant);
  }

  void setPrimary(void *ptr, size_t length);
  void setSpecial(void *ptr, size_t length);

  void expand(size_t newlength);

  int deviceId() const;
  void setDeviceId(int deviceId);

  //updates whether the buffer is the owner of its associated buffers or not.
  void markOwner(bool owner);

  int useCount() const;

  static void registerSpecialUse(const std::vector<const InteropDataBuffer *> &writeList,
                                 const std::vector<const InteropDataBuffer *> &readList);
  static void prepareSpecialUse(const std::vector<const InteropDataBuffer *> &writeList,
                                const std::vector<const InteropDataBuffer *> &readList,
                                bool synchronizeWritables = false);

  static void registerPrimaryUse(const std::vector<const InteropDataBuffer *> &writeList,
                                 const std::vector<const InteropDataBuffer *> &readList);
  static void preparePrimaryUse(const std::vector<const InteropDataBuffer *> &writeList,
                                const std::vector<const InteropDataBuffer *> &readList,
                                bool synchronizeWritables = false);
};
}  // namespace sd

#endif  // LIBND4J_INTEROPDATABUFFER_H
