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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef DEV_TESTS_DATABUFFER_H
#define DEV_TESTS_DATABUFFER_H

#include <array/DataType.h>
#include <execution/LaunchContext.h>
#include <memory/Workspace.h>
#include <system/common.h>
#include <system/op_boilerplate.h>

#include <cstring>
#include <mutex>
namespace sd {

class SD_LIB_EXPORT DataBuffer {
 private:
  void *_primaryBuffer = nullptr;
  void *_specialBuffer = nullptr;
  LongType _lenInBytes = 0;
  memory::Workspace *_workspace = nullptr;

  std::atomic<int> _deviceId;
  std::mutex _deleteMutex;
#ifndef __JAVACPP_HACK__
#if defined(__CUDABLAS__)
  mutable std::atomic<LongType> _counter;
  mutable std::atomic<LongType> _writePrimary;
  mutable std::atomic<LongType> _writeSpecial;
  mutable std::atomic<LongType> _readPrimary;
  mutable std::atomic<LongType> _readSpecial;
#endif

#if defined(SD_GCC_FUNCTRACE)
  StackTrace *allocationStackTracePrimary = nullptr;
  StackTrace *allocationStackTraceSpecial = nullptr;
  StackTrace *creationStackTrace = nullptr;

#endif


#endif

  bool closed = false;

  // Helper template function for printing host buffer content (implementation in .cpp)
  template <typename T>
  void printHostBufferContent(void* buffer, sd::LongType offset, sd::LongType length);



  void setCountersToZero();
  void copyCounters(const DataBuffer &other);
  void deleteSpecial();
  void deletePrimary();
  void deleteBuffers();
  void setAllocFlags(const bool isOwnerPrimary, const bool isOwnerSpecial = false);
  void allocateBuffers(const bool allocBoth = false);

  void setSpecial(void *special, const bool isOwnerSpecial);

  void copyBufferFromHost(const void *hostBuffer, size_t sizeToCopyinBytes = 0, const LongType offsetThis = 0,
                          const LongType offsetHostBuffer = 0);

 public:

  bool _isOwnerPrimary;
  bool _isOwnerSpecial;
  bool isConstant = false;
  DataType _dataType;

  DataBuffer(void *primary, void *special, const size_t lenInBytes, const DataType dataType,
             const bool isOwnerPrimary = false, const bool isOwnerSpecial = false,
             memory::Workspace *workspace = nullptr);

  DataBuffer(void *primary, const size_t lenInBytes, const DataType dataType, const bool isOwnerPrimary = false,
             memory::Workspace *workspace = nullptr);

  DataBuffer(const void *hostBuffer,  // copies data from hostBuffer to own memory buffer
             const DataType dataType, const size_t lenInBytes, memory::Workspace *workspace = nullptr);

  DataBuffer(const sd::LongType lenInBytes, const DataType dataType, memory::Workspace *workspace = nullptr,
             const bool allocBoth = false);

  DataBuffer(const DataBuffer &other);
  DataBuffer(DataBuffer &&other);
  explicit DataBuffer();
  ~DataBuffer();

  DataBuffer &operator=(const DataBuffer &other);
  DataBuffer &operator=(DataBuffer &&other) noexcept;

  DataType getDataType();
  void setDataType(DataType dataType);
  size_t getLenInBytes() const;

  size_t getNumElements();

  template <typename T>
  void *primaryAtOffset(const LongType offset);
  template <typename T>
  void *specialAtOffset(const LongType offset);

  void *primary();
  void *special();
  void printAllocationTrace();

  void allocatePrimary();
  void allocateSpecial();

  void writePrimary() const;
  void writeSpecial() const;
  void readPrimary() const;
  void readSpecial() const;
  bool isPrimaryActual() const;
  bool isSpecialActual() const;

  void expand(const uint64_t size);

  int deviceId() const;
  void setDeviceId(int deviceId);
  void migrate();

  template <typename T>
  SD_INLINE T *primaryAsT();
  template <typename T>
  SD_INLINE T *specialAsT();

  void markConstant(bool reallyConstant);


  void syncToPrimary(const LaunchContext *context, const bool forceSync = false);
  void syncToSpecial(const bool forceSync = false);

  void setToZeroBuffers(const bool both = false);

  void copyBufferFrom(const DataBuffer &other, size_t sizeToCopyinBytes = 0, const LongType offsetThis = 0,
                      const LongType offsetOther = 0);


  void setPrimaryBuffer(void *buffer, size_t length);
  void setSpecialBuffer(void *buffer, size_t length);


  void  showBufferLimited();
  //for Debug purposes
  void showCounters(const char* msg1, const char* msg2);

  /**
   * This method deletes buffers, if we're owners
   */
  void close();
  void printPrimaryAllocationStackTraces();
  void printSpecialAllocationTraces();
  DataBuffer  dup();
  void printHostDevice(long offset);
  static void memcpy(DataBuffer *dst, DataBuffer *src, sd::LongType startingOffset, sd::LongType dstOffset);
  /**
   * Print detailed buffer information including host and device content if available
   * @param msg - Optional message to display
   * @param offset - Starting offset for printing buffer contents
   * @param limit - Maximum number of elements to print
   */
#ifndef __JAVACPP_HACK__
  void printBufferDebug(const char* msg = nullptr, sd::LongType offset = 0, sd::LongType limit = 10);
#endif
};
///// IMPLEMENTATION OF INLINE METHODS /////

////////////////////////////////////////////////////////////////////////
template <typename T>
T *DataBuffer::primaryAsT() {
  return reinterpret_cast<T *>(_primaryBuffer);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T *DataBuffer::specialAsT() {
  return reinterpret_cast<T *>(_specialBuffer);
}

}  // namespace sd

#endif  // DEV_TESTS_DATABUFFER_H
