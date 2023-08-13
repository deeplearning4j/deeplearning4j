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
#include <array/DataTypeUtils.h>
#include <exceptions/allocation_exception.h>
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <memory/MemoryCounter.h>
#include <system/op_boilerplate.h>

#include "../DataBuffer.h"

namespace sd {
void DataBuffer::expand(const uint64_t size) {
  if (size > _lenInBytes) {
    // allocate new buffer
    int8_t* newBuffer = nullptr;
    int8_t* newSpecialBuffer = nullptr;
    ALLOCATE_SPECIAL(newSpecialBuffer, _workspace, size, int8_t);

    // copy data from existing buffer
    if (_primaryBuffer != nullptr) {
      // there's non-zero chance that primary buffer doesn't exist yet
      ALLOCATE(newBuffer, _workspace, size, int8_t);
      std::memcpy(newBuffer, _primaryBuffer, _lenInBytes);

      if (_isOwnerPrimary) {
        auto ipb = reinterpret_cast<int8_t*>(_primaryBuffer);
        RELEASE(ipb, _workspace);
      }

      _primaryBuffer = newBuffer;
      _isOwnerPrimary = true;
    }

    cudaMemcpy(newSpecialBuffer, _specialBuffer, _lenInBytes, cudaMemcpyDeviceToDevice);

    if (_isOwnerSpecial) {
      auto isb = reinterpret_cast<int8_t*>(_specialBuffer);
      RELEASE_SPECIAL(isb, _workspace);
    }

    _specialBuffer = newSpecialBuffer;
    _lenInBytes = size;
    _isOwnerSpecial = true;
  }
}

void DataBuffer::showBufferLimited() {

}

void DataBuffer::showCounters(const char* msg1, const char* msg2) {
  sd_debug("%s %s || primary %p special %p :: wP: %d wS: %d rP: %d rS: %d\n", msg1, msg2, _primaryBuffer,
           _specialBuffer, (int)_writePrimary.load(), (int)_writeSpecial.load(), (int)_readPrimary.load(),
           (int)_readSpecial.load());
}
////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {
#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    allocationStackTraceSpecial = new backward::StackTrace();
    allocationStackTraceSpecial->load_here(32);
  }

#endif

  if (_specialBuffer == nullptr) {
    auto deviceId = sd::AffinityManager::currentDeviceId();

    if (_workspace == nullptr) {
      if (!sd::memory::MemoryCounter::getInstance().validate(getLenInBytes()))
        throw sd::allocation_exception::build("Requested amount exceeds device limits",
                                              sd::memory::MemoryCounter::getInstance().deviceLimit(deviceId),
                                              getLenInBytes());
    }

    sd_printf("Allocating special buffer of size %lld on device %d\n", getLenInBytes(), deviceId);
    ALLOCATE_SPECIAL(_specialBuffer, _workspace, getLenInBytes(), int8_t);
    _isOwnerSpecial = true;

    sd_print("After allocated special\n");
    if (_workspace == nullptr) {
      sd::memory::MemoryCounter::getInstance().countIn(deviceId, getLenInBytes());
      sd::memory::MemoryCounter::getInstance().countIn(sd::memory::MemoryType::DEVICE, getLenInBytes());

    }
  } else if(getLenInBytes() == 0) {
    std::string errorMessage;
    errorMessage += "DataBuffer::allocateSpecial: ";
    errorMessage += "Special buffer is already allocated";
    errorMessage += " or length is 0";
    errorMessage += "Length is: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += "Special buffer is nullptr : ";
    errorMessage += std::to_string(_specialBuffer == nullptr);
    THROW_EXCEPTION(errorMessage.c_str());
  }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {
  if (isPrimaryActual() && !forceSync) {
    return;
  }

  if(_specialBuffer == nullptr)
    return;


  allocatePrimary();


  auto res = cudaStreamSynchronize(*context->getCudaStream());
  if (res != 0) throw cuda_exception::build("DataBuffer::syncToPrimary failed to to some previous kernel failre", res);

  res = cudaMemcpy(_primaryBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToHost);
  if (res != 0) throw cuda_exception::build("DataBuffer::syncToPrimary cudaMemcpy failed", res);

  readPrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial(const bool forceSync) {
  // in this case there's nothing to do here
  if (_primaryBuffer == nullptr) return;

  if (isSpecialActual() && !forceSync) {
    return;
  }

  allocateSpecial();

  if(_specialBuffer == nullptr || _primaryBuffer == nullptr)
    return;

  auto res = cudaMemcpy(_specialBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice);
  if (res != 0) throw cuda_exception::build("DataBuffer::syncToSpecial cudaMemcpy failed", res);

  readSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {

  if (_isOwnerSpecial && _specialBuffer != nullptr) {
    auto p = reinterpret_cast<int8_t*>(_specialBuffer);
#if defined(SD_GCC_FUNCTRACE)
    if(Environment::getInstance().isFuncTracePrintAllocate()) {
      sd_print("Beginning printing for allocation part of  deallocation event deleteSpecial\n");
      Printer p2;
      if(allocationStackTraceSpecial != nullptr && allocationStackTraceSpecial->size() > 0)
        p2.print(*allocationStackTraceSpecial);
      else {
        sd_print("No stack trace available for deletePrimary\n");
      }
      sd_print("End printing for allocation part of deallocation event deleteSpecial\n");
    }

    if(Environment::getInstance().isFuncTracePrintDeallocate()) {
      sd_print("Beginning printing for deallocation event deleteSpecial\n");
      Printer p2;
      StackTrace deallocTrace;
      deallocTrace.load_here();
      p2.print(deallocTrace);
      sd_print("End printing for deallocation event deleteSpecial\n");

    }

#endif
    RELEASE_SPECIAL(p, _workspace);
    _specialBuffer = nullptr;
    _isOwnerSpecial = false;

    // count out towards DataBuffer device, only if we're not in workspace
    if (_workspace == nullptr) {
      sd::memory::MemoryCounter::getInstance().countOut(_deviceId, getLenInBytes());
      sd::memory::MemoryCounter::getInstance().countOut(sd::memory::MemoryType::DEVICE, getLenInBytes());
    }
  }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {
  _counter.store(0L);
  _writePrimary.store(0L);
  _writeSpecial.store(0L);
  _readPrimary.store(0L);
  _readSpecial.store(0L);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyCounters(const DataBuffer& other) {
  _counter.store(other._counter);
  _writePrimary.store(other._readSpecial);
  _writeSpecial.store(other._readPrimary);
  _readPrimary.store(other._writeSpecial);
  _readSpecial.store(other._writePrimary);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFrom(const DataBuffer& other, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                const sd::LongType offsetOther) {  // copies only to special buffer

  if (other._primaryBuffer == nullptr && other._specialBuffer == nullptr) {
    return;
  }

  if (sizeToCopyinBytes == 0) {
    sizeToCopyinBytes = other.getLenInBytes();
  }
  if (sizeToCopyinBytes == 0) {
    return;
  }

  if (other.isPrimaryActual()) {
    auto res = cudaMemcpy(
        static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._primaryBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes, cudaMemcpyHostToDevice);
    if (res != 0)
      throw cuda_exception::build("DataBuffer::copyBufferFrom: cudaMemcpy_cudaMemcpyHostToDevice failed!", res);
    other.readPrimary();
  } else {
    auto res = cudaMemcpy(
        static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._specialBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes, cudaMemcpyDeviceToDevice);
    if (res != 0)
      throw cuda_exception::build("DataBuffer::copyBufferFrom: cudaMemcpy_cudaMemcpyDeviceToDevice failed!", res);
    other.readSpecial();
  }

  writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                    const sd::LongType offsetHostBuffer) {  // copies only to special buffer

  if (hostBuffer == nullptr) return;

  if (sizeToCopyinBytes == 0) sizeToCopyinBytes = getLenInBytes();
  if (sizeToCopyinBytes == 0) return;

  auto res =
      cudaMemcpy(static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
                 static_cast<const int8_t*>(hostBuffer) + offsetHostBuffer * DataTypeUtils::sizeOfElement(_dataType),
                 sizeToCopyinBytes, cudaMemcpyHostToDevice);
  if (res != 0)
    throw cuda_exception::build("DataBuffer::copyBufferFromHost: cudaMemcpy_cudaMemcpyHostToDevice failed!", res);

  writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setSpecial(void* special, const bool isOwnerSpecial) {
  //note we don't use locks here
  deleteSpecial();
  _specialBuffer = special;
  _isOwnerSpecial = isOwnerSpecial;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateBuffers(const bool allocBoth) {  // always allocate special buffer only (cuda case)
  allocateSpecial();
  if (allocBoth) allocatePrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setToZeroBuffers(const bool both) {
  if(getLenInBytes() < 1 || special() == nullptr)
    return;
  cudaMemsetAsync(special(), 0, getLenInBytes(), *LaunchContext::defaultContext()->getCudaStream());
  auto res = cudaStreamSynchronize(*LaunchContext::defaultContext()->getCudaStream());
  if (res != 0) throw cuda_exception::build("DataBuffer::setToZeroBuffers: streamSync failed!", res);

  writeSpecial();

  if (both) {
    memset(primary(), 0, getLenInBytes());
    readPrimary();
  }
}

/////////////////////////
void DataBuffer::memcpy(const DataBuffer& dst, const DataBuffer& src) {
  if (src._lenInBytes > dst._lenInBytes)
    THROW_EXCEPTION("DataBuffer::memcpy: Source data buffer is larger than destination");

  int res = 0;
  if (src.isSpecialActual()) {
    res = cudaMemcpyAsync(dst._specialBuffer, src._specialBuffer, src.getLenInBytes(), cudaMemcpyDeviceToDevice,
                          *LaunchContext::defaultContext()->getCudaStream());
  } else if (src.isPrimaryActual()) {
    res = cudaMemcpyAsync(dst._specialBuffer, src._primaryBuffer, src.getLenInBytes(), cudaMemcpyHostToDevice,
                          *LaunchContext::defaultContext()->getCudaStream());
  }

  if (res != 0) throw cuda_exception::build("DataBuffer::memcpy: cudaMemcpyAsync failed!", res);

  res = cudaStreamSynchronize(*LaunchContext::defaultContext()->getCudaStream());
  if (res != 0) throw cuda_exception::build("DataBuffer::memcpy: streamSync failed!", res);

  dst.writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::migrate() {
  memory::Workspace* newWorkspace = nullptr;
  void* newBuffer;
  ALLOCATE_SPECIAL(newBuffer, newWorkspace, getLenInBytes(), int8_t);
  auto res = cudaMemcpy(newBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToDevice);
  if (res != 0) throw cuda_exception::build("DataBuffer::migrate: cudaMemcpyAsync failed!", res);

  if (_isOwnerSpecial) {
    // now we're releasing original buffer
    RELEASE_SPECIAL(_specialBuffer, _workspace);
  }

  _isOwnerSpecial = true;
  _specialBuffer = newBuffer;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const { _writePrimary = ++_counter; }
void DataBuffer::writeSpecial() const { _writeSpecial = ++_counter; }
void DataBuffer::readPrimary() const { _readPrimary = ++_counter; }
void DataBuffer::readSpecial() const { _readSpecial = ++_counter; }
bool DataBuffer::isPrimaryActual() const {
  return (_writePrimary.load() > _writeSpecial.load() || _readPrimary.load() > _writeSpecial.load());
}
bool DataBuffer::isSpecialActual() const {
  return (_writeSpecial.load() > _writePrimary.load() || _readSpecial.load() > _writePrimary.load());
}

}  // namespace sd
