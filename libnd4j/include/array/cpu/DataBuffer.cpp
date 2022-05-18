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
#include <array/DataBuffer.h>
#include <array/DataTypeUtils.h>
#if defined(HAVE_VEDA)
#include <ops/declarable/platform/vednn/veda_helper.h>
#endif

namespace sd {
void DataBuffer::expand(const uint64_t size) {
  if (size > _lenInBytes) {
    // allocate new buffer
    int8_t* newBuffer = nullptr;
    ALLOCATE(newBuffer, _workspace, size, int8_t);

    // copy data from existing buffer
    std::memcpy(newBuffer, _primaryBuffer, _lenInBytes);

    if (_isOwnerPrimary) {
      RELEASE(reinterpret_cast<int8_t*>(_primaryBuffer), _workspace);
    }

    _primaryBuffer = newBuffer;
    _lenInBytes = size;
    _isOwnerPrimary = true;
  }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateBuffers(const bool allocBoth) {  // always allocate primary buffer only (cpu case)

  allocatePrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFrom(const DataBuffer& other, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                const sd::LongType offsetOther) {
  if (sizeToCopyinBytes == 0) sizeToCopyinBytes = other.getLenInBytes();
  if (sizeToCopyinBytes == 0) return;

  if (other._primaryBuffer != nullptr)
    std::memcpy(
        static_cast<int8_t*>(_primaryBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._primaryBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                    const sd::LongType offsetHostBuffer) {
  if (sizeToCopyinBytes == 0) sizeToCopyinBytes = getLenInBytes();
  if (sizeToCopyinBytes == 0) return;

  if (hostBuffer != nullptr)
    std::memcpy(static_cast<int8_t*>(_primaryBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
                static_cast<const int8_t*>(hostBuffer) + offsetHostBuffer * DataTypeUtils::sizeOfElement(_dataType),
                sizeToCopyinBytes);
}

/////////////////////////
void DataBuffer::memcpy(const DataBuffer& dst, const DataBuffer& src) {
  if (src._lenInBytes > dst._lenInBytes)
    throw std::runtime_error("DataBuffer::memcpy: Source data buffer is larger than destination");

  std::memcpy(dst._primaryBuffer, src._primaryBuffer, src._lenInBytes);
  dst.readPrimary();
}

////////////////////////////////////////////////////////////////////////

#if defined(HAVE_VEDA)
////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {
  // device id for now is 0
  if (_specialBuffer) {
#if defined(DEBUG_VEDA_LOGS)
    sd_debug("%s \n", "remove Veda Buffer");
#endif
    VEDAdeviceptr v = (VEDAdeviceptr)_specialBuffer;
    VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
    SCOPED_VEDA_CONTEXT scopedContext(handle.getDevice());
    VEDA_CALL_THROW(vedaMemFreeAsync(v, 0));
    // sync here
    // scopedContext.sync();
    _specialBuffer = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////
void** DataBuffer::getPtrToSpecial() const { return (void**)&_specialBuffer; }

void DataBuffer::showBufferLimited() {
#if defined(DEBUG_VEDA_LOGS)
  float* x = (float*)_primaryBuffer;
  size_t size = getLenInBytes();
  size = size > 80 ? 80 : 0;
  sd_debug("cpu: %p\n", (void*)x);
  for (int i = 0; i < size / sizeof(float); i++) sd_debug("%f, ", x[i]);
  sd_debug("%s", "\n");
#endif
}
////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {
  if (isPrimaryActual() && !forceSync) {
    return;
  }
  // do it if we have _specialBuffer otherwise escape this as no op
  if (_specialBuffer) {
    allocatePrimary();
    // lets copy from _specialBuffer and sync it back
    // we will take device 0 as usual and sync on it
    sd_debug("%s \n", "syncToPrimary Veda Buffer");
#if defined(DEBUG_VEDA_LOGS)
    sd_debug("syncToPrimary--%p ---%p---{\n", _primaryBuffer, _specialBuffer);
    showBufferLimited();
#endif
    VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
    SCOPED_VEDA_CONTEXT scopedContext(handle.getDevice());
    VEDA_CALL_THROW(vedaMemcpyDtoHAsync(_primaryBuffer, (VEDAdeviceptr)_specialBuffer, getLenInBytes(), 0));
    // sync ops here to read completed result
    scopedContext.sync();
    readPrimary();
#if defined(DEBUG_VEDA_LOGS)
    if (sd::Environment::getInstance().isDebug() && sd::Environment::getInstance().isVerbose()) {
      auto fshow = handle.getFunctionByConstPtrName("showBufferVe");
      VEDA_CALL_THROW(vedaLaunchKernel(fshow, 0, (VEDAdeviceptr)_specialBuffer));
      scopedContext.sync();
    }
    sd_debug("%s", "----after---\n");
    // show buffer
    showBufferLimited();
    sd_debug("%s", "----}\n");
#endif
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
  _writePrimary.store(other._writePrimary);
  _writeSpecial.store(other._writeSpecial);
  _readPrimary.store(other._readPrimary);
  _readSpecial.store(other._readSpecial);
}

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

#else

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyCounters(const DataBuffer& other) {}

void DataBuffer::writePrimary() const {}
void DataBuffer::writeSpecial() const {}
void DataBuffer::readPrimary() const {}
void DataBuffer::readSpecial() const {}
bool DataBuffer::isPrimaryActual() const { return true; }
bool DataBuffer::isSpecialActual() const { return false; }
#endif

////////////////////////////////////////////////////////////////////////
void DataBuffer::setSpecial(void* special, const bool isOwnerSpecail) {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setToZeroBuffers(const bool both) { memset(primary(), 0, getLenInBytes()); }

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial(const bool forceSync) {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::migrate() {}

void DataBuffer::showCounters(const char* msg1, const char* msg2) {
#if defined(HAVE_VEDA) && defined(DEBUG_VEDA_LOGS)
  sd_debug("%s %s || primary %p special %p :: wP: %d wS: %d rP: %d rS: %d\n", msg1, msg2, _primaryBuffer,
           _specialBuffer, (int)_writePrimary.load(), (int)_writeSpecial.load(), (int)_readPrimary.load(),
           (int)_readSpecial.load());
#endif
}

}  // namespace sd
