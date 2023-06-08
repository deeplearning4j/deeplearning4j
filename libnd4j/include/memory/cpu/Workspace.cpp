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
// CPU workspaces implementation
//
// @author raver119@gmail.com
//

#include "../Workspace.h"

#include <helpers/logger.h>
#include <math/templatemath.h>
#include <stdio.h>
#include <stdlib.h>
#include <system/op_boilerplate.h>

#include <atomic>
#include <cstring>

namespace sd {
namespace memory {
Workspace::Workspace(ExternalWorkspace *external) {
  if (external->sizeHost() > 0) {
    _ptrHost = (char *)external->pointerHost();
    _ptrDevice = (char *)external->pointerDevice();

    _initialSize = external->sizeHost();
    _currentSize = external->sizeHost();
    _offset = 0L;
    _offsetSecondary = 0L;
    this->_cycleAllocations = 0;
    this->_spillsSize = 0;

    _externalized = true;
  }
};

Workspace::Workspace(sd::LongType initialSize, sd::LongType secondaryBytes) {
  if (initialSize > 0) {
    this->_ptrHost = (char *)malloc(initialSize);

    CHECK_ALLOC(this->_ptrHost, "Failed to allocate new workspace", initialSize);

    memset(this->_ptrHost, 0, initialSize);
    this->_allocatedHost = true;
  } else
    this->_allocatedHost = false;

  this->_initialSize = initialSize;
  this->_currentSize = initialSize;
  this->_currentSizeSecondary = 0;
  this->_spillsSizeSecondary = 0;
  this->_offset = 0;
  this->_offsetSecondary = 0;
  this->_cycleAllocations = 0;
  this->_spillsSize = 0;
}

void Workspace::init(sd::LongType bytes, sd::LongType secondaryBytes) {
  if (this->_currentSize < bytes) {
    if (this->_allocatedHost && !_externalized) free((void *)this->_ptrHost);

    this->_ptrHost = (char *)malloc(bytes);

    CHECK_ALLOC(this->_ptrHost, "Failed to allocate new workspace", bytes);

    memset(this->_ptrHost, 0, bytes);
    this->_currentSize = bytes;
    this->_allocatedHost = true;
  }
}

void Workspace::expandBy(sd::LongType numBytes, sd::LongType secondaryBytes) {
  this->init(_currentSize + numBytes, _currentSizeSecondary + secondaryBytes);
}

void Workspace::expandTo(sd::LongType numBytes, sd::LongType secondaryBytes) { this->init(numBytes, secondaryBytes); }

void Workspace::freeSpills() {
  _spillsSize = 0;

  if (_spills.size() < 1) return;

  for (auto v : _spills) free(v);

  _spills.clear();
}

Workspace::~Workspace() {
  if (this->_allocatedHost && !_externalized) free((void *)this->_ptrHost);

  freeSpills();
}

sd::LongType Workspace::getUsedSize() { return getCurrentOffset(); }

sd::LongType Workspace::getCurrentSize() { return _currentSize; }

sd::LongType Workspace::getCurrentOffset() { return _offset.load(); }

void *Workspace::allocateBytes(sd::LongType numBytes) {
  if (numBytes < 1) throw allocation_exception::build("Number of bytes for allocation should be positive", numBytes);

  // numBytes += 32;
  void *result = nullptr;
  this->_cycleAllocations += numBytes;
  this->_mutexAllocation.lock();

  if (_offset.load() + numBytes > _currentSize) {
    sd_debug("Allocating %lld bytes in spills\n", numBytes);
    this->_mutexAllocation.unlock();
#if defined(SD_ALIGNED_ALLOC)
    void *p = aligned_alloc(SD_DESIRED_ALIGNMENT, (numBytes + SD_DESIRED_ALIGNMENT - 1) & (-SD_DESIRED_ALIGNMENT));
#else
    void *p = malloc(numBytes);
#endif
    CHECK_ALLOC(p, "Failed to allocate new workspace", numBytes);

    _mutexSpills.lock();
    _spills.push_back(p);
    _mutexSpills.unlock();

    _spillsSize += numBytes;

    return p;
  }

  result = (void *)(_ptrHost + _offset.load());
  _offset += numBytes;
  // memset(result, 0, (int) numBytes);

  sd_debug("Allocating %lld bytes from workspace; Current PTR: %p; Current offset: %lld\n", numBytes, result,
           _offset.load());

  this->_mutexAllocation.unlock();

  return result;
}

sd::LongType Workspace::getAllocatedSize() { return getCurrentSize() + getSpilledSize(); }

void Workspace::scopeIn() {
  freeSpills();
  init(_cycleAllocations.load());
  _cycleAllocations = 0;
}

void Workspace::scopeOut() {
  _offset = 0;
  _offsetSecondary = 0;
}

sd::LongType Workspace::getSpilledSize() { return _spillsSize.load(); }

void *Workspace::allocateBytes(sd::memory::MemoryType type, sd::LongType numBytes) {
  if (type == DEVICE) THROW_EXCEPTION("CPU backend doesn't have device memory");

  return this->allocateBytes(numBytes);
}

sd::LongType Workspace::getAllocatedSecondarySize() { return 0L; }

sd::LongType Workspace::getCurrentSecondarySize() { return 0L; }

sd::LongType Workspace::getCurrentSecondaryOffset() { return 0L; }

sd::LongType Workspace::getSpilledSecondarySize() { return 0L; }

sd::LongType Workspace::getUsedSecondarySize() { return 0L; }

Workspace *Workspace::clone() {
  // for clone we take whatever is higher: current allocated size, or allocated size of current loop
  return new Workspace(sd::math::sd_max<sd::LongType>(this->getCurrentSize(), this->_cycleAllocations.load()));
}
}  // namespace memory
}  // namespace sd
