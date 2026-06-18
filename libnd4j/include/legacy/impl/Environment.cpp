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
// Created by raver119 on 06.10.2017.
//
// Refactored: Environment now delegates to subsystem config classes.
// Each subsystem handles its own env var parsing via initFromEnvironment().
//

#include <system/Environment.h>
#include <memory/MemoryCounter.h>
#include <graph/gpu/DspCudaDispatch.h>
#include <mutex>

// Lifecycle tracker includes for enabling/disabling via Environment setters
#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <array/ShapeCacheLifecycleTracker.h>
#include <array/DeallocatorServiceLifecycleTracker.h>
#include <graph/OpContextLifecycleTracker.h>

namespace sd {

Environment::Environment() {
  // Initialize each subsystem from environment variables.
  // Order matters: core first (threading), then CUDA (device init), then the rest.
  _core.initFromEnvironment();

  _cuda.initFromEnvironment();

  // Sync BLAS version from CUDA subsystem to legacy public fields
  _blasMajorVersion = _cuda.blasMajorVersion();
  _blasMinorVersion = _cuda.blasMinorVersion();
  _blasPatchVersion = _cuda.blasPatchVersion();

  _memory.initFromEnvironment();
  _lifecycle.initFromEnvironment();
  _print.initFromEnvironment();

  // Triton and DSP are only meaningful on CUDA but their env var parsing is safe on CPU
  _triton.initFromEnvironment();
  _dsp.initFromEnvironment();
}

Environment::~Environment() {
  //
}

Environment &Environment::getInstance() {
  static Environment* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new Environment();
  });
  return *instance;
}

bool Environment::isCPU() {
  return !sd::graph::dspIsCudaBuild();
}

// --- Memory limits/counters (delegate to MemoryCounter singleton) ---
void Environment::setGroupLimit(int group, LongType numBytes) {
  memory::MemoryCounter::getInstance().setGroupLimit((memory::MemoryType)group, numBytes);
}

void Environment::setDeviceLimit(int deviceId, LongType numBytes) {
  memory::MemoryCounter::getInstance().setDeviceLimit(deviceId, numBytes);
}

LongType Environment::getGroupLimit(int group) {
  return memory::MemoryCounter::getInstance().groupLimit((memory::MemoryType)group);
}

LongType Environment::getDeviceLimit(int deviceId) {
  return memory::MemoryCounter::getInstance().deviceLimit(deviceId);
}

LongType Environment::getGroupCounter(int group) {
  return memory::MemoryCounter::getInstance().allocatedGroup((memory::MemoryType)group);
}

LongType Environment::getDeviceCounter(int deviceId) {
  return memory::MemoryCounter::getInstance().allocatedDevice(deviceId);
}

// --- Lifecycle tracker setters that also enable/disable tracker singletons ---
// These must stay in Environment because they touch tracker singletons that
// the subsystem headers should not depend on.

void Environment::setNDArrayTracking(bool enabled) {
  _lifecycle.setNDArrayTracking(enabled);
  array::NDArrayLifecycleTracker::getInstance().setEnabled(enabled);
}

void Environment::setDataBufferTracking(bool enabled) {
  _lifecycle.setDataBufferTracking(enabled);
  array::DataBufferLifecycleTracker::getInstance().setEnabled(enabled);
}

void Environment::setTADCacheTracking(bool enabled) {
  _lifecycle.setTADCacheTracking(enabled);
  array::TADCacheLifecycleTracker::getInstance().setEnabled(enabled);
}

void Environment::setShapeCacheTracking(bool enabled) {
  _lifecycle.setShapeCacheTracking(enabled);
  array::ShapeCacheLifecycleTracker::getInstance().setEnabled(enabled);
}

void Environment::setOpContextTracking(bool enabled) {
  _lifecycle.setOpContextTracking(enabled);
  graph::OpContextLifecycleTracker::getInstance().setEnabled(enabled);
}

}  // namespace sd
