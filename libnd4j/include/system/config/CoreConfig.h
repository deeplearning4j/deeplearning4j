/* ******************************************************************************
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

#ifndef LIBND4J_CORE_CONFIG_H
#define LIBND4J_CORE_CONFIG_H

#include <array/DataType.h>
#include <system/common.h>

#include <atomic>
#include <string>

namespace sd {
namespace config {

/**
 * Core runtime configuration: verbose/debug mode, threading, memory limits,
 * BLAS settings, data type, helpers, and general diagnostics.
 *
 * This subsystem covers the foundational settings that existed from the
 * earliest versions of libnd4j.
 */
class SD_LIB_EXPORT CoreConfig {
 private:
  std::atomic<int> _tadThreshold{1};
  std::atomic<int> _elementThreshold{1024};
  std::atomic<bool> _verbose{false};
  std::atomic<bool> _debug{false};
  std::atomic<bool> _leaks{false};
  std::atomic<bool> _profile{false};
  std::atomic<sd::DataType> _dataType{FLOAT32};
  std::atomic<bool> _precBoost{false};
  std::atomic<bool> _useONEDNN{true};
  std::atomic<bool> _useMPS{true};
  std::atomic<bool> _allowHelpers{true};
  std::atomic<bool> _funcTracePrintDeallocate{false};
  std::atomic<bool> _funcTracePrintAllocate{false};

  std::atomic<int> _maxThreads;
  std::atomic<int> _maxMasterThreads;
  std::atomic<bool> _deleteSpecial{true};
  std::atomic<bool> _deletePrimary{true};
  std::atomic<bool> _deleteShapeInfo{true};
  std::atomic<bool> _checkInputChange{false};
  std::atomic<bool> _checkOutputChange{false};
  std::atomic<bool> _logNDArrayEvents{false};
  std::atomic<bool> _logNativeNDArrayCreation{false};

  std::atomic<int64_t> _maxTotalPrimaryMemory{-1};
  std::atomic<int64_t> _maxTotalSpecialMemory{-1};
  std::atomic<int64_t> _maxDeviceMemory{-1};
  bool _blasFallback{false};
  std::atomic<bool> _enableBlasFall{true};

  std::atomic<bool> _serializeBlasCalls{true};
  std::atomic<bool> _serializeBlasCallsSet{false};
  std::atomic<int> _openBlasThreads{0};
  std::atomic<int> _cpuSoftLimitPercent{0};  // proactive memory soft limit (0=disabled, 1-100)

 public:
  CoreConfig();

  // --- Verbose / Debug / Profile ---
  bool isVerbose() { return _verbose.load(); }
  void setVerbose(bool v) { _verbose.store(v); }
  bool isDebug() { return _debug.load(); }
  void setDebug(bool v) { _debug.store(v); }
  bool isDebugAndVerbose() { return isDebug() && isVerbose(); }
  bool isProfiling() { return _profile.load(); }
  void setProfiling(bool v) { _profile.store(v); }
  bool isDetectingLeaks() { return _leaks.load(); }
  void setLeaksDetector(bool v) { _leaks.store(v); }

  // --- Helpers ---
  bool helpersAllowed() { return _allowHelpers.load(); }
  void allowHelpers(bool v) { _allowHelpers.store(v); }
  bool isUseONEDNN() { return _useONEDNN.load(); }
  void setUseONEDNN(bool v) { _useONEDNN.store(v); }
  bool isUseMPS() { return _useMPS.load(); }
  void setUseMPS(bool v) { _useMPS.store(v); }

  // --- Threading ---
  int maxThreads() { return _maxThreads.load(); }
  void setMaxThreads(int max) { _maxThreads.store(max); }
  int maxMasterThreads() { return _maxMasterThreads.load(); }
  void setMaxMasterThreads(int max);
  int tadThreshold() { return _tadThreshold.load(); }
  void setTadThreshold(int v) { _tadThreshold.store(v); }
  int elementwiseThreshold() { return _elementThreshold.load(); }
  void setElementwiseThreshold(int v) { _elementThreshold.store(v); }

  // --- Data type ---
  sd::DataType defaultFloatDataType() { return _dataType.load(); }
  void setDefaultFloatDataType(sd::DataType dtype);
  bool precisionBoostAllowed() { return _precBoost.load(); }
  void allowPrecisionBoost(bool v) { _precBoost.store(v); }

  // --- Memory limits ---
  void setMaxPrimaryMemory(uint64_t maxBytes) { _maxTotalPrimaryMemory.store(maxBytes); }
  void setMaxSpecialMemory(uint64_t maxBytes) { _maxTotalSpecialMemory.store(maxBytes); }
  void setMaxDeviceMemory(uint64_t maxBytes) { _maxDeviceMemory.store(maxBytes); }
  uint64_t maxPrimaryMemory() { return _maxTotalPrimaryMemory.load(); }
  uint64_t maxSpecialMemory() { return _maxTotalSpecialMemory.load(); }
  // Per-process device (GPU) memory budget in bytes; -1 means unbounded (default).
  // Returned as int64_t so the -1 "unset" sentinel survives (siblings return
  // uint64_t and rely on consumers handling the wrap). Read by CudaMemoryPool to
  // cap this process's device pool usage and fail over to host/peer past it.
  int64_t maxDeviceMemory() { return _maxDeviceMemory.load(); }

  // --- CPU soft limit ---
  int cpuSoftLimitPercent() { return _cpuSoftLimitPercent.load(); }
  void setCpuSoftLimitPercent(int percent);

  // --- BLAS ---
  bool blasFallback() { return _blasFallback; }
  bool isEnableBlas() { return _enableBlasFall.load(); }
  void setEnableBlas(bool v) { _enableBlasFall.store(v); }
  bool isSerializeBlasCalls();
  void setSerializeBlasCalls(bool serialize);
  int getOpenBlasThreads();
  void setOpenBlasThreads(int threads);

  // --- Deletion control ---
  bool isDeleteSpecial() { return _deleteSpecial.load(); }
  void setDeleteSpecial(bool v) { _deleteSpecial.store(v); }
  bool isDeletePrimary() { return _deletePrimary.load(); }
  void setDeletePrimary(bool v) { _deletePrimary.store(v); }
  bool isDeleteShapeInfo() { return _deleteShapeInfo.load(); }
  void setDeleteShapeInfo(bool v) { _deleteShapeInfo.store(v); }

  // --- Diagnostics ---
  bool isCheckOutputChange() { return _checkOutputChange.load(); }
  void setCheckOutputChange(bool v) { _checkOutputChange.store(v); }
  bool isCheckInputChange() { return _checkInputChange.load(); }
  void setCheckInputChange(bool v) { _checkInputChange.store(v); }
  bool isLogNDArrayEvents() { return _logNDArrayEvents.load(); }
  void setLogNDArrayEvents(bool v) { _logNDArrayEvents.store(v); }
  bool isLogNativeNDArrayCreation() { return _logNativeNDArrayCreation.load(); }
  void setLogNativeNDArrayCreation(bool v) { _logNativeNDArrayCreation.store(v); }
  bool isFuncTracePrintAllocate() { return _funcTracePrintAllocate.load(); }
  void setFuncTracePrintAllocate(bool v) { _funcTracePrintAllocate.store(v); }
  bool isFuncTracePrintDeallocate() { return _funcTracePrintDeallocate.load(); }
  void setFuncTracePrintDeallocate(bool v) { _funcTracePrintDeallocate.store(v); }

  // --- Path helpers ---
  std::string homeDirectory() const;
  std::string cudaToolkitPath() const;

  /**
   * Initialize from environment variables.
   * Called once during Environment construction.
   */
  void initFromEnvironment();
};

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_CORE_CONFIG_H
