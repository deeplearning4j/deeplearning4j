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

#ifndef LIBND4J_ENVIRONMENT_H
#define LIBND4J_ENVIRONMENT_H
#include <array/DataType.h>
#include <types/pair.h>

#include <atomic>
#include <stdexcept>
#include <vector>
#include <config.h>


namespace sd {
class SD_LIB_EXPORT Environment {
 private:
  std::atomic<int> _tadThreshold;
  std::atomic<int> _elementThreshold;
  std::atomic<bool> _verbose;
  std::atomic<bool> _debug;
  std::atomic<bool> _leaks;
  std::atomic<bool> _profile;
  std::atomic<sd::DataType> _dataType;
  std::atomic<bool> _precBoost;
  std::atomic<bool> _useONEDNN{true};
  std::atomic<bool> _allowHelpers{true};
  std::atomic<bool> funcTracePrintDeallocate;
  std::atomic<bool> funcTracePrintAllocate;
  std::atomic<int> _maxThreads;
  std::atomic<int> _maxMasterThreads;
  std::atomic<bool> deleteSpecial{true};
  std::atomic<bool> deletePrimary{true};
  std::atomic<bool> deleteShapeInfo{true};
  std::atomic<bool> _checkInputChange{false};
  std::atomic<bool> _checkOutputChange{false};
  std::atomic<bool> _logNDArrayEvenuts{false};
  std::atomic<bool> _logNativeNDArrayCreation{false};
  // these fields hold defaults
  std::atomic<int64_t> _maxTotalPrimaryMemory{-1};
  std::atomic<int64_t> _maxTotalSpecialMemory{-1};
  std::atomic<int64_t> _maxDeviceMemory{-1};
  bool _blasFallback = false;
  std::atomic<bool> _enableBlasFall{true};

#ifdef SD_EXPERIMENTAL_ENABLED
  const bool _experimental = true;
#else
  const bool _experimental = false;
#endif




  // device compute capability for CUDA
  std::vector<Pair> _capabilities;

  Environment();

 public:
  ~Environment();
  /**
   * These 3 fields are mostly for CUDA/cuBLAS version tracking
   */
  int _blasMajorVersion = 0;
  int _blasMinorVersion = 0;
  int _blasPatchVersion = 0;

  static Environment& getInstance();

  bool isEnableBlas() {
    return _enableBlasFall.load();
  }

  void setEnableBlas(bool reallyEnable) {
    _enableBlasFall.store(reallyEnable);
  }

  /**
   * When log ndarray evens is true in c++
   * certain features of ndarray logging will trigger such as what ndarray constructors are being called.
   *  A great use case for this is for detecting subtle changes in ndarrays like move constructor calls
   *  which  can cause the underlying data to change.
   * @return
   */
  bool isLogNativeNDArrayCreation();
  void setLogNativeNDArrayCreation(bool logNativeNDArrayCreation);

  /**
   * This is mostly a java feature. We can use this to build a framework
   * for logging ndarray events from c++ later.
   * @return
   */
  bool isLogNDArrayEvents();

  void setLogNDArrayEvents(bool logNDArrayEvents);

  /**
   * This is mainly for debugging. This toggles
   * deletion of shape info descriptors.
   * This can be used to isolate potential issues with shape info
   * memory management.
   * The next concern is why have this at all?
   * Historically, we had issues with shape descriptors and shape info
   * buffers being deallocated when they shouldn't be due to stack based deallocation.
   * By controlling everything with normal heap allocation, manual deletes and configurable behavior
   * we can keep memory management consistent and predictable.
   */

  bool isDeleteSpecial();
  void setDeleteSpecial(bool reallyDelete);
  bool isDeletePrimary();
  void setDeletePrimary(bool reallyDelete);


  /**
   * Checks whether the outputs of the op have changed
   * by duplicating them before and after the op runs
   * if it doesn't change it throws an exception.
   * @return
   */
  bool isCheckOutputChange();

  void setCheckOutputChange(bool reallyCheck);

  /**
   * Checks whether immutable ops changed their inputs by
   * duplicating each input and ensuring they're still equal after the op runs.
   * @return
   */
  bool isCheckInputChange();
  void setCheckInputChange(bool reallyCheck);

  bool isVerbose();
  void setVerbose(bool reallyVerbose);
  bool isDebug();
  bool isProfiling();
  bool isDetectingLeaks();
  bool isDebugAndVerbose();
  void setDebug(bool reallyDebug);
  void setProfiling(bool reallyProfile);
  void setLeaksDetector(bool reallyDetect);
  bool helpersAllowed();
  void allowHelpers(bool reallyAllow);

  bool blasFallback();

  int tadThreshold();
  void setTadThreshold(int threshold);

  int elementwiseThreshold();
  void setElementwiseThreshold(int threshold);

  int maxThreads();
  void setMaxThreads(int max);

  int maxMasterThreads();
  void setMaxMasterThreads(int max);

  /*
   * Legacy memory limits API, still used in new API as simplified version
   */
  void setMaxPrimaryMemory(uint64_t maxBytes);
  void setMaxSpecialyMemory(uint64_t maxBytes);
  void setMaxDeviceMemory(uint64_t maxBytes);

  uint64_t maxPrimaryMemory();
  uint64_t maxSpecialMemory();
  ////////////////////////

  /*
   * Methods for memory limits/counters
   */
  void setGroupLimit(int group, sd::LongType numBytes);
  void setDeviceLimit(int deviceId, sd::LongType numBytes);

  sd::LongType getGroupLimit(int group);
  sd::LongType getDeviceLimit(int deviceId);

  sd::LongType getGroupCounter(int group);
  sd::LongType getDeviceCounter(int deviceId);
  ////////////////////////

  bool isUseONEDNN() { return _useONEDNN.load(); }
  void setUseONEDNN(bool useMKLDNN) { _useONEDNN.store(useMKLDNN); }

  sd::DataType defaultFloatDataType();
  void setDefaultFloatDataType(sd::DataType dtype);

  bool precisionBoostAllowed();
  void allowPrecisionBoost(bool reallyAllow);

  bool isExperimentalBuild();

  bool isCPU();

  int blasMajorVersion();
  int blasMinorVersion();
  int blasPatchVersion();

  std::vector<Pair>& capabilities();


  bool isFuncTracePrintDeallocate();
  void setFuncTracePrintDeallocate(bool reallyPrint);
  bool isFuncTracePrintAllocate();
  void setFuncTracePrintAllocate(bool reallyPrint);

  bool isDeleteShapeInfo();
  void setDeleteShapeInfo(bool deleteShapeInfo);
};
}  // namespace sd

#endif  // LIBND4J_ENVIRONMENT_H
