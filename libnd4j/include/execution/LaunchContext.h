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
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_CUDACONTEXT_H
#define LIBND4J_CUDACONTEXT_H


#ifndef __JAVACPP_HACK__
#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "config.h"
#endif

// used for MKLDNN etc
#if !defined(__STANDALONE_BUILD__)
#include "config.h"
#endif

#include <execution/ContextBuffers.h>
#include <execution/ErrorReference.h>
#include <memory/Workspace.h>
#include <system/common.h>
#include <system/op_boilerplate.h>

#include <memory>
#include <mutex>
#include <vector>

namespace sd {

class SD_LIB_EXPORT LaunchContext {
 private:
  static std::vector<std::shared_ptr<LaunchContext>> _contexts;
  static std::mutex _mutex;

  static SD_MAP_IMPL<int, std::mutex*> _deviceMutexes;

  // used for MKLDNN
  void* _engine = nullptr;

#ifdef SD_CUDA

#ifndef __JAVACPP_HACK__

  void* _cublasHandle = nullptr;
  void* _cusolverHandle = nullptr;

#endif  // JCPP

  bool _isAllocated = false;
#endif  // CUDA
  memory::Workspace* _workspace = nullptr;
  int _deviceID = 0;

 public:
#ifdef SD_CUDA

#ifndef __JAVACPP_HACK__
  LaunchContext(cudaStream_t* cudaStream, cudaStream_t& specialCudaStream, void* reductionPointer = nullptr,
                void* scalarPointer = nullptr, int* allocationPointer = nullptr);

  void* getReductionPointer() const;
  void* getScalarPointer() const;
  LongType* getAllocationPointer() const;
  void* getCublasHandle() const;
  void* getCusolverHandle() const;
  void* getCuDnnHandle() const;
  cudaStream_t* getCudaStream() const;
  cudaStream_t* getCudaSpecialStream() const;

  void setReductionPointer(void* reductionPointer);
  void setScalarPointer(void* scalarPointer);
  void setAllocationPointer(int* allocationPointer);
  void setCudaStream(cudaStream_t* cudaStream);
  void setCudaSpecialStream(cudaStream_t* cudaStream);
  void setCublasHandle(void* handle);

#endif  // JCPP

#endif  // CUDA
  LaunchContext(Pointer cudaStream, Pointer reductionPointer = nullptr, Pointer scalarPointer = nullptr,
                Pointer allocationPointer = nullptr);
  LaunchContext();
  ~LaunchContext();
  memory::Workspace* getWorkspace() const {
    return _workspace;
  }
  void setWorkspace(memory::Workspace* theWorkspace) { _workspace = theWorkspace; }

  void* engine();

  int getDeviceID() const { return _deviceID; }
  void setDeviceID(int deviceID) { _deviceID = deviceID; }
  ErrorReference* errorReference();

#ifndef __JAVACPP_HACK__
  // this method returns mutex shared between all threads that use the same device
  static std::mutex* deviceMutex();

#endif

  static bool isInitialized();
  static void releaseBuffers();

  static LaunchContext* defaultContext();

  static void swapContextBuffers(ContextBuffers& buffers);
};

}  // namespace sd
#endif
#endif  // LIBND4J_CUDACONTEXT_H
