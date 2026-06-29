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

//
// Split from NativeOps.cu to reduce object file size for SD_GCC_FUNCTRACE builds
// Contains: shuffle
//

#include <cuda.h>

#include <execution/LaunchContext.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <legacy/NativeOps.h>
#include <loops/special_kernels.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <system/common.h>
#include <array/ArrayOptions.h>
#include <helpers/shape.h>
#include <execution/cuda/LaunchDims.h>

void shuffle(sd::Pointer *extras,
             OpaqueNDArrayArr x,
             OpaqueNDArrayArr z,
             int N,
             OpaqueNDArray dimension,
             OpaqueNDArray shuffleMap) {
  try {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extras[1]);

    auto xType = x[0]->dataType();
    dim3 launchDims = getLaunchDims("shuffle");

    // Extract buffers from each NDArray in the array (host-side vectors)
    std::vector<void*> xBuffers(N);
    std::vector<sd::LongType*> xShapeInfos(N);
    std::vector<sd::LongType*> tadShapeInfoBuffers(N);
    std::vector<sd::LongType*> tadOffsetsBuffers(N);
    std::vector<void*> zBuffers(N);
    std::vector<sd::LongType*> zShapeInfos(N);

    // Extract dimensions once
    sd::LongType* dimensions = reinterpret_cast<sd::LongType*>(dimension->buffer());
    sd::LongType dimLength = shape::length(dimension->shapeInfo());

    // Keep all TadPacks alive until after the kernel launches.
    // CUDA TadPack::~TadPack() calls delete _tadShape which frees the GPU shape
    // info buffer (non-cached CudaShapeBufferCreator output). If tadPackX goes
    // out of scope at the END of each loop body, the raw GPU pointer stored in
    // tadShapeInfoBuffers[i] becomes dangling before the kernel reads it →
    // CUDA error 700 (illegal memory access). Holding all shared_ptr refs here
    // keeps the CUDA device memory alive through the kernel launch.
    std::vector<std::shared_ptr<sd::TadPack>> tadPacks(N);

    for (int i = 0; i < N; ++i) {
      xBuffers[i] = x[i]->specialBuffer();
      xShapeInfos[i] = const_cast<sd::LongType*>(x[i]->specialShapeInfo());

      zBuffers[i] = z[i]->specialBuffer();
      zShapeInfos[i] = const_cast<sd::LongType*>(z[i]->specialShapeInfo());

      // Calculate TADs for each x — keep the shared_ptr alive in tadPacks[i]
      // so that _tadShape (and its GPU device buffer) is not freed until we
      // are done with the kernel launch below.
      tadPacks[i] = sd::ConstantTadHelper::getInstance().tadForDimensions(x[i]->shapeInfo(), dimensions, dimLength);
      tadShapeInfoBuffers[i] = const_cast<sd::LongType*>(tadPacks[i]->specialShapeInfo());
      tadOffsetsBuffers[i] = const_cast<sd::LongType*>(tadPacks[i]->specialOffsets());
    }

    // Ensure shuffleMap data is on device - use specialBuffer for CUDA kernel access
    shuffleMap->syncToDevice();

    // Allocate device memory for pointer arrays (kernel needs to read these from device memory)
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    auto& pool = sd::memory::CudaMemoryPool::getInstance();

    void** d_xBuffers = reinterpret_cast<void**>(pool.allocate(N * sizeof(void*), deviceId, *stream));
    sd::LongType** d_xShapeInfos = reinterpret_cast<sd::LongType**>(pool.allocate(N * sizeof(sd::LongType*), deviceId, *stream));
    void** d_zBuffers = reinterpret_cast<void**>(pool.allocate(N * sizeof(void*), deviceId, *stream));
    sd::LongType** d_tadShapeInfoBuffers = reinterpret_cast<sd::LongType**>(pool.allocate(N * sizeof(sd::LongType*), deviceId, *stream));
    sd::LongType** d_tadOffsetsBuffers = reinterpret_cast<sd::LongType**>(pool.allocate(N * sizeof(sd::LongType*), deviceId, *stream));

    // Copy pointer arrays from host to device
    cudaMemcpyAsync(d_xBuffers, xBuffers.data(), N * sizeof(void*), cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(d_xShapeInfos, xShapeInfos.data(), N * sizeof(sd::LongType*), cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(d_zBuffers, zBuffers.data(), N * sizeof(void*), cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(d_tadShapeInfoBuffers, tadShapeInfoBuffers.data(), N * sizeof(sd::LongType*), cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(d_tadOffsetsBuffers, tadOffsetsBuffers.data(), N * sizeof(sd::LongType*), cudaMemcpyHostToDevice, *stream);

    BUILD_SINGLE_SELECTOR(xType, sd::shuffleKernelGeneric,
                          (launchDims, stream, d_xBuffers, d_xShapeInfos, d_zBuffers, N,
                              reinterpret_cast<int*>(shuffleMap->specialBuffer()), d_tadShapeInfoBuffers,
                              d_tadOffsetsBuffers),
                          SD_COMMON_TYPES);

    sd::DebugHelper::checkErrorCode(stream, "shuffle(...) failed");

    // Free device memory for pointer arrays via pool (async)
    pool.free(d_xBuffers, deviceId, *stream);
    pool.free(d_xShapeInfos, deviceId, *stream);
    pool.free(d_zBuffers, deviceId, *stream);
    pool.free(d_tadShapeInfoBuffers, deviceId, *stream);
    pool.free(d_tadOffsetsBuffers, deviceId, *stream);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
