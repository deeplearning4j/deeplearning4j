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

/**
 * Utility functions split from NativeOps.cu to reduce object file size
 * and prevent linker relocation overflow errors in large binaries.
 */

#include <cuda.h>
#include <system/env_functions.h>
#include <system/buffer.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <helpers/shape.h>
#include <helpers/DebugHelper.h>
#include <helpers/StringUtils.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ConstantHelper.h>
#include <execution/LaunchContext.h>
#include <execution/AffinityManager.h>
#include <execution/cuda/LaunchDims.h>
#include <loops/special_kernels.h>
#include <system/type_boilerplate.h>
#include <system/op_boilerplate.h>

// External declarations from NativeOps.cu
extern cudaDeviceProp *deviceProperties;
extern bool allowedP2P;
extern bool supportedP2P;

// Forward declarations
void checkP2P();
void enableP2P(bool enable);

////////////////////////////////////////////////////////////////////////
void initializeDevicesAndFunctions() {
  try {
    int devCnt = 0;
    cudaGetDeviceCount(&devCnt);
    deviceProperties = new cudaDeviceProp[devCnt];
    for (int i = 0; i < devCnt; i++) {
      cudaSetDevice(i);
      cudaGetDeviceProperties(&deviceProperties[i], i);

      cudaDeviceSetLimit(cudaLimitStackSize, 8192);
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576 * 128);

      // Force CUDA context creation and module registration for this device
      cudaFree(0);
      cudaDeviceSynchronize();
      cudaGetLastError();
    }

    cudaSetDevice(0);
    cudaFree(0);
    cudaDeviceSynchronize();
    cudaGetLastError();

    checkP2P();

    if (supportedP2P && devCnt > 1) enableP2P(true);  // Enable P2P by default when supported
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void pullRows(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray z, sd::LongType n, OpaqueNDArray indexes, sd::LongType dimension) {
  try {
    x->prepareSpecialUse({z}, {x});

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    dim3 launchDims = getLaunchDims("pullRows");
    auto xType = x->dataType();

    std::vector<void*> xBuffers(n);
    std::vector<const sd::LongType*> tadShapeInfoBuffers(n);
    std::vector<const sd::LongType*> tadOffsetsBuffers(n);

    auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &dimension, 1);

    void* zBuffer = z->specialBuffer();
    sd::LongType* zShapeInfo = const_cast<sd::LongType*>(z->specialShapeInfo());

    auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), &dimension, 1);
    sd::LongType* zTadShapeInfoBuffer = const_cast<sd::LongType*>(tadPackZ->specialShapeInfo());
    sd::LongType* zTadOffsetsBuffer = const_cast<sd::LongType*>(tadPackZ->specialOffsets());

    sd::LongType* indexesBuffer = reinterpret_cast<sd::LongType*>(indexes->specialBuffer());

    BUILD_SINGLE_SELECTOR(xType, sd::pullRowsKernelGeneric,
                          (launchDims, stream, x->specialBuffer(), zBuffer, n,
                              indexes->specialBufferasT<sd::LongType>(),
                              const_cast<sd::LongType *>(tadPackX->specialShapeInfo()),
                              const_cast<sd::LongType *>(tadPackX->specialOffsets()), zTadShapeInfoBuffer, zTadOffsetsBuffer),
                          SD_COMMON_TYPES);

    DEBUG_KERNEL(stream, -1);

    for (int i = 0; i < n; ++i) {
      x->registerSpecialUse({z}, {x});
    }
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
__global__ void tryPointerKernel(void *p, int len) {
  auto buf = reinterpret_cast<int8_t *>(p);
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int b;
  if (tid < len) atomicAdd(&b, buf[tid]);

  __syncthreads();
}

void tryPointer(sd::Pointer extra, sd::Pointer p, int len) {
  try {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    tryPointerKernel<<<256, 512, len + 64, stream>>>(p, len);
    sd::DebugHelper::checkGlobalErrorCode("try pointer failed(...) failed");

    auto e = cudaStreamSynchronize(stream);

    if (e != 0) {
      THROW_EXCEPTION("CUDA error");
    }

    cudaStreamDestroy(stream);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
