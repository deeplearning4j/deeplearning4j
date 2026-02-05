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
// Contains: execRandom, execRandom2, execRandom3, initRandom, destroyRandom, refreshBuffer, reSeedBuffer
//

#include <cuda.h>
#include <curand.h>
#include <exceptions/cuda_exception.h>
#include <execution/LaunchContext.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/DebugHelper.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <system/common.h>
#include <helpers/shape.h>

void execRandom(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost, OpaqueNDArray z, void *extraArguments) {
  try {
    z->prepareSpecialUse({}, {z});

    auto lc = sd::LaunchContext::defaultContext();

    NativeOpExecutioner::execRandom(
        lc, opNum, stateHost,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        extraArguments);

    z->registerSpecialUse({}, {z});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom2(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost, OpaqueNDArray x, OpaqueNDArray z, void *extraArguments) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto lc = sd::LaunchContext::defaultContext();

    NativeOpExecutioner::execRandom(
        lc, opNum, stateHost,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(), extraArguments);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom3(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost, OpaqueNDArray x,
                 OpaqueNDArray y, OpaqueNDArray z, void *extraArguments) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execRandom(lc, opNum, stateHost,
                                    shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                    x->shapeInfo(),
                                    shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                    x->specialShapeInfo(),
                                    shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
                                    y->shapeInfo(),
                                    shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
                                    y->specialShapeInfo(),
                                    shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                    z->shapeInfo(),
                                    shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                    z->specialShapeInfo(),
                                    extraArguments);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

sd::Pointer initRandom(sd::Pointer *extraPointers, long seed, long bufferSize, sd::Pointer ptrToBuffer) {
  unsigned long long *ptrHost = reinterpret_cast<unsigned long long *>(extraPointers[0]);
  cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

  auto ptrDev = reinterpret_cast<unsigned long long *>(ptrToBuffer);
  auto buffer = new sd::random::RandomBuffer(seed, bufferSize, reinterpret_cast<uint64_t *>(ptrHost),
                                             reinterpret_cast<uint64_t *>(ptrDev));
  buffer->propagateToDevice(buffer, *stream);

  sd::DebugHelper::checkErrorCode(stream, "initRandom(...) failed A");

  sd::random::Xoroshiro128 generator(buffer);
  generator.refreshBuffer();

  cudaMemcpyAsync(ptrDev, ptrHost, bufferSize * 8, cudaMemcpyHostToDevice, *stream);
  sd::DebugHelper::checkErrorCode(stream, "initRandom(...) failed B");

  return buffer;
}

void destroyRandom(sd::Pointer ptrBuffer) {
  sd::random::RandomBuffer *buffer = reinterpret_cast<sd::random::RandomBuffer *>(ptrBuffer);
  auto stream = sd::LaunchContext::defaultContext()->getCudaStream();
  if (stream != nullptr)
    cudaStreamSynchronize(*stream);
  delete buffer;
}

void refreshBuffer(sd::Pointer *extraPointers, long seed, sd::Pointer ptrRandom) {
  sd::random::RandomBuffer *buffer = reinterpret_cast<sd::random::RandomBuffer *>(ptrRandom);

  unsigned long long *ptrHost = reinterpret_cast<unsigned long long *>(extraPointers[0]);
  cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
  cudaStreamSynchronize(*stream);

  uint64_t *ptrDev = buffer->getDeviceBuffer();

  buffer->setSeed(seed);
  buffer->setOffset(0);
  buffer->propagateToDevice(buffer, *stream);

  sd::random::Xoroshiro128 generator(buffer);
  generator.refreshBuffer();

  cudaMemcpyAsync(ptrDev, ptrHost, buffer->getSize() * 8, cudaMemcpyHostToDevice, *stream);
}

void reSeedBuffer(sd::Pointer *extraPointers, long seed, sd::Pointer ptrRandom) {
  sd::random::RandomBuffer *buffer = reinterpret_cast<sd::random::RandomBuffer *>(ptrRandom);

  cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
  cudaStreamSynchronize(*stream);

  buffer->reSeed(seed);
  buffer->setOffset(0);
  buffer->propagateToDevice(buffer, *stream);
}
