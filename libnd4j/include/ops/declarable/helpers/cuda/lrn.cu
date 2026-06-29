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
//  @author raver119@gmail.com
//
#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/helpers/lrn.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void lrnKernel(void* vx, LongType const* xTadShapeInfo, LongType const* xTadOffsets, void* vz,
                                LongType const* zTadShapeInfo, LongType const* zTadOffsets, LongType numTads,
                                LongType tadLength, int depth, double bias, double alpha,
                                double beta) {
  extern __shared__ char sharedChar[];
  T* shared = reinterpret_cast<T*>(sharedChar);

  auto xTadRank = shape::rank(xTadShapeInfo);
  auto zTadRank = shape::rank(zTadShapeInfo);
  auto xTadShape = shape::shapeOf(xTadShapeInfo);
  auto zTadShape = shape::shapeOf(zTadShapeInfo);
  auto xTadStride = shape::stride(xTadShapeInfo);
  auto zTadStride = shape::stride(zTadShapeInfo);

  const T tbias = static_cast<T>(bias);
  const T tbeta = static_cast<T>(beta);
  const T talpha = static_cast<T>(alpha);

  // one block of threads processes 1 example within batch
  for (LongType i = blockIdx.x; i < numTads; i += gridDim.x) {
    auto x = reinterpret_cast<T*>(vx) + xTadOffsets[i];
    auto z = reinterpret_cast<T*>(vz) + zTadOffsets[i];

    // load everything into shared memory using INDEX2COORDS/COORDS2INDEX
    LongType xCoords[SD_MAX_RANK];
    LongType xOff;
    INDEX2COORDS(threadIdx.x, xTadRank, xTadShape, xCoords);
    COORDS2INDEX(xTadRank, xTadStride, xCoords, xOff);
    shared[threadIdx.x] = x[xOff];
    __syncthreads();

    const LongType begin = sd::math::sd_max<int>(0, threadIdx.x - depth);
    const LongType last = depth + threadIdx.x + 1;
    const LongType end = sd::math::sd_min<int>(last, tadLength);

    T prev = static_cast<T>(0.);
    for (int s = begin; s < end; s++) prev = prev + shared[s] * shared[s];

    LongType zCoords[SD_MAX_RANK];
    LongType zOff;
    INDEX2COORDS(threadIdx.x, zTadRank, zTadShape, zCoords);
    COORDS2INDEX(zTadRank, zTadStride, zCoords, zOff);
    z[zOff] = shared[threadIdx.x] / math::sd_pow<T, T, T>(tbias + alpha * prev, tbeta);
  }
}

template <typename X, typename Z>
static SD_KERNEL void lrnBPKernel(void const* vx, LongType const* xTadShapeInfo, LongType const* xTadOffsets,
                                  void* vz,
                                  LongType const* zTadShapeInfo, LongType const* zTadOffsets, LongType numTads,
                                  LongType tadLength, int depth, double bias, double alpha,
                                  double beta) {
  extern __shared__ char sharedChar[];
  X* sharedX = reinterpret_cast<X*>(sharedChar);
  Z* sharedY = reinterpret_cast<Z*>(sharedX + blockDim.x);

  auto xTadRank = shape::rank(xTadShapeInfo);
  auto zTadRank = shape::rank(zTadShapeInfo);
  auto xTadShape = shape::shapeOf(xTadShapeInfo);
  auto zTadShape = shape::shapeOf(zTadShapeInfo);
  auto xTadStride = shape::stride(xTadShapeInfo);
  auto zTadStride = shape::stride(zTadShapeInfo);

  const Z tbias = static_cast<Z>(bias);
  const Z tbeta = static_cast<Z>(beta);
  const Z talpha = static_cast<Z>(alpha);
  const Z coeff = talpha * tbeta;

  for (LongType i = blockIdx.x; i < numTads; i += gridDim.x) {
    auto x = reinterpret_cast<X const*>(vx) + xTadOffsets[i];
    auto z = reinterpret_cast<Z*>(vz) + zTadOffsets[i];

    const LongType begin = sd::math::sd_max<int>(0, threadIdx.x - depth);
    const LongType last = depth + threadIdx.x + 1;
    const LongType end = sd::math::sd_min<int>(last, tadLength);

    // load everything into shared memory using INDEX2COORDS/COORDS2INDEX
    LongType xCoords[SD_MAX_RANK];
    LongType xOff;
    INDEX2COORDS(threadIdx.x, xTadRank, xTadShape, xCoords);
    COORDS2INDEX(xTadRank, xTadStride, xCoords, xOff);
    sharedX[threadIdx.x] = x[xOff];
    sharedY[threadIdx.x] = static_cast<Z>(0.f);
    __syncthreads();

    // we're operating in shared memory
    for (int s = begin; s < end; s++) sharedY[threadIdx.x] = sharedY[threadIdx.x] + sharedX[s] * sharedX[s];
    __syncthreads();

    Z factor[1024];
    Z init = tbias + talpha * sharedY[threadIdx.x];

    Z prev = static_cast<Z>(0.f);
    for (LongType s = begin; s < end; ++s) {
      factor[s] = math::sd_pow<Z, Z, Z>(tbias + talpha * sharedY[s], -tbeta - 1);
      prev = prev + sharedX[s] * factor[s];
    }

    LongType zCoords[SD_MAX_RANK];
    LongType zOff;
    INDEX2COORDS(threadIdx.x, zTadRank, zTadShape, zCoords);
    COORDS2INDEX(zTadRank, zTadStride, zCoords, zOff);
    z[zOff] = factor[threadIdx.x] * init - 2 * sharedX[threadIdx.x] * coeff * prev;
  }
}

template <typename X, typename Z>
static void lrnBP_(graph::Context& block, NDArray& input, NDArray& gradO, NDArray& gradI,
                   const int depth, const float bias, const float alpha, const float beta) {
  auto rank = input.rankOf();
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), {rank - 1});
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(gradI.shapeInfo(), {rank - 1});

  const auto tadLength = shape::length(packX->primaryShapeInfo());
  const int numThreads = tadLength;

  if (tadLength > 1024 || tadLength < 1) THROW_EXCEPTION("LRN: tadLength > 1024 isn't implemented yet");

  dim3 launchDims = lrnDims(tadLength,packX->numberOfTads(),DataTypeUtils::sizeOf(input.dataType()),DataTypeUtils::sizeOf(gradI.dataType()));

  NDArray::prepareSpecialUse({&gradI}, {&input});
  lrnBPKernel<X, Z><<<launchDims.y, launchDims.x, launchDims.z,
                      *block.launchContext()->getCudaStream()>>>(
      input.specialBuffer(), packX->platformShapeInfo(), packX->platformOffsets(), gradI.specialBuffer(),
      packZ->platformShapeInfo(), packZ->platformOffsets(), packX->numberOfTads(), tadLength, depth, bias, alpha, beta);
  NDArray::registerSpecialUse({&gradI}, {&input});

  gradI *= gradO;
}

void lrnBP(graph::Context& block, NDArray& input, NDArray& gradO, NDArray& gradI, const int depth,
           const float bias, const float alpha, const float beta) {
  BUILD_DOUBLE_SELECTOR(input.dataType(), gradO.dataType(), lrnBP_,
                        (block, input, gradO, gradI, depth, bias, alpha, beta), SD_FLOAT_TYPES, SD_FLOAT_TYPES);
}

template <typename T>
static void lrnFunctor_(graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha,
                        double beta) {
  auto rank = input->rankOf();
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), {rank - 1});
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), {rank - 1});

  const auto tadLength = shape::length(packX->primaryShapeInfo());
  const int numBlocks = sd::math::sd_min<LongType>(1024, packX->numberOfTads());
  const int numThreads = tadLength;
  dim3 launchDims = lrnDims(tadLength, packX->numberOfTads(), DataTypeUtils::sizeOf(input->dataType()),
                            DataTypeUtils::sizeOf(input->dataType()));

  if (tadLength > 1024 || tadLength < 1) THROW_EXCEPTION("LRN: tadLength > 1024 isn't implemented yet");

  lrnKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *block.launchContext()->getCudaStream()>>>(
      input->specialBuffer(), packX->platformShapeInfo(), packX->platformOffsets(), output->specialBuffer(),
      packZ->platformShapeInfo(), packZ->platformOffsets(), packX->numberOfTads(), tadLength, depth, bias, alpha, beta);
}

Status lrnFunctor(graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha,
                      double beta) {
  NDArray::prepareSpecialUse({output}, {input});

  BUILD_SINGLE_SELECTOR(input->dataType(), lrnFunctor_, (block, input, output, depth, bias, alpha, beta),
                        SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
