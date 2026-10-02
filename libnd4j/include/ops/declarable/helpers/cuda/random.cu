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
//  @author sgazeos@gmail.com
//
#include <array/NDArrayFactory.h>
#include <string>
#include <graph/Context.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/RandomLauncher.h>
#include <helpers/ShapeUtils.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/random.h>

#include <memory>
#include <vector>


#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {
// Samples in float, or in double for a double output.
template <typename Z>
using RandomComputeT = typename std::conditional<std::is_same<Z, double>::value, double, float>::type;

// The generator is passed by value: kernels only read it, and the host rewinds it after the launch.
template <typename Z, typename C>
static SD_KERNEL void fillGammaKernel(graph::RandomGenerator rng, C const* alphas, C const* rates, LongType parameters,
                                      Z* output, const LongType* outputShapeInfo) {
  const LongType length = shape::length(outputShapeInfo);
  const LongType rank = shape::rank(outputShapeInfo);
  LongType coords[SD_MAX_RANK];
  LongType offset;
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType p = i % parameters;
    const C sample = sampleGamma<C>(rng, i, alphas[p], rates != nullptr ? rates[p] : C(1));
    INDEX2COORDS(i, rank, shape::shapeOf(outputShapeInfo), coords);
    COORDS2INDEX(rank, shape::stride(outputShapeInfo), coords, offset);
    output[offset] = static_cast<Z>(sample);
  }
}

template <typename Z>
static void fillRandomGamma_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                             NDArray* output) {
  using C = RandomComputeT<Z>;
  // Output element i samples with parameters i % length of alpha and beta broadcast together.
  LongType* parameterShape = alpha->shapeInfo();
  if (beta != nullptr)
    ShapeUtils::evalBroadcastShapeInfo(alpha->shapeInfo(), beta->shapeInfo(), true, parameterShape,
                                       context->getWorkspace());
  NDArray* alphas = randomParameter(alpha, parameterShape, DataTypeUtils::fromT<C>(), context);
  NDArray* rates = beta != nullptr ? randomParameter(beta, parameterShape, DataTypeUtils::fromT<C>(), context) : nullptr;

  if (output->lengthOf() > 0) {
    NDArray::prepareSpecialUse({output}, {alphas, rates});
    dim3 launchDims = getLaunchDims("random_gamma");
    fillGammaKernel<Z, C><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        rng, reinterpret_cast<C const*>(alphas->specialBuffer()),
        rates != nullptr ? reinterpret_cast<C const*>(rates->specialBuffer()) : nullptr, alphas->lengthOf(),
        reinterpret_cast<Z*>(output->specialBuffer()), output->specialShapeInfo());
    DebugHelper::checkGlobalErrorCode("fillGammaKernel failed");
    NDArray::registerSpecialUse({output}, {alphas, rates});
    rng.rewindH(output->lengthOf());
  }

  delete alphas;
  delete rates;
}

void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                     NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomGamma_, (context, rng, alpha, beta, output), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE( void fillRandomGamma_,
                      (LaunchContext * context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                          NDArray* output),
                      SD_FLOAT_TYPES);

template <typename Z, typename C>
static SD_KERNEL void fillPoissonKernel(graph::RandomGenerator rng, C const* lambdas, LongType parameters, Z* output,
                                        const LongType* outputShapeInfo) {
  const LongType length = shape::length(outputShapeInfo);
  const LongType rank = shape::rank(outputShapeInfo);
  LongType coords[SD_MAX_RANK];
  LongType offset;
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const C sample = samplePoisson<C>(rng, i, lambdas[i % parameters]);
    INDEX2COORDS(i, rank, shape::shapeOf(outputShapeInfo), coords);
    COORDS2INDEX(rank, shape::stride(outputShapeInfo), coords, offset);
    output[offset] = static_cast<Z>(sample);
  }
}

template <typename Z>
static void fillRandomPoisson_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
  using C = RandomComputeT<Z>;
  // Output element i samples with lambda i % lambda's length.
  NDArray* lambdas = randomParameter(lambda, lambda->shapeInfo(), DataTypeUtils::fromT<C>(), context);

  if (output->lengthOf() > 0) {
    NDArray::prepareSpecialUse({output}, {lambdas});
    dim3 launchDims = getLaunchDims("random_poisson");
    fillPoissonKernel<Z, C><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        rng, reinterpret_cast<C const*>(lambdas->specialBuffer()), lambdas->lengthOf(),
        reinterpret_cast<Z*>(output->specialBuffer()), output->specialShapeInfo());
    DebugHelper::checkGlobalErrorCode("fillPoissonKernel failed");
    NDArray::registerSpecialUse({output}, {lambdas});
    rng.rewindH(output->lengthOf());
  }

  delete lambdas;
}

void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomPoisson_, (context, rng, lambda, output), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void fillRandomPoisson_,
                      (LaunchContext * context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output),
                      SD_FLOAT_TYPES);

template <typename T>
static SD_KERNEL void fillUniformKernel(graph::RandomGenerator rng, T from, T to, T* output,
                                        const LongType* outputShapeInfo) {
  const LongType length = shape::length(outputShapeInfo);
  const LongType rank = shape::rank(outputShapeInfo);
  LongType coords[SD_MAX_RANK];
  LongType offset;
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    INDEX2COORDS(i, rank, shape::shapeOf(outputShapeInfo), coords);
    COORDS2INDEX(rank, shape::stride(outputShapeInfo), coords, offset);
    output[offset] = rng.relativeT<T>(i, from, to);
  }
}

template <typename T>
static void fillRandomUniform_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max,
                               NDArray* output) {
  T minVal = T(0);
  T maxVal = DataTypeUtils::infOrMax<T>();
  if (min) minVal = min->t<T>(0);
  if (max) maxVal = max->t<T>(0);

  if (output->isR()) {
    RandomLauncher::fillUniform(context, rng, output, minVal, maxVal);
  } else if (output->lengthOf() > 0) {
    NDArray::prepareSpecialUse({output}, {});
    dim3 launchDims = getLaunchDims("random_uniform");
    fillUniformKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        rng, minVal, maxVal, reinterpret_cast<T*>(output->specialBuffer()), output->specialShapeInfo());
    DebugHelper::checkGlobalErrorCode("fillUniformKernel failed");
    NDArray::registerSpecialUse({output}, {});
    // The floating fill above rewinds inside the random launcher; the next fill must draw anew.
    rng.rewindH(output->lengthOf());
  }
}

void fillRandomUniform(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max,
                       NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomUniform_, (context, rng, min, max, output), SD_NUMERIC_TYPES);
}

///////////////////////////////////////////////////////////////////
// used https://en.wikipedia.org/wiki/Categorical_distribution
// methods: gumbel trick + softmax + argmax
template <typename X, typename Z>
SD_KERNEL static void fillMultiNomialCuda_(graph::RandomGenerator* devRng, const void* vx, const LongType* xShapeInfo,
                                           void* vz, const LongType* zShapeInfo, const LongType batchValue,
                                           const LongType numOfSamples, const LongType numOfClassX, const LongType dimA, const X minVal,
                                           const X maxVal) {
  const X* x = reinterpret_cast<const X*>(vx);
  Z* z = reinterpret_cast<Z*>(vz);

  __shared__ LongType xDimAstride, zDimAstride, xDimCstride, zDimCstride, dimC;

  if (0 == threadIdx.x) {
    dimC = (0 == dimA) ? 1 : 0;
    zDimAstride = shape::stride(zShapeInfo)[dimA];
    xDimAstride = shape::stride(xShapeInfo)[dimA];
    zDimCstride = shape::stride(zShapeInfo)[dimC];
    xDimCstride = shape::stride(xShapeInfo)[dimC];
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType index = tid; index < batchValue * numOfSamples; index += gridDim.x * blockDim.x) {
    LongType nBatchIndex = index / numOfSamples;
    LongType nSampleIndexInBatch = index - (nBatchIndex * numOfSamples);

    const X* xTad = x + (nBatchIndex * xDimCstride);
    Z* zTad = z + (nBatchIndex * zDimCstride);
    Z& arg = zTad[nSampleIndexInBatch * zDimAstride];

    X Max = -minVal;
    LongType nSamplesPerBatch = nBatchIndex * numOfClassX * numOfSamples;
    LongType nClassPerSamples = nSampleIndexInBatch * numOfClassX;

    for (LongType nClass = 0; nClass < numOfClassX; nClass++) {
      LongType nIndex = nSamplesPerBatch + nClassPerSamples + nClass;
      X tValue = (xTad[nClass * xDimAstride] -
                  math::sd_log<X, X>(-math::sd_log<X, X>(devRng->relativeT<X>(nIndex, minVal, maxVal))));
      if (tValue > Max) {
        Max = tValue;
        arg = nClass;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_HOST static void fillMultiNomialCudaLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                                const cudaStream_t* stream, graph::RandomGenerator* devRng,
                                                const void* vx, const LongType* xShapeInfo, void* vz,
                                                const LongType* zShapeInfo, const LongType batchValue,
                                                const LongType numOfSamples, const LongType numOfClassX,
                                                const LongType dimA) {
  const X minVal = DataTypeUtils::min<X>();
  const X maxVal = static_cast<X>(1.0);

  fillMultiNomialCuda_<X, Z><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(
      devRng, vx, xShapeInfo, vz, zShapeInfo, batchValue, numOfSamples, numOfClassX, dimA, minVal, maxVal);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "fillMultiNomialCuda_ failed");

}

///////////////////////////////////////////////////////////////////
void fillRandomMultiNomial(LaunchContext* context, graph::RandomGenerator& rng, NDArray& input, NDArray& output,
                           const LongType numOfSamples, const int dimC) {
  LongType dimA = (0 == dimC) ? 1 : 0;

  const LongType batchValue = output.sizeAt(dimC);
  const LongType numOfClassX = input.sizeAt(dimA);

  const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const int blocksPerGrid = (batchValue * numOfSamples + threadsPerBlock - 1) / threadsPerBlock;

  PointersManager manager(context, "fillMultinomial");
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  graph::RandomGenerator* devRng = reinterpret_cast<graph::RandomGenerator*>(
      memory::CudaMemoryPool::getInstance().allocate(sizeof(graph::RandomGenerator), deviceId, *context->getCudaStream()));
  if (devRng == nullptr) {
    std::string msg = "fillRandomMultiNomial: Cannot allocate device memory for random generator; Error code: [" + std::to_string(cudaErrorMemoryAllocation) + "]";
    THROW_EXCEPTION(msg.c_str());
  }
  auto err = cudaMemcpyAsync(devRng, &rng, sizeof(graph::RandomGenerator), cudaMemcpyHostToDevice, *context->getCudaStream());
  if (err != 0) {
    std::string msg = "fillRandomMultiNomial: Cannot copy random generator to device; Error code: [" + std::to_string(err) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  NDArray::prepareSpecialUse({&output}, {&input});
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), fillMultiNomialCudaLauncher,
                        (blocksPerGrid, threadsPerBlock, context->getCudaStream(), devRng, input.specialBuffer(),
                            input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), batchValue,
                            numOfSamples, numOfClassX, dimA),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({&output}, {&input});
  manager.synchronize();

  memory::CudaMemoryPool::getInstance().free(devRng, deviceId, *context->getCudaStream());
  rng.rewindH(output.lengthOf() * numOfClassX);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
