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
#include <exceptions/cuda_exception.h>
#include <graph/Context.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/RandomLauncher.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/random.h>

#include <memory>
#include <vector>


#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {
/**
 * gammaLess - compute gamma distributed value for shapes (alpha) from 0 to 1
 * @tparam T - any float types are acceptable
 * @param U - uniform random generated vals
 * @param alpha - shape of distribution
 * @param beta - scale of distributed values
 * @return gamma distributed value
 */
template <typename T>
T SD_DEVICE gammaLess(T const* U, LongType index, LongType maxLength, T const alpha, T const beta) {
  auto d = T(1.0334f) - T(0.0766f) * math::p_exp(T(2.2942f) * alpha);
  auto a = math::p_pow(T(2.f), alpha) * math::p_pow<T>(T(1.f) - math::p_exp(-d * T(0.5f)), alpha);
  auto b = alpha * math::p_pow(d, alpha - T(1.f)) * exp(-d);
  auto c = a + b;
  T rawX;
  auto indexV = index;
  auto underAlpha = T(1.f) / alpha;
  auto powerAlpha = math::p_pow<T>(T(2.f), alpha - T(1.f));

  for (;;) {
    auto u = (indexV < maxLength) ? U[indexV++] : U[0];
    if (indexV >= maxLength) indexV = 0LL;
    if (u <= a / c)
      rawX = -T(2.f) * math::p_log(T(1.f) - T(0.5f) * math::p_pow(c * u, underAlpha));
    else
      rawX = -math::p_log(c * (T(1.f) - u) / (alpha * math::p_pow(d, alpha - T(1.f))));

    T v = indexV < maxLength ? U[indexV++] : U[0];
    if (indexV >= maxLength) indexV = 0LL;
    //            math::atomics::sd_atomicAdd(index, 1LL);

    if (rawX <= d) {
      auto testVal = (math::p_pow<T>(rawX, alpha - 1.f) * math::p_exp(-T(0.5f) * rawX)) /
                     (powerAlpha * math::p_pow<T>(T(1.f) - math::p_exp(-T(0.5f) * rawX), alpha - T(1.f)));
      if (testVal < v) continue;
      break;
    } else {
      if (v <= math::p_pow<T>(d / rawX, T(1.f) - alpha)) break;
      continue;
    }
  }
  return rawX / beta;
}

/**
 * gammaGreat - generate gamma distributed value for shape (alpha) greater then 1
 * @tparam T - given type (any float type is accepted.)
 * @param rng  - random generator
 * @param alpha - shape of the gamma distribution (alpha)
 * @param beta  - scale of the gamma distribution (beta)
 * @return - gamma distributed value with given params
 */
template <typename T>
T SD_DEVICE gammaGreat(T const* U, LongType index, LongType maxLength, T const alpha, T const beta) {
  auto decreasedAlpha = alpha - T(1.f / 3.f);
  auto c = T(1.) / math::p_sqrt(T(9.f) * decreasedAlpha);
  auto indexV = index;
  T x;
  auto normalDistributed = [U, maxLength](LongType& index) {
    auto v1 = index < maxLength ? U[index++] : U[0];
    if (index >= maxLength) index = 0LL;
    auto v2 = index < maxLength ? U[index++] : U[0];
    if (index >= maxLength) index = 0LL;
    return math::p_cos(T(2.f * 3.141592f) * v2) * math::p_sqrt(T(-2.f) * math::p_log(v1));
  };

  float normalizedVar;
  for (;;) {
    do {
      x = normalDistributed(indexV);
      normalizedVar = T(1.f) + c * x;
    } while (normalizedVar < T(0.f));
    normalizedVar = normalizedVar * normalizedVar * normalizedVar;  // v * v * v;

    auto u = U[indexV++];
    if (indexV >= maxLength) indexV = 0LL;
    if (u < T(1.f) - T(.0331f) * (x * x) * (x * x)) break;  // return (d * v / b);
    if (log(u) < 0.5f * x * x + decreasedAlpha * (1. - normalizedVar + math::p_log(normalizedVar))) break;
  }
  return (decreasedAlpha * normalizedVar / beta);
}

/*
 * fillGammaKernel - fill up output with gamma distributed values
 *
 *  uList - uniformly distributed values set
 *  uLength - length of uList
 *  alpha - alpha param
 *  beta - beta param
 *  output - distributed output.
 * */
template <typename T>
static SD_KERNEL void fillGammaKernel(T const* uList, LongType uLength, T const* alpha,
                                      const LongType* alphaShape, T const* beta, const LongType* betaShape,
                                      T* output, const LongType* outputShape) {
  __shared__ LongType aLength;
  __shared__ LongType outLength;
  __shared__ LongType rankAlpha, rankBeta, rankOutput;
  __shared__ const LongType *alphaShapeArr, *alphaStride, *betaShapeArr, *betaStride, *outputShapeArr, *outputStride;

  if (threadIdx.x == 0) {
    aLength = shape::length(alphaShape);
    outLength = shape::length(outputShape) / aLength;

    rankAlpha = shape::rank(alphaShape);
    alphaShapeArr = shape::shapeOf(alphaShape);
    alphaStride = shape::stride(alphaShape);

    rankBeta = betaShape ? shape::rank(betaShape) : 0;
    betaShapeArr = betaShape ? shape::shapeOf(betaShape) : nullptr;
    betaStride = betaShape ? shape::stride(betaShape) : nullptr;

    rankOutput = shape::rank(outputShape);
    outputShapeArr = shape::shapeOf(outputShape);
    outputStride = shape::stride(outputShape);
  }
  __syncthreads();

  for (LongType k = blockIdx.x; k < outLength; k += gridDim.x) {
    const auto pos = k * aLength;

    for (LongType e = threadIdx.x; e < aLength; e += blockDim.x) {
      LongType aCoords[SD_MAX_RANK], bCoords[SD_MAX_RANK], zCoords[SD_MAX_RANK];
      LongType aIndex, bIndex = -1LL, zIndex;

      // Alpha coordinates and index
      INDEX2COORDS(e, rankAlpha, alphaShapeArr, aCoords);
      COORDS2INDEX(rankAlpha, alphaStride, aCoords, aIndex);

      // Beta coordinates and index (if beta is provided)
      if (betaShape) {
        INDEX2COORDS(e, rankBeta, betaShapeArr, bCoords);
        COORDS2INDEX(rankBeta, betaStride, bCoords, bIndex);
      }

      // Output coordinates and index
      INDEX2COORDS(e + pos, rankOutput, outputShapeArr, zCoords);
      COORDS2INDEX(rankOutput, outputStride, zCoords, zIndex);

      // Get beta value or default to 1
      const auto betaV = beta ? beta[bIndex] : T(1.f);

      // Fill the output with the computed gamma value
      output[zIndex] = alpha[aIndex] > T(1.f)
                           ? gammaGreat(uList, pos, uLength, alpha[aIndex], betaV)
                           : gammaLess(uList, pos, uLength, alpha[aIndex], betaV);
    }
  }
}

template <typename T>
static void fillRandomGamma_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                             NDArray* output) {
  // To fill up output need to broadcast alpha and beta to the same shape and in
  const LongType* broadcasted = nullptr;
  if (beta != nullptr)
    ShapeUtils::evalBroadcastShapeInfo(*alpha, *beta, true, broadcasted, context->getWorkspace());
  else
    broadcasted = alpha->shapeInfo();
  auto step = shape::length(broadcasted);
  auto shift = output->lengthOf() * 4LL;  // 2-wise greater case for uniform vals

  auto copyAlpha = alpha;
  auto copyBeta = beta;
  if (beta != nullptr) {
    NDArray alphaBroadcasted(broadcasted, alpha->dataType(), true, context);
    NDArray betaBroadcasted(broadcasted, beta->dataType(), true, context);

    copyAlpha = new NDArray(alphaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), *alpha));
    copyBeta = new NDArray(betaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), *beta));
  }

  auto stream = context->getCudaStream();
  NDArray uniform = NDArrayFactory::create<T>('c', {shift}, context);
  uniform.syncToDevice();
  // fill up uniform with given length
  RandomLauncher::fillUniform(context, rng, &uniform, 0.0000000001, 0.9999999999);
  uniform.syncToDevice();
  dim3 launchDims = getLaunchDims("random_gamma");
  fillGammaKernel<T><<<launchDims.x, launchDims.y,launchDims.z, *stream>>>(
      uniform.dataBuffer()->specialAsT<T>(), shift, copyAlpha->dataBuffer()->specialAsT<T>(),
      copyAlpha->specialShapeInfo(), beta ? copyBeta->dataBuffer()->specialAsT<T>() : (T const*)nullptr,
      beta ? copyBeta->specialShapeInfo() : (LongType const*)nullptr, output->dataBuffer()->specialAsT<T>(),
      output->specialShapeInfo());
  sd::DebugHelper::checkErrorCode(stream, "fillGammaKernel failed");

  if (beta != nullptr) {
    delete copyAlpha;
    delete copyBeta;
  }
}

void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                     NDArray* output) {
  if (beta)
    NDArray::prepareSpecialUse({output}, {alpha, beta});
  else
    NDArray::prepareSpecialUse({output}, {alpha});
  BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomGamma_, (context, rng, alpha, beta, output), SD_FLOAT_NATIVE);
  if (beta)
    NDArray::registerSpecialUse({output}, {alpha, beta});
  else
    NDArray::prepareSpecialUse({output}, {alpha});
}
BUILD_SINGLE_TEMPLATE(template void fillRandomGamma_,
                      (LaunchContext * context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                          NDArray* output),
                      SD_FLOAT_NATIVE);

/*
 * algorithm Poisson generator based upon the inversion by sequential search
 *
init:
     Let x ← 0, p ← e−λ, s ← p.
     using uniformly random sequence U (u in U) distributed at [0, 1].
while u > s do:
     x ← x + 1.
     p ← p * λ / x.
     s ← s + p.
return x.
 * */
template <typename T>
static SD_KERNEL void fillPoissonKernel(T* uList, LongType uLength, T* lambda, const LongType* lambdaShape,
                                        T* output, const LongType* outputShape) {
  __shared__ LongType lambdaLen;
  __shared__ LongType rankLambda, rankOutput;
  __shared__ const LongType *lambdaShapeArr, *lambdaStride, *outputShapeArr, *outputStride;

  if (threadIdx.x == 0) {
    lambdaLen = shape::length(lambdaShape);

    rankLambda = shape::rank(lambdaShape);
    rankOutput = shape::rank(outputShape);

    lambdaShapeArr = shape::shapeOf(lambdaShape);
    lambdaStride = shape::stride(lambdaShape);

    outputShapeArr = shape::shapeOf(outputShape);
    outputStride = shape::stride(outputShape);
  }
  __syncthreads();

  for (LongType k = blockIdx.x; k < uLength; k += gridDim.x) {
    const auto pos = k * lambdaLen;
    const auto u = uList[k];

    for (LongType e = threadIdx.x; e < lambdaLen; e += blockDim.x) {
      auto p = math::sd_exp<T, T>(-lambda[e]);
      auto s = p;
      auto x = T(0.);

      LongType lCoords[SD_MAX_RANK], zCoords[SD_MAX_RANK];
      LongType lIndex, zIndex;

      // Compute coordinates and indices for lambda
      INDEX2COORDS(e, rankLambda, lambdaShapeArr, lCoords);
      COORDS2INDEX(rankLambda, lambdaStride, lCoords, lIndex);

      // Compute coordinates and indices for output
      INDEX2COORDS(e + pos, rankOutput, outputShapeArr, zCoords);
      COORDS2INDEX(rankOutput, outputStride, zCoords, zIndex);

      // Compute Poisson distributed value
      while (u > s) {
        x += T(1.);
        p *= lambda[lIndex] / x;
        s += p;
      }

      // Assign computed value to output
      output[zIndex] = x;
    }
  }
}


template <typename T>
static void fillRandomPoisson_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
  auto shift = output->lengthOf() / lambda->lengthOf();
  std::vector<LongType> shape = {shift};
  NDArray uniform('c',shape, DOUBLE);
  PointersManager manager(context, "fillRandomPoisson");
  auto stream = context->getCudaStream();
  // fill up uniform with given length
  NDArray tempOutput = output->cast(DOUBLE);
  RandomLauncher::fillUniform(context, rng, &uniform, 0., 1.);

  NDArray tempLambda = lambda->cast(DOUBLE);
  NDArray::prepareSpecialUse({output,&tempOutput}, {lambda,&tempLambda});

  dim3 launchDims = getLaunchDims("random_poisson");
  fillPoissonKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(uniform.dataBuffer()->specialAsT<T>(), uniform.lengthOf(),
                                                                              tempLambda.dataBuffer()->specialAsT<T>(), tempLambda.specialShapeInfo(),
                                                                              tempOutput.dataBuffer()->specialAsT<T>(), tempOutput.specialShapeInfo());

  sd::DebugHelper::checkErrorCode(stream, "fillPoissonKernel failed");

  output->assign(tempOutput.cast(output->dataType()));
  NDArray::registerSpecialUse({output,&tempOutput}, {lambda,&tempLambda});

  manager.synchronize();
}

void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {lambda});
  BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomPoisson_, (context, rng, lambda, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {lambda});
}

BUILD_SINGLE_TEMPLATE(template void fillRandomPoisson_,
                      (LaunchContext * context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output),
                      SD_FLOAT_NATIVE);

template <typename T>
static SD_KERNEL void fillUniformKernel(graph::RandomGenerator* devRng, T from, T to, T* output,
                                        const LongType* outputShape) {
  const auto start = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  __shared__ LongType outputLen;
  __shared__ LongType rank;
  __shared__ const LongType *shape, *stride;

  if (threadIdx.x == 0) {
    outputLen = shape::length(outputShape);
    rank = shape::rank(outputShape);
    shape = shape::shapeOf(outputShape);
    stride = shape::stride(outputShape);
  }
  __syncthreads();

  LongType zCoords[SD_MAX_RANK];

  for (LongType i = start; i < outputLen; i += step) {
    LongType zIndex;

    // Calculate output index
    INDEX2COORDS(i, rank, shape, zCoords);
    COORDS2INDEX(rank, stride, zCoords, zIndex);

    // Fill output with a random value in the range [from, to]
    output[zIndex] = devRng->relativeT<T>(i, from, to);
  }
}

template <typename T>
static void fillRandomUniform_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max,
                               NDArray* output) {
  T minVal = T(0);
  T maxVal = DataTypeUtils::infOrMax<T>();
  if (min) minVal = min->t<T>(0);
  if (max) maxVal = max->t<T>(0);

  if (output->isR())
    RandomLauncher::fillUniform(context, rng, output, minVal, maxVal);
  else {
    auto stream = context->getCudaStream();
    graph::RandomGenerator* devRng;
    auto err = cudaMalloc(&devRng, sizeof(graph::RandomGenerator));
    if (err != 0) {
      cuda_exception::build("fillRandomUniform_: Cannot allocate device memory for random generator due error", err);
    }

    err = cudaMemcpy(devRng, &rng, sizeof(graph::RandomGenerator), cudaMemcpyHostToDevice);
    if (err != 0) {
      cuda_exception::build("fillRandomUniform_: Cannot copy random generator to device", err);
    }
    auto outputBuf = output->dataBuffer()->specialAsT<T>();
    auto outputShape = output->specialShapeInfo();
    dim3 launchDims = getLaunchDims("random_uniform");
    fillUniformKernel<T><<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(devRng, minVal, maxVal, outputBuf, outputShape);
    sd::DebugHelper::checkErrorCode(stream, "fillUniformKernel failed");

    err = cudaStreamSynchronize(*stream);
    if (err != 0) {
      cuda_exception::build("fillRandomUniform_: Cannot successfully finish kernel call", err);
    }

    err = cudaFree(devRng);
    if (err != 0) {
      cuda_exception::build("fillRandomUniform_: Cannot deallocate device memory for random generator", err);
    }
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
  const X maxVal = 1.0;

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
  graph::RandomGenerator* devRng;

  auto err = cudaMalloc(&devRng, sizeof(graph::RandomGenerator));
  if (err != 0) {
    cuda_exception::build("fillRandomMultiNomial: Cannot allocate device memory for random generator due error", err);
  }
  err = cudaStreamSynchronize(*context->getCudaStream());
  if (err != 0) {
    cuda_exception::build("fillRandomMultiNomial: Cannot synchronize stream for random generator due error", err);
  }
  err =
      cudaMemcpyAsync(devRng, &rng, sizeof(graph::RandomGenerator), cudaMemcpyHostToDevice, *context->getCudaStream());
  if (err != 0) {
    cuda_exception::build("fillRandomMultiNomial: Cannot copy random generator to device", err);
  }

  NDArray::prepareSpecialUse({&output}, {&input});
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), fillMultiNomialCudaLauncher,
                        (blocksPerGrid, threadsPerBlock, context->getCudaStream(), devRng, input.specialBuffer(),
                            input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), batchValue,
                            numOfSamples, numOfClassX, dimA),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({&output}, {&input});
  manager.synchronize();

  err = cudaFree(devRng);
  if (err != 0) {
    cuda_exception::build("fillRandomMultiNomial: Cannot deallocate device memory for random generator", err);
  }
  rng.rewindH(output.lengthOf() * numOfClassX);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
