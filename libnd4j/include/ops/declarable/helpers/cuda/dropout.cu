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
#include <legacy/NativeOps.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/dropout.h>
#include <helpers/DebugHelper.h>
#include <memory>
#include <vector>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void dropoutSimpleKernel(void const* inputBuf, LongType const* inputShape, void* outputBuf,
                                          LongType const* outputShape, void* maskBuf, LongType const* maskShape,
                                          double probVal, int inLen, RandomGenerator nodeRng) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  T const* input = reinterpret_cast<T const*>(inputBuf);
  T* output = reinterpret_cast<T*>(outputBuf);
  T* mask = (maskBuf != nullptr) ? reinterpret_cast<T*>(maskBuf) : nullptr;

  __shared__ LongType inputRank, outputRank, maskRank;
  __shared__ const LongType *inputShapePtr, *inputStridePtr;
  __shared__ const LongType *outputShapePtr, *outputStridePtr;
  __shared__ const LongType *maskShapePtr, *maskStridePtr;

  if (threadIdx.x == 0) {
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    if (maskShape != nullptr) {
      maskRank = shape::rank(maskShape);
      maskShapePtr = shape::shapeOf(maskShape);
      maskStridePtr = shape::stride(maskShape);
    }
  }
  __syncthreads();

  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];
  LongType maskCoords[SD_MAX_RANK];
  LongType inputOffset;
  LongType outputOffset;
  LongType maskOffset;

  for (LongType e = tid; e < inLen; e += step) {
    T val = nodeRng.relativeT(e, T(0.f), T(1.f));
    bool keep = double(val) < probVal;

    INDEX2COORDS(e, outputRank, outputShapePtr, outputCoords);
    COORDS2INDEX(outputRank, outputStridePtr, outputCoords, outputOffset);

    if (mask != nullptr) {
      INDEX2COORDS(e, maskRank, maskShapePtr, maskCoords);
      COORDS2INDEX(maskRank, maskStridePtr, maskCoords, maskOffset);
      mask[maskOffset] = keep ? T(1) : T(0);
    }

    // Every output element is written: the input when kept, 0 when dropped. The op does not pre-zero its
    // output, so a dropped element must be stored here. Each element is read before it is written at the
    // same offset, which keeps the in-place form (output aliasing the input) correct.
    INDEX2COORDS(e, inputRank, inputShapePtr, inputCoords);
    COORDS2INDEX(inputRank, inputStridePtr, inputCoords, inputOffset);
    output[outputOffset] = keep ? input[inputOffset] : T(0);
  }
}


template <typename T>
static void dropoutSimple(LaunchContext* context, RandomGenerator& nodeRng, NDArray* input, NDArray* output,
                          double probValue, NDArray* mask) {
  int inLen = input->lengthOf();
  if (inLen == 0) return;
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output, mask}, {input});

  void* maskBuf = (mask != nullptr) ? mask->specialBuffer() : nullptr;
  LongType const* maskShape = (mask != nullptr) ? mask->specialShapeInfo() : nullptr;

  // The generator is passed by value: the kernel only reads it.
  dim3 getDims = getLaunchDims("dropout");
  dropoutSimpleKernel<T><<<getDims.x, getDims.y, getDims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                                                                       output->specialBuffer(), output->specialShapeInfo(),
                                                                       maskBuf, maskShape, probValue,
                                                                       inLen, nodeRng);
  DebugHelper::checkGlobalErrorCode("dropoutSimpleKernel failed");
  NDArray::registerSpecialUse({output, mask}, {input});
}

template <typename T>
Status _dropOutFunctor(sd::graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                       double probValue, NDArray* mask) {
  // A nonzero seed fixes the mask. Seed 0 draws it from the context's generator, which SameDiff
  // seeds from Nd4j.getRandom(), and advances that generator, so each execution drops anew.
  RandomGenerator seeded(3019L, seed);
  RandomGenerator& rng = seed != 0 ? seeded : context.randomGenerator();
  if (reduceShape == nullptr) {
    dropoutSimple<T>(context.launchContext(), rng, input, output, probValue, mask);
  } else {
    REQUIRE_TRUE(reduceShape->lengthOf() <= input->rankOf(), 0, "dropout: Noise shape should be fittable to input");

    std::vector<LongType> dims(reduceShape->lengthOf());
    reduceShape->syncToHost();  // to ensure that follows are actual
    bool fit = true;

    for (int i = 0; i < dims.size(); i++) {
      if (fit) {
        dims[i] = reduceShape->e<LongType>(i);
        for (int e = 0; e < input->rankOf(); ++e)
          if (fit)
            if (input->sizeAt(e) % dims[i]) {
              fit = false;
            }
      }
    }

    // check dims to fit input
    REQUIRE_TRUE(fit, 0, "dropout: Noise shape should fit to input rank.");
    NDArray *chunk = new NDArray('c', dims, output->dataType(), context.launchContext());
    float one = 1.f;
    chunk->assign(one);

    dropoutSimple<T>(context.launchContext(), rng, chunk, chunk, probValue, nullptr);
    // broadcast the chunk's keep decisions (1 kept, 0 dropped) to the full mask: zeros plus the chunk
    mask->nullify();
    *mask += *chunk;
    delete chunk;

    NDArray* ret = (*input) * (*mask);
    output->assign(ret);
    delete ret;
  }
  rng.rewindH(input->lengthOf());

  return Status::OK;
}

Status dropOutFunctor(sd::graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue, NDArray* mask) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, return _dropOutFunctor, (context, input, output, reduceShape, seed, probValue, mask),
                        SD_FLOAT_TYPES);
}

/////////////////////////////////// backpropagations ///////////////////////////////////////////////
template <typename T>
static Status dropOutFunctorBP_(sd::graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                NDArray* reduceShape, int seed, double probValue, NDArray* mask) {
  // The forward passes kept elements through unscaled, and its mask (input 1) records which: the
  // gradient is gradOut where the mask is 1 and 0 where it is 0. Re-running the forward would
  // overwrite that input and, unseeded, drop other elements than the forward did.
  // The gradient may be computed in place (output aliasing gradOut): there is nothing to copy then.
  if (output != gradOut) output->assign(gradOut);
  *output *= *mask;
  return Status::OK;
}

template <typename T>
static SD_KERNEL void alphaDropoutSimpleKernel(void const* inputBuf, LongType const* inputShape, void* outputBuf,
                                               LongType const* outputShape, void* maskBuf, LongType const* maskShape,
                                               double probValue, double alpha, double alpha1, double beta, int inLen,
                                               RandomGenerator* nodeRng) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  T const* input = reinterpret_cast<T const*>(inputBuf);
  T* output = reinterpret_cast<T*>(outputBuf);
  T* mask = reinterpret_cast<T*>(maskBuf);

  __shared__ LongType inputRank, outputRank, maskRank;
  __shared__ const LongType *inputShapePtr, *inputStridePtr;
  __shared__ const LongType *outputShapePtr, *outputStridePtr;
  __shared__ const LongType *maskShapePtr, *maskStridePtr;

  if (threadIdx.x == 0) {
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    if (maskShape != nullptr) {
      maskRank = shape::rank(maskShape);
      maskShapePtr = shape::shapeOf(maskShape);
      maskStridePtr = shape::stride(maskShape);
    }
  }
  __syncthreads();

  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];
  LongType maskCoords[SD_MAX_RANK];
  LongType inputOffset;
  LongType outputOffset;
  LongType maskOffset;

  for (auto e = tid; e < inLen; e += step) {
    T val = nodeRng->relativeT(e, T(0.f), T(1.f));
    const bool keep = !(val >= T(probValue));

    INDEX2COORDS(e, inputRank, inputShapePtr, inputCoords);
    COORDS2INDEX(inputRank, inputStridePtr, inputCoords, inputOffset);

    INDEX2COORDS(e, outputRank, outputShapePtr, outputCoords);
    COORDS2INDEX(outputRank, outputStridePtr, outputCoords, outputOffset);

    // the mask records the keep decision (1 kept, 0 dropped): alpha_dropout_bp's gradient is gradOut * mask * alpha
    if (mask != nullptr) {
      INDEX2COORDS(e, maskRank, maskShapePtr, maskCoords);
      COORDS2INDEX(maskRank, maskStridePtr, maskCoords, maskOffset);
      mask[maskOffset] = keep ? T(1) : T(0);
    }

    output[outputOffset] = keep ? T(alpha * static_cast<double>(input[inputOffset]) + alpha1) : T(alpha * beta + alpha1);
  }
}

template <typename T>
static void alphaDropoutSimple(LaunchContext* context, NDArray * input, NDArray* output, NDArray* mask, int seed,
                               double probValue, double alpha, double alpha1, double beta) {
  RandomGenerator nodeRng(3019L, seed), *dRandom;
  auto stream = context->getCudaStream();
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  dRandom = reinterpret_cast<RandomGenerator*>(memory::CudaMemoryPool::getInstance().allocate(sizeof(RandomGenerator), deviceId, *stream));
  NDArray::prepareSpecialUse({output, mask}, {input});
  if (dRandom == nullptr) {
    THROW_EXCEPTION("helpers::alphaDropoutSimple: Cannot allocate device memory for random generator.");
  }
  auto err = cudaMemcpyAsync(dRandom, &nodeRng, sizeof(RandomGenerator), cudaMemcpyHostToDevice, *stream);
  if (err) {
    { std::string msg = "helpers::alphaDropoutSimple: Cannot set up device memory for random generator.; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  dim3 launchDims = getLaunchDims("dropout");
  alphaDropoutSimpleKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
      mask != nullptr ? mask->specialBuffer() : nullptr, mask != nullptr ? mask->specialShapeInfo() : nullptr,
      probValue, alpha, alpha1, beta, output->lengthOf(), dRandom);

  DebugHelper::checkGlobalErrorCode( "alphaDropoutSimpleKernel(...) failed");

  memory::CudaMemoryPool::getInstance().free(dRandom, deviceId, *stream);
  NDArray::registerSpecialUse({output, mask}, {input});
}

template <typename T>
static Status alphaDropOutFunctor_(sd::graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha,
                                       double alpha1, double beta, NDArray* mask) {
  if (reduceShape == nullptr) {
    alphaDropoutSimple<T>(context.launchContext(), input, output, mask, seed, probValue, alpha, alpha1, beta);
  } else {
    REQUIRE_TRUE(reduceShape->lengthOf() <= input->rankOf(), 0, "dropout: Noise shape should be fittable to input");

    std::vector<LongType> dims(reduceShape->lengthOf());
    reduceShape->syncToHost();  // to ensure that follows are actual
    bool fit = true;

    for (int i = 0; i < dims.size(); i++) {
      if (fit) {
        dims[i] = reduceShape->e<LongType>(i);
        for (int e = 0; e < input->rankOf(); ++e)
          if (fit)
            if (input->sizeAt(e) % dims[i]) {
              fit = false;
            }
      }
    }

    // check dims to fit input
    REQUIRE_TRUE(fit, 0, "alpha_dropout: Noise shape should fit to input rank.");
    NDArray *chunk = new NDArray('c', dims, output->dataType(), context.launchContext());
    NDArray *chunkMask = new NDArray('c', dims, output->dataType(), context.launchContext());
    float one = 1.f;
    chunk->assign(one);

    // one keep decision per chunk element, broadcast to the full mask (1 kept, 0 dropped)
    alphaDropoutSimple<T>(context.launchContext(), chunk, chunk, chunkMask, seed, probValue, alpha, alpha1, beta);
    mask->nullify();
    *mask += *chunkMask;
    delete chunk;
    delete chunkMask;

    // kept elements map to alpha * x + alpha1, dropped ones to alpha * beta + alpha1:
    // alpha * (mask * (x - beta) + beta) + alpha1
    NDArray* ret = (*input) - beta;
    *ret *= *mask;
    *ret += beta;
    *ret *= alpha;
    *ret += alpha1;
    output->assign(ret);
    delete ret;
  }

  return Status::OK;
}

template <typename T>
Status alphaDropOutFunctorBP_(sd::graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape,
                              int seed, double probValue, double alpha, double alpha1, double beta, NDArray* mask) {
  // The forward scales kept inputs by alpha (alpha * x + alpha1) and replaces dropped ones, so the
  // gradient is gradOut * alpha where the keep mask is 1 and 0 where it is 0.
  output->assign(gradOut);
  *output *= *mask;
  *output *= alpha;
  return Status::OK;
}

Status dropOutFunctorBP(sd::graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape,
                        int seed, double probValue, NDArray* mask) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return dropOutFunctorBP_,
                        (context, input, gradOut, output, reduceShape, seed, probValue, mask), SD_FLOAT_TYPES);
}

Status alphaDropOutFunctor(sd::graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                           double probValue, double alpha, double alpha1, double beta, NDArray* mask) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctor_,
                        (context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta, mask),
                        SD_FLOAT_TYPES);
}

Status alphaDropOutFunctorBP(sd::graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape, int seed, double probValue,
                                 double alpha, double alpha1, double beta, NDArray* mask) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctorBP_,
                        (context, input, gradOut, output, reduceShape, seed, probValue, alpha, alpha1, beta,mask),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
