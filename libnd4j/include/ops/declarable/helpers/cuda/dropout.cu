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
#include <exceptions/cuda_exception.h>
#include <legacy/NativeOps.h>
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
                                          LongType const* outputShape, double probVal, int inLen,
                                          RandomGenerator* nodeRng) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  T const* input = reinterpret_cast<T const*>(inputBuf);
  T* output = reinterpret_cast<T*>(outputBuf);

  __shared__ LongType inputRank, outputRank;
  __shared__ const LongType *inputShapePtr, *inputStridePtr;
  __shared__ const LongType *outputShapePtr, *outputStridePtr;

  if (threadIdx.x == 0) {
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);
  }
  __syncthreads();

  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];
  LongType inputOffset;
  LongType outputOffset;

  // Loop through all elements and nullify based on probability
  for (LongType e = tid; e < inLen; e += step) {
    T val = nodeRng->relativeT(e, T(0.f), T(1.f));

    // If probability is acceptable, save the scaled value
    if (double(val) < probVal) {
      INDEX2COORDS(e, outputRank, outputShapePtr, outputCoords);
      COORDS2INDEX(outputRank, outputStridePtr, outputCoords, outputOffset);

      INDEX2COORDS(e, inputRank, inputShapePtr, inputCoords);
      COORDS2INDEX(inputRank, inputStridePtr, inputCoords, inputOffset);

      output[outputOffset] = T(input[inputOffset] / probVal);
    }
  }
}


template <typename T>
static void dropoutSimple(LaunchContext* context, NDArray * input, NDArray* output, double probValue,
                          int seed) {
  RandomGenerator nodeRng(3019L, seed);
  int inLen = input->lengthOf();
  RandomGenerator* dRandom;
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input});

  auto err = cudaMalloc(&dRandom, sizeof(RandomGenerator));
  if (err) {
    throw cuda_exception::build("helpers::dropoutSimple: Cannot allocate device memory for random generator.", err);
  }
  err = cudaMemcpy(dRandom, &nodeRng, sizeof(RandomGenerator), cudaMemcpyHostToDevice);
  if (err) {
    throw cuda_exception::build("helpers::dropoutSimple: Cannot set up device memory for random generator.", err);
  }

  dim3 getDims = getLaunchDims("dropout");
  dropoutSimpleKernel<T><<<getDims.x, getDims.y, getDims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                                                                       output->specialBuffer(), output->specialShapeInfo(), probValue,
                                                                       inLen, dRandom);
  err = cudaFree(dRandom);
  if (err) {
    throw cuda_exception::build("helpers::dropoutSimple: Cannot deallocate device memory for random generator.", err);
  }
  NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
Status _dropOutFunctor(sd::graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                       double probValue) {
  if (reduceShape == nullptr) {
    dropoutSimple<T>(context.launchContext(), input, output, probValue, seed);
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
    std::unique_ptr<NDArray> chunk(new NDArray('c', dims, output->dataType(), context.launchContext()));
    float one = 1.f;
    chunk->assign(one);

    dropoutSimple<T>(context.launchContext(), chunk.get(), chunk.get(), probValue, seed);
    // broadcast chunk to full matrix
    std::unique_ptr<NDArray> dropOutMultiplier(new NDArray(*input));
    dropOutMultiplier->assign(one);

    *dropOutMultiplier += *chunk;

    // FIXME: we could do this in one step, aren't we?
    NDArray ret = *input * *dropOutMultiplier;
    output->assign(&ret);
  }

  return Status::OK;
}

Status dropOutFunctor(sd::graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue, NDArray* mask) {
  auto xType = input->dataType();
  NDArray::prepareSpecialUse({output}, {input});

  BUILD_SINGLE_SELECTOR(xType, return _dropOutFunctor, (context, input, output, reduceShape, seed, probValue),
                        SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({output}, {input});
}

/////////////////////////////////// backpropagations ///////////////////////////////////////////////
template <typename T>
static SD_KERNEL void dropoutBPKernel(void* outputBuf, LongType const* outputShape, void* gradOutBuf,
                                      LongType const* gradOutShape, double probValue) {
  __shared__ T* output;
  __shared__ T* input;
  __shared__ LongType len;
  __shared__ LongType outputRank, gradOutRank;
  __shared__ const LongType *outputShapePtr, *outputStridePtr;
  __shared__ const LongType *gradOutShapePtr, *gradOutStridePtr;

  if (threadIdx.x == 0) {
    len = shape::length(outputShape);

    output = reinterpret_cast<T*>(outputBuf);
    input = reinterpret_cast<T*>(gradOutBuf);

    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    gradOutRank = shape::rank(gradOutShape);
    gradOutShapePtr = shape::shapeOf(gradOutShape);
    gradOutStridePtr = shape::stride(gradOutShape);
  }
  __syncthreads();

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  LongType outputCoords[SD_MAX_RANK];
  LongType gradOutCoords[SD_MAX_RANK];
  LongType zOffset;
  LongType gradOutOffset;

  for (LongType e = tid; e < len; e += step) {
    INDEX2COORDS(e, outputRank, outputShapePtr, outputCoords);
    COORDS2INDEX(outputRank, outputStridePtr, outputCoords, zOffset);

    INDEX2COORDS(e, gradOutRank, gradOutShapePtr, gradOutCoords);
    COORDS2INDEX(gradOutRank, gradOutStridePtr, gradOutCoords, gradOutOffset);

    // Scale gradients back if the output wasn't zero
    if (output[zOffset] != T(0.)) {
      output[zOffset] = T(input[gradOutOffset] / probValue);
    }
  }
}

template <typename T>
static Status dropOutFunctorBP_(sd::graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                NDArray* reduceShape, int seed, double probValue, NDArray* mask) {
  // we're making additional FF run to see how probabilities played out with given seeds
  auto res = dropOutFunctor(context, input, output, reduceShape, seed, probValue,mask);
  auto stream = context.launchContext()->getCudaStream();

  NDArray::prepareSpecialUse({output}, {input, gradOut});


  if (Status::OK == res) {
    dim3 launchDims = getLaunchDims("dropout");
    dropoutBPKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        output->specialBuffer(), output->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        probValue);

    DebugHelper::checkGlobalErrorCode( "dropout_bp(...) failed");

  }
  NDArray::registerSpecialUse({output}, {input, gradOut});

  return res;
}

template <typename T>
static SD_KERNEL void alphaDropoutSimpleKernel(void const* inputBuf, LongType const* inputShape, void* outputBuf,
                                               LongType const* outputShape, double probValue, double alpha,
                                               double alpha1, double beta, int inLen, RandomGenerator* nodeRng) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  T const* input = reinterpret_cast<T const*>(inputBuf);
  T* output = reinterpret_cast<T*>(outputBuf);

  __shared__ LongType inputRank, outputRank;
  __shared__ const LongType *inputShapePtr, *inputStridePtr;
  __shared__ const LongType *outputShapePtr, *outputStridePtr;

  if (threadIdx.x == 0) {
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);
  }
  __syncthreads();

  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];
  LongType inputOffset;
  LongType outputOffset;

  for (auto e = tid; e < inLen; e += step) {
    T val = nodeRng->relativeT(e, T(0.f), T(1.f));

    INDEX2COORDS(e, inputRank, inputShapePtr, inputCoords);
    COORDS2INDEX(inputRank, inputStridePtr, inputCoords, inputOffset);

    INDEX2COORDS(e, outputRank, outputShapePtr, outputCoords);
    COORDS2INDEX(outputRank, outputStridePtr, outputCoords, outputOffset);

    output[outputOffset] = (val >= T(probValue)
                                ? T(alpha * beta + alpha1)
                                : T(alpha * static_cast<double>(input[inputOffset]) + alpha1));
  }
}

template <typename T>
static void alphaDropoutSimple(LaunchContext* context, NDArray * input, NDArray* output, int seed,
                               double probValue, double alpha, double alpha1, double beta) {
  RandomGenerator nodeRng(3019L, seed), *dRandom;
  auto stream = context->getCudaStream();
  auto err = cudaMalloc(&dRandom, sizeof(RandomGenerator));
  NDArray::prepareSpecialUse({output}, {input});
  if (err) {
    throw cuda_exception::build("helpers::alphaDropoutSimple: Cannot allocate device memory for random generator.",
                                err);
  }
  err = cudaMemcpy(dRandom, &nodeRng, sizeof(RandomGenerator), cudaMemcpyHostToDevice);
  if (err) {
    throw cuda_exception::build("helpers::alphaDropoutSimple: Cannot set up device memory for random generator.", err);
  }

  dim3 launchDims = getLaunchDims("dropout");
  alphaDropoutSimpleKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), probValue,
      alpha, alpha1, beta, output->lengthOf(), dRandom);

  DebugHelper::checkGlobalErrorCode( "alphaDropoutSimpleKernel(...) failed");

  err = cudaFree(dRandom);
  if (err) {
    throw cuda_exception::build("helpers::alphaDropoutSimple: Cannot deallocate device memory for random generator.",
                                err);
  }
  NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
static Status alphaDropOutFunctor_(sd::graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed, double probValue, double alpha,
                                       double alpha1, double beta, NDArray* mask) {
  if (reduceShape == nullptr) {
    alphaDropoutSimple<T>(context.launchContext(), input, output, seed, probValue, alpha, alpha1, beta);
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
    std::unique_ptr<NDArray> chunk(new NDArray('c', dims, output->dataType(), context.launchContext()));
    float one = 1.f;

    chunk->assign(one);

    alphaDropoutSimple<T>(context.launchContext(), chunk.get(), chunk.get(), seed, probValue, alpha, alpha1, beta);

    // broadcast chunk to full matrix
    std::unique_ptr<NDArray> dropOutMultiplier(new NDArray(*input));
    dropOutMultiplier->assign(one);

    *dropOutMultiplier += *chunk;
    NDArray ret = *input * *dropOutMultiplier;
    output->assign(&ret);

  }

  return Status::OK;
}

template <typename T>
Status alphaDropOutFunctorBP_(sd::graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output, NDArray* reduceShape,
                              int seed, double probValue, double alpha, double alpha1, double beta, NDArray* mask) {
  auto res = alphaDropOutFunctor(context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta, mask);
  if (res == Status::OK) {
    // FIXME: can we make it single-loop?
    (*output) *= alpha;
    (*output) *= (*gradOut);
  }
  return res;
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
