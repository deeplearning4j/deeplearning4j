/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Paul Dubs
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_layer_norm)

#include <math/templatemath.h>
#include <ops/declarable/headers/nn.h>
#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/reverse.h>
#include <ops/declarable/helpers/layer_norm.h>
#include <graph/DspDeviceDispatch.h>
#include <helpers/ShapeUtils.h>
#include <execution/Threads.h>
#include <cmath>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(layer_norm, 2, 1, false, 0, -1) {
  auto input = INPUT_VARIABLE(0);
  auto gain = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  // Handle empty input gracefully - nothing to normalize
  if (input->isEmpty() || input->lengthOf() == 0) {
    return sd::Status::OK;
  }

  std::vector<sd::LongType> axis = *block.getIArguments();

  const bool isNCHW = block.getBArguments()->size() > 0 ? B_ARG(0) : true;  // 0-NCHW,  1-NHWC
  const int dimC = isNCHW ? 1 : input->rankOf() - 1;

  REQUIRE_TRUE(gain->rankOf() == 1 && gain->sizeAt(0) == input->sizeAt(dimC), 0,
               "LAYER_NORM OP: wrong shape of gain array, expected is {%i}, but got %s instead !", input->sizeAt(dimC),
               ShapeUtils::shapeAsString(gain).c_str());

  NDArray *bias = nullptr;
  if (block.width() > 2) {
    bias = INPUT_VARIABLE(2);
    REQUIRE_TRUE(bias->rankOf() == 1 && bias->sizeAt(0) == input->sizeAt(dimC), 0,
                 "LAYER_NORM OP: wrong shape of bias array, expected is {%i}, but got %s instead !",
                 input->sizeAt(dimC), ShapeUtils::shapeAsString(bias).c_str());
  }

  std::vector<sd::LongType> longAxis = ArrayUtils::toLongVector(axis);
  shape::checkDimensions(input->rankOf(), &longAxis);

  // Fast path: normalizing over last dimension with contiguous row-major data
  // This is the common BERT case: [batch, seq, hidden] normalized over hidden
  const int rank = input->rankOf();
  const bool lastDimNorm = (longAxis.size() == 1 && longAxis[0] == rank - 1);
  const bool inputContiguous = input->ordering() == 'c' &&
                               shape::strideDescendingCAscendingF(input->shapeInfo());
  const bool outputContiguous = output->ordering() == 'c' &&
                                shape::strideDescendingCAscendingF(output->shapeInfo());
  const bool isContiguous = inputContiguous && outputContiguous;
  const bool isFloat = input->dataType() == DataType::FLOAT32;
  const bool isDouble = input->dataType() == DataType::DOUBLE;
  const bool isHalf = input->dataType() == DataType::HALF;
  const bool gainContiguous = shape::strideDescendingCAscendingF(gain->shapeInfo());
  const bool biasContiguous = bias == nullptr || shape::strideDescendingCAscendingF(bias->shapeInfo());
  const bool gainTypeSupported = gain->dataType() == DataType::FLOAT32 ||
                                 gain->dataType() == DataType::DOUBLE ||
                                 gain->dataType() == DataType::HALF;
  const bool biasTypeSupported = bias == nullptr ||
                                 bias->dataType() == DataType::FLOAT32 ||
                                 bias->dataType() == DataType::DOUBLE ||
                                 bias->dataType() == DataType::HALF;

  // CUDA fast path: fused kernel for last-dimension normalization
  // If input is non-contiguous (e.g. from permute view), dup to contiguous first —
  // one cudaMemcpy is far cheaper than 8 separate decomposed kernel launches
  if (sd::graph::dspIsCudaBuild() &&
      lastDimNorm && (isFloat || isDouble || isHalf) && gainContiguous && biasContiguous &&
      gainTypeSupported && biasTypeSupported) {
    NDArray* inputToUse = input;
    NDArray* contiguousInput = nullptr;
    NDArray* outputToUse = output;
    NDArray* contiguousOutput = nullptr;

    if (!inputContiguous) {
      contiguousInput = new NDArray(input->dup('c'));
      inputToUse = contiguousInput;
    }
    if (!outputContiguous) {
      contiguousOutput = new NDArray(output->dup('c'));
      outputToUse = contiguousOutput;
    }

    helpers::layerNorm(inputToUse, gain, bias, outputToUse, longAxis, 1e-5f, block.launchContext());

    if (contiguousOutput != nullptr) {
      output->assign(outputToUse);
      delete contiguousOutput;
    }
    if (contiguousInput != nullptr) {
      delete contiguousInput;
    }

    return sd::Status::OK;
  }

  const bool sameParamTypes = gain->dataType() == input->dataType() &&
                              (bias == nullptr || bias->dataType() == input->dataType());
  if (lastDimNorm && isContiguous && (isFloat || isDouble) && gainContiguous && biasContiguous &&
      sameParamTypes) {
    // Fused layer norm: 2 passes instead of 7-8
    // This is a CPU implementation, so we need to sync data from device to host first.
    input->syncToHost();
    gain->syncToHost();
    if (bias != nullptr) {
      bias->syncToHost();
    }

    const sd::LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const sd::LongType rowLen = input->sizeAt(-1);
    const double epsilon = 1e-5;

    if (isFloat) {
      const float* x = input->bufferAsT<float>();
      float* z = output->bufferAsT<float>();
      const float* g = gain->bufferAsT<float>();
      const float* b = bias != nullptr ? bias->bufferAsT<float>() : nullptr;
      const float eps = static_cast<float>(epsilon);

      auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
          const float* xRow = x + row * rowLen;
          float* zRow = z + row * rowLen;

          // Pass 1: compute mean and variance
          float sum = 0.0f;
          for (sd::LongType i = 0; i < rowLen; ++i) {
            sum += xRow[i];
          }
          const float mean = sum / static_cast<float>(rowLen);

          float varSum = 0.0f;
          for (sd::LongType i = 0; i < rowLen; ++i) {
            float diff = xRow[i] - mean;
            varSum += diff * diff;
          }
          const float invStd = 1.0f / sd::math::sd_sqrt<float, float>(varSum / static_cast<float>(rowLen) + eps);

          // Pass 2: normalize, scale, shift
          if (b != nullptr) {
            for (sd::LongType i = 0; i < rowLen; ++i) {
              zRow[i] = (xRow[i] - mean) * invStd * g[i] + b[i];
            }
          } else {
            for (sd::LongType i = 0; i < rowLen; ++i) {
              zRow[i] = (xRow[i] - mean) * invStd * g[i];
            }
          }
        }
      };
      samediff::Threads::parallel_tad(func, 0, numRows);
    } else {
      // Double precision
      const double* x = input->bufferAsT<double>();
      double* z = output->bufferAsT<double>();
      const double* g = gain->bufferAsT<double>();
      const double* b = bias != nullptr ? bias->bufferAsT<double>() : nullptr;

      auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
          const double* xRow = x + row * rowLen;
          double* zRow = z + row * rowLen;

          // Pass 1: compute mean and variance
          double sum = 0.0;
          for (sd::LongType i = 0; i < rowLen; ++i) {
            sum += xRow[i];
          }
          const double mean = sum / static_cast<double>(rowLen);

          double varSum = 0.0;
          for (sd::LongType i = 0; i < rowLen; ++i) {
            double diff = xRow[i] - mean;
            varSum += diff * diff;
          }
          const double invStd = 1.0 / sd::math::sd_sqrt<double, double>(varSum / static_cast<double>(rowLen) + epsilon);

          // Pass 2: normalize, scale, shift
          if (b != nullptr) {
            for (sd::LongType i = 0; i < rowLen; ++i) {
              zRow[i] = (xRow[i] - mean) * invStd * g[i] + b[i];
            }
          } else {
            for (sd::LongType i = 0; i < rowLen; ++i) {
              zRow[i] = (xRow[i] - mean) * invStd * g[i];
            }
          }
        }
      };
      samediff::Threads::parallel_tad(func, 0, numRows);
    }

    // After writing to HOST, mark output as written on HOST and sync to DEVICE
    output->tickWriteHost();
    output->syncToDevice();

    return sd::Status::OK;
  }

  // General path for non-contiguous or non-last-dimension normalization.

  // Pre-compute shape for mean/stdev with keepDims=true for broadcasting
  auto reducedShapeInfo = ShapeUtils::evalReduceShapeInfo('c', &longAxis, input->shapeInfo(), true, false, block.getWorkspace());

  // Create mean and stdev arrays on stack with pre-computed shape
  NDArray means(reducedShapeInfo, true, block.launchContext());
  NDArray stdev(reducedShapeInfo, true, block.launchContext());

  // Compute mean directly into pre-allocated array
  input->reduceAlongDimension(reduce::Mean, &means, &longAxis, true, false);

  // Compute variance directly into stdev array (we'll transform it to stdev in-place)
  input->varianceAlongDimension(variance::SummaryStatsVariance, stdev, false, &longAxis);

  // stdev = sqrt(variance + epsilon)
  stdev.applyScalar(scalar::Add, 1e-5, &stdev);
  stdev.applyTransform(transform::Sqrt, &stdev);

  // output = (input - mean) / stdev.
  // Use a temporary for (input - mean) to avoid read/write aliasing in the Divide broadcast.
  NDArray inputMinusMean(input->shapeInfo(), false, block.launchContext());
  input->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), &means, &inputMinusMean, false);
  inputMinusMean.applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), &stdev, output, false);

  // Apply gain and bias
  std::vector<sd::LongType> dimcVec = {dimC};
  output->applyBroadcast(sd::broadcast::Multiply, &dimcVec, gain, output);
  if (bias != nullptr) {
    helpers::addBias(block, *output, *bias, *output, isNCHW);
  }

  return sd::Status::OK;
}

DECLARE_TYPES(layer_norm) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
}

CUSTOM_OP_IMPL(layer_norm_bp, 3, -1, false, 0, -1) {
  auto input = INPUT_VARIABLE(0);
  auto gain = INPUT_VARIABLE(1);
  auto bias = block.width() == 4 ? INPUT_VARIABLE(2) : nullptr;
  auto eps = block.width() == 4 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  auto dLdx = OUTPUT_VARIABLE(0);
  auto dLdg = OUTPUT_VARIABLE(1);
  auto dLdb = block.width() == 4 ? OUTPUT_VARIABLE(2) : nullptr;

  // Handle empty input gracefully - nothing to backpropagate
  if (input->isEmpty() || input->lengthOf() == 0) {
    return sd::Status::OK;
  }

  const bool isNCHW = block.getBArguments()->size() > 0 ? B_ARG(0) : true;  //  0-NCHW,  1-NHWC
  const int dimC = isNCHW ? 1 : input->rankOf() - 1;

  REQUIRE_TRUE(gain->rankOf() == 1 && gain->sizeAt(0) == input->sizeAt(dimC), 0,
               "LAYER_NORM_BP OP: wrong shape of gain array, expected is {%i}, but got %s instead !",
               input->sizeAt(dimC), ShapeUtils::shapeAsString(gain).c_str());

  std::vector<sd::LongType> axis = *block.getIArguments();

  std::vector<sd::LongType> longAxis = ArrayUtils::toLongVector(axis);

  if (bias != nullptr) {
    REQUIRE_TRUE(bias->rankOf() == 1 && bias->sizeAt(0) == input->sizeAt(dimC), 0,
                 "LAYER_NORM_BP OP: wrong shape of bias array, expected is {%i}, but got %s instead !",
                 input->sizeAt(dimC), ShapeUtils::shapeAsString(bias).c_str());
    std::vector<sd::LongType> dimCVector = {dimC};
    auto vec = ShapeUtils::evalDimsToExclude(input->rankOf(),1,dimCVector.data());
    eps->reduceAlongDimension(sd::reduce::Sum, dLdb, vec);
    delete vec;
  }

  const double epsilon = 1e-5;

  // Compute mean and inverse standard deviation WITH epsilon — must match the forward pass exactly.
  // The forward computes: output = (input - mean) / sqrt(var + eps) * gain + bias
  // Using standardize_bp (which divides by sqrt(var), no epsilon) produces systematically
  // wrong gradients, especially when variance is near zero (testLayerNormNoDeviation).
  auto reducedShapeInfo = ShapeUtils::evalReduceShapeInfo('c', &longAxis, input->shapeInfo(), true, false, block.getWorkspace());
  NDArray means(reducedShapeInfo, true, block.launchContext());
  NDArray invStd(reducedShapeInfo, true, block.launchContext());

  input->reduceAlongDimension(reduce::Mean, &means, &longAxis, true, false);
  input->varianceAlongDimension(variance::SummaryStatsVariance, invStd, false, &longAxis);
  invStd.applyScalar(scalar::Add, epsilon, &invStd);
  invStd.applyTransform(transform::Sqrt, &invStd);

  // standardized = (input - mean) / sqrt(var + eps), matching forward exactly.
  // Compute in two steps using a separate temporary to avoid read/write aliasing
  // in the Divide broadcast (source == target on CUDA can produce wrong results).
  NDArray inputMinusMean(input->shapeInfo(), false, block.launchContext());
  input->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), &means, &inputMinusMean, false);
  NDArray standardized(input->shapeInfo(), false, block.launchContext());
  inputMinusMean.applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), &invStd, &standardized, false);

  // dLdg = sum(eps_upstream * standardized, over all dims except dimC)
  NDArray epsTimesStd(input->shapeInfo(), false, block.launchContext());
  standardized.applyPairwiseTransform(sd::pairwise::Multiply, eps, &epsTimesStd);
  std::vector<sd::LongType> dimCVector = {dimC};
  auto vec = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, dimCVector.data());
  epsTimesStd.reduceAlongDimension(sd::reduce::Sum, dLdg, vec);
  delete vec;

  // dLdx: analytical gradient of layer_norm w.r.t. input
  //
  // layer_norm(x) = (x - mean) / sqrt(var + eps) * g + b
  // Let xhat = (x - mean) / sqrt(var + eps),  dL/d(xhat) = eps_upstream * g
  //
  // dL/dx = (1/sqrt(var+eps)) * (dLdxhat - mean(dLdxhat) - xhat * mean(dLdxhat * xhat))
  //
  // where mean(...) is the mean over the normalization axes.
  //
  // Implementation note: all intermediate results are written to DISTINCT output arrays
  // (no read-write aliasing) to ensure correctness on CUDA where the same device
  // buffer appearing in both dX and dZ of a broadcast kernel can produce wrong results
  // when the broadcast shape info is non-trivial.

  // Step 1: dLdxHat = eps_upstream * gain (broadcast gain along dimC)
  //         Use a separate array; dLdx is written last to avoid read/write aliasing.
  NDArray dLdxHat(input->shapeInfo(), false, block.launchContext());
  std::vector<sd::LongType> dimvC = {dimC};
  eps->applyBroadcast(sd::broadcast::Multiply, &dimvC, gain, &dLdxHat);

  // Step 2: mean(dLdxHat) over normalization axes, keepDims=true for broadcasting
  NDArray dLdxHatMean(reducedShapeInfo, true, block.launchContext());
  dLdxHat.reduceAlongDimension(reduce::Mean, &dLdxHatMean, &longAxis, true, false);

  // Step 3: mean(dLdxHat * xhat) over normalization axes, keepDims=true
  NDArray dLdxHatTimesXhat(input->shapeInfo(), false, block.launchContext());
  dLdxHat.applyPairwiseTransform(sd::pairwise::Multiply, &standardized, &dLdxHatTimesXhat);
  NDArray dLdxHatTimesXhatMean(reducedShapeInfo, true, block.launchContext());
  dLdxHatTimesXhat.reduceAlongDimension(reduce::Mean, &dLdxHatTimesXhatMean, &longAxis, true, false);

  // Step 4: term = xhat * mean(dLdxHat * xhat)  [broadcast reducedShape → inputShape]
  NDArray term(input->shapeInfo(), false, block.launchContext());
  standardized.applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), &dLdxHatTimesXhatMean, &term, false);

  // Step 5: numerator = dLdxHat - mean(dLdxHat)  [broadcast subtract, distinct source/target]
  NDArray numerator(input->shapeInfo(), false, block.launchContext());
  dLdxHat.applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), &dLdxHatMean, &numerator, false);

  // Step 6: numerator2 = numerator - term  [elementwise subtract, distinct target]
  NDArray numerator2(input->shapeInfo(), false, block.launchContext());
  numerator.applyPairwiseTransform(sd::pairwise::Subtract, &term, &numerator2);

  // Step 7: dLdx = numerator2 / invStd  [broadcast divide; writes to output, no aliasing]
  numerator2.applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), &invStd, dLdx, false);

  return sd::Status::OK;
}

DECLARE_TYPES(layer_norm_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(layer_norm_bp) {
  if (inputShape->size() > 3) {
    return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)), CONSTANT(inputShape->at(2)));
  }
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)));
}

}  // namespace ops
}  // namespace sd

#endif
