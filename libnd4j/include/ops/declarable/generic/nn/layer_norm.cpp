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

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/reverse.h>
#include <helpers/ShapeUtils.h>
#include <execution/Threads.h>
#include <cmath>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(layer_norm, 2, 1, false, 0, -1) {
  auto input = INPUT_VARIABLE(0);
  auto gain = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

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
  const bool gainContiguous = shape::strideDescendingCAscendingF(gain->shapeInfo());
  const bool biasContiguous = bias == nullptr || shape::strideDescendingCAscendingF(bias->shapeInfo());

  if (lastDimNorm && isContiguous && (isFloat || isDouble) && gainContiguous && biasContiguous) {
    // Fused layer norm: 2 passes instead of 7-8
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
          const float invStd = 1.0f / std::sqrt(varSum / static_cast<float>(rowLen) + eps);

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
          const double invStd = 1.0 / std::sqrt(varSum / static_cast<double>(rowLen) + epsilon);

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

    return sd::Status::OK;
  }

  // General path for non-contiguous or non-last-dimension normalization
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

  // output = (input - mean) / stdev
  input->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), &means, output, false);
  output->applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), &stdev, output, false);

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
}

CUSTOM_OP_IMPL(layer_norm_bp, 3, -1, false, 0, -1) {
  auto input = INPUT_VARIABLE(0);
  auto gain = INPUT_VARIABLE(1);
  auto bias = block.width() == 4 ? INPUT_VARIABLE(2) : nullptr;
  auto eps = block.width() == 4 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  auto dLdx = OUTPUT_VARIABLE(0);
  auto dLdg = OUTPUT_VARIABLE(1);
  auto dLdb = block.width() == 4 ? OUTPUT_VARIABLE(2) : nullptr;

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

  NDArray standardized(input->shapeInfo(), false, block.launchContext());

  sd::ops::standardize standardizeOp;
  std::vector<NDArray *> inputs = {input};
  std::vector<NDArray *> outputs = {&standardized};
  std::vector<double> targs = {};
  std::vector<bool> bargs = {};

  auto status = standardizeOp.execute(inputs, outputs, targs, longAxis, bargs);
  if (status != sd::Status::OK) {
    std::string errorMessage;
    errorMessage += "LAYER_NORM_BP OP: standardize operation failed with status ";
    errorMessage += std::to_string(static_cast<int>(status));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  standardized.applyPairwiseTransform(sd::pairwise::Multiply, eps, &standardized);
  std::vector<sd::LongType> dimCVector = {dimC};
  auto vec = ShapeUtils::evalDimsToExclude(input->rankOf(),1,dimCVector.data());
  standardized.reduceAlongDimension(sd::reduce::Sum, dLdg, vec);
  delete vec;

  sd::ops::standardize_bp standardizeBp;
  std::vector<sd::LongType> dimvC = {dimC};
  eps->applyBroadcast(sd::broadcast::Multiply, &dimvC, gain, dLdx);

  auto dLdx_tmp = dLdx->dup();
  std::vector<NDArray *> standardizeBpArgs = {input, dLdx_tmp};
  std::vector<NDArray *> standardizeBpOut = {dLdx};
  status = standardizeBp.execute(standardizeBpArgs, standardizeBpOut, targs, longAxis, bargs);
  if (status != sd::Status::OK) {
    delete dLdx_tmp;
    std::string errorMessage;
    errorMessage += "LAYER_NORM_BP OP: standardize_bp operation failed with status ";
    errorMessage += std::to_string(static_cast<int>(status));
    THROW_EXCEPTION(errorMessage.c_str());
  }

  delete dLdx_tmp;

  return sd::Status::OK;
}

DECLARE_TYPES(layer_norm_bp) {
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
