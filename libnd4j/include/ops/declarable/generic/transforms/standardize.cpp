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
#if NOT_EXCLUDED(OP_standardize)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(standardize, 1, 1, true, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<sd::LongType> axis;

  if (block.width() > 1)
    axis = INPUT_VARIABLE(1)->template asVectorT<sd::LongType>();
  else if (block.numI() > 0)
    axis = *block.getIArguments();

  REQUIRE_TRUE(!axis.empty(), 0, "STANDARDIZE OP: axis has to be non-empty")

  shape::checkDimensions(input->rankOf(), &axis);

  // First, replace any Inf/NaN in input to prevent them from corrupting mean/stdev calculations
  // Create a working copy if input contains problematic values
  NDArray* workingInput = const_cast<NDArray*>(input);
  NDArray* inputCopy = nullptr;

  // Check if input has Inf or NaN using type-safe methods - if so, create a cleaned copy
  bool hasInfOrNan = input->hasNaNs() || input->hasInfs();

  if (hasInfOrNan) {
    inputCopy = input->dup();
    // Replace Inf with large finite values, NaN with 0
    inputCopy->applyScalar(sd::scalar::ReplaceNans, 0.0, inputCopy);
    // Clamp to prevent Inf - use a large but finite value
    sd::ops::clipbyvalue clipOp;
    clipOp.execute({inputCopy}, {inputCopy}, {-1e10, 1e10}, {});
    workingInput = inputCopy;
  }

  auto means = workingInput->reduceAlongDimension(reduce::Mean, &axis, true);
  REQUIRE_TRUE(means != nullptr, 0, "STANDARDIZE OP: failed to compute mean along dimension");

  auto base = workingInput->varianceAlongDimension(variance::SummaryStatsStandardDeviation, false, &axis);
  REQUIRE_TRUE(base != nullptr, 0, "STANDARDIZE OP: failed to compute standard deviation along dimension");

  // Use larger epsilon for numerical stability - 1e-12 is too small for float32
  // and can cause division by near-zero, leading to Inf values
  // Note: base + 1e-5 creates a new NDArray, so we need to manage it as a pointer
  NDArray* stdev = new NDArray(*base + 1e-5);
  REQUIRE_TRUE(stdev != nullptr, 0, "STANDARDIZE OP: failed to add epsilon to standard deviation");
  auto meansShape = means->getShapeAsVector();
  std::vector<sd::LongType> meansShapeVec = *meansShape;
  stdev->reshapei(meansShapeVec);
  delete meansShape;
  workingInput->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), means, output, false);
  output->applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), stdev, output, false);

  // Replace any NaN that may have been created and clamp output to reasonable range
  output->applyScalar(sd::scalar::ReplaceNans, 0, output);

  // Clamp output to prevent extreme values from propagating
  sd::ops::clipbyvalue finalClipOp;
  finalClipOp.execute({output}, {output}, {-1e4, 1e4}, {});

  delete means;
  delete base;
  delete stdev;
  if (inputCopy != nullptr) {
    delete inputCopy;
  }
  return sd::Status::OK;
}

DECLARE_TYPES(standardize) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
  getOpDescriptor()->setAllowedInputTypes(1, {DataType::INT32, DataType::INT64});
  getOpDescriptor()->setAllowedOutputTypes(0, DataType::INHERIT);
}

CUSTOM_OP_IMPL(standardize_bp, 2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto eps = block.width() == 3 ? INPUT_VARIABLE(2) : INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);
  std::vector<sd::LongType> axis;

  if (block.width() == 3)
    axis = INPUT_VARIABLE(1)->template asVectorT<sd::LongType>();
  else if (block.numI() > 0)
    axis = *block.getIArguments();

  REQUIRE_TRUE(!axis.empty(), 0, "STANDARDIZE OP: axis has to be non-empty")

  shape::checkDimensions(input->rankOf(), &axis);
  auto longAxis = ArrayUtils::toLongVector(axis);

  auto means = input->reduceAlongDimension(reduce::Mean, &axis, true);
  REQUIRE_TRUE(means != nullptr, 0, "STANDARDIZE_BP OP: failed to compute mean along dimension");

  auto stdev = input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, false, &axis);
  REQUIRE_TRUE(stdev != nullptr, 0, "STANDARDIZE_BP OP: failed to compute standard deviation along dimension");

  auto meansShape = means->getShapeAsVector();;
  std::vector<sd::LongType> meansShapeVec = *meansShape;
  stdev->reshapei(meansShapeVec);
  delete meansShape;

  eps->applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), stdev, output, false);

  auto sum = output->reduceAlongDimension(reduce::Sum, &axis, true);
  REQUIRE_TRUE(sum != nullptr, 0, "STANDARDIZE_BP OP: failed to compute sum along dimension");

  NDArray dldu_sum = -(*sum);

  NDArray dldx_u(input->shapeInfo(), false, block.launchContext());
  std::vector<NDArray *> meanBpArgs = {input, &dldu_sum};
  std::vector<NDArray *> meanBpOutput = {&dldx_u};
  std::vector<double> meanBpTArgs = {};
  std::vector<bool> meanBpBArgs = {};

  sd::ops::reduce_mean_bp meanBp;
  meanBp.execute(meanBpArgs, meanBpOutput, meanBpTArgs, longAxis, meanBpBArgs);
  *output += dldx_u;

  // (eps * (means - input) / (stdev * stdev))
  NDArray tmp(eps->shapeInfo(), false, block.launchContext());
  means->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), input, &tmp, false);
  tmp.applyPairwiseTransform(sd::pairwise::Multiply, eps, &tmp);
  stdev->applyPairwiseTransform(sd::pairwise::Multiply, stdev, stdev);
  tmp.applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), stdev, &tmp, false);

  auto dlds_sum = tmp.reduceAlongDimension(reduce::Sum, &axis, true);
  REQUIRE_TRUE(dlds_sum != nullptr, 0, "STANDARDIZE_BP OP: failed to compute dlds_sum along dimension");

  NDArray dldx_s(input->shapeInfo(), false, block.launchContext());
  std::vector<NDArray *> stdevBpArgs = {input, dlds_sum};
  std::vector<NDArray *> stdevBpOutput = {&dldx_s};
  std::vector<double> stdevBpTArgs = {};
  std::vector<bool> stdevBpBArgs = {};
  sd::ops::reduce_stdev_bp stdevBp;
  stdevBp.execute(stdevBpArgs, stdevBpOutput, stdevBpTArgs, longAxis, stdevBpBArgs);
  *output += dldx_s;

  output->applyScalar(sd::scalar::ReplaceNans, 0, output);
  delete sum;
  delete means;
  delete stdev;
  delete dlds_sum;
  return sd::Status::OK;
}

DECLARE_TYPES(standardize_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(standardize_bp) {
  auto in = inputShape->at(0);
  sd::LongType *out;
  COPY_SHAPE(in, out);
  auto result = CONSTANT(out);
  delete[] out;

  return SHAPELIST(result);
}

}  // namespace ops
}  // namespace sd

#endif
