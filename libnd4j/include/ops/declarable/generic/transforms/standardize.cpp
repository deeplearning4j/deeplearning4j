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
  output->nullify();
  std::vector<sd::LongType> axis;

  if (block.width() > 1)
    axis = INPUT_VARIABLE(1)->template asVectorT<sd::LongType>();
  else if (block.numI() > 0)
    axis = *block.getIArguments();

  REQUIRE_TRUE(!axis.empty(), 0, "STANDARDIZE OP: axis has to be non-empty")

  shape::checkDimensions(input->rankOf(), &axis);

  // Compute mean with keepDims=true for broadcasting
  auto means = input->reduceAlongDimension(reduce::Mean, &axis, true);

  // Compute VARIANCE (not stdev) - uses Welford's algorithm internally
  // biasCorrected=false gives population variance (divide by N, not N-1)
  auto varianceRaw = input->varianceAlongDimension(variance::SummaryStatsVariance, false, &axis);

  // Reshape variance to match means shape for broadcasting (use reshape instead of reshapei)
  auto meansShape = means->getShapeAsVector();
  auto variance = varianceRaw->reshape(varianceRaw->ordering(), *meansShape);
  delete meansShape;
  delete varianceRaw;

  // Add epsilon BEFORE sqrt: stdev = sqrt(variance + epsilon)
  // This is the numerically stable formula for LayerNorm
  NDArray* varPlusEps = *variance + 1e-5;
  NDArray* stdev = varPlusEps->transform(transform::Sqrt);

  // output = (input - mean) / stdev
  input->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), means, output, false);
  output->applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), stdev, output, false);

  delete means;
  delete variance;
  delete varPlusEps;
  delete stdev;

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

  auto stdevRaw = input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, false, &axis);
  REQUIRE_TRUE(stdevRaw != nullptr, 0, "STANDARDIZE_BP OP: failed to compute standard deviation along dimension");

  // Reshape stdev to match means shape for broadcasting (use reshape instead of reshapei)
  auto meansShape = means->getShapeAsVector();
  auto stdev = stdevRaw->reshape(stdevRaw->ordering(), *meansShape);
  delete meansShape;
  delete stdevRaw;

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
