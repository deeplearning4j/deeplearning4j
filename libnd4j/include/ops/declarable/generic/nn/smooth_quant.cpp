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
// @author Adam Gibson
//
// smooth_quant - SmoothQuant W8A8 quantized matmul
//
// Implements the SmoothQuant technique: Y = (X * diag(s)^-1) @ (diag(s) * W)
// where s is a per-channel smoothing scale computed offline via calibration.
// The smoothed activation quantizes onto the grid of the weight codes (INT8 or
// FP8) by the activation scale, and the product dequantizes by both scales.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_smooth_quant)
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/int8_gemm.h>
#include <helpers/ConstantShapeHelper.h>

#include <vector>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(smooth_quant, 2, 1, false, 0, 0) {
  const int width = block.width();
  REQUIRE_TRUE(width == 2 || width == 5 || width == 6, 0,
               "smooth_quant: expected X and the smoothing scale (2 inputs), or X, W, the smoothing, activation and "
               "weight scales and an optional bias (5 or 6 inputs), got %i inputs",
               width);
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  REQUIRE_TRUE(input->rankOf() >= 1, 0, "smooth_quant: X must have at least one axis");
  const LongType depth = input->sizeAt(-1);

  if (width == 2) {
    auto smoothScale = INPUT_VARIABLE(1);
    REQUIRE_TRUE(DataTypeUtils::isR(smoothScale->dataType()), 0,
                 "smooth_quant: the smoothing scale must be floating, got %s",
                 DataTypeUtils::asString(smoothScale->dataType()).c_str());
    REQUIRE_TRUE(smoothScale->lengthOf() == depth, 0,
                 "smooth_quant: the smoothing scale must hold one value per channel (%lld), got %lld", depth,
                 smoothScale->lengthOf());
    REQUIRE_TRUE(output->isSameShape(input), 0, "smooth_quant: the smoothed output must have X's shape");
    helpers::smoothActivation(block.launchContext(), input, smoothScale, output);
    return Status::OK;
  }

  auto weight = INPUT_VARIABLE(1);
  auto smoothScale = INPUT_VARIABLE(2);
  auto actScale = INPUT_VARIABLE(3);
  auto weightScale = INPUT_VARIABLE(4);
  // An empty bias input stands for an absent one.
  auto bias = width > 5 && !INPUT_VARIABLE(5)->isEmpty() ? INPUT_VARIABLE(5) : nullptr;
  const bool transposeWeight = INT_ARG_OR(0, 0) != 0;
  const DataType weightType = weight->dataType();
  REQUIRE_TRUE(DataTypeUtils::isR(weightType) || (DataTypeUtils::isZ(weightType) && !DataTypeUtils::isU(weightType)),
               0, "smooth_quant: W must hold signed integer or floating codes, got %s",
               DataTypeUtils::asString(weightType).c_str());
  REQUIRE_TRUE(weight->rankOf() == 2, 0, "smooth_quant: W must be a matrix, got rank %i", weight->rankOf());
  const LongType columns = weight->sizeAt(transposeWeight ? 1 : 0);
  const LongType weightDepth = weight->sizeAt(transposeWeight ? 0 : 1);
  REQUIRE_TRUE(weightDepth == depth, 0, "smooth_quant: X has %lld channels but W has %lld", depth, weightDepth);
  REQUIRE_TRUE(smoothScale->lengthOf() == depth, 0,
               "smooth_quant: the smoothing scale must hold one value per channel (%lld), got %lld", depth,
               smoothScale->lengthOf());
  REQUIRE_TRUE(actScale->lengthOf() == 1 || actScale->lengthOf() == depth, 0,
               "smooth_quant: the activation scale must hold one value or one per channel (%lld), got %lld", depth,
               actScale->lengthOf());
  REQUIRE_TRUE(weightScale->lengthOf() == 1 || weightScale->lengthOf() == columns, 0,
               "smooth_quant: the weight scale must hold one value or one per output channel (%lld), got %lld",
               columns, weightScale->lengthOf());
  REQUIRE_TRUE(bias == nullptr || bias->lengthOf() == columns, 0,
               "smooth_quant: bias must hold one value per output channel (%lld), got %lld", columns,
               bias == nullptr ? 0 : bias->lengthOf());
  std::vector<LongType> outputShape(input->shapeOf(), input->shapeOf() + input->rankOf());
  outputShape.back() = columns;
  REQUIRE_TRUE(output->isSameShape(outputShape), 0, "smooth_quant: output must be X's shape with %lld channels",
               columns);

  helpers::smoothQuantGemm(block.launchContext(), input, weight, smoothScale, actScale, weightScale, bias, output,
                           transposeWeight);
  return Status::OK;
}

DECLARE_TYPES(smooth_quant) {
  // Input 1 is the smoothing scale (2 inputs) or the weight codes, integer or
  // FP8 (5 or 6 inputs); the op checks which.
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, DataType::ANY)
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setShapeValueInputs({})
      ->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(smooth_quant) {
  auto in = inputShape->at(0);
  const int rank = shape::rank(in);
  REQUIRE_TRUE(rank >= 1, 0, "smooth_quant: X must have at least one axis");
  std::vector<LongType> dims(shape::shapeOf(in), shape::shapeOf(in) + rank);
  if (inputShape->size() >= 5) {
    auto w = inputShape->at(1);
    REQUIRE_TRUE(shape::rank(w) == 2, 0, "smooth_quant: W must be a matrix, got rank %i", shape::rank(w));
    dims.back() = shape::sizeAt(w, INT_ARG_OR(0, 0) != 0 ? 1 : 0);
  }
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(in), 'c', dims));
}

}  // namespace ops
}  // namespace sd

#endif
