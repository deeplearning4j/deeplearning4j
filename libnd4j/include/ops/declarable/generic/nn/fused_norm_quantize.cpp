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
// fused_norm_quantize - normalization followed by per-row symmetric quantization
//
// The rows of the last axis normalize, by RMSNorm (normType 0) or LayerNorm
// (normType 1), scale by gamma and shift by the optional beta. Each normalized
// row then quantizes symmetrically onto the grid of the quantized data type
// (INT8 by default; any signed integer or floating type, FP8 included):
//   scale = max|row| / qmax,  code = snap(clamp(row / scale, -qmax, qmax))
// so code * scale reconstructs the row. The op returns the codes and the
// per-row scales.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_fused_norm_quantize)
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/rms_norm.h>
#include <ops/declarable/helpers/symmetric_quant.h>
#include <ops/op_types.h>

#include <cmath>
#include <vector>

namespace sd {
namespace ops {

// The rows normalize in the aggregate type of the input and quantize from it,
// so each normalized value rounds once, into its code. LayerNorm centres each
// row on its mean and normalizes the centred row like RMSNorm: the mean square
// of a centred row is the row's variance. The codes use the row scales as
// rounded into the scales' data type, so the returned scales dequantize them
// exactly.
template <typename X>
static void fusedNormQuantize_(LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* beta,
                               NDArray* quantized, NDArray* scales, bool layerNorm, double epsilon) {
  using AccT = typename simdOps::AggregateType<X>::type;
  const DataType accType = DataTypeUtils::fromT<AccT>();
  const int rank = input->rankOf();
  std::vector<LongType> shape(input->shapeOf(), input->shapeOf() + rank);
  std::vector<LongType> rowShape(shape.begin(), shape.end() - 1);
  rowShape.push_back(1);

  NDArray* rows = input;
  if (layerNorm || input->dataType() != accType) {
    rows = new NDArray('c', shape, accType, context);
    rows->assign(input);
  }
  if (layerNorm) {
    const std::vector<LongType> lastAxis = {rank - 1};
    auto* mean = new NDArray('c', rowShape, accType, context);
    rows->reduceAlongDimension(reduce::Mean, mean, &lastAxis, true);
    *rows -= *mean;
    MmulHelper::deleteTemporary(mean);
  }
  auto* normed = new NDArray('c', shape, accType, context);
  helpers::rmsNorm(context, rows, gamma, normed, epsilon);
  if (rows != input) MmulHelper::deleteTemporary(rows);
  if (beta != nullptr) {
    // beta stages in its own shape, then lines up with the last axis.
    std::vector<LongType> betaShape(beta->shapeOf(), beta->shapeOf() + beta->rankOf());
    std::vector<LongType> shiftShape(rank, 1);
    shiftShape.back() = shape.back();
    auto* shift = new NDArray('c', betaShape, accType, context);
    shift->assign(beta);
    shift->reshapei('c', shiftShape);
    *normed += *shift;
    MmulHelper::deleteTemporary(shift);
  }

  NDArray* rowScale = new NDArray('c', rowShape, accType, context);
  helpers::symmetricRowScale(context, normed, quantized->dataType(), rowScale);
  if (scales->dataType() != accType) {
    auto* rounded = new NDArray('c', rowShape, scales->dataType(), context);
    rounded->assign(rowScale);
    MmulHelper::deleteTemporary(rowScale);
    rowScale = rounded;
  }
  helpers::symmetricQuantize(context, normed, rowScale, quantized);
  scales->assign(rowScale);
  MmulHelper::deleteTemporary(rowScale);
  MmulHelper::deleteTemporary(normed);
}

CUSTOM_OP_IMPL(fused_norm_quantize, 2, 2, false, 0, 0) {
  const int width = block.width();
  REQUIRE_TRUE(width <= 3, 0, "fused_norm_quantize: expected the input, gamma and an optional beta, got %i inputs",
               width);
  REQUIRE_TRUE(block.numI() <= 1, 0,
               "fused_norm_quantize: expected the normalization type as the only integer argument, got %i; the "
               "quantized data type is the data type argument",
               static_cast<int>(block.numI()));
  auto input = INPUT_VARIABLE(0);
  auto gamma = INPUT_VARIABLE(1);
  // An empty beta input stands for an absent one.
  auto beta = width > 2 && !INPUT_VARIABLE(2)->isEmpty() ? INPUT_VARIABLE(2) : nullptr;
  auto quantized = OUTPUT_VARIABLE(0);
  auto scales = OUTPUT_VARIABLE(1);
  const int normType = static_cast<int>(INT_ARG_OR(0, 0));
  const double epsilon = T_ARG_OR(0, 1e-5);

  REQUIRE_TRUE(normType == 0 || normType == 1, 0,
               "fused_norm_quantize: normType must be 0 (RMSNorm) or 1 (LayerNorm), got %i", normType);
  REQUIRE_TRUE(std::isfinite(epsilon) && epsilon >= 0, 0,
               "fused_norm_quantize: epsilon must be finite and nonnegative, got %f", epsilon);
  REQUIRE_TRUE(input->rankOf() >= 1, 0, "fused_norm_quantize: the input must have at least one axis");
  const LongType features = input->sizeAt(-1);
  REQUIRE_TRUE(gamma->rankOf() == 1 && gamma->lengthOf() == features, 0,
               "fused_norm_quantize: gamma must be a vector of one value per feature (%lld), got %s", features,
               ShapeUtils::shapeAsString(gamma).c_str());
  REQUIRE_TRUE(beta == nullptr || beta->lengthOf() == features, 0,
               "fused_norm_quantize: beta must hold one value per feature (%lld), got %s", features,
               beta == nullptr ? "" : ShapeUtils::shapeAsString(beta).c_str());
  const DataType quantizedType = quantized->dataType();
  REQUIRE_TRUE(DataTypeUtils::isR(quantizedType) ||
                   (DataTypeUtils::isZ(quantizedType) && !DataTypeUtils::isU(quantizedType)),
               0, "fused_norm_quantize: the codes need a symmetric grid, a signed integer or floating type, got %s",
               DataTypeUtils::asString(quantizedType).c_str());
  REQUIRE_TRUE(quantized->isSameShape(input), 0, "fused_norm_quantize: the codes must have the input's shape %s, got %s",
               ShapeUtils::shapeAsString(input).c_str(), ShapeUtils::shapeAsString(quantized).c_str());
  const std::vector<LongType> rows(input->shapeOf(), input->shapeOf() + input->rankOf() - 1);
  REQUIRE_TRUE(scales->rankOf() == input->rankOf() - 1 && (rows.empty() || scales->isSameShape(rows)), 0,
               "fused_norm_quantize: the scales must have the input's shape without the last axis, got %s",
               ShapeUtils::shapeAsString(scales).c_str());

  if (input->isEmpty()) {
    // Rows without elements have no magnitude, so their scale is zero.
    if (!scales->isEmpty()) scales->nullify();
    return Status::OK;
  }
  BUILD_SINGLE_SELECTOR(input->dataType(), fusedNormQuantize_,
                        (block.launchContext(), input, gamma, beta, quantized, scales, normType == 1, epsilon),
                        SD_FLOAT_TYPES);
  return Status::OK;
}

DECLARE_TYPES(fused_norm_quantize) {
  // The codes take any type with a symmetric grid (checked in the op).
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedOutputTypes(0, DataType::ANY)
      ->setAllowedOutputTypes(1, {ALL_FLOATS})
      ->setShapeValueInputs({})
      ->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(fused_norm_quantize) {
  auto in = inputShape->at(0);
  const int rank = shape::rank(in);
  REQUIRE_TRUE(rank >= 1, 0, "fused_norm_quantize: the input must have at least one axis");
  const DataType dtype = ArrayOptions::dataType(in);
  // The codes default to INT8; a data type argument chooses another grid.
  const DataType quantizedType = block.numD() > 0 ? D_ARG(0) : INT8;
  std::vector<LongType> dims(shape::shapeOf(in), shape::shapeOf(in) + rank);
  auto* codes = ConstantShapeHelper::getInstance().createShapeInfo(quantizedType, 'c', dims);
  dims.pop_back();
  auto* scales = dims.empty() ? ConstantShapeHelper::getInstance().scalarShapeInfo(dtype)
                              : ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', dims);
  return SHAPELIST(codes, scales);
}

}  // namespace ops
}  // namespace sd
#endif
