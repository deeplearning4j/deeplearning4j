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
// awq_matmul - matmul with AWQ group-quantized weights
//
// Y = X @ W + bias for a weight W [K, N] stored as numBits-wide unsigned codes
// packed along K: byte (c, n) of the packed weights [ceil(K * numBits / 8), N]
// holds the codes of rows c * (8 / numBits) + j in bits [j * numBits,
// (j + 1) * numBits). Rows group by groupSize, and each group dequantizes by its
// scale and zero point per output channel:
//   W[k, n] = (code(k, n) - zeros[k / groupSize, n]) * scales[k / groupSize, n]
// Without zeros the zero point is the middle of the code range, 2^(numBits - 1).
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_awq_matmul)
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/int8_gemm.h>
#include <ops/declarable/helpers/weight_dequant.h>
#include <ops/op_types.h>

#include <vector>

namespace sd {
namespace ops {

// W dequantizes in the aggregate type of the output, where the scaled GEMM
// accumulates, so each weight rounds once. The dequantization writes W as
// [N, K] and reads the packed weights, scales and zeros as [N, ...] transposed
// views, which it addresses through their strides.
template <typename Z>
static void awqMatmul_(LaunchContext* context, NDArray* input, NDArray* packed, NDArray* scales, NDArray* zeros,
                       NDArray* bias, NDArray* output, int groupSize, int numBits) {
  using AccT = typename simdOps::AggregateType<Z>::type;
  std::vector<LongType> transpose = {1, 0};
  NDArray* packedT = packed->permute(transpose, false, false);
  NDArray* scalesT = scales->permute(transpose, false, false);
  NDArray* zerosT = zeros != nullptr ? zeros->permute(transpose, false, false) : nullptr;
  std::vector<LongType> weightShape = {packed->sizeAt(1), input->sizeAt(-1)};
  auto* weight = new NDArray('c', weightShape, DataTypeUtils::fromT<AccT>(), context);
  helpers::awqDequantize(context, packedT, scalesT, zerosT, weight, groupSize, numBits);
  delete packedT;
  delete scalesT;
  delete zerosT;
  helpers::scaledGemm(context, input, weight, nullptr, nullptr, bias, output, false, true);
  MmulHelper::deleteTemporary(weight);
}

CUSTOM_OP_IMPL(awq_matmul, 3, 1, false, 0, 0) {
  const int width = block.width();
  REQUIRE_TRUE(width <= 5, 0,
               "awq_matmul: expected X, the packed weights, the scales, optional zeros and an optional bias, got %i "
               "inputs",
               width);
  auto input = INPUT_VARIABLE(0);
  auto packed = INPUT_VARIABLE(1);
  auto scales = INPUT_VARIABLE(2);
  // Empty zeros or bias inputs stand for absent ones.
  auto zeros = width > 3 && !INPUT_VARIABLE(3)->isEmpty() ? INPUT_VARIABLE(3) : nullptr;
  auto bias = width > 4 && !INPUT_VARIABLE(4)->isEmpty() ? INPUT_VARIABLE(4) : nullptr;
  auto output = OUTPUT_VARIABLE(0);
  const int groupSize = static_cast<int>(INT_ARG_OR(0, 128));
  const int numBits = static_cast<int>(INT_ARG_OR(1, 4));

  REQUIRE_TRUE(numBits > 0 && 8 % numBits == 0, 0, "awq_matmul: numBits must be 1, 2, 4 or 8, got %i", numBits);
  REQUIRE_TRUE(groupSize > 0, 0, "awq_matmul: groupSize must be positive, got %i", groupSize);
  REQUIRE_TRUE(DataTypeUtils::isZ(packed->dataType()) && packed->sizeOfT() == 1, 0,
               "awq_matmul: the packed weights must hold one-byte integer codes, got %s",
               DataTypeUtils::asString(packed->dataType()).c_str());
  REQUIRE_TRUE(zeros == nullptr || zeros->dataType() == scales->dataType(), 0,
               "awq_matmul: zeros must have the scales' data type %s, got %s",
               DataTypeUtils::asString(scales->dataType()).c_str(),
               zeros == nullptr ? "" : DataTypeUtils::asString(zeros->dataType()).c_str());
  REQUIRE_TRUE(input->rankOf() >= 1, 0, "awq_matmul: X must have at least one axis");
  REQUIRE_TRUE(packed->rankOf() == 2 && scales->rankOf() == 2, 0,
               "awq_matmul: the packed weights and scales must be matrices, got %s and %s",
               ShapeUtils::shapeAsString(packed).c_str(), ShapeUtils::shapeAsString(scales).c_str());
  const LongType depth = input->sizeAt(-1);
  const LongType columns = packed->sizeAt(1);
  const LongType codesPerByte = 8 / numBits;
  const LongType packedRows = (depth + codesPerByte - 1) / codesPerByte;
  const LongType groups = (depth + groupSize - 1) / groupSize;
  REQUIRE_TRUE(packed->sizeAt(0) == packedRows, 0,
               "awq_matmul: X has %lld channels, which pack %lld to a byte into %lld rows, but the packed weights "
               "are %s",
               depth, codesPerByte, packedRows, ShapeUtils::shapeAsString(packed).c_str());
  REQUIRE_TRUE(scales->sizeAt(0) == groups && scales->sizeAt(1) == columns, 0,
               "awq_matmul: the scales must hold one value per group of %i channels and output channel, [%lld, "
               "%lld], got %s",
               groupSize, groups, columns, ShapeUtils::shapeAsString(scales).c_str());
  REQUIRE_TRUE(zeros == nullptr || zeros->isSameShape(scales), 0,
               "awq_matmul: zeros must have the scales' shape %s, got %s", ShapeUtils::shapeAsString(scales).c_str(),
               zeros == nullptr ? "" : ShapeUtils::shapeAsString(zeros).c_str());
  REQUIRE_TRUE(bias == nullptr || bias->lengthOf() == columns, 0,
               "awq_matmul: bias must hold one value per output channel (%lld), got %lld", columns,
               bias == nullptr ? 0 : bias->lengthOf());
  std::vector<LongType> outputShape(input->shapeOf(), input->shapeOf() + input->rankOf());
  outputShape.back() = columns;
  REQUIRE_TRUE(output->isSameShape(outputShape), 0, "awq_matmul: output must be X's shape with %lld channels, got %s",
               columns, ShapeUtils::shapeAsString(output).c_str());
  if (output->isEmpty()) return Status::OK;

  BUILD_SINGLE_SELECTOR(output->dataType(), awqMatmul_,
                        (block.launchContext(), input, packed, scales, zeros, bias, output, groupSize, numBits),
                        SD_FLOAT_TYPES);
  return Status::OK;
}

DECLARE_TYPES(awq_matmul) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setShapeValueInputs({})
      ->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(awq_matmul) {
  auto in = inputShape->at(0);
  auto packed = inputShape->at(1);
  const int rank = shape::rank(in);
  REQUIRE_TRUE(rank >= 1, 0, "awq_matmul: X must have at least one axis");
  REQUIRE_TRUE(shape::rank(packed) == 2, 0, "awq_matmul: the packed weights must be a matrix, got rank %i",
               shape::rank(packed));
  std::vector<LongType> dims(shape::shapeOf(in), shape::shapeOf(in) + rank);
  dims.back() = shape::sizeAt(packed, static_cast<LongType>(1));
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(in), 'c', dims));
}

}  // namespace ops
}  // namespace sd
#endif
