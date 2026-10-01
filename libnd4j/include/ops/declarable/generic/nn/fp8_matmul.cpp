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
// fp8_matmul - scaled GEMM over FP8 operands
//
// C = (op(A) @ op(B)) * scaleA * scaleB + bias. A and B keep their storage
// types (FP8 E4M3 or E5M2, or a wider floating type); the storage type carries
// the format, and the product accumulates in the aggregate type of the output.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_fp8_matmul)
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/int8_gemm.h>
#include <helpers/ConstantShapeHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(fp8_matmul, 4, 1, false, 0, 0) {
  REQUIRE_TRUE(block.width() <= 5, 0, "fp8_matmul: expected A, B, scaleA, scaleB and an optional bias, got %i inputs",
               block.width());
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto scaleA = INPUT_VARIABLE(2);
  auto scaleB = INPUT_VARIABLE(3);
  // An empty bias input stands for an absent one.
  auto bias = block.width() > 4 && !INPUT_VARIABLE(4)->isEmpty() ? INPUT_VARIABLE(4) : nullptr;
  auto output = OUTPUT_VARIABLE(0);
  const bool transposeA = INT_ARG_OR(0, 0) != 0;
  const bool transposeB = INT_ARG_OR(1, 0) != 0;

  REQUIRE_TRUE(DataTypeUtils::isR(a->dataType()) && DataTypeUtils::isR(b->dataType()), 0,
               "fp8_matmul: A and B must hold floating values, got %s and %s",
               DataTypeUtils::asString(a->dataType()).c_str(), DataTypeUtils::asString(b->dataType()).c_str());
  REQUIRE_TRUE(a->rankOf() == 2 && b->rankOf() == 2, 0, "fp8_matmul: A and B must be matrices, got ranks %i and %i",
               a->rankOf(), b->rankOf());
  const LongType m = a->sizeAt(transposeA ? 1 : 0);
  const LongType k = a->sizeAt(transposeA ? 0 : 1);
  const LongType kB = b->sizeAt(transposeB ? 1 : 0);
  const LongType n = b->sizeAt(transposeB ? 0 : 1);
  REQUIRE_TRUE(k == kB, 0, "fp8_matmul: contraction dimensions differ, A has %lld and B has %lld", k, kB);
  REQUIRE_TRUE(scaleA->lengthOf() == 1 || scaleA->lengthOf() == m, 0,
               "fp8_matmul: scaleA must hold one scale or one per row (%lld), got %lld", m, scaleA->lengthOf());
  REQUIRE_TRUE(scaleB->lengthOf() == 1 || scaleB->lengthOf() == n, 0,
               "fp8_matmul: scaleB must hold one scale or one per column (%lld), got %lld", n, scaleB->lengthOf());
  REQUIRE_TRUE(bias == nullptr || bias->lengthOf() == n, 0,
               "fp8_matmul: bias must hold one value per column (%lld), got %lld", n,
               bias == nullptr ? 0 : bias->lengthOf());
  REQUIRE_TRUE(output->isSameShape({m, n}), 0, "fp8_matmul: output must be [%lld, %lld]", m, n);

  helpers::scaledGemm(block.launchContext(), a, b, scaleA, scaleB, bias, output, transposeA, transposeB);
  return Status::OK;
}

DECLARE_TYPES(fp8_matmul) {
  // A and B take any floating storage type, FP8 included (checked in the op).
  getOpDescriptor()
      ->setAllowedInputTypes(0, DataType::ANY)
      ->setAllowedInputTypes(1, DataType::ANY)
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setShapeValueInputs({})
      ->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(fp8_matmul) {
  auto a = inputShape->at(0);
  auto b = inputShape->at(1);
  REQUIRE_TRUE(shape::rank(a) == 2 && shape::rank(b) == 2, 0,
               "fp8_matmul: A and B must be matrices, got ranks %i and %i", shape::rank(a), shape::rank(b));
  const bool transposeA = INT_ARG_OR(0, 0) != 0;
  const bool transposeB = INT_ARG_OR(1, 0) != 0;
  const LongType m = shape::sizeAt(a, transposeA ? 1 : 0);
  const LongType n = shape::sizeAt(b, transposeB ? 0 : 1);
  // The output takes the requested type, else the type of the dequantization scales.
  const DataType dtype = block.numD() > 0 ? D_ARG(0) : ArrayOptions::dataType(inputShape->at(2));
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {m, n}));
}

}  // namespace ops
}  // namespace sd
#endif
