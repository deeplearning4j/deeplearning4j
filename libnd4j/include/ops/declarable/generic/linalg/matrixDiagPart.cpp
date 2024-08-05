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
// Created to use with batched tensor by GS <sgazeos@gmail.com> 3/21/2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matrix_diag_part.h>

#if NOT_EXCLUDED(OP_matrix_diag_part)

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(matrix_diag_part, 1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  const int inRank = input->rankOf();

  REQUIRE_TRUE(inRank >= 2, 0, "CUSTOM_OP matrix_diag_part: input array must have rank >= 2, but %i given!", inRank);

  output->nullify();
  return helpers::matrixDiagPart(block.launchContext(), input, output);
}

DECLARE_SHAPE_FN(matrix_diag_part) {
  LongType const* outShapeInfo = nullptr;
  auto in = inputShape->at(0);
  LongType inRank = shape::rank(in);

  REQUIRE_TRUE(inRank >= 2, 0, "CUSTOM_OP matrix_diag_part: input array must have rank >= 2, but %i given!", inRank);

  LongType outRank = inRank - 1;
  LongType lastDimension = sd::math::sd_min<LongType>(shape::sizeAt(in, static_cast<LongType>(-1)), shape::sizeAt(in, static_cast<LongType>(-2)));
  if (outRank == 1) {
    // output shape is a vector with size min(sizeAt(0), sizeAt(1))
    outShapeInfo = ConstantShapeHelper::getInstance().vectorShapeInfo(lastDimension, ArrayOptions::dataType(in));
  } else {
    LongType* anShapeInfo;
    ALLOCATE(anShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), sd::LongType);
    anShapeInfo[0] = outRank;
    for (LongType i = 0; i < outRank - 1; ++i) anShapeInfo[i + 1] = shape::sizeAt(in, i);
    anShapeInfo[outRank] = lastDimension;

    ShapeUtils::updateStridesAndType(anShapeInfo, in, shape::order(in));
    outShapeInfo = CONSTANT(anShapeInfo);
  }
  return SHAPELIST(outShapeInfo);
}

DECLARE_TYPES(matrix_diag_part) { getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true); }
}  // namespace ops
}  // namespace sd
#endif
