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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 07.06.2018
//
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_mirror_pad)

#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(mirror_pad, 2, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto paddings = INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  const int mode = INT_ARG(0);  // 0 - REFLECT, else - SYMMETRIC
  const int includeBorder = mode ? 0 : 1;
  helpers::mirrorPad(block.launchContext(), *input, *paddings, *output, mode);

  return sd::Status::OK;
}

DECLARE_TYPES(mirror_pad) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
  getOpDescriptor()->setAllowedInputTypes(1, {DataType::INT32, DataType::INT64});  // to conform with TF
  getOpDescriptor()->setAllowedOutputTypes(0, {ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_VALUE_DEPENDENT_SHAPE);
}

DECLARE_SHAPE_FN(mirror_pad) {
  auto inShapeInfo = inputShape->at(0);
  auto paddings = INPUT_VARIABLE(1);

  const int includeBorder = static_cast<bool>(INT_ARG(0)) ? 0 : 1;

  if (shape::isScalar(inShapeInfo)) {
    sd::LongType len = 1 + paddings->e<sd::LongType>(0) + paddings->e<sd::LongType>(1);
    return SHAPELIST(ConstantShapeHelper::getInstance().vectorShapeInfo(len, ArrayOptions::dataType(inShapeInfo)));
  }

  sd::LongType* outShapeInfo(nullptr);
  int rank = shape::rank(inShapeInfo);

  ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);
  outShapeInfo[0] = rank;
  if(shape::rank(inputShape->at(1)) == 1) {
    for (int i = 0; i < rank; ++i) {
      outShapeInfo[i + 1] = shape::sizeAt(inShapeInfo, static_cast<sd::LongType>(i)) + paddings->e<sd::LongType>(0) + paddings->e<sd::LongType>(1);

    }
  } else {
    for (int i = 0; i < rank; ++i) {
      outShapeInfo[i + 1] = shape::sizeAt(inShapeInfo, static_cast<sd::LongType>(i)) + paddings->e<sd::LongType>(i, 0) + paddings->e<sd::LongType>(i, 1);

    }
  }

  ShapeUtils::updateStridesAndType(outShapeInfo, inShapeInfo, shape::order(inShapeInfo));

  return SHAPELIST(CONSTANT(outShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
