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

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(mirror_pad, 2, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto paddings = INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  const int mode = INT_ARG(0);  // 0 - REFLECT, else - SYMMETRIC
  const int includeBorder = mode ? 0 : 1;

    REQUIRE_TRUE(paddings->rankOf() == 2, 0,
                 "MIRROR_PAD OP: the rank of paddings array must be equal 2, but got %i instead !", paddings->rankOf());
    REQUIRE_TRUE(paddings->sizeAt(0) == input->rankOf(), 0,
                 "MIRROR_PAD OP: zero dimension of paddings array must be equal to input array rank, but got %i and %i "
                 "correspondingly !",
                 paddings->sizeAt(0), input->rankOf());

    for (int i = 0; i < input->rankOf(); ++i)
      REQUIRE_TRUE((paddings->e<sd::LongType>(i, 0) <= (input->sizeAt(i) - includeBorder)) &&
                       (paddings->e<sd::LongType>(i, 1) <= (input->sizeAt(i) - includeBorder)),
                   0,
                   "MIRROR_PAD OP: wrong content of paddings array, its elements must be no grater then corresponding "
                   "dimension of input array for symmetric mode (or dimension-1 for reflect mode) !");


  helpers::mirrorPad(block.launchContext(), *input, *paddings, *output, mode);

  return sd::Status::OK;
}

DECLARE_TYPES(mirror_pad) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
  getOpDescriptor()->setAllowedInputTypes(1, {DataType::INT32, DataType::INT64});  // to conform with TF
  getOpDescriptor()->setAllowedOutputTypes(0, {ALL_FLOATS});
}

DECLARE_SHAPE_FN(mirror_pad) {
  auto input = INPUT_VARIABLE(0);
  auto paddings = INPUT_VARIABLE(1);

  const int includeBorder = static_cast<bool>(INT_ARG(0)) ? 0 : 1;

    REQUIRE_TRUE(paddings->rankOf() == 2, 0,
                 "MIRROR_PAD OP: the rank of paddings array must be equal 2, but got %i instead !", paddings->rankOf());
    REQUIRE_TRUE(paddings->sizeAt(0) == input->rankOf(), 0,
                 "MIRROR_PAD OP: zero dimension of paddings array must be equal to input array rank, but got %i and %i "
                 "correspondingly !",
                 paddings->sizeAt(0), input->rankOf());
    for (int i = 0; i < input->rankOf(); ++i)
      REQUIRE_TRUE((paddings->e<sd::LongType>(i, 0) <= (input->sizeAt(i) - includeBorder)) &&
                       (paddings->e<sd::LongType>(i, 1) <= (input->sizeAt(i) - includeBorder)),
                   0,
                   "MIRROR_PAD OP: wrong content of paddings array, its elements must be no grater then corresponding "
                   "dimension of input array for symmetric mode (or dimension-1 for reflect mode) !");


  if (input->isScalar()) {
    sd::LongType len = input->isScalar() ? 1 + paddings->e<sd::LongType>(0)  + paddings->e<sd::LongType>(1) : input->lengthOf() + paddings->e<sd::LongType>(0) + paddings->e<sd::LongType>(1);
    return SHAPELIST(ConstantShapeHelper::getInstance().vectorShapeInfo(len, input->dataType()));
  }

  sd::LongType* outShapeInfo(nullptr);
  int rank = input->rankOf();

  ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);
  outShapeInfo[0] = rank;
  for (int i = 0; i < rank; ++i)
    outShapeInfo[i + 1] = input->sizeAt(i) + paddings->e<sd::LongType>(i, 0) + paddings->e<sd::LongType>(i, 1);
  ShapeUtils::updateStridesAndType(outShapeInfo, input->shapeInfo(), input->ordering());

  return SHAPELIST(CONSTANT(outShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
