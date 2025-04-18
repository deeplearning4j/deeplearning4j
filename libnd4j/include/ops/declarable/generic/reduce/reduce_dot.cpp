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
// Created by george@skymind.io on 6/1/2018.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

#if NOT_EXCLUDED(OP_reduce_dot_bp)

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_dot_bp, -1, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto gradO = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  // L(x,y) = SUM(x_i * y_i)
  // dL/dx_i = y_i

  REQUIRE_TRUE(x->isSameShape(y), 0,
               "REDUCE_DOT_BP OP: both input arrays x and y should have same shapes, but got %s and %s correspondingly",
               ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());

  if (gradO->lengthOf() == 1) {  // scalar of reduced to scalar with keep dimensions
    NDArray assign1 = (*y) * (*gradO);
    gradX->assign(&assign1);
    NDArray assign2 = (*x) * (*gradO);
    gradY->assign(&assign2);
  } else {
    bool keepDims = false;
    auto dimensions = *block.getIArguments();

    if (block.width() > 3) {
      auto axesVector = INPUT_VARIABLE(3);
      helpers::adjustAxis(x->rankOf(), axesVector, dimensions);
    }

    if (block.getBArguments()->size())
      keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
      keepDims = (bool)T_ARG(0);

    REQUIRE_TRUE(
        dimensions.size() <= static_cast<size_t>(x->rankOf()), 0,
        "REDUCE_DOT_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
        dimensions.size());

    for (const auto& item : dimensions)
      REQUIRE_TRUE(
          item >= -x->rankOf() && item < x->rankOf(), 0,
          "REDUCE_DOT_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
          x->rankOf(), x->rankOf(), item);

    if (!keepDims) {
      auto gradOShapeKeepDims =
          ShapeUtils::evalReduceShapeInfo(gradO->ordering(), &dimensions, *x, true, false, block.getWorkspace());
      std::vector<sd::LongType> shape =  ShapeUtils::pullShapeFromShapeInfo(
          gradOShapeKeepDims);
      auto r = gradO->reshape(gradO->ordering(),
                              shape);  // for example could be something like [a,b] -> [1,a,1,b]

      // First case - for gradX
      NDArray gradXTemp1 = (*y) * r;
      gradX->assign(&gradXTemp1);

      // First case - for gradY
      NDArray gradYTemp1 = (*x) * r;
      gradY->assign(&gradYTemp1);

    } else {
      // Second case - for gradX
      NDArray gradXTemp2 = (*y) * (*gradO);
      gradX->assign(&gradXTemp2);

      // Second case - for gradY
      NDArray gradYTemp2 = (*x) * (*gradO);
      gradY->assign(&gradYTemp2);
    }
  }
  return sd::Status::OK;
}

DECLARE_SHAPE_FN(reduce_dot_bp) {
  if (shape::length(inputShape->at(2)) > 1) {
    auto dimensions = *block.getIArguments();

    if (block.width() > 3) {
      auto axesVector = INPUT_VARIABLE(3);
      helpers::adjustAxis(INPUT_VARIABLE(0)->rankOf(), axesVector, dimensions);
    }


    REQUIRE_TRUE(
        dimensions.size() <= static_cast<size_t>(inputShape->at(0)[0]), 0,
        "REDUCE_DOT_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
        dimensions.size());

    for (const auto& item : dimensions)
      REQUIRE_TRUE(
          item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0,
          "REDUCE_DOT_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
          inputShape->at(0)[0], inputShape->at(0)[0], item);
  }

  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)));
}

DECLARE_TYPES(reduce_dot_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
