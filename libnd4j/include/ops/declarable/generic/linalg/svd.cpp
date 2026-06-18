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
//  @author Yurii Shyrma (iuriish@yahoo.com), created on 20.01.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_svd)

#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/svd.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(svd, 1, -1, false, 0, 3) {
  auto x = INPUT_VARIABLE(0);

  const int rank = x->rankOf();
  REQUIRE_TRUE(rank >= 2, 0, "SVD OP: the rank of input array must be >=2, but got %i instead!", rank);

  bool fullUV = (bool)INT_ARG(0);
  const bool calcUV = (bool)INT_ARG(1);

  if (calcUV == false) fullUV = false;

  const int switchNum = INT_ARG(2);

  helpers::svd(block.launchContext(), x,
               {OUTPUT_VARIABLE(0), calcUV ? OUTPUT_VARIABLE(1) : nullptr, calcUV ? OUTPUT_VARIABLE(2) : nullptr},
               fullUV, calcUV, switchNum);

  return Status::OK;
  ;
}

DECLARE_TYPES(svd) {
  getOpDescriptor()->setAllowedInputTypes(0, {FLOAT32, DOUBLE, HALF})->setSameMode(true);
}

DECLARE_SHAPE_FN(svd) {
  auto inShapeInfo = inputShape->at(0);
  bool fullUV = (bool)INT_ARG(0);
  bool calcUV = (bool)INT_ARG(1);

  const int rank = inShapeInfo[0];
  REQUIRE_TRUE(rank >= 2, 0, "SVD OP: the rank of input array must be >=2, but got %i instead!", rank);

  const int diagSize = inShapeInfo[rank] < inShapeInfo[rank - 1] ? inShapeInfo[rank] : inShapeInfo[rank - 1];
  const auto dtype = ArrayOptions::dataType(inShapeInfo);
  const auto order = shape::order(inShapeInfo);

  auto sShape = ShapeUtils::shapeAsVector(inShapeInfo);
  sShape.erase(sShape.end() - 2);
  sShape.back() = diagSize;
  auto sShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(dtype, order, sShape);

  if (calcUV) {
    auto uShape = ShapeUtils::shapeAsVector(inShapeInfo);
    auto vShape = ShapeUtils::shapeAsVector(inShapeInfo);

    if (fullUV) {
      uShape[rank - 1] = uShape[rank - 2];
      vShape[rank - 2] = vShape[rank - 1];
    } else {
      uShape[rank - 1] = diagSize;
      vShape[rank - 2] = vShape[rank - 1];
      vShape[rank - 1] = diagSize;
    }

    auto uShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(dtype, order, uShape);
    auto vShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(dtype, order, vShape);
    return SHAPELIST(sShapeInfo, uShapeInfo, vShapeInfo);
  }

  return SHAPELIST(sShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
