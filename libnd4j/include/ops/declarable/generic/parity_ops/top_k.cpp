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
//  @author sgazeos@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_top_k)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/top_k.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(top_k, 1, 2, false, 0, -1) {
  auto x = INPUT_VARIABLE(0);
  int k = 1;  // from params
  bool needSort = true;

  auto values = OUTPUT_VARIABLE(0);
  auto indices = OUTPUT_VARIABLE(1);

  if (block.numB() == 1) {
      needSort = B_ARG(0);
  }

  if (block.width() == 1) {
    if (block.numI() > 0) {
      k = INT_ARG(0);
    }
  } else {
    k = INPUT_VARIABLE(1)->e<int>(0);
  }

  REQUIRE_TRUE(k <= x->sizeAt(-1), 0, "top_k: k should not be greater than last dimension");
  REQUIRE_TRUE(k > 0, 0, "top_k: k should be positive, but %i given.", k);

  auto res = helpers::topKFunctor(block.launchContext(), x, values, indices, k, needSort);
  return res;
}

DECLARE_SHAPE_FN(top_k) {
  auto shapeList = SHAPELIST();
  auto in = inputShape->at(0);
  int shapeRank = shape::rank(in);
  int k = 1;  // default output shape is size 1

  if (block.width() == 2) {
    k = INPUT_VARIABLE(1)->e<int>(0);
  } else if (block.numI() > 0) {
    k = INT_ARG(0);
  }


  REQUIRE_TRUE(k > 0, 0, "top_k: k should be positive, but %i given.", k);

  for (int e = 0; e < 2; e++) {  // 2 element tuple at output
    LongType* aShape;
    ALLOCATE(aShape, block.getWorkspace(), shape::shapeInfoLength(shapeRank), sd::LongType);
    aShape[0] = shapeRank;
    for (LongType i = 1; i < shapeRank; ++i) aShape[i] = shape::sizeAt(in, i - 1);
    aShape[shapeRank] = k;

    shape::updateStrides(aShape, shape::order(in), false);
    auto desc = new ShapeDescriptor(aShape, (e == 0 ? ArrayOptions::dataType(in) : INT64), false);
    shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));

    RELEASE(aShape, block.getWorkspace());
  }
  return shapeList;
}

DECLARE_TYPES(top_k) {
  getOpDescriptor()
      ->setAllowedInputTypes(ANY)
      ->setAllowedOutputTypes(0, ANY)
      ->setAllowedOutputTypes(1, {ALL_INDICES});
}
}  // namespace ops
}  // namespace sd

#endif
