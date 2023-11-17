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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_Where)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/where.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(Where, 1, 1, false, 0, 0) {
  auto condition = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);
  if (z->isEmpty()) return Status::OK;

  if (block.width() == 3) {
    auto x = INPUT_VARIABLE(1);
    auto y = INPUT_VARIABLE(2);

    REQUIRE_TRUE(x->isSameShape(y), 0, "X and Y must have equal shapes");

    // if cond matches x/y shape - we have per-element mask
    if (condition->isSameShape(x)) {
      // FIXME: for perf it might be better to issue memcpy here, and fill only mismatched values from either X or Y
      for (int e = 0; e < condition->lengthOf(); e++) {
        if (y->isR()) {
          auto r = !condition->e<bool>(e) ? y->e<double>(e) : x->e<double>(e);
          z->p(e, r);
        } else {
          auto r = !condition->e<bool>(e) ? y->e<LongType>(e) : x->e<LongType>(e);
          z->p(e, r);
        }
      }
    } else {
      REQUIRE_TRUE(condition->lengthOf() == x->sizeAt(0), 0,
                   "Condition length should be equal to the dim0 of x/y to act as TAD-mask, but got %d instead",
                   condition->lengthOf());

      std::vector<LongType> zero({0});
      auto dims = ShapeUtils::evalDimsToExclude(x->rankOf(), 1,zero.data());
      auto tadsX = x->allTensorsAlongDimension(*dims);
      auto tadsY = y->allTensorsAlongDimension(*dims);
      auto tadsZ = z->allTensorsAlongDimension(*dims);

      for (int e = 0; e < tadsX.size(); e++) {
        if (!condition->e<bool>(e)) {
          tadsZ.at(e)->assign(tadsY.at(e));
        } else {
          tadsZ.at(e)->assign(tadsX.at(e));
        }
      }

      delete dims;
    }
  } else {
    printf("where: second case\n");
    // in this case we return 2D matrix, which basically contains coordinates fo true
    REQUIRE_TRUE(block.width() == 1, 0, "Where op takes either 1 or 3 operands, But got %d operands instead",
                 block.width());
    auto output = OUTPUT_VARIABLE(0);
    std::vector<LongType> zero({0});

    int width = condition->rankOf();
    if (z->isEmpty()) return Status::OK;


    std::vector<LongType> *dims = ShapeUtils::evalDimsToExclude(width,1,zero.data());

    helpers::_where(block.launchContext(), *condition, *output, block.workspace());
    delete dims;
  }
  return Status::OK;
}

DECLARE_SHAPE_FN(Where) {
  if (block.width() == 3) {
    auto inShape = inputShape->at(1);
    LongType* newshape;
    COPY_SHAPE(inShape, newshape);

    return SHAPELIST(CONSTANT(newshape));
  } else {
    // FIXME: we can't estimate result here in this case
    // output shape is the 2D tensor num_true x rankOf (inShape)
    auto condition = INPUT_VARIABLE(0);
    auto inShape = inputShape->at(0);
    LongType numOfTrue = 0;  // condition->reduceNumber(reduce::CountNonZero, nullptr).e<sd::LongType>(0);
    for (LongType i = 0; i < condition->lengthOf(); i++)
      if (condition->e<bool>(i)) numOfTrue++;

    LongType const* theNewShape;
    if (numOfTrue > 0) {
      LongType* newShape;
      ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), sd::LongType);
      printf("where: num true is %d\n",numOfTrue);
      newShape[0] = 2;
      newShape[1] = numOfTrue;
      newShape[2] = shape::rank(inShape);
      newShape[3] = 1;
      newShape[4] = 1;
      newShape[5] = 0;
      newShape[6] = 1;
      newShape[7] = 99;
      ShapeUtils::updateStridesAndType(newShape, INT64, 'c');

      theNewShape = CONSTANT(newShape);
    } else {
      theNewShape = ConstantShapeHelper::getInstance().emptyShapeInfo(INT64);
    }

    return SHAPELIST(theNewShape);
  }
}

DECLARE_TYPES(Where) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)  // bool
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedInputTypes(2, ANY)
      ->setAllowedOutputTypes(0, {ALL_INTS, ALL_FLOATS});
}
}  // namespace ops
}  // namespace sd

#endif
