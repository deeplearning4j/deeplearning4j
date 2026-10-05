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
//  @author Adam Gibson
//

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_where_np)

#include <helpers/DenseOutputShape.h>
#include <ops/declarable/headers/boolean.h>
#include <ops/declarable/helpers/where.h>

namespace sd {
namespace ops {

static LongType whereNpTrueCount(NDArray& condition, LaunchContext* context) {
  if (condition.isEmpty()) return 0;
  NDArray count(DataType::INT64, context);
  condition.reduceNumber(reduce::CountNonZero, &count);
  return count.e<LongType>(0);
}

// Validate in shape inference as well: execution may short-circuit empty inputs after allocation.
static void validateWhereNpInputs(graph::Context& block) {
  REQUIRE_TRUE(block.width() == 1 || block.width() == 3, 0,
               "where_np: expected 1 or 3 inputs, got %i", block.width());
  auto condition = INPUT_VARIABLE(0);
  REQUIRE_TRUE(condition->dataType() == BOOL, 0, "where_np: condition must be BOOL");
  if (block.width() == 3) {
    auto x = INPUT_VARIABLE(1);
    auto y = INPUT_VARIABLE(2);
    if (condition->isSameShape(x)) {
      // This is replacement/gather, not an elementwise selection from y at the mask index.
      if (!y->isScalar() && y->lengthOf() < condition->lengthOf()) {
        const LongType matches = whereNpTrueCount(*condition, block.launchContext());
        REQUIRE_TRUE(y->lengthOf() >= matches, 0,
                     "where_np: replacement length %lld is smaller than true count %lld",
                     static_cast<long long>(y->lengthOf()), static_cast<long long>(matches));
      }
    } else {
      REQUIRE_TRUE(x->rankOf() > 0 && condition->lengthOf() == x->sizeAt(0), 0,
                   "where_np: row mask length must equal x dim0");
      REQUIRE_TRUE(x->isSameShape(y), 0, "where_np: row-mask x and y must have equal shapes");
    }
  }
}

CUSTOM_OP_IMPL(where_np, -1, 1, false, 0, 0) {
  validateWhereNpInputs(block);
  auto condition = INPUT_VARIABLE(0);
  if (block.width() == 3) {
    auto x = INPUT_VARIABLE(1);
    auto y = INPUT_VARIABLE(2);
    auto z = OUTPUT_VARIABLE(0);
    REQUIRE_TRUE(z->dataType() == x->dataType() && z->isSameShape(x), 0,
                 "where_np: output must have x's shape and dtype");
    if (z->isEmpty()) return Status::OK;
    if (condition->isSameShape(x)) {
      if (y->isScalar())
        helpers::_whereNpScalarBroadcast(block.launchContext(), *condition, *x, *y, *z);
      else
        helpers::_whereNpGather(block.launchContext(), *condition, *x, *y, *z);
    } else {
      // Preserve the row/TAD contract: true selects x, false selects y.
      helpers::_whereNpRows(block.launchContext(), *condition, *x, *y, *z);
    }
  } else {
    std::vector<NDArray*> outputs;
    for (LongType axis = 0; axis < condition->rankOf(); ++axis) {
      auto output = OUTPUT_VARIABLE(axis);
      REQUIRE_TRUE(output->dataType() == INT64, 0, "where_np: coordinate outputs must be INT64");
      if (output->isEmpty()) return Status::OK;
      outputs.push_back(output);
    }
    helpers::_whereNpCoordinates(block.launchContext(), *condition, outputs);
  }
  return Status::OK;
}

DECLARE_SHAPE_FN(where_np) {
  validateWhereNpInputs(block);
  if (block.width() == 3) return SHAPELIST(denseOutputShapeInfo(inputShape->at(1)));

  auto shapes = SHAPELIST();
  auto condition = INPUT_VARIABLE(0);
  const LongType numOfTrue = whereNpTrueCount(*condition, block.launchContext());
  if (numOfTrue) {
    for (LongType axis = 0; axis < condition->rankOf(); ++axis)
      shapes->push_back(ConstantShapeHelper::getInstance().vectorShapeInfo(numOfTrue, INT64));
  } else {
    // Preserve the existing single-empty-result contract when no coordinates exist.
    shapes->push_back(ConstantShapeHelper::getInstance().emptyShapeInfo(INT64));
  }
  return shapes;
}

samediff::EmptyHandling SD_BACKEND_OPS_CLASS(where_np)::emptyHandling() {
  // An empty replacement with an all-false mask still has to copy x to the output.
  return samediff::EmptyHandling::EMPTY_EXECUTE;
}

DECLARE_TYPES(where_np) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, BOOL)
      ->setAllowedInputTypes(1, {ALL_FLOATS, ALL_INTS, BOOL})
      ->setAllowedInputTypes(2, {ALL_FLOATS, ALL_INTS, BOOL})
      ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS, BOOL})
      ->setSameMode(false);
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_DYNAMIC_OUTPUT_SIZE | OP_TRAIT_FULLY_WRITING);
}
}  // namespace ops
}  // namespace sd

#endif
