/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Paul Dubs
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_additive_attention)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(additive_attention, 3, -1, false, 0, 2) {
  auto queries = INPUT_VARIABLE(0);
  auto keys = INPUT_VARIABLE(1);
  auto values = INPUT_VARIABLE(2);
  auto mask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

  auto scale = block.numT() > 0 ? T_ARG(0) : 1.0;

  auto output = OUTPUT_VARIABLE(0);
  NDArray *weights;
  bool outputWeights = INT_ARG(1);
  if (outputWeights) {
    weights = OUTPUT_VARIABLE(1);
  } else {
    auto weightShape = ShapeUtils::evalShapeForMatmul(keys->shapeInfo(), queries->shapeInfo(), true, false);
    weights = new NDArray('c', weightShape, values->dataType(), block.launchContext());
  }

  int normalization = INT_ARG(0);

  REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
               "dot_product_attention: Queries, Keys and Values must have same rank. "
               "But got queries = %s, keys = %s, values = %s",
               ShapeUtils::shapeAsString(queries).c_str(), ShapeUtils::shapeAsString(keys).c_str(),
               ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(queries->rankOf() == 3 || queries->rankOf() == 4, 0,
               "dot_product_attention: Queries, Keys and Values must be rank 3 arrays for single headed attention "
               "or rank 4 arrays for multi headed attention. But got rank = %i",
               queries->rankOf());

  REQUIRE_TRUE(queries->sizeAt(0) == keys->sizeAt(0) && keys->sizeAt(0) == values->sizeAt(0), 0,
               "dot_product_attention: Queries, Keys and Values must have the same mini batch size. "
               "But got queries = %i, keys = %i, values = %i",
               queries->sizeAt(0), keys->sizeAt(0), values->sizeAt(0));

  REQUIRE_TRUE(queries->sizeAt(-2) == keys->sizeAt(-2), 0,
               "dot_product_attention: Queries and Keys must have the same feature size. "
               "But got queries = %i, keys = %i",
               queries->sizeAt(-2), keys->sizeAt(-2));

  REQUIRE_TRUE(keys->sizeAt(-1) == values->sizeAt(-1), 0,
               "dot_product_attention: Keys and Values must have the same timestep length. "
               "But got keys = %i, values = %i",
               keys->sizeAt(-1), values->sizeAt(-1));


  //Based on keras implementation of additive attention
  sd::ops::expand_dims expandDims;
  auto keysReshaped = expandDims.evaluate({keys}, {}, {-2}).at(0);
  auto queryReshaped = expandDims.evaluate({queries},{-3}).at(0);

  sd::ops::tanh tanh;
  auto rawScores = (*queryReshaped + *keysReshaped);
  auto tanhOut = tanh.evaluate({&rawScores});
  auto scores = scale * *tanhOut.at(0);

  sd::ops::reduce_sum sum;
  auto reducedSum = sum.evaluate({&scores});

  if (mask != nullptr) {
    NDArray reshapedMask;
    if (weights->rankOf() == 4) {
      reshapedMask = mask->reshape(mask->ordering(), {mask->sizeAt(0), 1, mask->sizeAt(1), 1});
    } else {
      reshapedMask = mask->reshape(mask->ordering(), {mask->sizeAt(0), mask->sizeAt(1), 1});
    }
  }
 //TODO: add proper masks implementation that handles causal masks
  //TODO: add separate masks for q and v
  //TODO: add dropout

  return sd::Status::OK;
}

DECLARE_TYPES(additive_attention) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(additive_attention) {
  auto query_shape = inputShape->at(0);
  auto keys_shape = inputShape->at(1);
  auto values_shape = inputShape->at(2);

  auto weights_shape = ConstantShapeHelper::getInstance().createShapeInfo(
      sd::ArrayOptions::dataType(values_shape), 'c',
      ShapeUtils::evalShapeForMatmul(keys_shape, query_shape, true, false));
  auto output_shape = ConstantShapeHelper::getInstance().createShapeInfo(
      sd::ArrayOptions::dataType(values_shape), 'c',
      ShapeUtils::evalShapeForMatmul(values_shape, weights_shape, false, false));

  if (INT_ARG(1)) {
    return SHAPELIST(output_shape, weights_shape);
  } else {
    return SHAPELIST(output_shape);
  }
}

CUSTOM_OP_IMPL(additive_attention_bp, 4, 3, false, 0, 1) {
  auto queries = INPUT_VARIABLE(0);
  auto keys = INPUT_VARIABLE(1);
  auto values = INPUT_VARIABLE(2);
  auto eps = INPUT_VARIABLE(3);
  auto mask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

  auto dLdq = OUTPUT_VARIABLE(0);
  auto dLdk = OUTPUT_VARIABLE(1);
  auto dLdv = OUTPUT_VARIABLE(2);

  int normalization = INT_ARG(0);

  REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
               "dot_product_attention: Queries, Keys and Values must have same rank. "
               "But got queries = %s, keys = %s, values = %s",
               ShapeUtils::shapeAsString(queries).c_str(), ShapeUtils::shapeAsString(keys).c_str(),
               ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(queries->rankOf() == 3 || queries->rankOf() == 4, 0,
               "dot_product_attention: Queries, Keys and Values must be rank 3 arrays for single headed attention "
               "or rank 4 arrays for multi headed attention. But got rank = %i",
               queries->rankOf());

  REQUIRE_TRUE(queries->sizeAt(0) == keys->sizeAt(0) && keys->sizeAt(0) == values->sizeAt(0), 0,
               "dot_product_attention: Queries, Keys and Values must have the same mini batch size. "
               "But got queries = %i, keys = %i, values = %i",
               queries->sizeAt(0), keys->sizeAt(0), values->sizeAt(0));

  REQUIRE_TRUE(queries->sizeAt(-2) == keys->sizeAt(-2), 0,
               "dot_product_attention: Queries and Keys must have the same feature size. "
               "But got queries = %i, keys = %i",
               queries->sizeAt(-2), keys->sizeAt(-2));

  REQUIRE_TRUE(keys->sizeAt(-1) == values->sizeAt(-1), 0,
               "dot_product_attention: Keys and Values must have the same timestep length. "
               "But got keys = %i, values = %i",
               keys->sizeAt(-1), values->sizeAt(-1));

  double factor;
  if (normalization) factor = sqrt((double)keys->sizeAt(-2));

  auto weightShape = ShapeUtils::evalShapeForMatmul(keys->shapeInfo(), queries->shapeInfo(), true, false);

  sd::ops::matmul mmul;
  NDArray preSoftmax('c', weightShape, values->dataType(), block.launchContext());
  mmul.execute({keys, queries}, {&preSoftmax}, {}, {1}, {});

  if (normalization) preSoftmax /= factor;

  if (mask != nullptr) {
    NDArray reshapedMask;
    if (preSoftmax.rankOf() == 4) {
      reshapedMask = mask->reshape(mask->ordering(), {mask->sizeAt(0), 1, mask->sizeAt(1), 1});
    } else {
      reshapedMask = mask->reshape(mask->ordering(), {mask->sizeAt(0), mask->sizeAt(1), 1});
    }
    preSoftmax += (reshapedMask - 1) * 1e9;
  }

  NDArray weights('c', weightShape, values->dataType(), block.launchContext());
  sd::ops::softmax softmax;
  softmax.execute({&preSoftmax}, {&weights}, {}, {-2}, {});

  sd::ops::matmul_bp mmul_bp;
  NDArray dLdw(weights.shapeInfo(), block.workspace());
  mmul_bp.execute({values, &weights, eps}, std::vector<NDArray *>{dLdv, &dLdw}, {}, {}, {});

  NDArray dLds(preSoftmax.shapeInfo(), block.workspace());
  sd::ops::softmax_bp softmax_bp;
  softmax_bp.execute({&preSoftmax, &dLdw}, {&dLds}, {}, {-2}, {});

  if (normalization) dLds /= factor;

  mmul_bp.execute({keys, queries, &dLds}, std::vector<NDArray *>{dLdk, dLdq}, {}, {1}, {});

  return sd::Status::OK;
}

DECLARE_TYPES(additive_attention_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(additive_attention_bp) {
  sd::LongType *dLdq_shape;
  COPY_SHAPE(inputShape->at(0), dLdq_shape);
  sd::LongType *dLdk_shape;
  COPY_SHAPE(inputShape->at(1), dLdk_shape);
  sd::LongType *dLdv_shape;
  COPY_SHAPE(inputShape->at(2), dLdv_shape);

  return SHAPELIST(CONSTANT(dLdq_shape), CONSTANT(dLdk_shape), CONSTANT(dLdv_shape));
}

}  // namespace ops
}  // namespace sd

#endif
