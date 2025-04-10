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
#if NOT_EXCLUDED(OP_multi_head_dot_product_attention)

#include <helpers/AttentionHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(multi_head_dot_product_attention, 7, -1, false, 0, 2) {
  auto queries = INPUT_VARIABLE(0);  //[batch, nIn, timeSteps]
  auto keys = INPUT_VARIABLE(1);     //[batch, nIn, timeSteps]
  auto values = INPUT_VARIABLE(2);   //[batch, nIn, timeSteps]
  auto Wq = INPUT_VARIABLE(3);       //[nHeads, headSize, nIn]
  auto Wk = INPUT_VARIABLE(4);       //[nHeads, headSize, nIn]
  auto Wv = INPUT_VARIABLE(5);       //[nHeads, headSize, nIn]
  auto Wo = INPUT_VARIABLE(6);       //[nHeads * headSize, nOut]
  auto mask = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;

  auto output = OUTPUT_VARIABLE(0);
  int normalization = INT_ARG(0);
  int weights = INT_ARG(1);

  auto numHeads = Wk->sizeAt(0);
  auto miniBatchSize = queries->sizeAt(0);
  auto queryCount = queries->sizeAt(2);
  auto projectedValuesSize = Wv->sizeAt(1);
  auto outSize = Wo->sizeAt(1);

  REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
               "multi_head_dot_product_attention: Queries, Keys and Values must have same rank. "
               "But got queries = %s, keys = %s, values = %s",
               ShapeUtils::shapeAsString(queries).c_str(), ShapeUtils::shapeAsString(keys).c_str(),
               ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(queries->rankOf() == 3, 0,
               "multi_head_dot_product_attention: Queries, Keys and Values must be rank 3 arrays"
               "But got rank = %i",
               queries->rankOf());

  REQUIRE_TRUE(Wq->rankOf() == Wk->rankOf() && Wk->rankOf() == Wv->rankOf(), 0,
               "multi_head_dot_product_attention: Input projections weights must have the same rank. "
               "But got Wq = %s, Wk = %s, Wv = %s",
               ShapeUtils::shapeAsString(Wq).c_str(), ShapeUtils::shapeAsString(Wk).c_str(),
               ShapeUtils::shapeAsString(Wv).c_str());

  REQUIRE_TRUE(Wq->sizeAt(0) == Wk->sizeAt(0) && Wk->sizeAt(0) == Wv->sizeAt(0), 0,
               "multi_head_dot_product_attention: Projections weights must have the same number of attention heads. "
               "But got Wq = %s, Wk = %s, Wv = %s",
               ShapeUtils::shapeAsString(Wq).c_str(), ShapeUtils::shapeAsString(Wk).c_str(),
               ShapeUtils::shapeAsString(Wv).c_str());

  REQUIRE_TRUE(Wo->rankOf() == 2, 0,
               "multi_head_dot_product_attention: Output projection weights must have rank 2. "
               "But got Wo = %s",
               ShapeUtils::shapeAsString(Wo).c_str());

  REQUIRE_TRUE(Wq->sizeAt(2) == queries->sizeAt(1), 0,
               "multi_head_dot_product_attention: Query projection matrix Wq has incompatible size to queries matrix."
               "Expected Wq[2] = queries[1] = %i, but got Wq = %s, queries = %s ",
               queries->sizeAt(1), ShapeUtils::shapeAsString(Wq).c_str(), ShapeUtils::shapeAsString(queries).c_str());

  REQUIRE_TRUE(Wk->sizeAt(2) == keys->sizeAt(1), 0,
               "multi_head_dot_product_attention: Key projection matrix Wk has incompatible size to keys matrix."
               "Expected Wk[2] = keys[1] = %i, but got Wk = %s, keys = %s ",
               keys->sizeAt(1), ShapeUtils::shapeAsString(Wk).c_str(), ShapeUtils::shapeAsString(keys).c_str());

  REQUIRE_TRUE(Wv->sizeAt(2) == values->sizeAt(1), 0,
               "multi_head_dot_product_attention: Value projection matrix Wv has incompatible size to values matrix."
               "Expected Wv[2] = values[1] = %i, but got Wv = %s, values = %s ",
               values->sizeAt(1), ShapeUtils::shapeAsString(Wv).c_str(), ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(
      Wo->sizeAt(0) == (Wv->sizeAt(1) * Wv->sizeAt(0)), 0,
      "multi_head_dot_product_attention: Output projection matrix Wo has incompatible size to attention result."
      "Expected Wo[0] = Wv[0] * Wv[1] = %i, but got Wo = %s, Wv = %",
      (Wv->sizeAt(1) * Wv->sizeAt(0)), ShapeUtils::shapeAsString(Wo).c_str(), ShapeUtils::shapeAsString(Wv).c_str());

  // Project queries, keys, values
  auto projectedQueries = AttentionHelper::multiHeadProject(
      queries, Wq, block.launchContext());  //[minibatch, numHeads, projectedSize, seqLength]
  auto projectedKeys = AttentionHelper::multiHeadProject(
      keys, Wk, block.launchContext());  //[minibatch, numHeads, projectedSize, seqLength]
  auto projectedValues = AttentionHelper::multiHeadProject(
      values, Wv, block.launchContext());  //[minibatch, numHeads, projectedSize, seqLength]

  std::vector<sd::LongType> shape = {projectedQueries.sizeAt(0), projectedValues.sizeAt(1), projectedValues.sizeAt(2), projectedQueries.sizeAt(3)};
  // Apply Attention
  // attnResults = [minibatch, numHeads, projectedSize, seqLenth
  NDArray attnResults(
      'c',
      shape,
      projectedValues.dataType(), block.launchContext());
  sd::ops::dot_product_attention attention;
  attention.execute({&projectedQueries, &projectedKeys, &projectedValues, mask},
                    {&attnResults, weights ? OUTPUT_VARIABLE(1) : nullptr}, {}, {normalization, weights}, {});

  // Project attention results
  attnResults.permutei({0, 3, 1, 2}, 0, false);
  attnResults.reshapei(attnResults.ordering(), {miniBatchSize * queryCount, numHeads * projectedValuesSize});

  sd::ops::matmul mmul;
  std::vector<sd::LongType> projShape ={attnResults.sizeAt(0), Wo->sizeAt(1)};
  NDArray projRes('c', projShape, values->dataType(), block.launchContext());
  mmul.execute({&attnResults, Wo}, {&projRes}, {}, {}, {});
  projRes.reshapei(projRes.ordering(), {miniBatchSize, queryCount, outSize});
  projRes.permutei({0, 2, 1}, 0, false);

  // FIXME: bad for performance
  output->assign(&projRes);

  return sd::Status::OK;
}

DECLARE_TYPES(multi_head_dot_product_attention) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(multi_head_dot_product_attention) {
  auto queryShape = inputShape->at(0);
  auto keysShape = inputShape->at(1);
  auto valuesShape = inputShape->at(2);
  auto WkShape = inputShape->at(3);
  auto WoShape = inputShape->at(6);

  auto batchSize = shape::sizeAt(queryShape, static_cast<sd::LongType>(0));
  auto outSize = shape::sizeAt(WoShape, static_cast<sd::LongType>(1));
  auto queryCount = shape::sizeAt(queryShape, static_cast<sd::LongType>(2));
  auto numHeads = shape::sizeAt(WkShape, static_cast<sd::LongType>(0));
  auto timeSteps = shape::sizeAt(keysShape, static_cast<sd::LongType>(2));

  auto weightsShape = ConstantShapeHelper::getInstance().createShapeInfo(sd::ArrayOptions::dataType(valuesShape), 'c',
                                                                         {batchSize, numHeads, timeSteps, queryCount});
  auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(sd::ArrayOptions::dataType(valuesShape), 'c',
                                                                        {batchSize, outSize, queryCount});

  if (INT_ARG(1)) {
    return SHAPELIST(outputShape, weightsShape);
  } else {
    return SHAPELIST(outputShape);
  }
}

CUSTOM_OP_IMPL(multi_head_dot_product_attention_bp, 8, 7, false, 0, 1) {
  auto queries = INPUT_VARIABLE(0);
  auto keys = INPUT_VARIABLE(1);
  auto values = INPUT_VARIABLE(2);
  auto Wq = INPUT_VARIABLE(3);
  auto Wk = INPUT_VARIABLE(4);
  auto Wv = INPUT_VARIABLE(5);
  auto Wo = INPUT_VARIABLE(6);
  auto eps = INPUT_VARIABLE(7);
  auto mask = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;

  auto dLdq = OUTPUT_VARIABLE(0);
  auto dLdk = OUTPUT_VARIABLE(1);
  auto dLdv = OUTPUT_VARIABLE(2);
  auto dLdWq = OUTPUT_VARIABLE(3);
  auto dLdWk = OUTPUT_VARIABLE(4);
  auto dLdWv = OUTPUT_VARIABLE(5);
  auto dLdWo = OUTPUT_VARIABLE(6);

  int normalization = INT_ARG(0);

  auto numHeads = Wk->sizeAt(0);
  auto miniBatchSize = queries->sizeAt(0);
  auto queryCount = queries->sizeAt(2);
  auto outSize = Wo->sizeAt(1);
  auto projectedValuesSize = Wv->sizeAt(1);

  REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
               "multi_head_dot_product_attention: Queries, Keys and Values must have same rank. "
               "But got queries = %s, keys = %s, values = %s",
               ShapeUtils::shapeAsString(queries).c_str(), ShapeUtils::shapeAsString(keys).c_str(),
               ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(queries->rankOf() == 3, 0,
               "multi_head_dot_product_attention: Queries, Keys and Values must be rank 3 arrays"
               "But got rank = %i",
               queries->rankOf());

  REQUIRE_TRUE(Wq->rankOf() == Wk->rankOf() && Wk->rankOf() == Wv->rankOf(), 0,
               "multi_head_dot_product_attention: Input projections weights must have the same rank. "
               "But got Wq = %s, Wk = %s, Wv = %s",
               ShapeUtils::shapeAsString(Wq).c_str(), ShapeUtils::shapeAsString(Wk).c_str(),
               ShapeUtils::shapeAsString(Wv).c_str());

  REQUIRE_TRUE(Wq->sizeAt(0) == Wk->sizeAt(0) && Wk->sizeAt(0) == Wv->sizeAt(0), 0,
               "multi_head_dot_product_attention: Projections weights must have the same number of attention heads. "
               "But got Wq = %s, Wk = %s, Wv = %s",
               ShapeUtils::shapeAsString(Wq).c_str(), ShapeUtils::shapeAsString(Wk).c_str(),
               ShapeUtils::shapeAsString(Wv).c_str());

  REQUIRE_TRUE(Wo->rankOf() == 2, 0,
               "multi_head_dot_product_attention: Output projection weights must have rank 2. "
               "But got Wo = %s",
               ShapeUtils::shapeAsString(Wo).c_str());

  REQUIRE_TRUE(Wq->sizeAt(2) == queries->sizeAt(1), 0,
               "multi_head_dot_product_attention: Query projection matrix Wq has incompatible size to queries matrix."
               "Expected Wq[2] = queries[1] = %i, but got Wq = %s, queries = %s ",
               queries->sizeAt(1), ShapeUtils::shapeAsString(Wq).c_str(), ShapeUtils::shapeAsString(queries).c_str());

  REQUIRE_TRUE(Wk->sizeAt(2) == keys->sizeAt(1), 0,
               "multi_head_dot_product_attention: Key projection matrix Wk has incompatible size to keys matrix."
               "Expected Wk[2] = keys[1] = %i, but got Wk = %s, keys = %s ",
               keys->sizeAt(1), ShapeUtils::shapeAsString(Wk).c_str(), ShapeUtils::shapeAsString(keys).c_str());

  REQUIRE_TRUE(Wv->sizeAt(2) == values->sizeAt(1), 0,
               "multi_head_dot_product_attention: Value projection matrix Wv has incompatible size to values matrix."
               "Expected Wv[2] = values[1] = %i, but got Wv = %s, values = %s ",
               values->sizeAt(1), ShapeUtils::shapeAsString(Wv).c_str(), ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(
      Wo->sizeAt(0) == (Wv->sizeAt(1) * Wv->sizeAt(0)), 0,
      "multi_head_dot_product_attention: Output projection matrix Wo has incompatible size to attention result."
      "Expected Wo[0] = Wv[0] * Wv[1] = %i, but got Wo = %s, Wv = %",
      (Wv->sizeAt(1) * Wv->sizeAt(0)), ShapeUtils::shapeAsString(Wo).c_str(), ShapeUtils::shapeAsString(Wv).c_str());

  // Project queries, keys, values
  auto projectedQueries = AttentionHelper::multiHeadProject(queries, Wq, block.launchContext());
  auto projectedKeys = AttentionHelper::multiHeadProject(keys, Wk, block.launchContext());
  auto projectedValues = AttentionHelper::multiHeadProject(values, Wv, block.launchContext());
  std::vector<sd::LongType> shape = {projectedQueries.sizeAt(0), projectedValues.sizeAt(1), projectedValues.sizeAt(2), projectedQueries.sizeAt(3)};
  // Apply Attention
  NDArray attnResults(
      'c',
      shape,
      projectedValues.dataType(), block.launchContext());
  sd::ops::dot_product_attention attention;
  attention.execute({&projectedQueries, &projectedKeys, &projectedValues, mask}, {&attnResults}, {}, {normalization, 0},
                    {});

  // Project attention results
  attnResults.permutei({0, 3, 1, 2}, 0, false);
  attnResults.reshapei(attnResults.ordering(), {miniBatchSize * queryCount, numHeads * projectedValuesSize});

  std::vector<sd::LongType> perm = {0,2,1};
  // dLdWo
  auto epsPerm = eps->permute(perm, false, false);
  std::vector<sd::LongType> epsShape =  {miniBatchSize * queryCount, outSize};
  auto epsPostReshape = epsPerm.reshape(eps->ordering(), epsShape);
  sd::ops::matmul_bp matmulBp;
  NDArray dLdPreWo(attnResults.shapeInfo(), false, block.launchContext());
  matmulBp.execute({&attnResults, Wo, &epsPostReshape}, std::vector<NDArray *>{&dLdPreWo, dLdWo}, {}, {}, {});

  // dLdAttn
  dLdPreWo.reshapei({miniBatchSize, queryCount, numHeads, projectedValues.sizeAt(2)});
  dLdPreWo.permutei({0, 2, 3, 1}, 0, false);

  sd::ops::dot_product_attention_bp attentionBp;
  NDArray dLdProjectedQueries(projectedQueries.shapeInfo(), false, block.launchContext());
  NDArray dLdProjectedKeys(projectedKeys.shapeInfo(), false, block.launchContext());
  NDArray dLdProjectedValues(projectedValues.shapeInfo(), false, block.launchContext());
  attentionBp.execute({&projectedQueries, &projectedKeys, &projectedValues, &dLdPreWo, mask},
                      {&dLdProjectedQueries, &dLdProjectedKeys, &dLdProjectedValues}, {}, {normalization}, {});

  AttentionHelper::multiHeadProjectBp(queries, Wq, &dLdProjectedQueries, dLdq, dLdWq, block.launchContext());
  AttentionHelper::multiHeadProjectBp(keys, Wk, &dLdProjectedKeys, dLdk, dLdWk, block.launchContext());
  AttentionHelper::multiHeadProjectBp(values, Wv, &dLdProjectedValues, dLdv, dLdWv, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(multi_head_dot_product_attention_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(multi_head_dot_product_attention_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)), CONSTANT(inputShape->at(2)), CONSTANT(inputShape->at(3)),
                   CONSTANT(inputShape->at(4)), CONSTANT(inputShape->at(5)), CONSTANT(inputShape->at(6)));
}

}  // namespace ops
}  // namespace sd

#endif
