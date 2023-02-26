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
#if NOT_EXCLUDED(OP_dot_product_attention_v2)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>
#include <helpers/AttentionHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dot_product_attention_v2, -2, -1, false, 0, -2) {
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = block.width() > 2  ? INPUT_VARIABLE(2) : values;



  REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
               "dot_product_attention: Queries, Keys and Values must have same rank. "
               "But got queries = %s, keys = %s, values = %s", ShapeUtils::shapeAsString(queries).c_str(),
               ShapeUtils::shapeAsString(keys).c_str(), ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(queries->rankOf() == 3 || queries->rankOf() == 4, 0,
               "dot_product_attention: Queries, Keys and Values must be rank 3 arrays for single headed attention "
               "or rank 4 arrays for multi headed attention. But got rank = %i", queries->rankOf());

  REQUIRE_TRUE(queries->sizeAt(0) == keys->sizeAt(0) && keys->sizeAt(0) == values->sizeAt(0), 0,
               "dot_product_attention: Queries, Keys and Values must have the same mini batch size. "
               "But got queries = %i, keys = %i, values = %i", queries->sizeAt(0), keys->sizeAt(0), values->sizeAt(0));

  REQUIRE_TRUE(queries->sizeAt(-2) == keys->sizeAt(-2), 0,
               "dot_product_attention: Queries and Keys must have the same feature size. "
               "But got queries = %i, keys = %i", queries->sizeAt(-2), keys->sizeAt(-2));

  REQUIRE_TRUE(keys->sizeAt(-1) == values->sizeAt(-1), 0,
               "dot_product_attention: Keys and Values must have the same timestep length. "
               "But got keys = %i, values = %i", keys->sizeAt(-1), values->sizeAt(-1));

  auto qMask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  auto vMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

  auto dropout = block.numT() > 0 ? T_ARG(0) : 0.0;

  auto scale = block.numT() > 1 ? T_ARG(1) : 1.0;

  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto returnAttentionScores = block.numB() > 1 ? B_ARG(1) : false;
  //permute the inputs before processing. This is to allow the old shapes of batch size x dim x Tv


  int attentionType = block.numI() > 0 ? I_ARG(0) : ATTENTION_TYPE_DOT_PRODUCT;


  std::vector<sd::NDArray*> inputs = {queries,values,keys};
  std::vector<sd::NDArray *> masks2 = {qMask,vMask};

  auto outputScores = OUTPUT_VARIABLE(0);
  outputScores->printShapeInfo("Output scores shape");
  NDArray calculatedScores('c',{queries->sizeAt(0),queries->sizeAt(1),values->sizeAt(1)},queries->dataType());
  sd_printf("Doing attention\n",0);

   int batchSize = queries->sizeAt(0);
   int tq = queries->sizeAt(1);
   int tv = values->sizeAt(1);
   //attention scores will be computed anyways, pass in a created array regardless
   auto attentionScores = returnAttentionScores ? OUTPUT_VARIABLE(1) : new NDArray(queries->dataType(),{batchSize,tq,tv});

   sd_printf("Return Attention scores %d\n",returnAttentionScores);


  AttentionHelper::doAttention(inputs,
                               masks2,
                               false,
                               returnAttentionScores,
                               useCausalMask,
                               dropout,
                               ATTENTION_SCORE_MODE_DOT,
                               attentionType,
                               scale,
                               &calculatedScores,
                                attentionScores);


  sd_printf("Done with attention\n",0);
  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention_v2) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_v2) {
  auto firstInputType = INPUT_VARIABLE(0)->dataType();
  auto query_shape = inputShape->at(0);
  auto values_shape = inputShape->at(1);
  auto keys_shape = block.inputs()->size() > 2 ? inputShape->at(2) : values_shape;



  int attentionType = block.numI() > 0 ? I_ARG(0) : ATTENTION_SCORE_MODE_DOT;

  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto returnAttentionScores = block.numB() > 1 ? B_ARG(1) : false;

  if(attentionType == ATTENTION_SCORE_MODE_DOT) {
    //inputs: batchSize,Tq,dim batchSize,Tq,Tv
    //outputs: batchSize,Tq,Tv
    auto qShape = shape::shapeOf(query_shape);
    auto keyShape = shape::shapeOf(keys_shape);
    auto valueShape = shape::shapeOf(values_shape);
    ShapeDescriptor *descriptor = new ShapeDescriptor(firstInputType,'c',{qShape[0],qShape[1],valueShape[2]});
    auto constOutputScores = ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
    if(returnAttentionScores) {
      sd_printf("Returning scores\n",0);
      ShapeDescriptor *scoresShape = new ShapeDescriptor(firstInputType,'c',{qShape[0],values_shape[1],valueShape[1]});
      auto attentionScoresShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(scoresShape)->primary();
      delete descriptor;
      delete scoresShape;
      return SHAPELIST(constOutputScores,attentionScoresShape);
    }

    delete descriptor;
    return SHAPELIST(constOutputScores);


  } else if(attentionType == ATTENTION_SCORE_MODE_CONCAT) {
    //inputs: batchSize,Tq,dim batchSize,Tq,Tv
    //outputs: batchSize,Tq,Tv
    auto qShape = shape::shapeOf(query_shape);
    auto keyShape = shape::shapeOf(keys_shape);
    auto valueShape = shape::shapeOf(values_shape);
    ShapeDescriptor *descriptor = new ShapeDescriptor(firstInputType,'c',{qShape[0],qShape[1],valueShape[2]});
    auto constOutputScores = ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
    if(returnAttentionScores) {
      ShapeDescriptor *scoresShape = new ShapeDescriptor(firstInputType,'c',{qShape[0],values_shape[1],valueShape[2]});
      auto attentionScoresShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(scoresShape)->primary();
      delete descriptor;
      delete scoresShape;
      return SHAPELIST(constOutputScores,attentionScoresShape);
    }

    delete descriptor;
    return SHAPELIST(constOutputScores);

  } else {
    throw std::runtime_error("dot_product_attention: Calculate output shape: Invalid attention type.");
  }
}

CUSTOM_OP_IMPL(dot_product_attention_v2_bp, -2, 3, false, 0, -2) {
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = INPUT_VARIABLE(2);
  auto eps = INPUT_VARIABLE(3);
  auto qMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
  auto vMask = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;


  auto dLdq = OUTPUT_VARIABLE(0);
  auto dLdv = OUTPUT_VARIABLE(1);
  auto dLdk = OUTPUT_VARIABLE(2);


  REQUIRE_TRUE(queries->rankOf() == 3,0,"Input rank of queries must be 3.");
  REQUIRE_TRUE(values->rankOf() == 3,0,"Input rank of values must be 3.");
  REQUIRE_TRUE(values->rankOf() == 3,0,"Input rank of keys must be 3.");

  auto dropout = block.numT() > 0 ? T_ARG(0) : 0.0;

  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto returnAttentionScores = block.numB() > 1 ? B_ARG(1) : false;

  int attentionType = block.numI() > 0 ? I_ARG(0) : ATTENTION_TYPE_DOT_PRODUCT;


  eps->printShapeInfo("Eps  shape info");

  std::vector<sd::NDArray*> inputs = {queries,values,keys,eps};
  std::vector<sd::NDArray *> masks2 = {qMask,vMask};
  std::vector<sd::NDArray *> outputs = {dLdq,dLdv,dLdk};
  AttentionHelper::doAttentionBp(
      inputs,
      masks2,
      false,
      returnAttentionScores,
      useCausalMask,
      dropout,
      ATTENTION_SCORE_MODE_DOT,
      attentionType,
      1.0,
      outputs);


  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention_v2_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_v2_bp) {
  sd::LongType *dLdq_shape;
  COPY_SHAPE(inputShape->at(0), dLdq_shape);
  sd::LongType *dLdv_shape;
  COPY_SHAPE(inputShape->at(1), dLdv_shape);
  sd::LongType *dLdk_shape;
  COPY_SHAPE(inputShape->at(2), dLdk_shape);

  return SHAPELIST(CONSTANT(dLdq_shape), CONSTANT(dLdk_shape), CONSTANT(dLdv_shape));
}

}  // namespace ops
}  // namespace sd

#endif
