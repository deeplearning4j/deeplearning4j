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
#if NOT_EXCLUDED(OP_dot_product_attention)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>
#include <helpers/AttentionHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dot_product_attention, -2, -1, false, 0, -2) {
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = block.width() > 2  ? INPUT_VARIABLE(2) : values;


  REQUIRE_TRUE(queries->rankOf() == 3,0,"Input rank of queries must be 3.");
  REQUIRE_TRUE(values->rankOf() == 3,0,"Input rank of values must be 3.");
  REQUIRE_TRUE(values->rankOf() == 3,0,"Input rank of keys must be 3.");

  auto qMask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  auto vMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

  auto dropout = block.numT() > 0 ? T_ARG(0) : 0.0;

  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto returnAttentionScores = block.numB() > 1 ? B_ARG(1) : false;
  //permute the inputs before processing. This is to allow the old shapes of batch size x dim x Tv
  auto permuteInputs = block.numB() > 2 ? B_ARG(2) : false;


  int attentionType = block.numI() > 0 ? I_ARG(0) : ATTENTION_TYPE_DOT_PRODUCT;

  auto inputKeys = permuteInputs ? new NDArray(keys->permute({0,2,1})) : keys;
  auto inputQueries = permuteInputs ? new NDArray(queries->permute({0,2,1})) : queries;


  auto qMaskInput = permuteInputs  && qMask != nullptr ? new NDArray(qMask->permute({0,2,1})) : qMask;
  auto vMaskInput = permuteInputs  && vMask != nullptr ? new NDArray(vMask->permute({0,2,1})) : vMask;

  std::vector<sd::NDArray*> inputs = {inputQueries,inputKeys,values};
  std::vector<sd::NDArray *> masks2 = {qMaskInput,vMaskInput};

  auto output2 = AttentionHelper::doAttention(inputs,
                                              masks2,
                                              false,
                                              returnAttentionScores,
                                              useCausalMask,
                                              dropout,
                                              ATTENTION_SCORE_MODE_DOT,
                                              attentionType,
                                              true);

  auto firstOutput = const_cast<sd::NDArray * const>(output2[0][0]);
  OUTPUT_VARIABLE(0)->assign(firstOutput);

  if(returnAttentionScores) {
    OUTPUT_VARIABLE(1)->assign(output2[0][1]);
  }

  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention) {
  auto firstInputType = INPUT_VARIABLE(0)->dataType();
  auto query_shape = inputShape->at(0);
  REQUIRE_TRUE(query_shape[0] == 3,0,"Query input must be rank 3.");
  auto values_shape = inputShape->at(1);
  REQUIRE_TRUE(values_shape[0] == 3,0,"Values input must be rank 3.");
  auto keys_shape = block.inputs()->size() > 2 ? inputShape->at(2) : values_shape;
  REQUIRE_TRUE(keys_shape[0] == 3,0,"Key input must be rank 3.");



  int attentionType = block.numI() > 0 ? I_ARG(0) : ATTENTION_TYPE_DOT_PRODUCT;

  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto returnAttentionScores = block.numB() > 1 ? B_ARG(1) : false;

  if(attentionType == ATTENTION_SCORE_MODE_DOT) {
    //inputs: batchSize,Tq,dim batchSize,Tq,Tv
    //outputs: batchSize,Tq,Tv
    auto qShape = shape::shapeOf(query_shape);
    auto keyShape = shape::shapeOf(keys_shape);
    auto valueShape = shape::shapeOf(values_shape);
    ShapeDescriptor *descriptor = new ShapeDescriptor(firstInputType,'c',{qShape[0],valueShape[1],valueShape[2]});
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


  } else if(attentionType == ATTENTION_SCORE_MODE_CONCAT) {
    //inputs: batchSize,Tq,dim batchSize,Tq,Tv
    //outputs: batchSize,Tq,Tv
    auto qShape = shape::shapeOf(query_shape);
    auto keyShape = shape::shapeOf(keys_shape);
    auto valueShape = shape::shapeOf(values_shape);
    ShapeDescriptor *descriptor = new ShapeDescriptor(firstInputType,'c',{qShape[0],valueShape[1],valueShape[2]});
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

CUSTOM_OP_IMPL(dot_product_attention_bp, 4, 3, false, 0, 1) {
  auto queries = INPUT_VARIABLE(0);
  auto keys = INPUT_VARIABLE(1);
  auto values = INPUT_VARIABLE(2);
  auto qMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
  auto vMask = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
  auto eps = INPUT_VARIABLE(block.width() - 1);


  auto dLdq = OUTPUT_VARIABLE(0);
  auto dLdk = OUTPUT_VARIABLE(1);
  auto dLdv = OUTPUT_VARIABLE(2);

  int normalization = INT_ARG(0);
  REQUIRE_TRUE(queries->rankOf() == 3,0,"Input rank of queries must be 3.");
  REQUIRE_TRUE(values->rankOf() == 3,0,"Input rank of values must be 3.");
  REQUIRE_TRUE(values->rankOf() == 3,0,"Input rank of keys must be 3.");

  auto dropout = block.numT() > 0 ? T_ARG(0) : 0.0;

  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto returnAttentionScores = block.numB() > 1 ? B_ARG(1) : false;
  //permute the inputs before processing. This is to allow the old shapes of batch size x dim x Tv
  auto permuteInputs = block.numB() > 2 ? B_ARG(2) : false;


  int attentionType = block.numI() > 0 ? I_ARG(0) : ATTENTION_TYPE_DOT_PRODUCT;

  auto inputKeys = permuteInputs ? new NDArray(keys->permute({0,2,1})) : keys;
  auto inputQueries = permuteInputs ? new NDArray(queries->permute({0,2,1})) : queries;

  auto qMaskInput = permuteInputs  && qMask != nullptr ? new NDArray(qMask->permute({0,2,1})) : qMask;
  auto vMaskInput = permuteInputs  && vMask != nullptr ? new NDArray(vMask->permute({0,2,1})) : vMask;

  std::vector<sd::NDArray*> inputs = {inputQueries,inputKeys,values,eps};
  std::vector<sd::NDArray *> masks2 = {qMaskInput,vMaskInput};
  std::vector<sd::NDArray *> outputs = {dLdq,dLdk,dLdv};
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

DECLARE_TYPES(dot_product_attention_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_bp) {
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
