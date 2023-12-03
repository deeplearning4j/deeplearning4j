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

CUSTOM_OP_IMPL(dot_product_attention_v2, -2, -1, false, -2, -2) {
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);

  REQUIRE_TRUE(queries->rankOf() == values->rankOf(),0,"Queries and values must be same ranks!");

  bool reshapedQ = false;
  if(queries->rankOf() == 2) {
    reshapedQ = true;
    queries = new NDArray(queries->reshape('c', {1,queries->sizeAt(0), queries->sizeAt(-1)}));
    values = new NDArray(values->reshape('c', {1,queries->sizeAt(0), queries->sizeAt(-1)}));
  }





  auto keys = block.width() > 2  ? INPUT_VARIABLE(2) : values;
  if(reshapedQ && block.width() > 2) {
    keys = new NDArray(keys->reshape('c', {1,keys->sizeAt(0), keys->sizeAt(-1)}));
  }

  auto qMask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  auto vMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

  //reshape to handle different shapes
  if(qMask != nullptr && reshapedQ) {
    qMask = new NDArray(qMask->reshape('c', {1,qMask->sizeAt(0), qMask->sizeAt(-1)}));
  }

  if(vMask != nullptr && reshapedQ) {
    vMask = new NDArray(vMask->reshape('c', {1,vMask->sizeAt(0), vMask->sizeAt(-1)}));
  }


  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

  auto scale = block.numT() > 0 ? T_ARG(0) : 1.0;

  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;



  std::vector<NDArray *> inputs = {queries,values,keys};
  std::vector<NDArray *> masks2 = {qMask,vMask};




  //TODO: handle reshape for rank 2 for each variable here
  auto applyScoresOut = OUTPUT_VARIABLE(0);
  auto attentionScores = OUTPUT_VARIABLE(1);
  auto attentionLogits = OUTPUT_VARIABLE(2);
  auto dropoutMask = dropout > 0.0 ? OUTPUT_VARIABLE(3) : nullptr;
  if(reshapedQ) {
    applyScoresOut->reshapei('c', {1,applyScoresOut->sizeAt(0), applyScoresOut->sizeAt(1)});
    attentionLogits->reshapei('c', {1,attentionLogits->sizeAt(0), attentionLogits->sizeAt(1)});
    attentionScores->reshapei('c', {1,attentionScores->sizeAt(0), attentionScores->sizeAt(1)});

  }
  AttentionHelper::doAttention(inputs, masks2, training, useCausalMask, dropout, scale, attentionScores,
                               block.randomSeed(), applyScoresOut, attentionLogits, dropoutMask);


  //delete extra memory
  if(reshapedQ) {
    delete queries;
    delete values;
    if(block.width() > 2) {
      delete keys;
    }
    if(qMask != nullptr)
      delete qMask;

    if(vMask != nullptr)
      delete vMask;

    applyScoresOut->reshapei('c', {applyScoresOut->sizeAt(1), applyScoresOut->sizeAt(-1)});
    attentionLogits->reshapei('c', {attentionLogits->sizeAt(1), attentionLogits->sizeAt(-1)});
    attentionScores->reshapei('c', {attentionScores->sizeAt(1), attentionScores->sizeAt(-1)});

  }

  return Status::OK;
}

DECLARE_TYPES(dot_product_attention_v2) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_v2) {
  auto firstInputType = INPUT_VARIABLE(0)->dataType();
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = block.width() > 2  ? INPUT_VARIABLE(2) : values;

  int batchSize = queries->sizeAt(0);
  int tq = queries->rankOf() == 3 ? queries->sizeAt(-2) : queries->sizeAt(-1);
  int tv = queries->rankOf() == 3 ? values->sizeAt(-2) : values->sizeAt(-1);
  int dim = queries->rankOf() == 3 ? values->sizeAt(-1) : 1;

  auto dropout = block.numT() > 1 ? block.getTArguments()->at(1) : 0.0;
  //inputs: batchSize,Tq,dim batchSize,Tq,Tv
  //outputs: batchSize,Tq, dim batchSize,Tq,Tv
  std::vector<LongType> outShape;
  std::vector<LongType> scoresShape1;


  if(queries->rankOf() == 3) {
    outShape.push_back(batchSize);
    outShape.push_back(tq);

    scoresShape1.push_back(batchSize);
    scoresShape1.push_back(tq);

    outShape.push_back(dim);
    scoresShape1.push_back(dim);
  } else {
    outShape.push_back(batchSize);
    outShape.push_back(tv);

    scoresShape1.push_back(batchSize);
    scoresShape1.push_back(dim);

  }




  ShapeDescriptor *descriptor = new ShapeDescriptor(firstInputType,'c',outShape);
  auto constOutputScores = ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor)->primary();
  ShapeDescriptor *scoresShape = new ShapeDescriptor(firstInputType,'c',scoresShape1);
  auto attentionScoresShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(scoresShape)->primary();
  auto attentionLogitsShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(scoresShape)->primary();
  if(dropout > 0) {
    if (Environment::getInstance().isDeleteShapeInfo()) {
      delete descriptor;
      delete scoresShape;
    }
    return SHAPELIST(constOutputScores,attentionScoresShape,attentionLogitsShape,attentionScoresShape);

  } else {
    if (Environment::getInstance().isDeleteShapeInfo()) {
      delete descriptor;
      delete scoresShape;
    }
    return SHAPELIST(constOutputScores,attentionScoresShape,attentionLogitsShape);

  }



}

CUSTOM_OP_IMPL(dot_product_attention_v2_bp, -2, 3, false, 0, -2) {
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = INPUT_VARIABLE(2);
  bool reshapedQ = false;
  if(queries->rankOf() == 2) {
    reshapedQ = true;
    queries = new NDArray(queries->reshape('c', {1,queries->sizeAt(0), queries->sizeAt(-1)}));
    values = new NDArray(values->reshape('c', {1,queries->sizeAt(0), queries->sizeAt(-1)}));
    keys = new NDArray(keys->reshape('c', {1,keys->sizeAt(0), keys->sizeAt(-1)}));
  }



  auto attentionScoresOut = INPUT_VARIABLE(3);
  auto attentionScoresWeights = INPUT_VARIABLE(4);
  auto attentionScoreLogits = INPUT_VARIABLE(5);
  if(reshapedQ) {
    attentionScoresOut->reshapei('c', {1,attentionScoresOut->sizeAt(0), attentionScoresOut->sizeAt(1)});
    attentionScoreLogits->reshapei('c', {1,attentionScoreLogits->sizeAt(0), attentionScoreLogits->sizeAt(1)});
    attentionScoresWeights->reshapei('c', {1,attentionScoresWeights->sizeAt(0), attentionScoresWeights->sizeAt(1)});

  }


  auto eps = INPUT_VARIABLE(6);
  if(reshapedQ) {
    eps->reshapei('c', {1,eps->sizeAt(0), eps->sizeAt(1)});
  }
  auto dropoutMask = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;

  auto qMask = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;
  auto vMask = block.width() > 9 ? INPUT_VARIABLE(9) : nullptr;

  if(qMask != nullptr && qMask->rankOf() == 2) {
    qMask = new NDArray(qMask->reshape('c', {1,qMask->sizeAt(0), qMask->sizeAt(-1)}));
  }

  if(vMask != nullptr && vMask->rankOf() == 2) {
    vMask = new NDArray(vMask->reshape('c', {1,vMask->sizeAt(0), vMask->sizeAt(-1)}));
  }

  auto dLdq = OUTPUT_VARIABLE(0);
  auto dLdv = OUTPUT_VARIABLE(1);
  auto dLdk = OUTPUT_VARIABLE(2);
  if(reshapedQ) {
    dLdq->reshapei('c', {1,dLdq->sizeAt(0), dLdq->sizeAt(1)});
    dLdv->reshapei('c', {1,dLdv->sizeAt(0), dLdv->sizeAt(1)});
    dLdk->reshapei('c', {1,dLdk->sizeAt(0), dLdk->sizeAt(1)});
  }
  auto scale = block.numT() > 1 ? T_ARG(0) : 1.0;
  auto dropout = block.numT() > 0 ? T_ARG(1) : 0.0;


  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;



  std::vector<NDArray *> inputs = {queries,values,keys,attentionScoresOut,attentionScoresWeights,attentionScoreLogits,eps};
  if(dropoutMask != nullptr) {
    inputs.push_back(dropoutMask);
  }

  std::vector<NDArray *> masks2 = {qMask,vMask};
  std::vector<NDArray *> outputs = {dLdq,dLdv,dLdk};

  int seed = block.randomSeed();
  AttentionHelper::dotProductAttentionBpHelper(queries, keys, values, scale, dLdq, dLdk, dLdv, eps, seed, qMask, vMask,
                                               useCausalMask, dropout, training, attentionScoresWeights,
                                               attentionScoreLogits, dropoutMask);

  if(reshapedQ) {
    delete queries;
    delete values;
    delete keys;
    if(qMask != nullptr)
      delete qMask;
    if(vMask != nullptr)
      delete vMask;

    dLdq->reshapei('c', {dLdq->sizeAt(1), dLdq->sizeAt(2)});
    dLdv->reshapei('c', {dLdv->sizeAt(1), dLdv->sizeAt(2)});
    dLdk->reshapei('c', {dLdk->sizeAt(1), dLdk->sizeAt(2)});
    eps->reshapei('c', {eps->sizeAt(1), eps->sizeAt(2)});
  }

  return Status::OK;
}

DECLARE_TYPES(dot_product_attention_v2_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_v2_bp) {
  LongType *dLdq_shape;
  COPY_SHAPE(inputShape->at(0), dLdq_shape);
  LongType *dLdv_shape;
  COPY_SHAPE(inputShape->at(1), dLdv_shape);
  LongType *dLdk_shape;
  COPY_SHAPE(inputShape->at(2), dLdk_shape);

  return SHAPELIST(CONSTANT(dLdq_shape), CONSTANT(dLdk_shape), CONSTANT(dLdv_shape));
}

}  // namespace ops
}  // namespace sd

#endif
