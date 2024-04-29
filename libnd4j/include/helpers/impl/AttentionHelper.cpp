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
// @author Adam Gibson
//

#ifndef LIBND4J_ATTENTIONHELPER_CPP
#define LIBND4J_ATTENTIONHELPER_CPP
#include "../AttentionHelper.h"
#include <indexing/NDIndexUtils.h>
#include <helpers/AttentionHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/batched_gemm.h>
#if NOT_EXCLUDED(OP_multi_head_dot_product_attention)

namespace sd {

NDArray AttentionHelper::multiHeadProject(const NDArray *input, const NDArray *projectionMatrix,
                                          LaunchContext *context) {
  auto miniBatchSize = input->sizeAt(0);
  auto seqLength = input->sizeAt(2);
  auto numHeads = projectionMatrix->sizeAt(0);
  auto projectedSize = projectionMatrix->sizeAt(1);

  auto inputPerm = input->permute({1, 0, 2});  //[batch, nIn, timeSteps] -> [nIn, batch, timeSteps]
  auto inputPrep = inputPerm.reshape('c', {input->sizeAt(1), (miniBatchSize * seqLength)});  //[nIn, batch*timeSteps]
  auto projectionPrep = projectionMatrix->reshape(
      'c',
      {numHeads * projectionMatrix->sizeAt(1), projectionMatrix->sizeAt(2)});  //[nHeads, hS, nIn] -> [nHeads*hS, nIn]

  NDArray projected('c', {numHeads * projectionMatrix->sizeAt(1), (miniBatchSize * seqLength)}, input->dataType(),
                    context);  //[nHeads*hS, batch*timeSteps]
  ops::matmul mmul;
  mmul.execute({&projectionPrep, &inputPrep}, {&projected});

  projected.reshapei({numHeads, projectedSize, miniBatchSize, seqLength});
  projected.permutei({2, 0, 1, 3});  //[minibatch, numHeads, projectedSize, seqLength]

  return projected;
}


/**
 * @param shape
 * @return
 */
NDArray * AttentionHelper::lowerTriangularMask(std::vector<LongType> *shape) {
  auto rowIndexOnes = NDArrayFactory::valueOf(*shape,1,'c');
  auto colIndexOnes = NDArrayFactory::valueOf(*shape, 1, 'c');
  ops::cumsum cumsum;
  auto rowCumSum = cumsum.evaluate({rowIndexOnes},{},{-2,0},{});
  auto colsCumSum = cumsum.evaluate({colIndexOnes}, {}, {-1, 0}, {});
  ops::greater_equal greaterEqual;
  auto ret = greaterEqual.evaluate({rowCumSum.at(0),colsCumSum.at(0)});
  return ret[0];
}

/**
 * @param query
 * @param value
 * @return
 */
NDArray *AttentionHelper::computeCasualMask(NDArray *query, NDArray *value, bool multiHead) {
  if(multiHead) {
    auto qSeqLength = query->sizeAt(1);
    auto vSeqLength = value != nullptr ? value->sizeAt(1) : qSeqLength;
    ops::matrix_band_part matrixBandPart;
    auto ones = NDArrayFactory::create('c',{1,qSeqLength,vSeqLength}, INT32);
    ones.assign(1);
    auto lower = matrixBandPart.evaluate({&ones},{},{-1,0});
    auto ret = new NDArray(lower.at(0)->cast(BOOL));
    return ret;

  } else {
    std::vector<LongType> causalMaskShape2;
    causalMaskShape2.push_back(query->sizeAt(0));
    //4d
    if(query->rankOf() > 3)
      causalMaskShape2.push_back(query->sizeAt(1));

    causalMaskShape2.push_back(query->sizeAt(-2));
    causalMaskShape2.push_back(value->sizeAt(-2));

    auto ret  = lowerTriangularMask(&causalMaskShape2);
    return ret;

  }

}


/**
 * @param query
 * @param value
 * @param attentionMask
 * @param useCausalMask
 * @return
 */
NDArray *AttentionHelper::computeAttentionMask(NDArray *query, NDArray *value, NDArray *queryMask, NDArray *valueMask,
                                               NDArray *attentionMask, bool useCausalMask) {
  auto internalQueryMask = queryMask;
  auto internalValueMask = valueMask;
  NDArray *autoMask = nullptr;
  ops::create_view createView;
  ops::boolean_and booleanAnd;
  auto all = NDIndexUtils::createAll();
  auto newAxis = NDIndexUtils::createNewAxis();

  if (internalQueryMask != nullptr && !internalQueryMask->isEmpty()) {
    internalQueryMask = new NDArray(queryMask->cast(BOOL));
    if (autoMask != nullptr && !autoMask->isEmpty()) {
      autoMask = createView.evaluate({internalQueryMask, &all, &all, &newAxis}).at(0);
    }
  }

  if (valueMask != nullptr && !valueMask->isEmpty()) {
    internalValueMask = new NDArray(valueMask->cast(BOOL));
    auto mask = createView.evaluate({internalValueMask, &all, &newAxis, &all}).at(0);
    if (autoMask == nullptr || autoMask->isEmpty()) {
      autoMask = mask;
    } else {
      autoMask = new NDArray(booleanAnd.evaluate({autoMask, mask}).at(0));
    }
  }

  if (useCausalMask) {
    auto mask = computeCasualMask(query, value, false);
    if (autoMask == nullptr) {
      autoMask = new NDArray(mask);
    } else {
      autoMask = new NDArray(booleanAnd.evaluate({autoMask, mask}).at(0));
    }
  }

  if (autoMask != nullptr && !autoMask->isEmpty()) {
    if (attentionMask == nullptr || attentionMask->isEmpty()) {
      return autoMask;
    } else {
      auto ret = new NDArray(booleanAnd.evaluate({attentionMask, autoMask}).at(0));
      return ret;
    }
  }

  return autoMask;
}

NDArray * AttentionHelper::mergeMasks(NDArray *x, NDArray *y) {
  if(x == nullptr || x->isEmpty()) {
    return y;
  }

  if (y == nullptr || y->isEmpty()) {
    return x;
  }

  ops::boolean_and booleanAnd;
  auto ret = booleanAnd.evaluate({x,y});
  return ret.at(0);
}

void AttentionHelper::applyAttentionScores(NDArray *scores, NDArray *value, NDArray *scoresMask,
                                           double dropout, int randomSeed, NDArray *applyScoresOut, NDArray *attentionLogits,
                                           NDArray *dropoutMask) {
  ops::boolean_not booleanNot;
  ops::softmax softmax;
  ops::dropout dropoutOp;
  ops::matmul matmul;

  int softmaxDim = -1;
  if (scoresMask != nullptr && !scoresMask->isEmpty()) {
    REQUIRE_TRUE(scoresMask->sizeAt(-2) == 1 || scoresMask->sizeAt(-2) == scores->sizeAt(-2),0,
                 "Scores mask must be either broadcastable or equal to scores shape. scores size at -2: was: %i scores size at -2 was: %i",scoresMask->sizeAt(-2),scores->sizeAt(-2));

    REQUIRE_TRUE(scoresMask->sizeAt(-1) == scores->sizeAt(-1),0,
                 "Scores mask must be either broadcastable or equal to scores shape. scores size at -1: was: %i scores size at -1 was: %i",scoresMask->sizeAt(-1),scores->sizeAt(-1));

    auto castedScoresMask = scoresMask->cast(BOOL);
    auto paddingMask = booleanNot.evaluate({&castedScoresMask}).at(0);
    if (attentionLogits->dataType() == BFLOAT16) {
      *attentionLogits -= 65504 * paddingMask->cast(scores->dataType());
    } else {
      *attentionLogits -= 1.0e9 * paddingMask->cast(scores->dataType());
    }
  }

  softmax.execute({attentionLogits},{scores},{},{softmaxDim});
  auto weights = scores;

  if (dropout > 0) {
    dropoutOp.execute({weights},{weights,dropoutMask},{dropout},{randomSeed});
  }

  //batch size, tq tv
  //batch size tv dim
  //output: batch size, tq dim
  matmul.execute({weights,value},{applyScoresOut});

}

void AttentionHelper::dotProductAttentionBpHelper(NDArray *query, NDArray *key, NDArray *values,
                                                  double scale,
                                                  NDArray *dLdq, NDArray *dLdk, NDArray *dLdv, NDArray *eps, LongType dropoutSeed, NDArray *qMask, NDArray *vMask, bool useCausalMask, double dropout, bool training,
                                                  NDArray *attentionScoresWeights, NDArray *attentionLogits,
                                                  NDArray *dropoutMask) {
  ops::matmul_bp matMulBp;
  ops::softmax_bp softmaxBp;
  NDArray dldW(attentionScoresWeights->shapeInfo());
  NDArray dldS(attentionScoresWeights->shapeInfo());
  NDArray * mask = nullptr;
  NDArray *causalPointer = nullptr;

  if(useCausalMask) {
    std::vector<LongType> causalMaskShape2;
    causalMaskShape2.push_back(attentionLogits->sizeAt(0));
    //4d
    if(attentionLogits->rankOf() > 3)
      causalMaskShape2.push_back(attentionLogits->sizeAt(1));

    for(int i = attentionLogits->rankOf() - 2; i < attentionLogits->rankOf(); i++) {
      causalMaskShape2.push_back(attentionLogits->sizeAt(i));
    }
    causalPointer = lowerTriangularMask(&causalMaskShape2);
  }

  mask = mergeMasks(vMask,causalPointer);



  matMulBp.execute({attentionScoresWeights,values,eps},{&dldW,dLdv},{},{});
  if(dropout > 0.0 && training) {
    ops::dropout_bp dropoutOp;
    auto inputs = {attentionScoresWeights,dropoutMask,&dldW};
    dropoutOp.execute(inputs,{&dldW},{dropout},{dropoutSeed},{false});
  }


  softmaxBp.execute({attentionLogits,&dldW,attentionScoresWeights},{&dldS},{},{-1},{});



  if(scale != 0.0 && scale != 1.0) {
    dldS *= scale;
  }

  NDArray times;
  if(mask != nullptr && !mask->isEmpty()) {
    ops::expand_dims expandDims;
    auto maskCast = mask->cast(query->dataType());
    times = maskCast * 1e9;
    dldS *= times;

  }



  matMulBp.execute({query,key,&dldS},{dLdq,dLdk},{},{0,1,0});
}




/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
void AttentionHelper::attentionBpHelper(NDArray *query, NDArray *key, NDArray *values, double scale, NDArray *dLdq,
                                        NDArray *dLdk, NDArray *dLdv, NDArray *eps,
                                        LongType dropoutSeed,
                                        NDArray *qMask, NDArray *vMask,
                                        bool useCausalMask, double dropout, bool training, NDArray *attentionScoresOut,
                                        NDArray *attentionScoresWeights,
                                        NDArray *attentionScoresLogits,
                                        NDArray *dropoutMask) {
  dotProductAttentionBpHelper(query, key, values, scale, dLdq, dLdk, dLdv, eps, dropoutSeed, qMask, vMask,
                              useCausalMask, dropout, training, attentionScoresWeights, attentionScoresLogits,
                              dropoutMask);


}

/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
void AttentionHelper::attentionHelper(NDArray *query, NDArray *key, double scale, NDArray *attentionLogits) {
  ops::matmul matmul3;
  matmul3.execute({query,key},{attentionLogits},{},{0,1});
  if(scale != 0.0 && scale != 1.0) {
    *attentionLogits *= scale;
  }
}




/**
 * @param inputs
 * @param mask
 * @param training
 * @param returnAttentionScores
 * @param useCausalMask
 */
void AttentionHelper::doAttentionBp(std::vector<NDArray *> &inputs, std::vector<NDArray *> &masks, bool training,
                                    bool useCausalMask, double dropout, double scale, std::vector<NDArray *> outputs,
                                    LongType dropoutSeed) {
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs[2];
  auto attentionScoresOut = inputs[3];
  auto attentionScoresWeights = inputs[4];
  auto attentionScoresLogits = inputs[5];
  auto eps = inputs[6];

  auto dropoutMask = inputs.size() > 7 ? inputs[7] : inputs[7];

  ops::expand_dims expandDims;
  ops::ones_as onesAs;
  ops::shape_of shapeOf;
  ops::concat concatOp;
  ops::create_view createView;
  auto qMask = masks.size() > 0 ? masks[0] : nullptr;
  auto vMask = masks.size() > 1 ? masks[1] : nullptr;
  auto vmaskInternal = vMask;
  auto qMaskInternal = qMask;
  if(vMask != nullptr && !vMask->isEmpty() && vMask->rankOf() < v->rankOf()) {
    vmaskInternal = expandDims.evaluate({vMask},{},{-2}).at(0);
  }

  if(qMask != nullptr && !qMask->isEmpty()) {
    qMaskInternal = expandDims.evaluate({qMaskInternal},{},{-1}).at(0);
  }


  auto dLdq = outputs[0];
  auto dLdv = outputs[1];
  auto dLdk = outputs[2];
  attentionBpHelper(q, k, v, scale, dLdq, dLdk, dLdv, eps, dropoutSeed, qMaskInternal, vmaskInternal, useCausalMask,
                    dropout, training, attentionScoresOut, attentionScoresWeights, attentionScoresLogits, dropoutMask);

}


/**
 * @param inputs
 * @param mask
 * @param training
 * @param returnAttentionScores
 * @param useCausalMask
 */
void AttentionHelper::doAttention(std::vector<NDArray *> &inputs, std::vector<NDArray *> &masks, bool training,
                                  bool useCausalMask, double dropout, double scale, NDArray *attentionScores,
                                  int dropoutSeed, NDArray *applyScoresOut, NDArray *attentionLogits,
                                  NDArray *dropoutMask) {
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs.size() > 2 ? inputs[2]  : v;
  auto concatWeights = inputs.size() > 3 ? inputs[3] : nullptr;

  ops::expand_dims expandDims;
  ops::ones_as onesAs;
  ops::shape_of shapeOf;
  ops::concat concatOp;
  ops::create_view createView;
  auto qMask = masks.size() > 0 ? masks[0] : nullptr;
  auto vMask = masks.size() > 1 ? masks[1] : nullptr;
  auto vmaskInternal = vMask;
  auto qMaskInternal = qMask;

  NDArray *casualPointer = nullptr;
  //inputs: query and value
  //shape: batch_size Tq dim (batch_size Tv dim)
  //note this does not apply softmax yet, we are just computing logits here
  attentionHelper(q, k, scale, attentionLogits);

  if(vMask != nullptr && !vMask->isEmpty() && vMask->rankOf() < v->rankOf()) {
    vmaskInternal = expandDims.evaluate({vMask},{},{-2}).at(0);
  }

  if(useCausalMask) {
    std::vector<LongType> causalMaskShape2;
    causalMaskShape2.push_back(attentionScores->sizeAt(0));
    //4d
    if(attentionScores->rankOf() > 3)
      causalMaskShape2.push_back(attentionScores->sizeAt(1));

    for(int i = attentionScores->rankOf() - 2; i < attentionScores->rankOf(); i++) {
      causalMaskShape2.push_back(attentionScores->sizeAt(i));
    }
    casualPointer = lowerTriangularMask(&causalMaskShape2);
  }

  auto scoresMask = mergeMasks(vmaskInternal,casualPointer);

  //compute actual softmax now
  if(training) {
    applyAttentionScores(attentionScores, v, scoresMask, dropout, dropoutSeed, applyScoresOut, attentionLogits,
                         dropoutMask);
  } else {
    applyAttentionScores(attentionScores, v, scoresMask, 0, dropoutSeed, applyScoresOut, attentionLogits, dropoutMask);
  }
  //inputs: scores:  batch size tq tv value:batch size, tv,dim scoresmask: batch size 1 tv or batch size tq tv
  if(qMask != nullptr && !qMask->isEmpty()) {
    qMaskInternal = expandDims.evaluate({qMaskInternal},{},{-1}).at(0);
    auto casted = qMaskInternal->cast(attentionScores->dataType());
    *attentionScores *= casted;
  }

}


void AttentionHelper::multiHeadProjectBp(const NDArray *input, const NDArray *projectionMatrix,
                                         const NDArray *eps,
                                         NDArray *dLdInput, NDArray *dLdProjectionMatrix, LaunchContext *context) {
  auto miniBatchSize = input->sizeAt(0);
  auto seqLength = input->sizeAt(2);
  auto numHeads = projectionMatrix->sizeAt(0);
  auto projectedSize = projectionMatrix->sizeAt(1);

  auto epsPerm = eps->permute({1, 2, 0, 3});
  auto epsReshaped = epsPerm.reshape('c', {numHeads * projectedSize, miniBatchSize * seqLength});

  auto inputPerm = input->permute({1, 0, 2});
  auto inputPrep = inputPerm.reshape('c', {input->sizeAt(1), (miniBatchSize * seqLength)});
  auto projectionPrep =
      projectionMatrix->reshape('c', {numHeads * projectionMatrix->sizeAt(1), projectionMatrix->sizeAt(2)});

  ops::matmul_bp mmulBp;
  NDArray dLdProjectionPrep(projectionPrep.shapeInfo(), false, context);
  NDArray dLdInputPrep(inputPrep.shapeInfo(), false, context);
  mmulBp.execute({&projectionPrep, &inputPrep, &epsReshaped}, std::vector<NDArray *>{&dLdProjectionPrep, &dLdInputPrep},
                 {}, {}, {});

  dLdProjectionPrep.reshapei({numHeads, projectionMatrix->sizeAt(1), projectionMatrix->sizeAt(2)});
  dLdProjectionMatrix->assign(dLdProjectionPrep);

  dLdInputPrep.reshapei({input->sizeAt(1), miniBatchSize, seqLength});
  dLdInputPrep.permutei({1, 0, 2});
  dLdInput->assign(dLdInputPrep);
}
}  // namespace sd
#endif

#endif
