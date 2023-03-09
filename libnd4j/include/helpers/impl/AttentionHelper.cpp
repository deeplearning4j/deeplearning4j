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

sd::NDArray AttentionHelper::multiHeadProject(const sd::NDArray *input, const sd::NDArray *projectionMatrix,
                                              sd::LaunchContext *context) {
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
  sd::ops::matmul mmul;
  mmul.execute({&projectionPrep, &inputPrep}, {&projected});

  projected.reshapei({numHeads, projectedSize, miniBatchSize, seqLength});
  projected.permutei({2, 0, 1, 3});  //[minibatch, numHeads, projectedSize, seqLength]

  return projected;
}


/**
   * @param shape
   * @return
   */
sd::NDArray* AttentionHelper::lowerTriangularMask(std::vector<sd::LongType> *shape) {
  auto rowIndexOnes = sd::NDArrayFactory::valueOf(*shape,1,'c');
  auto colIndexOnes = sd::NDArrayFactory::valueOf(*shape,1,'c');
  sd::ops::cumsum cumsum;
  auto rowCumSum = cumsum.evaluate({rowIndexOnes},{},{-2},{});
  auto colsCumSum = cumsum.evaluate({colIndexOnes},{},{-1},{});
  sd::ops::greater_equal greaterEqual;
  auto ret = greaterEqual.evaluate({rowCumSum.at(0),colsCumSum.at(0)});
  return ret[0];
}

/**
 * @param query
 * @param value
 * @return
 */
NDArray *AttentionHelper::computeCasualMask(sd::NDArray *query, sd::NDArray *value, bool multiHead) {
  if(multiHead) {
    auto qSeqLength = query->sizeAt(1);
    auto vSeqLength = value != nullptr ? value->sizeAt(1) : qSeqLength;
    sd::ops::matrix_band_part matrixBandPart;
    auto ones = NDArrayFactory::create('c',{1,qSeqLength,vSeqLength},sd::DataType::INT32);
    ones.assign(1);
    auto lower = matrixBandPart.evaluate({&ones},{},{-1,0});
    auto ret = new NDArray(lower.at(0)->cast(sd::DataType::BOOL));
    return ret;

  } else {
    std::vector<sd::LongType> causalMaskShape2;
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
NDArray *AttentionHelper::computeAttentionMask(sd::NDArray *query, sd::NDArray *value, sd::NDArray *queryMask,
                                               sd::NDArray *valueMask, sd::NDArray *attentionMask, bool useCausalMask) {

  auto internalQueryMask = queryMask;
  auto internalValueMask = valueMask;
  sd::NDArray *autoMask = nullptr;
  sd::ops::create_view createView;
  sd::ops::boolean_and booleanAnd;
  auto all = sd::NDIndexUtils::createAll();
  auto newAxis = sd::NDIndexUtils::createNewAxis();

  if(queryMask != nullptr && !queryMask->isEmpty()) {
    internalQueryMask = new NDArray(queryMask->cast(sd::DataType::BOOL));
    if(autoMask != nullptr) {
      //  auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
      autoMask = createView.evaluate({internalQueryMask,&all,&all,&newAxis}).at(0);
    }

  }

  if(valueMask != nullptr && !valueMask->isEmpty()) {
    internalValueMask = new NDArray(valueMask->cast(sd::DataType::BOOL));
    // mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
    //                                    auto_mask = mask if auto_mask is None else auto_mask & mask
    auto mask = createView.evaluate({internalValueMask,&all,&newAxis,&all}).at(0);
    if(autoMask == nullptr) {
      autoMask = mask;
    } else {
      autoMask = new NDArray(booleanAnd.evaluate({autoMask,mask}).at(0));
    }

  }


  if(useCausalMask) {
    auto mask = computeCasualMask(query, value, false);
    if(autoMask == nullptr) {
      autoMask = new NDArray(mask);
    } else {
      autoMask = new NDArray(booleanAnd.evaluate({autoMask,mask}).at(0));
    }
  }


  if(autoMask != nullptr) {
    if(attentionMask == nullptr) {
      return autoMask;
    } else {
      auto ret = new NDArray(booleanAnd.evaluate({attentionMask,autoMask}).at(0));
      return ret;
    }
  }


  return autoMask;
}

sd::NDArray * AttentionHelper::mergeMasks(sd::NDArray *x,sd::NDArray *y) {
  if(x == nullptr || x->isEmpty()) {
    return y;
  }

  if(y == nullptr || y->isEmpty()) {
    return x;
  }

  sd::ops::boolean_and booleanAnd;
  auto ret = booleanAnd.evaluate({x,y});
  return ret.at(0);
}



void AttentionHelper::applyAttentionScores(sd::NDArray *scores,
                                           sd::NDArray *value,
                                           sd::NDArray *scoresMask,
                                           double dropout,
                                           int randomSeed,
                                           sd::NDArray *applyScoresOut) {
  sd::ops::boolean_not booleanNot;
  sd::ops::softmax softmax;
  sd::ops::dropout dropoutOp;
  sd::ops::matmul matmul;

  if (scoresMask != nullptr && !scoresMask->isEmpty()) {
    REQUIRE_TRUE(scoresMask->sizeAt(-2) == 1 || scoresMask->sizeAt(-2) == scores->sizeAt(-2),0,
                 "Scores mask must be either broadcastable or equal to scores shape. scores size at -2: was: %i scores size at -2 was: %i",scoresMask->sizeAt(-2),scores->sizeAt(-2));

    REQUIRE_TRUE(scoresMask->sizeAt(-1) == scores->sizeAt(-1),0,
                 "Scores mask must be either broadcastable or equal to scores shape. scores size at -1: was: %i scores size at -1 was: %i",scoresMask->sizeAt(-1),scores->sizeAt(-1));

    auto castedScoresMask = scoresMask->cast(sd::DataType::BOOL);
    auto paddingMask = booleanNot.evaluate({&castedScoresMask}).at(0);
    if (scores->dataType() == DataType::BFLOAT16) {
      *scores -= 65504 * paddingMask->cast(scores->dataType());
    } else {
      *scores -= 1.0e9 * paddingMask->cast(scores->dataType());
    }
  }

  auto softmaxOutput = softmax.evaluate({scores},{},{-2});
  auto weights = softmaxOutput.at(0);
  if (dropout > 0) {
    auto dropout2 = dropoutOp.evaluate({weights}, {dropout}, {randomSeed});
    auto dropoutResult = dropout2.at(0);
    weights = dropoutResult;
  }

  //batch size, tq tv
  //batch size tv dim
  //output: batch size, tq dim
  matmul.execute({weights,value},{applyScoresOut});

}

void AttentionHelper::additiveAttentionBpHelper(sd::NDArray *query, sd::NDArray *key, sd::NDArray *values, double scale,
                                                sd::NDArray *concatWeights, sd::NDArray *dLdq, sd::NDArray *dLdk,
                                                sd::NDArray *dLdv, sd::NDArray *eps, LongType dropoutSeed,
                                                sd::NDArray *qMask, sd::NDArray *vMask, bool useCausalMask,
                                                double dropout, bool training) {

  sd::ops::matmul matMul;
  sd::ops::matmul_bp matMulBp;
  sd::ops::reduce_sum reduceSum;
  sd::ops::add_bp addBp;
  sd::ops::tanh tanh1;
  /**
   * Each bp needs the original inputs + the backwards pass.
   */
  sd::ops::reduce_sum reduceSum1;
  sd::ops::reduce_sum_bp reduceSumBp1;
  sd::ops::tanh_bp tanhBp;
  sd::ops::squeeze squeeze;

  //A: value, B: weights
  //note we permute already and do not need to do so again here
  auto weightShapeInfo = ShapeUtils::evalShapeForMatmul(
      query->shapeInfo(),
      key->shapeInfo(),
      false,
      true);


  const sd::LongType *weightShapeInfoInput = ConstantShapeHelper::getInstance().createShapeInfo(
      query->dataType(),
      'c',
      weightShapeInfo);


  NDArray preSoftmax( weightShapeInfoInput);
  preSoftmax.printShapeInfo("Pre softmax shape info");
  sd::ops::expand_dims expandDims;

  auto qReshaped = expandDims.evaluate({query},{},{-2}).at(0);
  auto kReshaped = expandDims.evaluate({key},{},{-3}).at(0);
  auto scaled =  scale > 0 ? ( scale * (*qReshaped + *kReshaped)) : ((*qReshaped + *kReshaped));
  auto qPlusK = tanh1.evaluate({&scaled}).at(0);
  reduceSum.execute({qPlusK},{&preSoftmax},{},{-1});
  if(concatWeights != nullptr)
    preSoftmax *= *concatWeights;



  auto mask = AttentionHelper::computeAttentionMask(query, values, qMask, vMask, nullptr, useCausalMask);


  if(scale != 0.0 && scale != 1.0) {
    preSoftmax /= scale;
  }

  NDArray times;
  if(mask != nullptr) {
    auto maskCast = mask->cast(query->dataType());
    if (preSoftmax.rankOf() == 4) {
      maskCast = maskCast.reshape(mask->ordering(), {mask->sizeAt(0), 1,mask->sizeAt(-1), 1,});
    } else {
      maskCast = maskCast.reshape(mask->ordering(), {mask->sizeAt(0), mask->sizeAt(-1),1});
    }

    times = maskCast * 1e9;
    preSoftmax -= times;
  }
  //end masking pre query/key matrix multiply section

  NDArray weights(weightShapeInfoInput);
  sd::ops::softmax softmax;
  softmax.execute({&preSoftmax}, {&weights},{}, {-2}, {});


  if(dropout > 0.0 && training) {
    sd::ops::dropout dropoutOp;
    dropoutOp.execute({&weights},{&weights},{dropout},{dropoutSeed},{});
  }



  auto weightInput = weights;
  //begin dldw
  NDArray dLdw(weightInput.shapeInfo());
  weights.printShapeInfo("Weights shape info");

  matMulBp.execute({&weights,values,eps},{&dLdw,dLdv},{},{});

  dLdv->printShapeInfo("DLDV shape info");
  dLdk->printShapeInfo("DLDK shape info");
  dLdw.printShapeInfo("DLDW shape info");
  qPlusK->printShapeInfo("QPLUS k shape info");
  auto expandDimsDLDW = expandDims.evaluate({dLdk},{},{1},{}).at(0);
  expandDimsDLDW->printShapeInfo("Expand dims dldw shape");
  auto dTanh = (*expandDimsDLDW * scale);
  sd_printf("After dtanh\n",0);

  //TODO: need to figure out correct output shape here. The dout could be eps or another thing.
  //TODO: see if numpy can do some extra  broadcasting we can't here.
  //TODO: theoretically d_out shoul dbe the same as the output from calculate scores: 10,3,4?
  //ideally this should be 10,1,4 with expand dims is 10,1,1,4
  //TODO: q plus k ends up being 10,1,3,4
  tanhBp.execute({qPlusK,expandDimsDLDW},{&dTanh},{},{});


  /**
   * sd::ops::tanh tanh1;
   * 10,1,1,4
auto qReshaped = expandDims.evaluate({query},{},{-2}).at(0);
   10,1,3,4
auto kReshaped = expandDims.evaluate({key},{},{-3}).at(0);
   10,1,3,4
auto scaled =  scale > 0 ? ( scale * (*qReshaped + *kReshaped)) : ((*qReshaped + *kReshaped));
auto qPlusK = tanh1.evaluate({&scaled});
reduceSum.execute({qPlusK.at(0)},{attentionScoresOut},{},{-1});


   */
  sd_printf("End  tanhbp\n",0);
  qReshaped->printShapeInfo("Q reshaped");
  kReshaped->printShapeInfo("K reshaped");
  dTanh.printShapeInfo("Dtanh shape info");

  auto squeezedQReshaped = expandDims.evaluate({qReshaped},{},{-1},{}).at(0);
  auto squeezedKReshaped = expandDims.evaluate({kReshaped},{},{-2},{}).at(0);
  sd_printf("Before execution eval test\n",0);
  //TODO: tanh tries to reshape to a 40 length array. Assuming 10,3,4 but elements are only 40 in length?
  auto reduceSumQBpOutput = reduceSumBp1.evaluate({squeezedQReshaped,&dTanh},{},{-1});
  auto reduceSumKBpOutput = reduceSumBp1.evaluate({squeezedKReshaped,&dTanh},{},{-1},{});
  sd_printf("After execution eval test\n",0);
  reduceSumBp1.execute({squeezedQReshaped,&dTanh},{dLdq},{},{-1},{});
  reduceSumBp1.execute({squeezedKReshaped,&dTanh},{dLdk},{},{-1},{});
  sd_printf("End  reduce sum bp\n",0);

  sd_printf("End  squeeze\n",0);



  NDArray dLds(preSoftmax.shapeInfo());
  sd::ops::softmax_bp softmax_bp;
  softmax_bp.execute({&preSoftmax, &dLdw}, {&dLds}, {}, {-2}, {});
  //first matrix multiply  backprop end
  if(dropout > 0.0 && training) {
    sd::ops::dropout_bp dropoutOp;
    dropoutOp.execute({&weights,&dLdw},{&dLdw},{dropout},{dropoutSeed},{});
  }

  if(scale != 0.0 && scale != 1.0)
    dLds /= scale;

  if(mask != nullptr) {
    dLds *= times;
  }
  //inputs: values, weights, eps
  //output is dldv, dldw


  //inputs: keys,queries,dlds
  //outputs: dldk, dldq







  dLdk->transposei();


  if(vMask != nullptr) {
    *dLdv *= *vMask;
  }

  if(qMask != nullptr) {
    *dLdq *= *qMask;
  }

}

void AttentionHelper::dotProductAttentionBpHelper(sd::NDArray *query, sd::NDArray *key, sd::NDArray *values,
                                                  double scale, sd::NDArray *dLdq, sd::NDArray *dLdk, sd::NDArray *dLdv,
                                                  sd::NDArray *eps, LongType dropoutSeed, sd::NDArray *qMask,
                                                  sd::NDArray *vMask, bool useCausalMask, double dropout,
                                                  bool training) {
  sd::ops::matmul matMul;
  sd::ops::matmul_bp matMulBp;
  sd::ops::reduce_sum reduceSum;
  sd::ops::tanh_bp tanh1;
  sd::ops::add_bp addBp;

  //A: value, B: weights
  //note we permute already and do not need to do so again here
  auto weightShapeInfo = ShapeUtils::evalShapeForMatmul(
      query->shapeInfo(),
      key->shapeInfo(),
      false,
      true);


  const sd::LongType *weightShapeInfoInput = ConstantShapeHelper::getInstance().createShapeInfo(
      query->dataType(),
      'c',
      weightShapeInfo);


  NDArray preSoftmax( weightShapeInfoInput);

  int transA = 0;
  int transB = 1;
  matMul.execute({query,key},{&preSoftmax},{transA,transB});


  auto mask = AttentionHelper::computeAttentionMask(query, values, qMask, vMask, nullptr, useCausalMask);


  if(scale != 0.0 && scale != 1.0) {
    preSoftmax /= scale;
  }

  NDArray times;
  if(mask != nullptr) {
    auto maskCast = mask->cast(query->dataType());
    if (preSoftmax.rankOf() == 4) {
      maskCast = maskCast.reshape(mask->ordering(), {mask->sizeAt(0), 1,mask->sizeAt(-1), 1,});
    } else {
      maskCast = maskCast.reshape(mask->ordering(), {mask->sizeAt(0), mask->sizeAt(-1),1});
    }

    times = maskCast * 1e9;
    preSoftmax -= times;
  }
  //end masking pre query/key matrix multiply section

  NDArray weights(weightShapeInfoInput);
  sd::ops::softmax softmax;
  softmax.execute({&preSoftmax}, {&weights},{}, {-2}, {});


  if(dropout > 0.0 && training) {
    sd::ops::dropout dropoutOp;
    dropoutOp.execute({&weights},{&weights},{dropout},{dropoutSeed},{});
  }



  auto weightInput = weights;
  //begin dldw
  NDArray dLdw(weightInput.shapeInfo());

  matMulBp.execute({&weights,values,eps},{&dLdw,dLdv},{},{});

  NDArray dLds(preSoftmax.shapeInfo());
  sd::ops::softmax_bp softmax_bp;
  softmax_bp.execute({&preSoftmax, &dLdw}, {&dLds}, {}, {-2}, {});
  //first matrix multiply  backprop end
  if(dropout > 0.0 && training) {
    sd::ops::dropout_bp dropoutOp;
    dropoutOp.execute({&weights,&dLdw},{&dLdw},{dropout},{dropoutSeed},{});
  }

  if(scale != 0.0 && scale != 1.0)
    dLds /= scale;

  if(mask != nullptr) {
    dLds *= times;
  }
  //inputs: values, weights, eps
  //output is dldv, dldw


  //inputs: keys,queries,dlds
  //outputs: dldk, dldq
  matMulBp.execute({key,query,&dLds},{dLdk,dLdq},{},{0,1,1});



  dLdk->transposei();


  if(vMask != nullptr) {
    *dLdv *= *vMask;
  }

  if(qMask != nullptr) {
    *dLdq *= *qMask;
  }


}

/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
void AttentionHelper::attentionBpHelper(sd::NDArray *query, sd::NDArray *key, sd::NDArray *values, double scale,
                                        sd::NDArray *concatWeights, int scoreMode, sd::NDArray *dLdq, sd::NDArray *dLdk,
                                        sd::NDArray *dLdv, sd::NDArray *eps, LongType dropoutSeed,
                                        sd::NDArray *qMask, sd::NDArray *vMask, bool useCausalMask, double dropout,
                                        bool training) {

  if(scoreMode == ATTENTION_SCORE_MODE_CONCAT) {
    additiveAttentionBpHelper(query, key, values, scale, concatWeights, dLdq, dLdk, dLdv, eps, dropoutSeed, qMask,
                              vMask, useCausalMask, dropout, training);
  } else if(scoreMode == ATTENTION_SCORE_MODE_DOT) {
    dotProductAttentionBpHelper(query, key, values, scale, dLdq, dLdk, dLdv, eps, dropoutSeed, qMask, vMask,
                                useCausalMask, dropout, training);
  }


}

/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
void AttentionHelper::attentionHelper(sd::NDArray *query,
                                      sd::NDArray *key,
                                      int scoreMode,
                                      double scale,
                                      sd::NDArray *concatWeights,
                                      sd::NDArray *attentionScoresOut) {

  sd::ops::matmul matmul3;

  if(scoreMode == ATTENTION_SCORE_MODE_DOT) {
    matmul3.execute({query,key},{attentionScoresOut},{0,1});
    if(scale != 0.0 && scale != 1.0) {
      *attentionScoresOut *= scale;
    }

  } else if(scoreMode == ATTENTION_SCORE_MODE_CONCAT) {
    sd::ops::expand_dims expandDims;
    sd::ops::reduce_sum reduceSum;
    sd::ops::tanh tanh1;
    auto qReshaped = expandDims.evaluate({query},{},{-2}).at(0);
    auto kReshaped = expandDims.evaluate({key},{},{-3}).at(0);
    auto scaled =  scale > 0 ? ( scale * (*qReshaped + *kReshaped)) : ((*qReshaped + *kReshaped));
    auto qPlusK = tanh1.evaluate({&scaled});
    reduceSum.execute({qPlusK.at(0)},{attentionScoresOut},{},{-1});
    if(concatWeights != nullptr)
      *attentionScoresOut *= *concatWeights;
  } else {
    throw std::runtime_error("Illegal attention type passed. Please pass either 0 or 1 for a valid attention type. 0 for dot product, 1 for concat.");
  }

}




/**
 * @param inputs
 * @param mask
 * @param training
 * @param returnAttentionScores
 * @param useCausalMask
 */
void AttentionHelper::doAttentionBp(std::vector<NDArray *> &inputs,
                                    std::vector<sd::NDArray *> &masks,
                                    bool training,
                                    bool returnAttentionScores,
                                    bool useCausalMask,
                                    double dropout,
                                    int attentionType,
                                    double scale,
                                    std::vector<NDArray *> outputs, LongType dropoutSeed) {
  sd_printf("In method\n",0);
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs[2];
  auto eps = inputs.size() > 3 ? inputs[3] : inputs[2];

  auto concatWeights = inputs.size() > 4 ? inputs[4] : nullptr;

  sd::ops::expand_dims expandDims;
  sd::ops::ones_as onesAs;
  sd::ops::shape_of shapeOf;
  sd::ops::concat concatOp;
  sd::ops::create_view createView;
  auto qMask = masks.size() > 0 ? masks[0] : nullptr;
  auto vMask = masks.size() > 1 ? masks[1] : nullptr;
  auto vmaskInternal = vMask;
  auto qMaskInternal = qMask;

  auto dLdq = outputs[0];
  auto dLdk = outputs[1];
  auto dLdv = outputs[2];
  sd_printf("Before helper \n",0);
  attentionBpHelper(q,
                    k,
                    v,
                    scale,
                    concatWeights,
                    attentionType,
                    dLdq,
                    dLdk,
                    dLdv,
                    eps,
                    dropoutSeed,
                    qMaskInternal,
                    vmaskInternal,
                    useCausalMask,
                    dropout,
                    training);
  sd_printf("After helper \n",0);

}


/**
 * @param inputs
 * @param mask
 * @param training
 * @param returnAttentionScores
 * @param useCausalMask
 */
void AttentionHelper::doAttention(std::vector<NDArray *> &inputs, std::vector<sd::NDArray *> &masks, bool training,
                                  bool returnAttentionScores, bool useCausalMask, double dropout, int attentionType,
                                  double scale, sd::NDArray *attentionScores, int dropoutSeed,
                                  sd::NDArray *applyScoresOut) {
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs.size() > 2 ? inputs[2]  : v;
  auto concatWeights = inputs.size() > 3 ? inputs[3] : nullptr;



  sd::ops::expand_dims expandDims;
  sd::ops::ones_as onesAs;
  sd::ops::shape_of shapeOf;
  sd::ops::concat concatOp;
  sd::ops::create_view createView;
  auto qMask = masks.size() > 0 ? masks[0] : nullptr;
  auto vMask = masks.size() > 1 ? masks[1] : nullptr;
  auto vmaskInternal = vMask;
  auto qMaskInternal = qMask;

  NDArray *casualPointer = nullptr;
  //inputs: query and value
  //shape: batch_size Tq dim (batch_size Tv dim)
  attentionHelper(q, k, attentionType, scale, concatWeights, attentionScores);

  if(vMask != nullptr && !vMask->isEmpty()) {
    vmaskInternal = expandDims.evaluate({vMask},{},{-2}).at(0);
  }

  if(useCausalMask) {
    std::vector<sd::LongType> causalMaskShape2;
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

  if(training) {
    applyAttentionScores(attentionScores, v, scoresMask, dropout, dropoutSeed, applyScoresOut);
  } else {
    applyAttentionScores(attentionScores, v, scoresMask, 0, dropoutSeed, applyScoresOut);
  }
  //inputs: scores:  batch size tq tv value:batch size, tv,dim scoresmask: batch size 1 tv or batch size tq tv
  if(qMask != nullptr && !qMask->isEmpty()) {
    qMaskInternal = expandDims.evaluate({qMaskInternal},{},{-1}).at(0);
    auto casted = qMaskInternal->cast(attentionScores->dataType());
    *attentionScores *= casted;
  }

}


void AttentionHelper::multiHeadProjectBp(const sd::NDArray *input, const sd::NDArray *projectionMatrix,
                                         const sd::NDArray *eps, sd::NDArray *dLdInput,
                                         sd::NDArray *dLdProjectionMatrix, sd::LaunchContext *context) {
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

  sd::ops::matmul_bp mmulBp;
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
