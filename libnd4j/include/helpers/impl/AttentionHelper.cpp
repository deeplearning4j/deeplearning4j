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
sd::NDArray AttentionHelper::computeCasualMask(sd::NDArray *query,sd::NDArray *value) {
  auto qSeqLength = query->sizeAt(1);
  auto vSeqLength = value != nullptr ? value->sizeAt(1) : qSeqLength;
  sd::ops::matrix_band_part matrixBandPart;
  auto ones = NDArrayFactory::create('c',{1,qSeqLength,vSeqLength},sd::DataType::INT32);
  ones.assign(1);
  auto lower = matrixBandPart.evaluate({&ones},{},{-1,0});
  auto ret = lower.at(0)->cast(sd::DataType::BOOL);
  return ret;
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
    auto mask = computeCasualMask(query,value);
    if(autoMask == nullptr) {
      autoMask = new NDArray(mask);
    } else {
      autoMask = new NDArray(booleanAnd.evaluate({autoMask,&mask}).at(0));
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



/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
void AttentionHelper::attentionBpHelper(sd::NDArray *query,
                                        sd::NDArray *key,
                                        sd::NDArray *values,
                                        double scale,
                                        sd::NDArray *concatWeights,
                                        int scoreMode,
                                        sd::NDArray *dLdq,
                                        sd::NDArray *dLdk,
                                        sd::NDArray *dLdv,
                                        sd::NDArray *eps,
                                        LaunchContext *launchContext,
                                        sd::NDArray *qMask,
                                        sd::NDArray *vMask,
                                        bool useCausalMask) {

  sd::ops::matmul matMul;
  sd::ops::matmul_bp matMulBp;
  sd::ops::reduce_sum_bp reduceSum;
  sd::ops::tanh_bp tanh1;
  sd::ops::add_bp addBp;



  if(scoreMode == ATTENTION_SCORE_MODE_DOT) {
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

    if(mask != nullptr) {
      auto maskCast = mask->cast(query->dataType());
      if (preSoftmax.rankOf() == 4) {
        maskCast = maskCast.reshape(mask->ordering(), {mask->sizeAt(0), 1,mask->sizeAt(-1), 1,});
      } else {
        maskCast = maskCast.reshape(mask->ordering(), {mask->sizeAt(0), mask->sizeAt(-1),1});
      }

      auto times = (maskCast - 1) * 1e9;
      preSoftmax -= times;
    }
    //end masking pre query/key matrix multiply section

    NDArray weights(weightShapeInfoInput);
    sd::ops::softmax softmax;
    softmax.execute({&preSoftmax}, {&weights},{}, {-2}, {});



    //permuted due to keys being permuted. Weights are query * keys permuted by 0,2,1 note we do this
    //instead of doing transb true
    auto weightInput = weights;
    //begin dldw
    NDArray dLdw(weightInput.shapeInfo());

    //weights * value?
    matMulBp.execute({&weights,values,eps},{&dLdw,dLdv},{},{});

    NDArray dLds(preSoftmax.shapeInfo());
    sd::ops::softmax_bp softmax_bp;
    softmax_bp.execute({&preSoftmax, &dLdw}, {&dLds}, {}, {-2}, {});
    //first matrix multiply  backprop end

    /* if(scale != 0.0)
      dLds /= scale; */
    //inputs: values, weights, eps
    //output is dldv, dldw


    //inputs: keys,queries,dlds
    //outputs: dldk, dldq
    matMulBp.execute({key,query,&dLds},{dLdk,dLdq},{},{0,1,1});
    dLdk->transposei();

  } else if(scoreMode == ATTENTION_SCORE_MODE_CONCAT) {
    REQUIRE_TRUE(concatWeights != nullptr,0,"Concat weights required when using attention score mode concat!");
    /**
     * squeeze both outputs
     * divide by scale
     * add_bp
     * tanh bp
     * reduce sum broadcast
     *
     * TODO: remember to factor in k,v and q gradients.
     */
    sd::ops::expand_dims expandDims;


    auto qReshaped = expandDims.evaluate({query},{},{-1}).at(0);
    auto kReshaped = expandDims.evaluate({key},{},{-3}).at(0);
    auto weights = scale > 0 ? (*kReshaped + *qReshaped) / scale : (*kReshaped + *qReshaped);
    auto epsExpanded = reduceSum.evaluate({eps},{},{},{},{}).at(0);

    tanh1.execute({epsExpanded},{epsExpanded});

    NDArray dLdw(weights.shapeInfo(), false,launchContext);
    addBp.execute({values,&weights,epsExpanded},std::vector<NDArray *>{dLdv,&dLdw},{},{},{});
    addBp.execute({key,query, &dLdw},std::vector<NDArray *>{dLdk, dLdq}, {}, {}, {});

    auto qPlusK = tanh1.evaluate({&weights}).at(0);
    auto scores2 = reduceSum.evaluate({qPlusK}).at(0);
    auto scoresResult = *concatWeights * (*scores2);

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
void AttentionHelper::doDotProductAttention(sd::NDArray *query,
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
    REQUIRE_TRUE(concatWeights != nullptr,0,"Concat weights required when using attention score mode concat!");
    sd::ops::expand_dims expandDims;
    sd::ops::reduce_sum reduceSum;
    sd::ops::create_view createView;
    sd::ops::tanh tanh1;
    auto qReshaped = expandDims.evaluate({query},{},{-1}).at(0);
    auto kReshaped = expandDims.evaluate({key},{},{-3}).at(0);
    auto scaled =  scale > 0 ? ( scale * (*qReshaped + *kReshaped)) : ((*qReshaped + *kReshaped));
    auto qPlusK = tanh1.evaluate({&scaled});
    auto scores2 = reduceSum.evaluate({qPlusK.at(0)});
    reduceSum.execute({qPlusK.at(0)},{attentionScoresOut});
    if(concatWeights != nullptr)
      *attentionScoresOut *= *concatWeights;
  } else {
    throw std::runtime_error("Illegal attention type passed. Please pass either 0 or 1 for a valid attention type. 0 for dot product, 1 for concat.");
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
void AttentionHelper::doAdditiveAttention(sd::NDArray *query, sd::NDArray *key, double scale, sd::NDArray *scores) {
  sd::ops::matmul matmul2;
  sd::ops::expand_dims expandDims;
  sd::ops::reduce_sum reduceSum;
  sd::ops::tanh tanh1;
  auto qReshaped = expandDims.evaluate({query},{},{-1}).at(0);
  auto kReshaped = expandDims.evaluate({key},{},{-3}).at(0);
  auto scaled =  scale > 0 ? ( scale * (*qReshaped + *kReshaped)) : ((*qReshaped + *kReshaped));
  auto qPlusK = tanh1.evaluate({&scaled});
  reduceSum.execute({qPlusK.at(0)},{scores});
}




/**
 * def call(
self,
inputs,
mask=None,
training=None,
return_attention_scores=False,
use_causal_mask=False,
):
self._validate_call_args(inputs=inputs, mask=mask)
q = inputs[0]
v = inputs[1]
k = inputs[2] if len(inputs) > 2 else v
q_mask = mask[0] if mask else None
v_mask = mask[1] if mask else None
scores = self._calculate_scores(query=q, key=k)
if v_mask is not None:
  # Mask of shape [batch_size, 1, Tv].
  v_mask = tf.expand_dims(v_mask, axis=-2)
if self.causal or use_causal_mask:
  # Creates a lower triangular mask, so position i cannot attend to
  # positions j>i. This prevents the flow of information from the
  # future into the past.
  scores_shape = tf.shape(scores)
  # causal_mask_shape = [1, Tq, Tv].
  causal_mask_shape = tf.concat(
      [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0
  )
  causal_mask = _lower_triangular_mask(causal_mask_shape)
else:
  causal_mask = None
scores_mask = _merge_masks(v_mask, causal_mask)
result, attention_scores = self._apply_scores(
  scores=scores, value=v, scores_mask=scores_mask, training=training
)
if q_mask is not None:
  # Mask of shape [batch_size, Tq, 1].
  q_mask = tf.expand_dims(q_mask, axis=-1)
  result *= tf.cast(q_mask, dtype=result.dtype)
if return_attention_scores:
  return result, attention_scores
return result
 * @param inputs
 * @param mask
 * @param training
 * @param returnAttentionScores
 * @param useCausalMask
 */
void AttentionHelper::doAttentionBp(std::vector<NDArray *> &inputs, std::vector<sd::NDArray *> &masks, bool training,
                                    bool returnAttentionScores, bool useCausalMask, double dropout, int attentionType,
                                    double scale, std::vector<NDArray *> outputs, LaunchContext *context) {
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs[2];
  auto eps = inputs.size() > 3 ? inputs[3] : inputs[2];


  int batchSize = q->sizeAt(0);
  int tq = q->sizeAt(1);
  int tv = v->sizeAt(1);
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


  NDArray *casualPointer = nullptr;

  NDArray *attentionScoresOut = nullptr;
  NDArray *scores = nullptr;
  auto dLdq = outputs[0];
  auto dLdk = outputs[1];
  auto dLdv = outputs[2];

  attentionBpHelper(q, k, v, scale, concatWeights, attentionType, dLdq, dLdk, dLdv, eps, context, qMaskInternal,
                    vmaskInternal, useCausalMask);

  scores = attentionScoresOut;

  if(vmaskInternal != nullptr && !vmaskInternal->isEmpty()) {
    vmaskInternal = expandDims.evaluate({vMask},{},{-2}).at(0);
  }

  if(useCausalMask) {
    auto scoresShape = shapeOf.evaluate({scores}).at(0);
    /*
     * # causal_mask_shape = [1, Tq, Tv].
 causal_mask_shape = tf.concat(
     [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0
 )
 causal_mask = _lower_triangular_mask(causal_mask_shape)
     */
    auto interval = NDIndexUtils::createInterval(0,-2);
    auto intervalBegin = NDIndexUtils::createInterval(-2,-1);
    auto scoresBegin = createView.evaluate({scoresShape,&intervalBegin}).at(0);
    auto scoresEnd = createView.evaluate({scoresShape,&interval});
    auto onesLike = onesAs.evaluate({scoresEnd.at(0)}).at(0);
    auto causalMaskShape = concatOp.evaluate({onesLike,scoresEnd.at(0)});
    auto lowerTriangleMaskInput = causalMaskShape.at(0)->asT(sd::DataType::INT64);
    std::vector<sd::LongType> *lowerTriangleMaskShape = new std::vector<sd::LongType>(lowerTriangleMaskInput.lengthOf());
    for(int i = 0; i < lowerTriangleMaskInput.lengthOf(); i++) {
      lowerTriangleMaskShape->push_back(lowerTriangleMaskInput.bufferAsT<sd::LongType>()[i]);
    }
    casualPointer = lowerTriangularMask(lowerTriangleMaskShape);
    delete lowerTriangleMaskShape;
    auto scoresMask = mergeMasks(vMask,casualPointer);

    sd::NDArray attentionScores(scores->dataType(),{batchSize,tq,tv});
    //inputs: scores:  batch size tq tv value:batch size, tv,dim scoresmask: batch size 1 tv or batch size tq tv
    applyAttentionScores(scores, v, scoresMask, dropout, 0, nullptr);

    if(qMask != nullptr && !qMask->isEmpty()) {
      qMaskInternal = expandDims.evaluate({qMaskInternal},{},{-1}).at(0);
      auto casted = qMaskInternal->cast(attentionScores.dataType());
      *scores *= casted;
    }

  }

}


/**
 * def call(
self,
inputs,
mask=None,
training=None,
return_attention_scores=False,
use_causal_mask=False,
):
self._validate_call_args(inputs=inputs, mask=mask)
q = inputs[0]
v = inputs[1]
k = inputs[2] if len(inputs) > 2 else v
q_mask = mask[0] if mask else None
v_mask = mask[1] if mask else None
scores = self._calculate_scores(query=q, key=k)
if v_mask is not None:
  # Mask of shape [batch_size, 1, Tv].
  v_mask = tf.expand_dims(v_mask, axis=-2)
if self.causal or use_causal_mask:
  # Creates a lower triangular mask, so position i cannot attend to
  # positions j>i. This prevents the flow of information from the
  # future into the past.
  scores_shape = tf.shape(scores)
  # causal_mask_shape = [1, Tq, Tv].
  causal_mask_shape = tf.concat(
      [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0
  )
  causal_mask = _lower_triangular_mask(causal_mask_shape)
else:
  causal_mask = None
scores_mask = _merge_masks(v_mask, causal_mask)
result, attention_scores = self._apply_scores(
  scores=scores, value=v, scores_mask=scores_mask, training=training
)
if q_mask is not None:
  # Mask of shape [batch_size, Tq, 1].
  q_mask = tf.expand_dims(q_mask, axis=-1)
  result *= tf.cast(q_mask, dtype=result.dtype)
if return_attention_scores:
  return result, attention_scores
return result
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
  if(attentionType == ATTENTION_TYPE_ADDITIVE) {
    doAdditiveAttention(q, v, scale, attentionScores);
  } else if(attentionType == ATTENTION_TYPE_DOT_PRODUCT) {
    //inputs: query and value
    //shape: batch_size Tq dim (batch_size Tv dim)
    doDotProductAttention(q,
                          k,
                          attentionType,
                          scale,
                          concatWeights,
                          attentionScores);
  }

  if(vMask != nullptr && !vMask->isEmpty()) {
    vmaskInternal = expandDims.evaluate({vMask},{},{-2}).at(0);
  }

  if(useCausalMask) {
    auto scoresShape = shapeOf.evaluate({attentionScores}).at(0);
    /*
     * # causal_mask_shape = [1, Tq, Tv].
 causal_mask_shape = tf.concat(
     [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0
 )
 causal_mask = _lower_triangular_mask(causal_mask_shape)
     */
    auto interval = NDIndexUtils::createInterval(0,-2);
    auto intervalBegin = NDIndexUtils::createInterval(-2,-1);
    auto scoresBegin = createView.evaluate({scoresShape,&intervalBegin}).at(0);
    auto scoresEnd = createView.evaluate({scoresShape,&interval});
    auto onesLike = onesAs.evaluate({scoresEnd.at(0)}).at(0);
    auto causalMaskShape = concatOp.evaluate({onesLike,scoresEnd.at(0)});
    auto lowerTriangleMaskInput = causalMaskShape.at(0)->asT(sd::DataType::INT64);
    std::vector<sd::LongType> *lowerTriangleMaskShape = new std::vector<sd::LongType>(lowerTriangleMaskInput.lengthOf());
    for(int i = 0; i < lowerTriangleMaskInput.lengthOf(); i++) {
      lowerTriangleMaskShape->push_back(lowerTriangleMaskInput.bufferAsT<sd::LongType>()[i]);
    }
    casualPointer = lowerTriangularMask(lowerTriangleMaskShape);
    //delete lowerTriangleMaskShape;
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
