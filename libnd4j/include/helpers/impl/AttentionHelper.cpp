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
   * def _lower_triangular_mask(shape):
"""Creates a lower-triangular boolean mask over the last 2 dimensions."""
row_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-2)
col_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-1)
return tf.greater_equal(row_index, col_index)
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
 * def _compute_causal_mask(self, query, value=None):
"""Computes a causal mask (e.g., for masked self-attention layers).
For example, if query and value both contain sequences of length 4,
this function returns a boolean `Tensor` equal to:
```
[[[True,  False, False, False],
[True,  True,  False, False],
[True,  True,  True,  False],
[True,  True,  True,  True]]]
```
Args:
query: query `Tensor` of shape `(B, T, ...)`.
value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
query).
Returns:
mask: a boolean `Tensor` of shape [1, T, S] containing a lower
      triangular matrix of shape [T, S].
"""
q_seq_length = tf.shape(query)[1]
v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
return tf.linalg.band_part(  # creates a lower triangular matrix
  tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
)
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
 * def _compute_attention_mask(
self, query, value, key=None, attention_mask=None, use_causal_mask=False
):
"""Computes the attention mask, using the Keras masks of the inputs.
* The `query`'s mask is reshaped from [B, T] to [B, T, 1].
* The `value`'s mask is reshaped from [B, S] to [B, 1, S].
* The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
mask is ignored if `key` is `None` or if `key is value`.
* If `use_causal_mask=True`, then the causal mask is computed. Its shape
is [1, T, S].
All defined masks are merged using a logical AND operation (`&`).
In general, if the `query` and `value` are masked, then there is no need
to define the `attention_mask`.
Args:
query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
  attention to certain positions.
use_causal_mask: A boolean to indicate whether to apply a causal mask
  to prevent tokens from attending to future tokens (e.g., used in a
  decoder Transformer).
Returns:
attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
  attention to certain positions, based on the Keras masks of the
  `query`, `key`, `value`, and `attention_mask` tensors, and the
  causal mask if `use_causal_mask=True`.
"""
query_mask = getattr(query, "_keras_mask", None)
value_mask = getattr(value, "_keras_mask", None)
key_mask = getattr(key, "_keras_mask", None)
auto_mask = None
if query_mask is not None:
  query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
  # B = batch size, T = max query length
  auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]


if value_mask is not None:
  value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
  # B = batch size, S == max value length
  mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
  auto_mask = mask if auto_mask is None else auto_mask & mask
if key_mask is not None:
  key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
  # B == batch size, S == max key length == max value length
  mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
  auto_mask = mask if auto_mask is None else auto_mask & mask
if use_causal_mask:
  # the shape of the causal mask is [1, T, S]
  mask = self._compute_causal_mask(query, value)
  auto_mask = mask if auto_mask is None else auto_mask & mask
if auto_mask is not None:
  # merge attention_mask & automatic mask, to shape [B, T, S]
  attention_mask = (
      auto_mask
      if attention_mask is None
      else tf.cast(attention_mask, bool) & auto_mask
  )
return
 * @param query
 * @param value
 * @param attentionMask
 * @param useCausalMask
 * @return
 */
NDArray *AttentionHelper::computeAttentionMask(sd::NDArray *query,
                                               sd::NDArray *value,
                                               sd::NDArray *queryMask,
                                               sd::NDArray *keyMask,
                                               sd::NDArray *valueMask,
                                               sd::NDArray *attentionMask,
                                               bool useCausalMask) {

  auto internalQueryMask = queryMask;
  auto internalKeyMask = keyMask;
  auto internalValueMask = valueMask;
  sd::NDArray *autoMask = nullptr;
  sd::ops::create_view createView;
  sd::ops::boolean_and booleanAnd;
  auto all = sd::NDIndexUtils::createAll();
  auto newAxis = sd::NDIndexUtils::createNewAxis();
  if(queryMask != nullptr) {
    internalQueryMask = new NDArray(queryMask->cast(sd::DataType::BOOL));
    if(autoMask != nullptr) {
      //  auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
      autoMask = createView.evaluate({internalQueryMask,&all,&all,&newAxis}).at(0);
    }

  }

  sd_printf("After query mask creation\n",0);

  if(valueMask != nullptr) {
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

  sd_printf("After value mask creation\n",0);



  if(keyMask != nullptr) {
    internalKeyMask = new NDArray(keyMask->cast(sd::DataType::BOOL));
    //mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
    //                                 auto_mask = mask if auto_mask is None else auto_mask & mask
    auto mask = createView.evaluate({internalKeyMask,&all,&newAxis,&all}).at(0);
    if(autoMask == nullptr) {
      autoMask = mask;
    } else {
      autoMask = new NDArray(booleanAnd.evaluate({autoMask,mask}).at(0));
    }

  }

  sd_printf("After key mask creation\n",0);


  if(useCausalMask) {
    auto mask = computeCasualMask(query,value);
    if(autoMask == nullptr) {
      autoMask = new NDArray(mask);
    } else {
      autoMask = new NDArray(booleanAnd.evaluate({autoMask,&mask}).at(0));
    }
  }


  sd_printf("After causal creation\n",0);

  if(autoMask != nullptr) {
    if(attentionMask == nullptr) {
      sd_printf("Returning auto mask\n",0);
      return autoMask;
    } else {
      auto ret = new NDArray(booleanAnd.evaluate({attentionMask,autoMask}).at(0));
      sd_printf("Returning ret\n",0);
      return ret;
    }
  }

  sd_printf("Returning auto mask at end\n",0);

  return autoMask;
}

sd::NDArray * AttentionHelper::mergeMasks(sd::NDArray *x,sd::NDArray *y) {
  if(x == nullptr) {
    return y;
  }

  if(y == nullptr) {
    return x;
  }

  sd::ops::boolean_and booleanAnd;
  auto ret = booleanAnd.evaluate({x,y});
  return ret.at(0);
}


/**
 * def _apply_scores(self, scores, value, scores_mask=None, training=None):
"""Applies attention scores to the given value tensor.
To use this method in your attention layer, follow the steps:
* Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of
shape `[batch_size, Tv]` to calculate the attention `scores`.
* Pass `scores` and `value` tensors to this method. The method applies
`scores_mask`, calculates `attention_distribution = softmax(scores)`,
then returns `matmul(attention_distribution, value).
* Apply `query_mask` and return the result.
Args:
scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
value: Value tensor of shape `[batch_size, Tv, dim]`.
scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
  `[batch_size, Tq, Tv]`. If given, scores at positions where
  `scores_mask==False` do not contribute to the result. It must
  contain at least one `True` value in each line along the last
  dimension.
training: Python boolean indicating whether the layer should behave in
  training mode (adding dropout) or in inference mode (no dropout).
Returns:
Tensor of shape `[batch_size, Tq, dim]`.
Attention scores after masking and softmax with shape
  `[batch_size, Tq, Tv]`.
"""
if scores_mask is not None:
  padding_mask = tf.logical_not(scores_mask)
  # Bias so padding positions do not contribute to attention
  # distribution.  Note 65504. is the max float16 value.
  if scores.dtype is tf.float16:
      scores -= 65504.0 * tf.cast(padding_mask, dtype=scores.dtype)
  else:
      scores -= 1.0e9 * tf.cast(padding_mask, dtype=scores.dtype)
if training is None:
  training = backend.learning_phase()
weights = tf.nn.softmax(scores)

if self.dropout > 0:

def dropped_weights():
                  return self._random_generator.dropout(
                      weights, rate=self.dropout
                      )

                      weights = control_flow_util.smart_cond(
                                                     training, dropped_weights, lambda: tf.identity(weights)
                                                         )
                                    return tf.matmul(weights, value), weights
 * @return
 */
void AttentionHelper::applyAttentionScores(sd::NDArray *scores, sd::NDArray *value,
                                                             sd::NDArray *scoresMask, double dropout,
                                                             sd::NDArray *attentionScores) {
  sd::ops::boolean_not booleanNot;
  sd::ops::softmax softmax;
  sd::ops::dropout dropoutOp;
  sd::ops::batched_gemm matmul;

  if (scoresMask != nullptr) {
    auto paddingMaks = booleanNot.evaluate({scoresMask}).at(0);
    if (scores->dataType() == DataType::BFLOAT16) {
      *scores -= 65504 * paddingMaks->cast(scores->dataType());
    } else {
      *scores -= 1.0e9 * paddingMaks->cast(scores->dataType());
    }
  }

  auto softmaxOutput = softmax.evaluate({scores});
  auto weights = softmaxOutput.at(0);
  if (dropout > 0) {
    auto dropout2 = dropoutOp.evaluate({weights}, {dropout}, {});
    auto dropoutResult = dropout2.at(0);
    weights = dropoutResult;
  }

  /**
   * TODO: create new batch gemm helper and then execute usual batch gemm create view batches
   */

  int M = scores->sizeAt(1);
  int N = value->sizeAt(-1);
  int k = scores->sizeAt(-1);
  int lda = scores->sizeAt(1);
  int ldb = value->sizeAt(1);
  int ldc = M;


  auto alpha = NDArrayFactory::valueOf<double>({1},1.0);
  auto alphaCasted = alpha->cast(scores->dataType());
  auto beta = NDArrayFactory::valueOf<double>({1},0.0);
  auto betaCasted = beta->cast(scores->dataType());
  auto all = NDIndexUtils::createAll();

  sd::ops::helpers::bgemm(weights,
                          value,
                          attentionScores,
                          &alphaCasted,
                          &betaCasted,
                          0,
                          0,
                          M,
                          N,
                          k,
                          lda,
                          ldb,
                          ldc,
                          &all);

}

/**
 if scores_mask is not None:
 padding_mask = tf.logical_not(scores_mask)
 # Bias so padding positions do not contribute to attention
 # distribution.  Note 65504. is the max float16 value.
 if scores.dtype is tf.float16:
     scores -= 65504.0 * tf.cast(padding_mask, dtype=scores.dtype)
 else:
     scores -= 1.0e9 * tf.cast(padding_mask, dtype=scores.dtype)
if training is None:
 training = backend.learning_phase()
weights = tf.nn.softmax(scores)

if self.dropout > 0:

     def dropped_weights():
                         return self._random_generator.dropout(
                             weights, rate=self.dropout
                             )

                             weights = control_flow_util.smart_cond(
                                                            training, dropped_weights, lambda: tf.identity(weights)
                                                                )
                                           return tf.matmul(weights, value), weights
 */


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
                                        sd::NDArray *kMask,
                                        sd::NDArray *vMask,
                                        bool useCausalMask) {

  query->printShapeInfo("Query shape info begin ");
  values->printShapeInfo("Values shape info begin");
  key->printShapeInfo("Key shape info begin");
  sd::ops::batched_gemm_bp batchedGemmBp;
  sd::ops::batched_gemm batchedGemm;
  sd::ops::expand_dims expandDims;
  sd::ops::reduce_sum_bp reduceSum;
  sd::ops::tanh_bp tanh1;
  sd::ops::add_bp addBp;
  sd::ops::create_view createView;

  std::vector<sd::NDArray> pointIndices;
  int batchSize = query->sizeAt(0) / 2;
  for(int i = 0; i < batchSize; i++) {
    pointIndices.push_back(NDIndexUtils::createPoint(i));
  }

  if(scoreMode == ATTENTION_SCORE_MODE_DOT) {
    int transA = 0;
    int transB = 1;

    auto keyInput = key->permute({0,2,1});
    keyInput.printShapeInfo("Key input post permute");
    //A: query, B: key
    int M = query->sizeAt(1);
    int N = keyInput.sizeAt(-1);
    int k = query->sizeAt(-1);
    int lda = query->sizeAt(1);
    int ldb = keyInput.sizeAt(1);
    int ldc = M;


    auto alpha = NDArrayFactory::valueOf<double>({1},1.0);
    auto alphaCasted = alpha->cast(query->dataType());
    auto beta = NDArrayFactory::valueOf<double>({1},0.0);
    auto betaCasted = beta->cast(query->dataType());
    auto all = NDIndexUtils::createAll();

    std::vector<NDArray *>queryInputs;
    std::vector<NDArray *> keyInputs;
    std::vector<NDArray *> valueInputs;

    std::vector<NDArray *> epsInputs;
    std::vector<NDArray *> dldKSlices;
    std::vector<NDArray *> dldVSlices;
    std::vector<NDArray *> dldQSlices;




    //add alpha and beta before the batch gemm, this just needs to be broadcasted
    //divide by 2: queries and keys
    for(int i = 0; i < batchSize; i++) {
      auto point = pointIndices[i];
      auto querySlice = createView.evaluate({query,&point,&all,&all},{},{});
      auto keySlice = createView.evaluate({&keyInput,&point,&all,&all},{},{});
      auto epsSlice = createView.evaluate({eps,&point,&all,&all},{},{});
      auto valueSlice = createView.evaluate({values,&point,&all,&all},{},{});
      auto dLdQSlice = createView.evaluate({dLdq,&point,&all,&all},{},{});
      auto dLdKSlice = createView.evaluate({dLdk,&point,&all,&all},{},{});
      auto dLdVSlice = createView.evaluate({dLdv,&point,&all,&all},{},{});

      querySlice.at(0)->printShapeInfo("Query slice: %d");
      keySlice.at(0)->printShapeInfo("Key slice: %d");
      epsSlice.at(0)->printShapeInfo("Eps slice: %d");
      valueSlice.at(0)->printShapeInfo("Values slice: %d");

      queryInputs.push_back(querySlice.at(0));
      keyInputs.push_back(keySlice.at(0));
      epsInputs.push_back(epsSlice.at(0));
      valueInputs.push_back(valueSlice.at(0));

    }

    queryInputs[0]->printShapeInfo("Query shape info");
    keyInputs[0]->printShapeInfo("Key shape info");
    //note we permute already and do not need to do so again here
    auto weightShapeInfo = ShapeUtils::evalShapeForMatmul(
        queryInputs[0]->shapeInfo(),
        keyInputs[0]->shapeInfo(),
        false,
        false);
    std::vector<sd::LongType> weightShape  = {batchSize * 2,weightShapeInfo[0],weightShapeInfo[1]};
    NDArray preSoftmax('c', weightShape, values->dataType(), launchContext);
    preSoftmax.printShapeInfo("Pre softmax shape");
    std::vector<NDArray *> weightsInputs;
    weightsInputs.push_back(&alphaCasted);
    weightsInputs.push_back(&betaCasted);

    for(int i = 0; i < queryInputs.size(); i++) {
      weightsInputs.push_back(queryInputs[i]);
    }

    for(int i = 0; i < keyInputs.size(); i++) {
      weightsInputs.push_back(keyInputs[i]);
    }




    std::vector<sd::NDArray *> preSoftmaxSlices;
    for(int i = 0; i < keyInputs.size(); i++) {
      auto point = pointIndices[i];
      auto preSoftmaxSlice = createView.evaluate({&preSoftmax,&point,&all,&all},{},{});
      preSoftmaxSlices.push_back(preSoftmaxSlice.at(0));
    }



    sd_printf("M: %d N: %d k: %d lda: %d ldb %d ldc: %d batchSize: %d transA: %d transB: %d\n",M,N,k,lda,ldb,ldc,batchSize,transA,transB);

    //    scores = matmul2.evaluate(inputs,{},{transA,transB,M,N,k,lda,ldb,ldc,batchSize}).at(0);
    batchedGemm.execute(weightsInputs,
                        preSoftmaxSlices,
                        {},{
                            transA,
                            transB,
                            M,
                            N,
                            k,
                            lda,
                            ldb,
                            ldc,
                            batchSize
                        });

    sd_printf("Done Computing first batch gemm\n",0);



    sd_printf("Computing attention mask\n",0);
    auto mask = AttentionHelper::computeAttentionMask(query,
                                                      values,
                                                      qMask,
                                                      kMask,
                                                      vMask,
                                                      nullptr,
                                                      useCausalMask);
    sd_printf("Done Computing attention mask\n",0);

    if(scale != 0.0)
      preSoftmax /= scale;
    preSoftmax += (*mask - 1) * 1e9;

    //end masking pre query/key matrix multiply section

    NDArray weights('c', weightShape, values->dataType(), launchContext);
    sd::ops::softmax softmax;
    softmax.execute({&preSoftmax}, {&weights},{}, {-2}, {});

    sd_printf("Done Computing first batch softmax\n",0);


    //begin dldw
    NDArray dLdw(weights.shapeInfo());

    std::vector<NDArray *> dldwSlices;
    for(int i = 0; i < batchSize; i++) {
      auto point = pointIndices[i];
      auto weightSlice = createView.evaluate({&dLdw,&point,&all,&all},{},{});
      dldwSlices.push_back(weightSlice.at(0));
    }

    sd_printf("Done Computing dldwslices\n",0);

    std::vector<NDArray *> weightsSlices;
    for(int i = 0; i < batchSize; i++) {
      auto point = pointIndices[i];
      auto weightSlice = createView.evaluate({&weights,&point,&all,&all},{},{});
      dldwSlices.push_back(weightSlice.at(0));
    }

    std::vector<sd::NDArray *> dldWInputs;
    for(int i = 0; i < valueInputs.size(); i++) {
      dldWInputs.push_back(valueInputs[i]);
    }

    for(int i = 0; i < dldwSlices.size(); i++) {
      dldWInputs.push_back(weightsSlices[i]);
    }

    for(int i = 0; i < dldwSlices.size(); i++) {
      dldWInputs.push_back(epsInputs[i]);
    }


    //outputs for dldv dldw
    std::vector<sd::NDArray *> dldWdlDv;
    for(int i = 0; i < dldVSlices.size(); i++) {
      dldWdlDv.push_back(dldVSlices[i]);
    }

    for(int i = 0; i < dldwSlices.size(); i++) {
      dldWdlDv.push_back(dldwSlices[i]);
    }

    sd_printf("About to execute first batch gemm bp\n",0);


    //inputs: values, weights, eps
    //output is dldv, dldw
    batchedGemmBp.execute({dldWInputs},{dldWdlDv},{},{transA,transB,M,N,k,lda,ldb,ldc,batchSize});

    //first matrix multiply  backprop end

    NDArray dLds(preSoftmax.shapeInfo());
    sd::ops::softmax_bp softmax_bp;
    softmax_bp.execute({&preSoftmax, &dLdw}, {&dLds}, {}, {-2}, {});
    sd_printf("Computed softmax bp\n",0);

    if(scale != 0.0)
      dLds /= scale;


    std::vector<sd::NDArray *> dldSSlices;
    for(int i = 0; i < batchSize; i++) {
      auto point = pointIndices[i];
      auto weightSlice = createView.evaluate({&dLds,&point,&all,&all},{},{});
      dldSSlices.push_back(weightSlice.at(0));
    }


    std::vector<sd::NDArray *> dldV;
    std::vector<sd::NDArray *> dldW;
    for(int i = 0; i < batchSize; i++) {
      dldV.push_back(dldWdlDv.at(i));
      dldW.push_back(dldWdlDv.at(i + batchSize));
    }

    std::vector<NDArray *> dldKdlDqInputs;
    for(int i = 0; i < keyInputs.size(); i++) {
      dldKdlDqInputs.push_back(keyInputs[i]);
    }

    for(int i = 0; i < queryInputs.size(); i++) {
      dldKdlDqInputs.push_back(queryInputs[i]);
    }

    for(int i = 0; i < dldSSlices.size(); i++) {
      dldKdlDqInputs.push_back(dldSSlices[i]);
    }


    std::vector<sd::NDArray *> dldKdldQOutputs;
    for(int i  = 0; i < dldKSlices.size(); i++) {
      dldKdldQOutputs.push_back(dldKSlices[i]);
    }

    for(int i  = 0; i < dldQSlices.size(); i++) {
      dldKdldQOutputs.push_back(dldQSlices[i]);
    }


    //inputs: keys,queries,dlds
    //outputs: dldk, dldq
    batchedGemmBp.execute(dldKdlDqInputs,dldKdlDqInputs,{},{transA,transB,M,N,k,lda,ldb,ldc,batchSize});


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
void AttentionHelper::doDotProductAttention(sd::NDArray *query, sd::NDArray *key, int scoreMode, double scale,
                                            sd::NDArray *concatWeights, sd::NDArray *attentionScoresOut) {

  sd::ops::batched_gemm matmul2;
  sd::ops::expand_dims expandDims;
  sd::ops::reduce_sum reduceSum;
  sd::ops::create_view createView;
  sd::ops::tanh tanh1;
  if(scoreMode == ATTENTION_SCORE_MODE_DOT) {
    int transA = 0;
    int transB = 1;

    auto keyInput = key->permute({0,2,1});
    keyInput.printShapeInfo("Key input post permute:");
    //A: query, B: key
    int M = query->sizeAt(1);
    int N = keyInput.sizeAt(-1);
    int k = query->sizeAt(-1);
    int lda = query->sizeAt(1);
    int ldb = keyInput.sizeAt(1);
    int ldc = M;
    auto alpha = NDArrayFactory::valueOf<double>({1},1.0);
    auto alphaCasted = alpha->cast(query->dataType());
    auto beta = NDArrayFactory::valueOf<double>({1},0.0);
    auto betaCasted = beta->cast(query->dataType());

    sd::NDArray all = NDIndexUtils::createAll();
    sd::ops::helpers::bgemm(query,
                            &keyInput,
                            attentionScoresOut,
                            &alphaCasted,
                            &betaCasted,
                            transA,
                            transB,
                            M,
                            N,
                            k,
                            lda,
                            ldb,
                            ldc,
                            &all);
    sd_printf("After matmul\n",0);
    if(scale != 0.0) {
      *attentionScoresOut *= scale;
    }

  } else if(scoreMode == ATTENTION_SCORE_MODE_CONCAT) {
    REQUIRE_TRUE(concatWeights != nullptr,0,"Concat weights required when using attention score mode concat!");
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
void AttentionHelper::doAttentionBp(std::vector<NDArray *> &inputs,
                                    std::vector<sd::NDArray *> &masks, bool training,
                                    bool returnAttentionScores, bool useCausalMask, double dropout,
                                    int attentionType, int dotProductType, double scale,
                                    std::vector<NDArray *> outputs,
                                    sd::LaunchContext *launchContext) {
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs[2];
  auto eps = inputs.size() > 3 ? inputs[3] : inputs[3];


  int batchSize = q->sizeAt(0);
  int tq = q->sizeAt(1);
  int tv = v->sizeAt(1);

  q->printShapeInfo("In attention bP: Input queries shape info");
  v->printShapeInfo("In attention bP: Input values shape info");
  k->printShapeInfo("In attention bP: Input keys shape info");
  eps->printShapeInfo("In attention bP: Eps  shape info");
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
                    launchContext,
                    qMaskInternal,
                    nullptr,
                    vmaskInternal,
                    useCausalMask);

  scores = attentionScoresOut;

  if(vmaskInternal != nullptr) {
    vmaskInternal = expandDims.evaluate({vMask},{},{-1}).at(0);
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
    applyAttentionScores(scores, v, scoresMask, dropout, &attentionScores);

    if(qMask != nullptr) {
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
void AttentionHelper::doAttention(std::vector<NDArray *> &inputs,
                                  std::vector<sd::NDArray *> &masks,
                                  bool training,
                                  bool returnAttentionScores,
                                  bool useCausalMask,
                                  double dropout,
                                  int attentionType,
                                  int dotProductType,
                                  double scale,
                                  sd::NDArray *scores,
                                  sd::NDArray *attentionScores) {
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs.size() > 2 ? inputs[2]  : v;
  auto concatWeights = inputs.size() > 3 ? inputs[3] : nullptr;

  int batchSize = q->sizeAt(0);
  int tq = q->sizeAt(1);
  int tv = v->sizeAt(1);

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
    doAdditiveAttention(q, v, scale, scores);
  } else if(attentionType == ATTENTION_TYPE_DOT_PRODUCT) {
    //inputs: query and value
    //shape: batch_size Tq dim (batch_size Tv dim)
    sd_printf("Before do dot product attention\n",0);
    doDotProductAttention(q,
                          k,
                          dotProductType,
                          scale,
                          concatWeights,
                          scores);
    sd_printf("After do dot product attention\n",0);
  }

  if(vMask != nullptr) {
    vmaskInternal = expandDims.evaluate({vMask},{},{-1}).at(0);
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
  }


  auto scoresMask = mergeMasks(vMask,casualPointer);

  applyAttentionScores(scores, v, scoresMask, dropout, attentionScores);
  //inputs: scores:  batch size tq tv value:batch size, tv,dim scoresmask: batch size 1 tv or batch size tq tv
  if(qMask != nullptr) {
    qMaskInternal = expandDims.evaluate({qMaskInternal},{},{-1}).at(0);
    auto casted = qMaskInternal->cast(scores->dataType());
    *scores *= casted;
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
