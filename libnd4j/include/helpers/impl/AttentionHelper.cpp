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

#ifndef LIBND4J_ATTENTIONHELPER_CPP
#define LIBND4J_ATTENTIONHELPER_CPP
#include "../AttentionHelper.h"
#include <indexing/NDIndexUtils.h>
#include <helpers/AttentionHelper.h>
#include <ops/declarable/CustomOperations.h>
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
  auto rowIndexOnes = sd::NDArrayFactory::create(sd::DataType::INT32,shape);
  auto colIndexOnes = sd::NDArrayFactory::create(sd::DataType::INT32,shape);
  sd::ops::cumsum cumsum;
  auto rowCumSum = cumsum.evaluate({&rowIndexOnes},{},{-2},{});
  auto colsCumSum = cumsum.evaluate({&colIndexOnes},{},{-1},{});
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
sd::NDArray AttentionHelper::computeAttentionMask(sd::NDArray *query,sd::NDArray *value,
                                                  sd::NDArray *queryMask,
                                                  sd::NDArray *keyMask,
                                                  sd::NDArray *valueMask,
                                                  sd::NDArray *attentionMask,
                                                  bool useCausalMask) {

  auto internalQueryMask = queryMask;
  auto internalKeyMask = keyMask;
  auto internalValueMask = valueMask;
  auto internalAttentionMask = attentionMask;
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

  if(valueMask != nullptr) {
    internalValueMask = new NDArray(valueMask->cast(sd::DataType::BOOL));
    // mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
    //                                    auto_mask = mask if auto_mask is None else auto_mask & mask
    auto mask = createView.evaluate({internalValueMask,&all,&newAxis,&all}).at(0);
    if(autoMask == nullptr) {
      autoMask = mask;
    } else {
      autoMask = booleanAnd.evaluate({autoMask,mask}).at(0);
    }
  }




  if(keyMask != nullptr) {
    internalKeyMask = new NDArray(keyMask->cast(sd::DataType::BOOL));
    //mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
    //                                 auto_mask = mask if auto_mask is None else auto_mask & mask
    auto mask = createView.evaluate({internalKeyMask,&all,&newAxis,&all}).at(0);
    if(autoMask == nullptr) {
      autoMask = mask;
    } else {
      autoMask = booleanAnd.evaluate({autoMask,mask}).at(0);
    }

  }

  if(useCausalMask) {
    auto mask = computeCasualMask(query,value);
    if(autoMask == nullptr) {
      autoMask = &mask;
    } else {
      autoMask = booleanAnd.evaluate({autoMask,&mask}).at(0);
    }
  }

  if(autoMask != nullptr) {
    if(attentionMask == nullptr) {
      return *autoMask;
    } else {
      auto ret = booleanAnd.evaluate({attentionMask,autoMask}).at(0);
      return *ret;
    }

  }

  return *attentionMask;
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
std::vector<sd::NDArray *> AttentionHelper::applyAttentionScores(sd::NDArray *scores,sd::NDArray *value, sd::NDArray *scoresMask,double dropout) {
  auto internalScores = scores;
  sd::ops::boolean_not booleanNot;
  sd::ops::softmax softmax;
  sd::ops::dropout dropoutOp;
  sd::ops::matmul matmul;
  if(scoresMask != nullptr) {
    auto paddingMaks = booleanNot.evaluate({scoresMask}).at(0);
    if(scores->dataType() == DataType::BFLOAT16) {
      *scores -= 65504 * paddingMaks->cast(scores->dataType());
    } else {
      *scores -= 1.0e9 * paddingMaks->cast(scores->dataType());
    }
  }

  auto weights = softmax.evaluate({scores}).at(0);
  if(dropout > 0) {
    *weights = dropoutOp.evaluate({weights},{dropout},{});
  }

  auto ret = matmul.evaluate({weights,value}).at(0);
  std::vector<sd::NDArray *> _content;
  _content.push_back(ret);
  _content.push_back(weights);




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
}

/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
NDArray *AttentionHelper::doDotProductAttentionBp(sd::NDArray *query, sd::NDArray *key, sd::NDArray *values, bool scale,
                                                  sd::NDArray *concatWeights, int scoreMode, sd::NDArray *dLdq,
                                                  sd::NDArray *dLdk, sd::NDArray *dLdv, sd::NDArray *eps,
                                                  LaunchContext *launchContext) {
  sd::ops::matmul_bp matmul2;
  sd::ops::expand_dims expandDims;
  sd::ops::reduce_sum_bp reduceSum;
  sd::ops::tanh_bp tanh1;
  sd::ops::add_bp addBp;
  NDArray *scores = nullptr;
  if(scoreMode == ATTENTION_SCORE_MODE_DOT) {
    sd::ops::matmul mmul;
    auto weightShape = ShapeUtils::evalShapeForMatmul(key->shapeInfo(), query->shapeInfo(), true, false);
    NDArray weights('c', weightShape, values->dataType(), launchContext);
    mmul.execute({key, query}, {&weights}, {}, {1}, {});


    sd::ops::matmul_bp mmul_bp;
    NDArray dLdw(weights.shapeInfo(), false,launchContext);
    mmul_bp.execute({values, &weights, eps}, std::vector<NDArray *>{dLdv, &dLdw}, {}, {}, {});


    scores = eps;
    if(scale != 0.0) {
      *scores /= scale;
    }

    mmul_bp.execute({key, query, &dLdw}, std::vector<NDArray *>{dLdk, dLdq}, {}, {1}, {});


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
    auto epsExpanded = reduceSum.evaluate({eps}).at(0);
    tanh1.evaluate({epsExpanded},{epsExpanded});

    NDArray dLdw(weights.shapeInfo(), false,launchContext);
    addBp.execute({values,&weights,epsExpanded},std::vector<NDArray *>{dLdv,&dLdw},{},{},{});
    addBp.execute({key,query, &dLdw},std::vector<NDArray *>{dLdk, dLdq}, {}, {}, {});

    auto qPlusK = tanh1.evaluate({&weights});
    auto scores2 = reduceSum.evaluate({qPlusK.at(0)});
    auto scoresResult = *concatWeights * *scores2.at(0);
    scores = &scoresResult;
  }

  return scores;
}

/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
NDArray *AttentionHelper::doDotProductAttention(sd::NDArray *query,sd::NDArray *key,int scoreMode,bool scale,sd::NDArray *concatWeights) {
  sd::ops::matmul matmul2;
  sd::ops::expand_dims expandDims;
  sd::ops::reduce_sum reduceSum;
  sd::ops::tanh tanh1;
  NDArray *scores = nullptr;
  if(scoreMode == ATTENTION_SCORE_MODE_DOT) {
    scores = matmul2.evaluate({query,key},{},{0,1}).at(0);
    if(scale != 0.0) {
      *scores *= scale;
    }
  } else if(scoreMode == ATTENTION_SCORE_MODE_CONCAT) {
    REQUIRE_TRUE(concatWeights != nullptr,0,"Concat weights required when using attention score mode concat!");
    auto qReshaped = expandDims.evaluate({query},{},{-1}).at(0);
    auto kReshaped = expandDims.evaluate({key},{},{-3}).at(0);
    auto scaled =  scale > 0 ? ( scale * (*qReshaped + *kReshaped)) : ((*qReshaped + *kReshaped));
    auto qPlusK = tanh1.evaluate({&scaled});
    auto scores2 = reduceSum.evaluate({qPlusK.at(0)});
    auto scoresResult = *concatWeights * *scores2.at(0);
    scores = &scoresResult;
  }

  return scores;
}
/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
NDArray *AttentionHelper::doAdditiveAttention(sd::NDArray *query, sd::NDArray *key, double scale) {
  sd::ops::matmul matmul2;
  sd::ops::expand_dims expandDims;
  sd::ops::reduce_sum reduceSum;
  sd::ops::tanh tanh1;
  NDArray *scores = nullptr;
  auto qReshaped = expandDims.evaluate({query},{},{-1}).at(0);
  auto kReshaped = expandDims.evaluate({key},{},{-3}).at(0);
  auto scaled =  scale > 0 ? ( scale * (*qReshaped + *kReshaped)) : ((*qReshaped + *kReshaped));
  auto qPlusK = tanh1.evaluate({&scaled});
  auto scores2 = reduceSum.evaluate({qPlusK.at(0)});
  scores = scores2.at(0);
  return scores;
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
std::vector<NDArray *> * AttentionHelper::doAttentionBp(std::vector<NDArray *>  &inputs,
                                                        std::vector<sd::NDArray *> &masks,
                                                        bool training,
                                                        bool returnAttentionScores,
                                                        bool useCausalMask,
                                                        double dropout,
                                                        int attentionType,
                                                        int dotProductType,
                                                        bool scale) {
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

  NDArray *attentionScoresOut = nullptr;
  NDArray *scores = nullptr;
  if(attentionType == ATTENTION_TYPE_ADDITIVE) {
    attentionScoresOut = doAdditiveAttention(q, v, scale);
  } else if(attentionType == ATTENTION_TYPE_DOT_PRODUCT) {
    attentionScoresOut = doDotProductAttention(q,k,dotProductType,scale,concatWeights);
  }

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
    auto vecConvert = lowerTriangleMaskInput.vectorasT<sd::LongType>();
    casualPointer = lowerTriangularMask(vecConvert);
    delete vecConvert;
    auto scoresMask = mergeMasks(vMask,casualPointer);
    auto appliedScores = applyAttentionScores(scores,v,scoresMask,dropout);
    auto result = appliedScores[0];
    auto attentionScores = appliedScores[1];
    if(qMask != nullptr) {
      qMaskInternal = expandDims.evaluate({qMaskInternal},{},{-1}).at(0);
      auto casted = qMaskInternal->cast(result->dataType());
      *result *= casted;
    }

    std::vector<NDArray *> *ret2 = new std::vector<NDArray *>();
    ret2->push_back(result);

    if(returnAttentionScores) {
      ret2->push_back(appliedScores[1]);
    }


    return ret2;


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
std::vector<NDArray *> * AttentionHelper::doAttention(std::vector<NDArray *>  &inputs,
                                                      std::vector<sd::NDArray *> &masks,
                                                      bool training,
                                                      bool returnAttentionScores,
                                                      bool useCausalMask,
                                                      double dropout,
                                                      int attentionType,
                                                      int dotProductType,
                                                      bool scale) {
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

  NDArray *attentionScoresOut = nullptr;
  NDArray *scores = nullptr;
  if(attentionType == ATTENTION_TYPE_ADDITIVE) {
    attentionScoresOut = doAdditiveAttention(q, v, scale);
  } else if(attentionType == ATTENTION_TYPE_DOT_PRODUCT) {
    attentionScoresOut = doDotProductAttention(q,k,dotProductType,scale,concatWeights);
  }

  scores = attentionScoresOut;

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
    auto vecConvert = lowerTriangleMaskInput.vectorasT<sd::LongType>();
    casualPointer = lowerTriangularMask(vecConvert);
    delete vecConvert;
    auto scoresMask = mergeMasks(vMask,casualPointer);
    auto appliedScores = applyAttentionScores(scores,v,scoresMask,dropout);
    auto result = appliedScores[0];
    auto attentionScores = appliedScores[1];
    if(qMask != nullptr) {
      qMaskInternal = expandDims.evaluate({qMaskInternal},{},{-1}).at(0);
      auto casted = qMaskInternal->cast(result->dataType());
      *result *= casted;
    }

    std::vector<NDArray *> *ret2 = new std::vector<NDArray *>();
    ret2->push_back(result);

    if(returnAttentionScores) {
      ret2->push_back(appliedScores[1]);
    }


    return ret2;


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
