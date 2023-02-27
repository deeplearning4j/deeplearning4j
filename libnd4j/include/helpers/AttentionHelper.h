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

#ifndef LIBND4J_ATTENTIONHELPER_H
#define LIBND4J_ATTENTIONHELPER_H



#include "array/NDArray.h"

#define ATTENTION_TYPE_DOT_PRODUCT 0
#define ATTENTION_TYPE_ADDITIVE 1

#define ATTENTION_SCORE_MODE_DOT 0
#define ATTENTION_SCORE_MODE_CONCAT 1


namespace sd {
class SD_LIB_EXPORT AttentionHelper {
 public:
  static sd::NDArray multiHeadProject(const sd::NDArray* input, const sd::NDArray* projectionMatrix,
                                      sd::LaunchContext* context = sd::LaunchContext ::defaultContext());
  static void multiHeadProjectBp(const sd::NDArray* input, const sd::NDArray* projectionMatrix, const sd::NDArray* eps,
                                 sd::NDArray* dLdInput, sd::NDArray* dLdProjectionMatrix,
                                 sd::LaunchContext* context = sd::LaunchContext ::defaultContext());

  /**
   * def _lower_triangular_mask(shape):
"""Creates a lower-triangular boolean mask over the last 2 dimensions."""
row_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-2)
col_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-1)
return tf.greater_equal(row_index, col_index)
   * @param shape
   * @return
   */
  static sd::NDArray * lowerTriangularMask(std::vector<sd::LongType> *shape);

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
  static sd::NDArray computeCasualMask(sd::NDArray *query,sd::NDArray *value = nullptr);


  static sd::NDArray * mergeMasks(sd::NDArray *x,sd::NDArray *y);

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
  static NDArray *computeAttentionMask(sd::NDArray *query, sd::NDArray *value, sd::NDArray *queryMask,
                                       sd::NDArray *valueMask, sd::NDArray *attentionMask, bool useCausalMask);


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
  static void applyAttentionScores(sd::NDArray *scores, sd::NDArray *value, sd::NDArray *scoresMask, double dropout,
                                   int randomSeed, sd::NDArray *applyScoresOut);




  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
   */
  static void doDotProductAttention(sd::NDArray *query, sd::NDArray *key, int scoreMode, double scale,
                                     sd::NDArray *concatWeights, sd::NDArray *attentionScoresOut);

  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @param concatWeights
   * @return
   */
  static void attentionBpHelper(sd::NDArray *query, sd::NDArray *key, sd::NDArray *values, double scale,
                                sd::NDArray *concatWeights, int scoreMode, sd::NDArray *dLdq, sd::NDArray *dLdk,
                                sd::NDArray *dLdv, sd::NDArray *eps, LaunchContext *launchContext, sd::NDArray *qMask,
                                sd::NDArray *vMask, bool useCausalMask);

  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
   */
  static void doAdditiveAttention(sd::NDArray *query, sd::NDArray *key, double scale, sd::NDArray *scores);



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
  static void doAttention(std::vector<NDArray *> &inputs, std::vector<sd::NDArray *> &masks, bool training,
                          bool returnAttentionScores, bool useCausalMask, double dropout, int attentionType,
                          double scale, sd::NDArray *attentionScores, int dropoutSeed, sd::NDArray *applyScoresOut);



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
  static void doAttentionBp(std::vector<NDArray *> &inputs, std::vector<sd::NDArray *> &masks, bool training,
                            bool returnAttentionScores, bool useCausalMask, double dropout, int attentionType,
                            double scale, std::vector<NDArray *> outputs,
                            LaunchContext *context = sd::LaunchContext::defaultContext());


};
}  // namespace sd

#endif
