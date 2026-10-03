/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// weighted_cross_entropy_with_logits (TensorFlow's): with logits x, targets z and positive weight w,
//   loss = (1 - z) x + (1 + (w - 1) z) softplus(-x),
// which is w z (-log sigmoid(x)) + (1 - z) (-log(1 - sigmoid(x))) without overflow. w is a scalar or one weight per
// class, broadcast along the last axis. Everything is done with array operations, which run on the arrays' own device.
// The op may run in place on the targets, so the output is written only once the targets have been read.
//
#include <ops/BroadcastOpsTuple.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_weighted_cross_entropy_with_logits)
namespace sd {
namespace ops {
namespace helpers {

void weightedCrossEntropyWithLogitsFunctor(LaunchContext* context, NDArray* targets, NDArray* input, NDArray* weights,
                                           NDArray* output) {
  const DataType type = output->dataType();
  NDArray* targetsCast = targets->dataType() == type ? nullptr : targets->cast(type);
  NDArray* inputCast = input->dataType() == type ? nullptr : input->cast(type);
  NDArray* z = targetsCast != nullptr ? targetsCast : targets;
  NDArray* x = inputCast != nullptr ? inputCast : input;

  // w - 1, as a vector over the classes unless it is a scalar
  NDArray* weightMinusOne = weights->cast(type);
  if (!weights->isScalar()) weightMinusOne->reshapei({weights->lengthOf()});
  weightMinusOne->applyScalar(scalar::Subtract, 1.0, weightMinusOne);

  // (1 + (w - 1) z) softplus(-x)
  NDArray positive(z->shapeInfo(), type, false, context, false);
  z->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightMinusOne, &positive);
  positive.applyScalar(scalar::Add, 1.0, &positive);
  NDArray softplus(x->shapeInfo(), type, false, context, false);
  x->applyTransform(transform::Neg, &softplus);
  softplus.applyTransform(transform::SoftPlus, &softplus);
  positive.applyPairwiseTransform(pairwise::Multiply, &softplus, &positive);

  // (1 - z) x
  NDArray negative(z->shapeInfo(), type, false, context, false);
  z->applyScalar(scalar::ReverseSubtract, 1.0, &negative);
  negative.applyPairwiseTransform(pairwise::Multiply, x, &negative);

  negative.applyPairwiseTransform(pairwise::Add, &positive, output);

  delete weightMinusOne;
  delete targetsCast;
  delete inputCast;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
