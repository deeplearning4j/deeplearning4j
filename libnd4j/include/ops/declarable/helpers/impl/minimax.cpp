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
// Gradients of z = max(x, y) and z = min(x, y), x and y broadcast against each other.
//
// dL/dz flows to the input z takes its value from: for max to x where x > y and to y where y > x, for min the other
// way round. Where x == y both inputs are the result and each takes half, so the two gradients still sum to dL/dz
// (max(x, x) has derivative 1). Each gradient is summed over the axes its input was broadcast along. Everything is
// done with array operations, which run on the array's own device.
//
#include <helpers/ShapeUtils.h>
#include <ops/BroadcastBoolOpsTuple.h>
#include <ops/declarable/helpers/minimax.h>
#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_maximum) || NOT_EXCLUDED(OP_minimum)
namespace sd {
namespace ops {
namespace helpers {

// grad = dL/dz * (1 where x `wins` y, 1/2 where `tied`, 0 elsewhere), summed over the axes `input` was broadcast along.
static void shareOfGradient(NDArray* x, NDArray* y, BroadcastBoolOpsTuple wins, NDArray* tied, NDArray* epsNext,
                            NDArray* input, NDArray* grad) {
  const DataType type = grad->dataType();

  NDArray won(epsNext->shapeInfo(), BOOL, false, epsNext->getContext(), false);
  x->applyTrueBroadcast(wins, y, &won);
  NDArray* share = won.cast(type);
  NDArray* half = tied->cast(type);
  half->applyScalar(scalar::Multiply, 0.5, half);
  share->applyPairwiseTransform(pairwise::Add, half, share);
  delete half;

  NDArray* epsCast = epsNext->dataType() == type ? nullptr : epsNext->cast(type);
  share->applyPairwiseTransform(pairwise::Multiply, epsCast != nullptr ? epsCast : epsNext, share);
  delete epsCast;

  std::vector<LongType> axes = ShapeUtils::evalBroadcastBackwardAxis(input->shapeInfo(), epsNext->shapeInfo());
  if (axes.empty()) {
    grad->assign(share);
  } else {
    NDArray* sum = share->reduceAlongDimension(reduce::Sum, &axes);
    grad->assign(sum);
    delete sum;
  }
  delete share;
}

static void extremumBP(NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY,
                       BroadcastBoolOpsTuple xWins, BroadcastBoolOpsTuple yWins) {
  NDArray tied(epsNext->shapeInfo(), BOOL, false, epsNext->getContext(), false);
  x->applyTrueBroadcast(BroadcastBoolOpsTuple::custom(scalar::EqualTo, pairwise::EqualTo, broadcast::EqualTo), y,
                        &tied);
  shareOfGradient(x, y, xWins, &tied, epsNext, x, gradX);
  shareOfGradient(x, y, yWins, &tied, epsNext, y, gradY);
}

static BroadcastBoolOpsTuple greaterThan() {
  return BroadcastBoolOpsTuple::custom(scalar::GreaterThan, pairwise::GreaterThan, broadcast::GreaterThan);
}

static BroadcastBoolOpsTuple lessThan() {
  return BroadcastBoolOpsTuple::custom(scalar::LessThan, pairwise::LessThan, broadcast::LessThan);
}

#if NOT_EXCLUDED(OP_minimum)
void minimumBPFunctor(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX,
                      NDArray* gradY) {
  extremumBP(x, y, epsNext, gradX, gradY, lessThan(), greaterThan());
}
#endif

#if NOT_EXCLUDED(OP_maximum)
void maximumBPFunctor(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX,
                      NDArray* gradY) {
  extremumBP(x, y, epsNext, gradX, gradY, greaterThan(), lessThan());
}
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
