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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.06.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_softmax_cross_entropy_loss_with_logits)

#include <ops/declarable/headers/loss.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(softmax_cross_entropy_loss_with_logits, 2, 1, false, 0, 0) {
  auto logits = INPUT_VARIABLE(0);
  auto labels = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  const int classesDim = block.getIArguments()->size() > 0 ? INT_ARG(0) : logits->rankOf() - 1;

  // input validation
  REQUIRE_TRUE(labels->isSameShape(logits), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: labels and logits arrays must have the same shapes, but got "
               "%s and %s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  REQUIRE_TRUE(classesDim < logits->rankOf(), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: class dimension must be smaller than rank of logits, but "
               "got %i and %i correspondingly !",
               classesDim, logits->rankOf());

  std::vector<LongType> dimension = {classesDim};
  auto ctx = block.launchContext();

  // maxAlongDim: heap (reduceAlongDimension)
  NDArray* maxAlongDim = logits->reduceAlongDimension(reduce::Max, &dimension, true);

  // shiftedLogits = logits - maxAlongDim  (broadcast subtract for keepDims shape)
  NDArray shiftedLogits(logits->shapeInfo(), false, ctx);
  logits->applyTrueBroadcast(BroadcastOpsTuple::Subtract(), maxAlongDim, &shiftedLogits, false);

  // logExp = exp(shiftedLogits)  (stack)
  NDArray logExp(shiftedLogits.shapeInfo(), false, ctx);
  shiftedLogits.applyTransform(transform::Exp, &logExp);

  // sumLogExp: heap (reduceAlongDimension)
  NDArray* sumLogExp = logExp.reduceAlongDimension(reduce::Sum, &dimension, true);

  // softmaxRatio = logExp / sumLogExp  (broadcast divide for keepDims shape)
  NDArray softmaxRatio(logExp.shapeInfo(), false, ctx);
  logExp.applyTrueBroadcast(BroadcastOpsTuple::Divide(), sumLogExp, &softmaxRatio, false);

  // logSoftMax = log(softmaxRatio)  (stack)
  NDArray logSoftMax(softmaxRatio.shapeInfo(), false, ctx);
  softmaxRatio.applyTransform(transform::Log, &logSoftMax);

  // negLabels = -labels  (stack)
  NDArray negLabels(labels->shapeInfo(), false, ctx);
  labels->applyTransform(transform::Neg, &negLabels);

  // product = negLabels * logSoftMax  (stack)
  NDArray product(negLabels.shapeInfo(), false, ctx);
  negLabels.applyPairwiseTransform(pairwise::Multiply, &logSoftMax, &product);

  product.reduceAlongDimension(reduce::Sum, output, &dimension);

  // Clean up heap-allocated intermediates
  delete maxAlongDim;
  delete sumLogExp;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss_with_logits) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(softmax_cross_entropy_loss_with_logits) {
  auto logitsShapeInfo = inputShape->at(0);
  auto labelsShapeInfo = inputShape->at(1);

  const int classesDim = block.getIArguments()->size() > 0 ? INT_ARG(0) : -1;
  std::vector<LongType> dimensions = {classesDim};

  // labels and logits must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(logitsShapeInfo, labelsShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: labels and logits arrays must have the same shapes, but got "
               "%s and %s correspondingly!",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

  auto outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));
  auto reducedShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(labelsShapeInfo), &dimensions, labelsShapeInfo,
                                                          outType, false, false, block.getWorkspace());

  return SHAPELIST(reducedShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(softmax_cross_entropy_loss_with_logits_grad, 2, 2, false, 0, 0) {
  auto logits = INPUT_VARIABLE(0);
  auto labels = INPUT_VARIABLE(1);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dlogits
  auto dLdl = OUTPUT_VARIABLE(1);  // dL/dlabels

  const int classesDim = block.getIArguments()->size() > 0 ? INT_ARG(0) : logits->rankOf() - 1;

  // input validation
  REQUIRE_TRUE(labels->isSameShape(logits), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: labels and logits arrays must have the same shapes, "
               "but got %s and %s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  REQUIRE_TRUE(classesDim < logits->rankOf(), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: class dimension must be smaller than rank of logits, "
               "but got %i and %i correspondingly !",
               classesDim, logits->rankOf());

  std::vector<LongType> dimension = {classesDim};
  auto ctx = block.launchContext();

  // Compute softmax
  // maxAlongDim: heap
  NDArray* maxAlongDim = logits->reduceAlongDimension(reduce::Max, &dimension, true);

  // shiftedLogits = logits - maxAlongDim  (broadcast subtract for keepDims shape)
  NDArray shiftedLogits(logits->shapeInfo(), false, ctx);
  logits->applyTrueBroadcast(BroadcastOpsTuple::Subtract(), maxAlongDim, &shiftedLogits, false);

  // softmax = exp(shiftedLogits)  (stack, then divide in-place)
  NDArray softmax(shiftedLogits.shapeInfo(), false, ctx);
  shiftedLogits.applyTransform(transform::Exp, &softmax);

  // sumSoftmax: heap
  NDArray* sumSoftmax = softmax.reduceAlongDimension(reduce::Sum, &dimension, true);

  // Normalise: softmax /= sumSoftmax  (broadcast divide for keepDims shape)
  softmax.applyTrueBroadcast(BroadcastOpsTuple::Divide(), sumSoftmax, &softmax, false);

  // dEdp = softmax * sum_i(labels_i) - labels
  // labelsPlusEps = labels + 1e-6  (stack)
  NDArray labelsPlusEps(labels->shapeInfo(), false, ctx);
  labels->applyScalar(scalar::Add, (double)1e-6, &labelsPlusEps);

  // labelSum: heap
  NDArray* labelSum = labelsPlusEps.reduceAlongDimension(reduce::Sum, &dimension, true);

  // softmaxTimesLabelSum = softmax * labelSum  (broadcast multiply for keepDims shape)
  NDArray softmaxTimesLabelSum(softmax.shapeInfo(), false, ctx);
  softmax.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), labelSum, &softmaxTimesLabelSum, false);

  // dLdp = softmaxTimesLabelSum - labels  (write directly to output)
  softmaxTimesLabelSum.applyPairwiseTransform(pairwise::Subtract, labels, dLdp);

  // dEdl = -log(softmax)
  // logSoftmax = log(softmax)  (stack)
  NDArray logSoftmax(softmax.shapeInfo(), false, ctx);
  softmax.applyTransform(transform::Log, &logSoftmax);

  // dLdl = -logSoftmax  (negate directly into output)
  logSoftmax.applyTransform(transform::Neg, dLdl);

  // Clean up heap-allocated intermediates
  delete maxAlongDim;
  delete sumSoftmax;
  delete labelSum;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss_with_logits_grad) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(softmax_cross_entropy_loss_with_logits_grad) {
  auto logitsShapeInfo = inputShape->at(0);
  auto labelsShapeInfo = inputShape->at(1);

  // labels and logits must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(logitsShapeInfo, labelsShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: labels and logits arrays must have the same shapes, "
               "but got %s and %s correspondingly!",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));

  auto dLdpShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(logitsShapeInfo),
                                                                             shape::rank(logitsShapeInfo),
                                                                             shape::shapeOf(logitsShapeInfo))->primary();

  auto dLdlShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(labelsShapeInfo),
                                                                             shape::rank(labelsShapeInfo),
                                                                             shape::shapeOf(labelsShapeInfo))->primary();
  return SHAPELIST(dLdpShapeInfo, dLdlShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
