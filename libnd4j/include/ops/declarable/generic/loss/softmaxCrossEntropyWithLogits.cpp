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

  // Compute softmax log - keep all intermediate results alive
  NDArray* maxAlongDim = logits->reduceAlongDimension(reduce::Max, &dimension, true);
  NDArray* shiftedLogits = (*logits) - (*maxAlongDim);
  NDArray* logExp = shiftedLogits->transform(transform::Exp);
  NDArray* sumLogExp = logExp->reduceAlongDimension(reduce::Sum, &dimension, true);
  NDArray* softmaxRatio = (*logExp) / (*sumLogExp);
  NDArray* logSoftMax = softmaxRatio->transform(transform::Log);
  NDArray* negLabels = new NDArray(-(*labels));
  NDArray* product = (*negLabels) * (*logSoftMax);
  
  product->reduceAlongDimension(reduce::Sum, output, &dimension);
  
  // Clean up all intermediates at once
  delete maxAlongDim;
  delete shiftedLogits;
  delete logExp;
  delete sumLogExp;
  delete softmaxRatio;
  delete logSoftMax;
  delete negLabels;
  delete product;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss_with_logits) {
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
  auto output = OUTPUT_VARIABLE(0);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dlogits
  auto dLdl = OUTPUT_VARIABLE(1);  // dL/dlabels

  const int classesDim = block.getIArguments()->size() > 0 ? INT_ARG(0) : logits->rankOf()-1;

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

  // Compute softmax - keep all intermediate results alive
  NDArray* maxAlongDim = logits->reduceAlongDimension(reduce::Max, &dimension, true);
  NDArray* shiftedLogits = (*logits) - (*maxAlongDim);
  NDArray* softmax = shiftedLogits->transform(transform::Exp);
  NDArray* sumSoftmax = softmax->reduceAlongDimension(reduce::Sum, &dimension, true);
  (*softmax) /= (*sumSoftmax);

  // dEdp = softmax * sum_i(labels_i) - labels
  NDArray* labelsPlusEps = (*labels) + (double)1e-6;
  NDArray* labelSum = labelsPlusEps->reduceAlongDimension(reduce::Sum, &dimension, true);
  NDArray* softmaxTimesLabelSum = (*softmax) * (*labelSum);
  NDArray* dLdpTemp = (*softmaxTimesLabelSum) - (*labels);
  
  dLdp->assign(dLdpTemp);

  // dEdl = -log(softmax)
  NDArray* logSoftmax = softmax->transform(transform::Log);
  dLdl->assign(logSoftmax);
  delete logSoftmax;
  dLdl->applyTransform(transform::Neg, dLdl);
  
  // Clean up all intermediates at once
  delete maxAlongDim;
  delete shiftedLogits;
  delete softmax;
  delete sumSoftmax;
  delete labelsPlusEps;
  delete labelSum;
  delete softmaxTimesLabelSum;
  delete dLdpTemp;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss_with_logits_grad) {
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
