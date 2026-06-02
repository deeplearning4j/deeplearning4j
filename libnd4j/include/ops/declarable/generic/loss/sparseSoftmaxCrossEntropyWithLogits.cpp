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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 29.08.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sparse_softmax_cross_entropy_loss_with_logits)

#include <ops/declarable/headers/loss.h>
#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sparse_softmax_cross_entropy_loss_with_logits, 2, 1, false, 0, 0) {
  auto labels = INPUT_VARIABLE(0);
  auto logits = INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  const int labelsRank = labels->rankOf();
  const int logitsRank = logits->rankOf();

  // input validation
  REQUIRE_TRUE(labelsRank == logitsRank - 1, 0,
               "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: input arrays should satisfy relation (labels_rank = "
               "logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !",
               labelsRank, logitsRank);

  auto* labelsShapePtr = labels->getShapeAsVector();
  std::vector<LongType> labelsShape = *labelsShapePtr;
  delete labelsShapePtr;
  auto* logitsShapePtr = logits->getShapeAsVector();
  std::vector<LongType> logitsShape = *logitsShapePtr;
  delete logitsShapePtr;
  logitsShape.pop_back();
  bool equalSoft = logitsShape == labelsShape;

  REQUIRE_TRUE(
      equalSoft, 0,
      "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: wrong shape of labels array, its shape should be the same as "
      "logits shape with last dimension excluded, however got labels_shape = %s and logits_shape = %s instead !",
      ShapeUtils::shapeAsString(labelsShape).c_str(), ShapeUtils::shapeAsString(logitsShape).c_str());

  std::vector<LongType> dimension = {-1};

  // Compute -log(softmax(logits)) using raw pointers (unique_ptr is banned in libnd4j).
  // All operator-(), operator/(), reduceAlongDimension(), and transform() return heap NDArray*.
  // We delete each intermediate immediately after it is no longer needed.
  NDArray* maxAlongDim = logits->reduceAlongDimension(reduce::Max, &dimension, true);
  NDArray* shiftedLogits = (*logits) - (*maxAlongDim);
  delete maxAlongDim;

  NDArray* logitsExp = shiftedLogits->transform(transform::Exp, nullptr);
  delete shiftedLogits;

  NDArray* sumLogitsExp = logitsExp->reduceAlongDimension(reduce::Sum, &dimension, true);
  NDArray* softmaxRatio = (*logitsExp) / (*sumLogitsExp);
  delete logitsExp;
  delete sumLogitsExp;

  NDArray* logSoftmax = softmaxRatio->transform(transform::Log);
  delete softmaxRatio;

  // Negate in-place: logSoftmax = -log(softmax). Avoids any copy-constructor call.
  logSoftmax->applyScalar(scalar::Multiply, -1.0, logSoftmax);

  helpers::scatterForLoss(block.launchContext(), *labels, *logSoftmax, *output, false);
  delete logSoftmax;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(sparse_softmax_cross_entropy_loss_with_logits) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(sparse_softmax_cross_entropy_loss_with_logits) {
  auto labelsShapeInfo = inputShape->at(0);
  auto logitsShapeInfo = inputShape->at(1);

  REQUIRE_TRUE(labelsShapeInfo[0] == logitsShapeInfo[0] - 1, 0,
               "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: input arrays should satisfy relation (labels_rank = "
               "logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !",
               labelsShapeInfo[0], logitsShapeInfo[0]);

  bool equalSoft = true;
  for (int i = 1; i < labelsShapeInfo[0]; ++i)
    if (labelsShapeInfo[i] != logitsShapeInfo[i]) {
      equalSoft = false;
      break;
    }

  REQUIRE_TRUE(
      equalSoft, 0,
      "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: wrong shape of labels array, its shape should be the same as "
      "logits shape with last dimension excluded, however got labels_shape = %s and logits_shape = %s instead !",
      ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

  auto outShapeInfo =
      ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, logitsShapeInfo, false, block.getWorkspace());

  return SHAPELIST(CONSTANT(outShapeInfo));
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sparse_softmax_cross_entropy_loss_with_logits_grad, 2, 1, false, 0, 0) {
  auto labels = INPUT_VARIABLE(0);
  auto logits = INPUT_VARIABLE(1);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dlogits

  const int labelsRank = labels->rankOf();
  const int logitsRank = logits->rankOf();

  // input validation
  REQUIRE_TRUE(labelsRank == logitsRank - 1, 0,
               "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: input arrays should satisfy relation "
               "(labels_rank = logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !",
               labelsRank, logitsRank);

  auto* labelsShapePtr = labels->getShapeAsVector();
  std::vector<LongType> labelsShape = *labelsShapePtr;
  delete labelsShapePtr;
  auto* logitsShapePtr = logits->getShapeAsVector();
  std::vector<LongType> logitsShape = *logitsShapePtr;
  delete logitsShapePtr;
  logitsShape.pop_back();
  bool equalSoft = logitsShape == labelsShape;

  REQUIRE_TRUE(equalSoft, 0,
               "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: wrong shape of labels array, its shape should "
               "be the same as logits shape with last dimension excluded, however got labels_shape = %s and "
               "logits_shape = %s instead !",
               ShapeUtils::shapeAsString(labelsShape).c_str(), ShapeUtils::shapeAsString(logitsShape).c_str());

  std::vector<LongType> dimension = {-1};

  // Compute softmax using raw pointers (unique_ptr is banned in libnd4j).
  NDArray* maxAlongDim = logits->reduceAlongDimension(reduce::Max, &dimension, true);
  NDArray* shiftedLogits = (*logits) - (*maxAlongDim);
  delete maxAlongDim;

  NDArray* softmax = shiftedLogits->transform(transform::Exp);
  delete shiftedLogits;

  NDArray* sumSoftmax = softmax->reduceAlongDimension(reduce::Sum, &dimension, true);
  *softmax /= *sumSoftmax;
  delete sumSoftmax;

  // dEdp = softmax - 1 (or 0)
  dLdp->assign(softmax);
  delete softmax;

  // subtract unities at appropriate indexes of dLdp array
  helpers::scatterForLoss(block.launchContext(), *labels, *dLdp,
                          *labels /*actually third array is unnecessary for gradient calculation*/, true);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(sparse_softmax_cross_entropy_loss_with_logits_grad) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(sparse_softmax_cross_entropy_loss_with_logits_grad) {
  auto labelsShapeInfo = inputShape->at(0);
  auto logitsShapeInfo = inputShape->at(1);

  REQUIRE_TRUE(labelsShapeInfo[0] == logitsShapeInfo[0] - 1, 0,
               "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: input arrays should satisfy relation "
               "(labels_rank = logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !",
               labelsShapeInfo[0], logitsShapeInfo[0]);

  bool equalSoft = true;
  for (int i = 1; i < labelsShapeInfo[0]; ++i)
    if (labelsShapeInfo[i] != logitsShapeInfo[i]) {
      equalSoft = false;
      break;
    }

  REQUIRE_TRUE(equalSoft, 0,
               "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: wrong shape of labels array, its shape should "
               "be the same as logits shape with last dimension excluded, however got labels_shape = %s and "
               "logits_shape = %s instead !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));

  LongType *dLdpShapeInfo =
      ShapeBuilders::copyShapeInfoAndType(logitsShapeInfo, outType, false, block.getWorkspace());

  return SHAPELIST(CONSTANT(dLdpShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
