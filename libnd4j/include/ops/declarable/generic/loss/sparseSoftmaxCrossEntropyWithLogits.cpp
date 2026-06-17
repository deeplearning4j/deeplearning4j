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

#include <cmath>
#include <math/templatemath.h>
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

  // Compute sparse softmax cross-entropy via a simple per-example scalar loop.
  // loss[i] = log(sum_c exp(logits[i,c])) - logits[i, labels[i]]
  // Numerically stable: subtract per-row max before computing exp and log-sum-exp.
  const LongType batch = labels->lengthOf();
  const LongType numClasses = logits->sizeAt(logitsRank - 1);

  for (LongType i = 0; i < batch; i++) {
    // Compute per-row max for numerical stability
    double maxVal = logits->e<double>(i, 0);
    for (LongType c = 1; c < numClasses; c++) {
      double v = logits->e<double>(i, c);
      if (v > maxVal) maxVal = v;
    }
    // Compute sum(exp(logits[i,c] - maxVal)) for all c
    double sumE = 0.0;
    for (LongType c = 0; c < numClasses; c++) {
      sumE += sd::math::sd_exp<double, double>(logits->e<double>(i, c) - maxVal);
    }
    // log-partition function (shifted): log(sumE)
    double logSumExp = sd::math::sd_log<double, double>(sumE);
    // Gather shifted logit at label position
    LongType label = labels->e<LongType>(i);
    double logitAtLabel = logits->e<double>(i, label) - maxVal;
    // loss[i] = logSumExp - logitAtLabel
    output->p(i, logSumExp - logitAtLabel);
  }

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

  // dEdp = softmax(logits) - one_hot(labels)
  // Compute per-example softmax via scalar loop for numerical stability.
  // dLdp[i, c] = exp(logits[i,c] - max_i) / sum_c exp(logits[i,c] - max_i) - 1{c == labels[i]}
  const LongType batchGrad = labels->lengthOf();
  const LongType numClassesGrad = logits->sizeAt(logitsRank - 1);

  for (LongType i = 0; i < batchGrad; i++) {
    // per-row max for numerical stability
    double maxValGrad = logits->e<double>(i, 0);
    for (LongType c = 1; c < numClassesGrad; c++) {
      double v = logits->e<double>(i, c);
      if (v > maxValGrad) maxValGrad = v;
    }
    // sum of shifted exp
    double sumEGrad = 0.0;
    for (LongType c = 0; c < numClassesGrad; c++) {
      sumEGrad += sd::math::sd_exp<double, double>(logits->e<double>(i, c) - maxValGrad);
    }
    // write softmax(logits)[i,c] - one_hot(labels[i])[c]
    LongType labelGrad = labels->e<LongType>(i);
    for (LongType c = 0; c < numClassesGrad; c++) {
      double sm = sd::math::sd_exp<double, double>(logits->e<double>(i, c) - maxValGrad) / sumEGrad;
      double grad = sm - (c == labelGrad ? 1.0 : 0.0);
      dLdp->p(i, c, grad);
    }
  }

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
