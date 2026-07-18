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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 25.11.2017
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sigm_cross_entropy_loss)

#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/loss.h>
#include <ops/declarable/helpers/legacy_helpers.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sigm_cross_entropy_loss, 3, 1, false, 1, 1) {
  auto logits = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  auto labelsSmoothing = T_ARG(0);

  // input validation
  REQUIRE_TRUE(labels->isSameShape(logits), 0,
               "SIGM_CROSS_ENTROPY_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s "
               "correspondingly!",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "SIGM_CROSS_ENTROPY_LOSS OP: weights array should be scalar or have the same rank as labels array, but "
               "got %i and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "SIGM_CROSS_ENTROPY_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got "
               "weights = %s and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "SIGM_CROSS_ENTROPY_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, "
               "but got %i instead!",
               reductionMode);

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(logits))
    weightsBroad = new NDArray(weights->tileToShape(logits->shapeInfo()));

  // If labelsSmoothing is nonzero, smooth the labels towards 1/2.
  // IMPORTANT: must allocate a fresh buffer (NOT the copy constructor which creates a VIEW sharing
  // the same DataBuffer as labels). Writing smoothed values into a VIEW would corrupt the labels
  // input array, causing wrong results on repeated forward evaluations (e.g. gradient checks).
  auto newLabels = labels;
  if (labelsSmoothing != 0.) {
    newLabels = new NDArray(labels->shapeInfo(), labels->dataType(), false, block.launchContext());
    newLabels->assign(labels);
    newLabels->applyScalar(scalar::SXELogitsSmoother, labelsSmoothing, newLabels);
  }

  // Numerically stable sigmoid cross entropy:
  // E = max(x, 0) - x * y + log(1 + exp(-|x|))
  // Uses explicit NDArray operations instead of LAMBDA_TT which has CUDA issues.
  auto ctx = block.launchContext();
  NDArray E(logits->shapeInfo(), logits->dataType(), false, ctx);
  {
    // max(x, 0) = (x + |x|) / 2
    NDArray absX(logits->shapeInfo(), logits->dataType(), false, ctx);
    logits->applyTransform(transform::Abs, &absX);
    NDArray maxX0(logits->shapeInfo(), logits->dataType(), false, ctx);
    logits->applyPairwiseTransform(pairwise::Add, &absX, &maxX0);
    maxX0.applyScalar(scalar::Divide, 2.0, &maxX0);

    NDArray xTimesY(logits->shapeInfo(), logits->dataType(), false, ctx);
    logits->applyPairwiseTransform(pairwise::Multiply, newLabels, &xTimesY);

    // log(1 + exp(-|x|))
    NDArray negAbsX(logits->shapeInfo(), logits->dataType(), false, ctx);
    absX.applyTransform(transform::Neg, &negAbsX);
    NDArray expNegAbsX(logits->shapeInfo(), logits->dataType(), false, ctx);
    negAbsX.applyTransform(transform::Exp, &expNegAbsX);
    NDArray onePlusExp(logits->shapeInfo(), logits->dataType(), false, ctx);
    expNegAbsX.applyScalar(scalar::Add, 1.0, &onePlusExp);
    NDArray logTerm(logits->shapeInfo(), logits->dataType(), false, ctx);
    onePlusExp.applyTransform(transform::Log, &logTerm);

    NDArray temp(logits->shapeInfo(), logits->dataType(), false, ctx);
    maxX0.applyPairwiseTransform(pairwise::Subtract, &xTimesY, &temp);
    temp.applyPairwiseTransform(pairwise::Add, &logTerm, &E);
  }

  // multiply E on weights — stack-allocated result
  NDArray EWeighted(labels->shapeInfo(), false, ctx);
  if (weightsBroad->isScalar()) {
    E.applyScalarArr(scalar::Multiply, weightsBroad, &EWeighted);
  } else {
    E.applyPairwiseTransform(pairwise::Multiply, weightsBroad, &EWeighted);
  }

  switch (reductionMode) {
    case 0:  // 0 - "none", un-reduced weighted losses with the same shape as labels.
      output->assign(&EWeighted);
      break;

    case 1: {  // 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
      NDArray sumResult(output->dataType(), block.launchContext());
      EWeighted.reduceNumber(reduce::Sum, &sumResult);
      output->assign(&sumResult);
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
      // all elements of weightsBroad array
      NDArray sum(output->dataType(), block.launchContext());
      if (weights->isScalar()) {
        // sum = weights * EWeighted.lengthOf()
        double lengthVal = static_cast<double>(EWeighted.lengthOf());
        sum.assign(lengthVal);
        weights->applyPairwiseTransform(pairwise::Multiply, &sum, &sum);
      } else {
        weightsBroad->reduceNumber(reduce::Sum, &sum);
      }

      if (sum.e<double>(0) == 0.) {
        double zeroVal = 0.;
        output->assign(zeroVal);
      } else {
        NDArray sumE(output->dataType(), block.launchContext());
        EWeighted.reduceNumber(reduce::Sum, &sumE);
        // output = sumE / sum
        sumE.applyPairwiseTransform(pairwise::Divide, &sum, &sumE);
        output->assign(&sumE);
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
      // array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = EWeighted.lengthOf();
      } else {
        NDArray countResult(DataType::INT64, block.launchContext());
        weightsBroad->reduceNumber(reduce::CountNonZero, &countResult);
        numOfNonZeroWeights = countResult.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zeroVal = 0.;
        output->assign(zeroVal);
      } else {
        NDArray sumE(output->dataType(), block.launchContext());
        EWeighted.reduceNumber(reduce::Sum, &sumE);
        // output = sumE / numOfNonZeroWeights
        sumE.applyScalar(scalar::Divide, static_cast<double>(numOfNonZeroWeights), &sumE);
        output->assign(&sumE);
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;
  if (newLabels != labels) delete newLabels;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(sigm_cross_entropy_loss) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(sigm_cross_entropy_loss) {
  auto logitsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and logits must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, logitsShapeInfo), 0,
               "SIGM_CROSS_ENTROPY_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "SIGM_CROSS_ENTROPY_LOSS OP: weights array should be scalar or have the same rank as labels array, but "
               "got %i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "SIGM_CROSS_ENTROPY_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
      "and labels = %s instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));
  LongType* outShapeInfo = nullptr;

  if (INT_ARG(0) != 0)  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  else {  // in this case output has the same shape as labels and logits
    outShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(labelsShapeInfo),
                                                                         shape::rank(labelsShapeInfo),
                                                                         shape::shapeOf(labelsShapeInfo))->primary();
  }
  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sigm_cross_entropy_loss_grad, 3, 3, false, 1, 1) {
  auto logits = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dlogits
  auto dLdw = OUTPUT_VARIABLE(1);  // dL/dweights
  auto dLdl = OUTPUT_VARIABLE(2);  // dL/dlabels

  NDArray labelsSmoothing(logits->dataType(), block.launchContext());
  labelsSmoothing.assign(T_ARG(0));

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
  if (reductionMode == 0) reductionMode = 1;

  // input validation
  REQUIRE_TRUE(labels->isSameShape(logits), 0,
               "SIGM_CROSS_ENTROPY_LOSS_GRAD OP: labels and logits arrays must have the same shapes, but got %s and %s "
               "correspondingly!",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "SIGM_CROSS_ENTROPY_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, "
               "but got %i and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "SIGM_CROSS_ENTROPY_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got "
               "weights = %s and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "SIGM_CROSS_ENTROPY_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, "
               "3, but got %i instead!",
               reductionMode);

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(logits))
    weightsBroad = new NDArray(weights->tileToShape(logits->shapeInfo()));

  // If labelsSmoothing is nonzero, smooth the labels towards 1/2.
  // IMPORTANT: must allocate a fresh buffer (NOT the copy constructor which creates a VIEW sharing
  // the same DataBuffer as labels). Writing smoothed values into a VIEW would corrupt the labels
  // input array, causing wrong results on repeated evaluations (e.g. gradient checks).
  auto newLabels = labels;
  if (labelsSmoothing.e<float>(0) != 0.f) {
    newLabels = new NDArray(labels->shapeInfo(), labels->dataType(), false, block.launchContext());
    newLabels->assign(labels);
    newLabels->applyScalar(scalar::SXELogitsSmoother, labelsSmoothing.e<float>(0), newLabels);
  }

  auto ctx = block.launchContext();

  // E = max(x,0) - x*y + log(1+exp(-|x|))  (numerically stable sigmoid cross entropy)
  NDArray E(logits->shapeInfo(), logits->dataType(), false, ctx);
  {
    // max(x, 0) = (x + |x|) / 2
    NDArray absX(logits->shapeInfo(), logits->dataType(), false, ctx);
    logits->applyTransform(transform::Abs, &absX);
    NDArray maxX0(logits->shapeInfo(), logits->dataType(), false, ctx);
    logits->applyPairwiseTransform(pairwise::Add, &absX, &maxX0);
    maxX0.applyScalar(scalar::Divide, 2.0, &maxX0);

    NDArray xTimesY(logits->shapeInfo(), logits->dataType(), false, ctx);
    logits->applyPairwiseTransform(pairwise::Multiply, newLabels, &xTimesY);

    // log(1 + exp(-|x|))
    NDArray negAbsX(logits->shapeInfo(), logits->dataType(), false, ctx);
    absX.applyTransform(transform::Neg, &negAbsX);
    NDArray expNegAbsX(logits->shapeInfo(), logits->dataType(), false, ctx);
    negAbsX.applyTransform(transform::Exp, &expNegAbsX);
    NDArray onePlusExp(logits->shapeInfo(), logits->dataType(), false, ctx);
    expNegAbsX.applyScalar(scalar::Add, 1.0, &onePlusExp);
    NDArray logTerm(logits->shapeInfo(), logits->dataType(), false, ctx);
    onePlusExp.applyTransform(transform::Log, &logTerm);

    NDArray temp(logits->shapeInfo(), logits->dataType(), false, ctx);
    maxX0.applyPairwiseTransform(pairwise::Subtract, &xTimesY, &temp);
    temp.applyPairwiseTransform(pairwise::Add, &logTerm, &E);
  }

  // dL/dlogits = 1 - labels - 1/(1+exp(logits))  (numerically stable)
  // For x <= 0: 1 - y - 1/(1+exp(x))
  // For x > 0:  1 - y - exp(-x)/(1+exp(-x))  =  1 - y - 1/(1+exp(x))  [same formula]
  // Using sigmoid: dL/dlogits = sigmoid(logits) - labels
  {
    // sigmoid(x) = 1 / (1 + exp(-x))
    NDArray negLogits(logits->shapeInfo(), logits->dataType(), false, ctx);
    logits->applyTransform(transform::Neg, &negLogits);
    NDArray expNegLogits(logits->shapeInfo(), logits->dataType(), false, ctx);
    negLogits.applyTransform(transform::Exp, &expNegLogits);
    NDArray onePlusExpNeg(logits->shapeInfo(), logits->dataType(), false, ctx);
    expNegLogits.applyScalar(scalar::Add, 1.0, &onePlusExpNeg);
    NDArray sigmoid(logits->shapeInfo(), logits->dataType(), false, ctx);
    onePlusExpNeg.applyTransform(transform::Reciprocal, &sigmoid);

    // dLdp = sigmoid - labels
    sigmoid.applyPairwiseTransform(pairwise::Subtract, newLabels, dLdp);
  }

  // dLdl = logits * (labelsSmoothing - 1) = -logits * (1 - labelsSmoothing)
  {
    double smoothingMinus1 = labelsSmoothing.e<double>(0) - 1.0;
    logits->applyScalar(scalar::Multiply, smoothingMinus1, dLdl);
  }

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

      {
        NDArray dLdpW(dLdp->shapeInfo(), false, ctx);
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpW, false);
        dLdp->assign(&dLdpW);
        NDArray dLdlW(dLdl->shapeInfo(), false, ctx);
        dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdlW, false);
        dLdl->assign(&dLdlW);
      }

      if (weights->isScalar()) {
        NDArray sumE(dLdw->dataType(), ctx);
        E.reduceNumber(reduce::Sum, &sumE);
        dLdw->assign(&sumE);
      } else if (weights != weightsBroad) {
        std::vector<LongType> axesToReduceAlong =
            ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
        E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
      } else {
        dLdw->assign(&E);
      }
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
      // all elements of weightsBroad array
      double sum;
      if (weights->isScalar()) {
        sum = weights->e<double>(0) * E.lengthOf();
      } else {
        NDArray sumScalar(dLdp->dataType(), ctx);
        weightsBroad->reduceNumber(reduce::Sum, &sumScalar);
        sum = sumScalar.e<double>(0);
      }

      if (sum == 0.) {
        double zeroVal = 0.;
        dLdp->assign(zeroVal);
        dLdl->assign(zeroVal);
        dLdw->assign(zeroVal);
      } else {
        // weightsDivSum = weightsBroad / sum  (weightsBroad shape, may be scalar or array)
        NDArray weightsDivSum(weightsBroad->shapeInfo(), false, ctx);
        weightsBroad->applyScalar(scalar::Divide, sum, &weightsDivSum);

        {
          NDArray dLdpW(dLdp->shapeInfo(), false, ctx);
          dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivSum, &dLdpW, false);
          dLdp->assign(&dLdpW);
          NDArray dLdlW(dLdl->shapeInfo(), false, ctx);
          dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivSum, &dLdlW, false);
          dLdl->assign(&dLdlW);
        }

        if (weights->isScalar()) {
          double zeroVal = 0.;
          dLdw->assign(zeroVal);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());

          NDArray ETimesSum(labels->shapeInfo(), false, ctx);
          NDArray ETimesWeights(labels->shapeInfo(), false, ctx);
          NDArray ETimesWeightsSum(dLdp->dataType(), ctx);
          NDArray numerator(labels->shapeInfo(), false, ctx);
          NDArray result(labels->shapeInfo(), false, ctx);

          E.applyScalar(scalar::Multiply, sum, &ETimesSum);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &ETimesWeights, false);
          ETimesWeights.reduceNumber(reduce::Sum, &ETimesWeightsSum);
          double eTWSum = ETimesWeightsSum.e<double>(0);
          ETimesSum.applyScalar(scalar::Subtract, eTWSum, &numerator);
          double sumSquared = sum * sum;
          numerator.applyScalar(scalar::Divide, sumSquared, &result);
          result.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else {
          NDArray ETimesSum(labels->shapeInfo(), false, ctx);
          NDArray ETimesWeights(labels->shapeInfo(), false, ctx);
          NDArray ETimesWeightsSum(dLdp->dataType(), ctx);
          NDArray numerator(labels->shapeInfo(), false, ctx);
          NDArray result(labels->shapeInfo(), false, ctx);

          E.applyScalar(scalar::Multiply, sum, &ETimesSum);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &ETimesWeights, false);
          ETimesWeights.reduceNumber(reduce::Sum, &ETimesWeightsSum);
          double eTWSum = ETimesWeightsSum.e<double>(0);
          ETimesSum.applyScalar(scalar::Subtract, eTWSum, &numerator);
          double sumSquared = sum * sum;
          numerator.applyScalar(scalar::Divide, sumSquared, &result);
          dLdw->assign(&result);
        }
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
      // array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E.lengthOf();
      } else {
        NDArray countResult(DataType::INT64, ctx);
        weightsBroad->reduceNumber(reduce::CountNonZero, &countResult);
        numOfNonZeroWeights = countResult.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zeroVal = 0.;
        dLdp->assign(zeroVal);
        dLdl->assign(zeroVal);
        dLdw->assign(zeroVal);
      } else {
        double numNZW = static_cast<double>(numOfNonZeroWeights);

        if (weights->isScalar()) {
          NDArray eSum(dLdw->dataType(), ctx);
          E.reduceNumber(reduce::Sum, &eSum);
          double eSumVal = eSum.e<double>(0);
          double result = eSumVal / numNZW;
          dLdw->assign(result);

          double scaleFactor = weights->e<double>(0) / numNZW;
          dLdp->applyScalar(scalar::Multiply, scaleFactor, dLdp);
          dLdl->applyScalar(scalar::Multiply, scaleFactor, dLdl);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          dLdw->applyScalar(scalar::Divide, numNZW, dLdw);

          NDArray weightsDivNum(weightsBroad->shapeInfo(), false, ctx);
          weightsBroad->applyScalar(scalar::Divide, numNZW, &weightsDivNum);
          {
            NDArray dLdpW(dLdp->shapeInfo(), false, ctx);
            dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivNum, &dLdpW, false);
            dLdp->assign(&dLdpW);
            NDArray dLdlW(dLdl->shapeInfo(), false, ctx);
            dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivNum, &dLdlW, false);
            dLdl->assign(&dLdlW);
          }
        } else {
          NDArray EDivNum(labels->shapeInfo(), false, ctx);
          E.applyScalar(scalar::Divide, numNZW, &EDivNum);
          dLdw->assign(&EDivNum);

          NDArray weightsDivNum(weightsBroad->shapeInfo(), false, ctx);
          weightsBroad->applyScalar(scalar::Divide, numNZW, &weightsDivNum);
          {
            NDArray dLdpW(dLdp->shapeInfo(), false, ctx);
            dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivNum, &dLdpW, false);
            dLdp->assign(&dLdpW);
            NDArray dLdlW(dLdl->shapeInfo(), false, ctx);
            dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivNum, &dLdlW, false);
            dLdl->assign(&dLdlW);
          }
        }
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;
  if (newLabels != labels) delete newLabels;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(sigm_cross_entropy_loss_grad) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(sigm_cross_entropy_loss_grad) {
  auto logitsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and logits must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, logitsShapeInfo), 0,
               "SIGM_CROSS_ENTROPY_LOSS_GRAD OP: labels and logits arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "SIGM_CROSS_ENTROPY_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, "
               "but got %i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "SIGM_CROSS_ENTROPY_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = "
      "%s and labels = %s instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));

  auto dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(logitsShapeInfo, outType, false, block.getWorkspace());
  auto dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());
  auto dLdlShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

  return SHAPELIST(CONSTANT(dLdpShapeInfo), CONSTANT(dLdwShapeInfo), CONSTANT(dLdlShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
