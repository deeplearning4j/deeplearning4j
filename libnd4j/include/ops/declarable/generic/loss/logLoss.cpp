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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 23.11.2017
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_log_loss)

#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/loss.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(log_loss, 3, 1, false, 1, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  double epsilon = T_ARG(0);

  // input validation
  REQUIRE_TRUE(
      labels->isSameShape(predictions), 0,
      "LOG_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "LOG_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i "
               "correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "LOG_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
               "labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "LOG_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!",
      reductionMode);

  // perform weights broadcasting/tile to predictions if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(predictions))
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));

  // E = -labels * log(predictions + epsilon) - (1 - labels) * log(1 + epsilon - predictions)
  // All intermediates are stack-allocated using the same shapeInfo as predictions.
  auto ctx = block.launchContext();
  NDArray predPlusEps(predictions->shapeInfo(), false, ctx);
  NDArray logPredPlusEps(predictions->shapeInfo(), false, ctx);
  NDArray negLabels(predictions->shapeInfo(), false, ctx);
  NDArray term1(predictions->shapeInfo(), false, ctx);
  NDArray oneMinusLabels(predictions->shapeInfo(), false, ctx);
  NDArray onePlusEpsMinusPred(predictions->shapeInfo(), false, ctx);
  NDArray logOnePlusEpsMinusPred(predictions->shapeInfo(), false, ctx);
  NDArray term2(predictions->shapeInfo(), false, ctx);
  NDArray E(predictions->shapeInfo(), false, ctx);

  // predPlusEps = predictions + epsilon
  predictions->applyScalar(scalar::Add, epsilon, &predPlusEps);
  // logPredPlusEps = log(predictions + epsilon)
  predPlusEps.applyTransform(transform::Log, &logPredPlusEps);
  // negLabels = -labels
  labels->applyTransform(transform::Neg, &negLabels);
  // term1 = -labels * log(predictions + epsilon)
  negLabels.applyPairwiseTransform(pairwise::Multiply, &logPredPlusEps, &term1);
  // oneMinusLabels = 1 - labels
  labels->applyScalar(scalar::ReverseSubtract, 1.0, &oneMinusLabels);
  // onePlusEpsMinusPred = (1 + epsilon) - predictions
  predictions->applyScalar(scalar::ReverseSubtract, (1.0 + epsilon), &onePlusEpsMinusPred);
  // logOnePlusEpsMinusPred = log(1 + epsilon - predictions)
  onePlusEpsMinusPred.applyTransform(transform::Log, &logOnePlusEpsMinusPred);
  // term2 = (1 - labels) * log(1 + epsilon - predictions)
  oneMinusLabels.applyPairwiseTransform(pairwise::Multiply, &logOnePlusEpsMinusPred, &term2);
  // E = term1 - term2
  term1.applyPairwiseTransform(pairwise::Subtract, &term2, &E);

  // multiply E by weights
  {
    NDArray EWeighted(E.shapeInfo(), false, block.launchContext());
    E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);
    E.assign(&EWeighted);
  }

  switch (reductionMode) {
    case 0: {  // 0 - "none", un-reduced weighted losses with the same shape as labels.
      output->assign(&E);
      break;
    }
    case 1: {  // 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
      E.reduceNumber(reduce::Sum, output);
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
      // all elements of weightsBroad array
      double sum;
      if (weights->isScalar()) {
        sum = weights->e<double>(0) * E.lengthOf();
      } else {
        NDArray sumScalar(output->dataType(), ctx);
        weightsBroad->reduceNumber(reduce::Sum, &sumScalar);
        sum = sumScalar.e<double>(0);
      }

      if (sum == 0.) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray eSum(output->dataType(), ctx);
        E.reduceNumber(reduce::Sum, &eSum);
        double eSumVal = eSum.e<double>(0);
        double result = eSumVal / sum;
        output->assign(result);
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
      // array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E.lengthOf();
      } else {
        NDArray countScalar(DataType::INT64, ctx);
        weightsBroad->reduceNumber(reduce::CountNonZero, &countScalar);
        numOfNonZeroWeights = countScalar.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray eSum(output->dataType(), ctx);
        E.reduceNumber(reduce::Sum, &eSum);
        double eSumVal = eSum.e<double>(0);
        double result = eSumVal / double(numOfNonZeroWeights);
        output->assign(result);
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(log_loss) { getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS}); }

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(log_loss) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(
      shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
      "LOG_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "LOG_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i "
               "correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "LOG_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s "
      "instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
  LongType* outShapeInfo = nullptr;

  if (INT_ARG(0) != 0)  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  else {  // in this case output has the same shape as labels and predictions
    outShapeInfo = ConstantShapeHelper::getInstance()
                       .bufferForShapeInfo(outType, shape::order(labelsShapeInfo), shape::rank(labelsShapeInfo),
                                           shape::shapeOf(labelsShapeInfo))
                       ->primary();
  }
  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(log_loss_grad, 3, 3, false, 1, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dpredictions
  auto dLdw = OUTPUT_VARIABLE(1);  // dL/dweights
  auto dLdl = OUTPUT_VARIABLE(2);  // dL/dlabels

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
  if (reductionMode == 0) reductionMode = 1;

  double epsilon = T_ARG(0);

  // input validation
  REQUIRE_TRUE(
      labels->isSameShape(predictions), 0,
      "LOG_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "LOG_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i and "
               "%i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "LOG_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
               "and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "LOG_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!",
      reductionMode);

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(predictions))
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));

  auto ctx = block.launchContext();

  // Stack-allocated intermediates shaped like predictions
  NDArray predictPlusEps(predictions->shapeInfo(), false, ctx);
  NDArray oneMinusLabels(predictions->shapeInfo(), false, ctx);
  NDArray onePlusEpsMinusPredict(predictions->shapeInfo(), false, ctx);

  // predictPlusEps = predictions + epsilon
  predictions->applyScalar(scalar::Add, epsilon, &predictPlusEps);
  // oneMinusLabels = 1 - labels
  labels->applyScalar(scalar::ReverseSubtract, 1.0, &oneMinusLabels);
  // onePlusEpsMinusPredict = (1 + epsilon) - predictions
  predictions->applyScalar(scalar::ReverseSubtract, (1.0 + epsilon), &onePlusEpsMinusPredict);

  // dE_i/dp_i = (1-y_i)/(1-p_i+eps) - y_i/(p_i+eps)
  {
    NDArray oneMinusDiv(predictions->shapeInfo(), false, ctx);
    NDArray labelsDiv(predictions->shapeInfo(), false, ctx);
    oneMinusLabels.applyPairwiseTransform(pairwise::Divide, &onePlusEpsMinusPredict, &oneMinusDiv);
    labels->applyPairwiseTransform(pairwise::Divide, &predictPlusEps, &labelsDiv);
    oneMinusDiv.applyPairwiseTransform(pairwise::Subtract, &labelsDiv, dLdp);
  }

  // dE_i/dy_i = log((1+2eps)/(p_i+eps) - 1)
  {
    double onePlus2Eps = 1. + 2. * epsilon;
    NDArray ratio(predictions->shapeInfo(), false, ctx);
    NDArray ratioMinus1(predictions->shapeInfo(), false, ctx);
    predictPlusEps.applyScalar(scalar::ReverseDivide, onePlus2Eps, &ratio);
    ratio.applyScalar(scalar::Subtract, 1.0, &ratioMinus1);
    ratioMinus1.applyTransform(transform::Log, dLdl);
  }

  // Compute E for gradient calculations
  // E = -labels * log(predictions + epsilon) - (1 - labels) * log(1 + epsilon - predictions)
  NDArray E(predictions->shapeInfo(), false, ctx);
  {
    NDArray logPredPlusEps(predictions->shapeInfo(), false, ctx);
    NDArray logOnePlusEpsMinusPred(predictions->shapeInfo(), false, ctx);
    NDArray negLabels(predictions->shapeInfo(), false, ctx);
    NDArray term1(predictions->shapeInfo(), false, ctx);
    NDArray term2(predictions->shapeInfo(), false, ctx);

    predictPlusEps.applyTransform(transform::Log, &logPredPlusEps);
    onePlusEpsMinusPredict.applyTransform(transform::Log, &logOnePlusEpsMinusPred);
    labels->applyTransform(transform::Neg, &negLabels);
    negLabels.applyPairwiseTransform(pairwise::Multiply, &logPredPlusEps, &term1);
    oneMinusLabels.applyPairwiseTransform(pairwise::Multiply, &logOnePlusEpsMinusPred, &term2);
    term1.applyPairwiseTransform(pairwise::Subtract, &term2, &E);
  }

  // process 3 possible reduction modes below
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
        NDArray eSum(dLdw->dataType(), ctx);
        E.reduceNumber(reduce::Sum, &eSum);
        dLdw->assign(&eSum);
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
        NDArray sumScalar(dLdw->dataType(), ctx);
        weightsBroad->reduceNumber(reduce::Sum, &sumScalar);
        sum = sumScalar.e<double>(0);
      }

      if (sum == 0.) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdl->assign(zero);
        dLdw->assign(zero);
      } else {
        // weightsDivSum = weightsBroad / sum  (weightsBroad shape, may be scalar)
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
          double zero = 0.0;
          dLdw->assign(zero);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());

          NDArray ETimesSum(predictions->shapeInfo(), false, ctx);
          NDArray ETimesWeights(predictions->shapeInfo(), false, ctx);
          NDArray ETimesWeightsSum(dLdw->dataType(), ctx);
          NDArray numerator(predictions->shapeInfo(), false, ctx);
          NDArray result(predictions->shapeInfo(), false, ctx);

          E.applyScalar(scalar::Multiply, sum, &ETimesSum);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &ETimesWeights, false);
          ETimesWeights.reduceNumber(reduce::Sum, &ETimesWeightsSum);
          double eTWSum = ETimesWeightsSum.e<double>(0);
          ETimesSum.applyScalar(scalar::Subtract, eTWSum, &numerator);
          double sumSquared = sum * sum;
          numerator.applyScalar(scalar::Divide, sumSquared, &result);
          result.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else {
          NDArray ETimesSum(predictions->shapeInfo(), false, ctx);
          NDArray ETimesWeights(predictions->shapeInfo(), false, ctx);
          NDArray ETimesWeightsSum(dLdw->dataType(), ctx);
          NDArray numerator(predictions->shapeInfo(), false, ctx);
          NDArray result(predictions->shapeInfo(), false, ctx);

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
        NDArray countScalar(DataType::INT64, ctx);
        weightsBroad->reduceNumber(reduce::CountNonZero, &countScalar);
        numOfNonZeroWeights = countScalar.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdl->assign(zero);
        dLdw->assign(zero);
      } else {
        double numNZW = double(numOfNonZeroWeights);

        if (weights->isScalar()) {
          NDArray eSum(dLdw->dataType(), ctx);
          E.reduceNumber(reduce::Sum, &eSum);
          double eSumVal = eSum.e<double>(0);
          double result = eSumVal / numNZW;
          dLdw->assign(result);

          // Scale prediction and label gradients by weight / numOfNonZeroWeights
          double scaleFactor = weights->e<double>(0) / numNZW;
          dLdp->applyScalar(scalar::Multiply, scaleFactor, dLdp);
          dLdl->applyScalar(scalar::Multiply, scaleFactor, dLdl);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          dLdw->applyScalar(scalar::Divide, numNZW, dLdw);

          // Scale prediction and label gradients by weightsBroad / numOfNonZeroWeights
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
          NDArray EDivNum(predictions->shapeInfo(), false, ctx);
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

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(log_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(log_loss_grad) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(
      shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
      "LOG_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "LOG_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i and "
               "%i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "LOG_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels "
      "= %s instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

  auto dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(predictionsShapeInfo, outType, false, block.getWorkspace());
  auto dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());
  auto dLdlShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

  return SHAPELIST(CONSTANT(dLdpShapeInfo), CONSTANT(dLdwShapeInfo), CONSTANT(dLdlShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
