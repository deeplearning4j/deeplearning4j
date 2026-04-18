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
#if NOT_EXCLUDED(OP_huber_loss)

#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/loss.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(huber_loss, 3, 1, false, 1, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  // FIXME: double?
  double delta = T_ARG(0);

  // input validation
  REQUIRE_TRUE(
      labels->isSameShape(predictions), 0,
      "HUBER_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "HUBER_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i "
               "correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "HUBER_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
               "labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "HUBER_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!",
      reductionMode);

  // perform weights broadcasting/tile to predictions if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(predictions))
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));

  auto ctx = block.launchContext();

  // error = |predictions - labels|
  NDArray error(predictions->shapeInfo(), false, ctx);
  predictions->applyPairwiseTransform(pairwise::Subtract, labels, &error);
  error.applyTransform(transform::Abs, &error);

  // quadratic = min(|error|, delta)
  NDArray quadratic(error.shapeInfo(), false, ctx);
  error.applyScalar(scalar::MinPairwise, delta, &quadratic);

  // scaledQuadratic = 0.5 * quadratic^2
  NDArray quadraticSquared(quadratic.shapeInfo(), false, ctx);
  quadratic.applyPairwiseTransform(pairwise::Multiply, &quadratic, &quadraticSquared);
  NDArray scaledQuadratic(quadraticSquared.shapeInfo(), false, ctx);
  double half = 0.5;
  quadraticSquared.applyScalar(scalar::Multiply, half, &scaledQuadratic);

  // linearTerm = (|error| - quadratic) * delta
  NDArray errorMinusQuadratic(error.shapeInfo(), false, ctx);
  error.applyPairwiseTransform(pairwise::Subtract, &quadratic, &errorMinusQuadratic);
  NDArray linearTerm(errorMinusQuadratic.shapeInfo(), false, ctx);
  errorMinusQuadratic.applyScalar(scalar::Multiply, delta, &linearTerm);

  // E = scaledQuadratic + linearTerm
  NDArray E(scaledQuadratic.shapeInfo(), false, ctx);
  scaledQuadratic.applyPairwiseTransform(pairwise::Add, &linearTerm, &E);

  // EWeighted = E * weightsBroad (broadcast multiply for scalar weights)
  NDArray EWeighted(E.shapeInfo(), false, ctx);
  E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);

  switch (reductionMode) {
    case 0: {  // 0 - "none", un-reduced weighted losses with the same shape as labels.
      output->assign(&EWeighted);
      break;
    }
    case 1: {  // 1 - "weighted_sum", output is scalar and equal to sum of all elements of EWeighted array
      EWeighted.reduceNumber(reduce::Sum, output);
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of EWeighted array divided by
      // sum of all elements of weightsBroad array
      NDArray sum(output->shapeInfo(), false, ctx);
      if (weights->isScalar()) {
        double wVal = weights->e<double>(0) * static_cast<double>(EWeighted.lengthOf());
        sum.assign(wVal);
      } else {
        weightsBroad->reduceNumber(reduce::Sum, &sum);
      }

      double sumVal = sum.e<double>(0);
      if (sumVal == 0.) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray eSum(output->shapeInfo(), false, ctx);
        EWeighted.reduceNumber(reduce::Sum, &eSum);
        double val = eSum.e<double>(0) / sumVal;
        output->assign(val);
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of
      // EWeighted array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = EWeighted.lengthOf();
      } else {
        NDArray countResult(DataType::INT64, ctx);
        weightsBroad->reduceNumber(reduce::CountNonZero, &countResult);
        numOfNonZeroWeights = countResult.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray eSum(output->shapeInfo(), false, ctx);
        EWeighted.reduceNumber(reduce::Sum, &eSum);
        double val = eSum.e<double>(0) / static_cast<double>(numOfNonZeroWeights);
        output->assign(val);
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(huber_loss) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(huber_loss) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(
      shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
      "HUBER_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "HUBER_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i "
               "correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "HUBER_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = "
      "%s instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
  LongType * outShapeInfo = nullptr;

  if (INT_ARG(0) != 0)  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  else {  // in this case output has the same shape as labels and predictions
    outShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(labelsShapeInfo),
                                                                         shape::rank(labelsShapeInfo),
                                                                         shape::shapeOf(labelsShapeInfo))->primary();
  }
  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(huber_loss_grad, 3, 3, false, 1, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dpredictions
  auto dLdw = OUTPUT_VARIABLE(1);  // dL/dweights
  auto dLdl = OUTPUT_VARIABLE(2);  // dL/dlabels

  auto delta = T_ARG(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
  if (reductionMode == 0) reductionMode = 1;

  // inputs validation
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "HUBER_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "HUBER_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i "
               "and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "HUBER_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
               "and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "HUBER_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!",
      reductionMode);

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(predictions))
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));

  auto ctx = block.launchContext();

  // diff = predictions - labels
  NDArray diff(predictions->shapeInfo(), false, ctx);
  predictions->applyPairwiseTransform(pairwise::Subtract, labels, &diff);

  // absDiff = |diff|
  NDArray absDiff(diff.shapeInfo(), false, ctx);
  diff.applyTransform(transform::Abs, &absDiff);

  // quadratic = min(|diff|, delta)
  NDArray quadratic(absDiff.shapeInfo(), false, ctx);
  absDiff.applyScalar(scalar::MinPairwise, delta, &quadratic);

  // scaledQuadratic = 0.5 * quadratic^2
  NDArray quadraticSquared(quadratic.shapeInfo(), false, ctx);
  quadratic.applyPairwiseTransform(pairwise::Multiply, &quadratic, &quadraticSquared);
  NDArray scaledQuadratic(quadraticSquared.shapeInfo(), false, ctx);
  double half = 0.5;
  quadraticSquared.applyScalar(scalar::Multiply, half, &scaledQuadratic);

  // linearTerm = (|diff| - quadratic) * delta
  NDArray absDiffMinusQuadratic(absDiff.shapeInfo(), false, ctx);
  absDiff.applyPairwiseTransform(pairwise::Subtract, &quadratic, &absDiffMinusQuadratic);
  NDArray linearTerm(absDiffMinusQuadratic.shapeInfo(), false, ctx);
  absDiffMinusQuadratic.applyScalar(scalar::Multiply, delta, &linearTerm);

  // E = scaledQuadratic + linearTerm
  NDArray E(scaledQuadratic.shapeInfo(), false, ctx);
  scaledQuadratic.applyPairwiseTransform(pairwise::Add, &linearTerm, &E);

  // Huber gradient: dH/dx = min(|x|, delta) * sign(x) = quadratic * sign(diff)
  // In the quadratic region (|diff| <= delta): quadratic = |diff|, so result = diff
  // In the linear region (|diff| > delta): quadratic = delta, so result = delta * sign(diff)
  NDArray signDiff(diff.shapeInfo(), false, ctx);
  diff.applyTransform(transform::Sign, &signDiff);

  // dLdp = quadratic * sign(diff)
  NDArray dLdpTemp(diff.shapeInfo(), false, ctx);
  quadratic.applyPairwiseTransform(pairwise::Multiply, &signDiff, &dLdpTemp);
  dLdp->assign(&dLdpTemp);

  // dLdl = -dLdp
  NDArray dLdlTemp(diff.shapeInfo(), false, ctx);
  dLdpTemp.applyTransform(transform::Neg, &dLdlTemp);
  dLdl->assign(&dLdlTemp);

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum"
      NDArray dLdpWeighted(dLdp->shapeInfo(), false, ctx);
      dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpWeighted, false);
      dLdp->assign(&dLdpWeighted);

      NDArray dLdlWeighted(dLdl->shapeInfo(), false, ctx);
      dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdlWeighted, false);
      dLdl->assign(&dLdlWeighted);

      if (weights->isScalar()) {
        E.reduceNumber(reduce::Sum, dLdw);
      } else if (weights != weightsBroad) {
        std::vector<LongType> axesToReduceAlong =
            ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
        E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
      } else {
        dLdw->assign(&E);
      }
      break;
    }
    case 2: {  // 2 - "weighted_mean"
      NDArray sum(DataTypeUtils::pickFloatingType(weights->dataType()), ctx);
      if (weights->isScalar()) {
        double wVal = weights->e<double>(0) * static_cast<double>(E.lengthOf());
        sum.assign(wVal);
      } else {
        weightsBroad->reduceNumber(reduce::Sum, &sum);
      }

      double sumVal = sum.e<double>(0);
      if (sumVal == 0.) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdl->assign(zero);
        dLdw->assign(zero);
      } else {
        NDArray weightsDivSum(weightsBroad->shapeInfo(), false, ctx);
        weightsBroad->applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sum, &weightsDivSum, false);

        NDArray dLdpResult(dLdp->shapeInfo(), false, ctx);
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivSum, &dLdpResult, false);
        dLdp->assign(&dLdpResult);

        NDArray dLdlResult(dLdl->shapeInfo(), false, ctx);
        dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivSum, &dLdlResult, false);
        dLdl->assign(&dLdlResult);

        if (weights->isScalar()) {
          double zero = 0.0;
          dLdw->assign(zero);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          // numerator = E * sum - E * weightsBroad (summed)
          NDArray EWeighted(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);
          NDArray EWeightedSum(DataTypeUtils::pickFloatingType(E.dataType()), ctx);
          EWeighted.reduceNumber(reduce::Sum, &EWeightedSum);
          NDArray ESum(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &sum, &ESum, false);
          NDArray numerator(E.shapeInfo(), false, ctx);
          ESum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &EWeightedSum, &numerator, false);
          NDArray sumSquared(DataTypeUtils::pickFloatingType(sum.dataType()), ctx);
          sum.applyPairwiseTransform(pairwise::Multiply, &sum, &sumSquared);
          NDArray gradTemp(E.shapeInfo(), false, ctx);
          numerator.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sumSquared, &gradTemp, false);
          gradTemp.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else {
          NDArray EWeighted(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);
          NDArray EWeightedSum(DataTypeUtils::pickFloatingType(E.dataType()), ctx);
          EWeighted.reduceNumber(reduce::Sum, &EWeightedSum);
          NDArray ESum(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &sum, &ESum, false);
          NDArray numerator(E.shapeInfo(), false, ctx);
          ESum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &EWeightedSum, &numerator, false);
          NDArray sumSquared(DataTypeUtils::pickFloatingType(sum.dataType()), ctx);
          sum.applyPairwiseTransform(pairwise::Multiply, &sum, &sumSquared);
          NDArray gradTemp(E.shapeInfo(), false, ctx);
          numerator.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sumSquared, &gradTemp, false);
          dLdw->assign(&gradTemp);
        }
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights"
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E.lengthOf();
      } else {
        NDArray countResult(DataType::INT64, ctx);
        weightsBroad->reduceNumber(reduce::CountNonZero, &countResult);
        numOfNonZeroWeights = countResult.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdl->assign(zero);
        dLdw->assign(zero);
      } else {
        double numNonZeroD = static_cast<double>(numOfNonZeroWeights);

        if (weights->isScalar()) {
          NDArray sumE(DataTypeUtils::pickFloatingType(E.dataType()), ctx);
          E.reduceNumber(reduce::Sum, &sumE);
          double val = sumE.e<double>(0) / numNonZeroD;
          dLdw->assign(val);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          NDArray dLdwResult(dLdw->shapeInfo(), false, ctx);
          dLdw->applyScalar(scalar::Divide, numNonZeroD, &dLdwResult);
          dLdw->assign(&dLdwResult);
        } else {
          NDArray result(E.shapeInfo(), false, ctx);
          E.applyScalar(scalar::Divide, numNonZeroD, &result);
          dLdw->assign(&result);
        }

        // Scale dLdp and dLdl by weightsBroad / numOfNonZeroWeights
        NDArray temp(weightsBroad->shapeInfo(), false, ctx);
        weightsBroad->applyScalar(scalar::Divide, numNonZeroD, &temp);

        NDArray dLdpResult(dLdp->shapeInfo(), false, ctx);
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdpResult, false);
        dLdp->assign(&dLdpResult);

        NDArray dLdlResult(dLdl->shapeInfo(), false, ctx);
        dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdlResult, false);
        dLdl->assign(&dLdlResult);
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

DECLARE_TYPES(huber_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(huber_loss_grad) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "HUBER_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "HUBER_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i "
               "and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "HUBER_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
      "labels = %s instead!",
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
