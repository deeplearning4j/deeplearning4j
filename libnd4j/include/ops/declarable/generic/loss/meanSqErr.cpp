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
#if NOT_EXCLUDED(OP_mean_sqerr_loss)

#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/loss.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(mean_sqerr_loss, 3, 1, false, 0, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"

  // inputs validation
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "MEAN_SQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "MEAN_SQERR_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i "
               "and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "MEAN_SQERR_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
               "and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "MEAN_SQERR_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!",
      reductionMode);

  auto ctx = block.launchContext();

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(predictions))
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));

  // E = (predictions - labels)^2
  NDArray E(labels->shapeInfo(), false, ctx);
  predictions->applyPairwiseTransform(pairwise::SquaredSubtract, labels, &E);

  // EWeighted = E * weightsBroad
  NDArray EWeighted(E.shapeInfo(), false, ctx);
  E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);

  switch (reductionMode) {
    case 0:  // 0 - "none", un-reduced weighted losses with the same shape as labels.
      output->assign(&EWeighted);
      break;

    case 1: {  // 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
      EWeighted.reduceNumber(reduce::Sum, output);
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
      // all elements of weightsBroad array
      DataType floatType = DataTypeUtils::pickFloatingType(predictions->dataType());
      NDArray sum(floatType, ctx);
      if (weights->isScalar()) {
        double wVal = weights->e<double>(0);
        double sVal = wVal * static_cast<double>(EWeighted.lengthOf());
        sum.assign(sVal);
      } else {
        weightsBroad->reduceNumber(reduce::Sum, &sum);
      }

      if (sum.e<double>(0) == 0.) {
        double zeroVal = 0.;
        output->assign(zeroVal);
      } else {
        NDArray eSum(floatType, ctx);
        EWeighted.reduceNumber(reduce::Sum, &eSum);
        double val = eSum.e<double>(0) / sum.e<double>(0);
        output->assign(val);
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
      // array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = EWeighted.lengthOf();
      } else {
        NDArray countResult(DataType::INT64, ctx);
        weightsBroad->reduceNumber(reduce::CountNonZero, &countResult);
        numOfNonZeroWeights = countResult.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zeroVal = 0.;
        output->assign(zeroVal);
      } else {
        DataType floatType = DataTypeUtils::pickFloatingType(predictions->dataType());
        NDArray eSum(floatType, ctx);
        EWeighted.reduceNumber(reduce::Sum, &eSum);
        double val = eSum.e<double>(0) / static_cast<double>(numOfNonZeroWeights);
        output->assign(val);
      }
      break;
    }
  }

  STORE_RESULT(*output);

  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

DECLARE_TYPES(mean_sqerr_loss) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(mean_sqerr_loss) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "MEAN_SQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "MEAN_SQERR_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i "
               "and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "MEAN_SQERR_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
      "labels = %s instead!",
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
CUSTOM_OP_IMPL(mean_sqerr_loss_grad, 3, 3, false, 0, 1) {
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

  // inputs validation
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "MEAN_SQERR_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "MEAN_SQERR_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got "
               "%i and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "MEAN_SQERR_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights "
               "= %s and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "MEAN_SQERR_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but "
               "got %i instead!",
               reductionMode);

  auto ctx = block.launchContext();
  DataType floatType = DataTypeUtils::pickFloatingType(predictions->dataType());

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(predictions))
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));

  // diff = predictions - labels
  NDArray diff(predictions->shapeInfo(), false, ctx);
  predictions->applyPairwiseTransform(pairwise::Subtract, labels, &diff);

  // dE_i/dp_i = 2 * (p_i - y_i)
  // dLdp = diff * 2
  diff.applyScalar(scalar::Multiply, static_cast<double>(2.), dLdp);

  // E = diff * diff  (squared difference, for dLdw)
  NDArray E(diff.shapeInfo(), false, ctx);
  diff.applyPairwiseTransform(pairwise::Multiply, &diff, &E);

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

      // dLdp = dLdp * weightsBroad
      NDArray dLdpWeighted(dLdp->shapeInfo(), false, ctx);
      dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpWeighted, false);
      dLdp->assign(&dLdpWeighted);

      if (weights->isScalar()) {
        NDArray sumE(floatType, ctx);
        E.reduceNumber(reduce::Sum, &sumE);
        dLdw->assign(&sumE);
      }
      else if (weights != weightsBroad) {
        std::vector<LongType> axesToReduceAlong =
            ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
        E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
      }
      else {
        dLdw->assign(&E);
      }
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
      // all elements of weightsBroad array

      NDArray sum(floatType, ctx);
      if (weights->isScalar()) {
        double wVal = weights->e<double>(0);
        double sVal = wVal * static_cast<double>(E.lengthOf());
        sum.assign(sVal);
      } else {
        weightsBroad->reduceNumber(reduce::Sum, &sum);
      }

      if (sum.e<double>(0) == 0.) {
        double zeroVal = 0.;
        dLdp->assign(zeroVal);
        dLdw->assign(zeroVal);
      } else {
        // weightsDivSum = weightsBroad / sum
        NDArray weightsDivSum(weightsBroad->shapeInfo(), false, ctx);
        double sumVal = sum.e<double>(0);
        weightsBroad->applyScalar(scalar::Divide, sumVal, &weightsDivSum);

        // dLdp = dLdp * weightsDivSum
        NDArray dLdpResult(dLdp->shapeInfo(), false, ctx);
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivSum, &dLdpResult, false);
        dLdp->assign(&dLdpResult);

        if (weights->isScalar()) {
          double zeroVal = 0.;
          dLdw->assign(zeroVal);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());

          // EWeighted = E * weightsBroad
          NDArray EWeighted(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);

          // EWeightedSum = sum(EWeighted)
          NDArray EWeightedSum(floatType, ctx);
          EWeighted.reduceNumber(reduce::Sum, &EWeightedSum);

          // numerator = E * sum - EWeightedSum  =>  E*sumVal - EWeightedSum_scalar
          double eWeightedSumVal = EWeightedSum.e<double>(0);
          // ESum = E * sumVal
          NDArray ESum(E.shapeInfo(), false, ctx);
          E.applyScalar(scalar::Multiply, sumVal, &ESum);
          // numerator = ESum - EWeightedSum (scalar broadcast)
          NDArray numerator(ESum.shapeInfo(), false, ctx);
          ESum.applyScalar(scalar::Subtract, eWeightedSumVal, &numerator);

          // gradTemp = numerator / (sum * sum)
          double sumSq = sumVal * sumVal;
          NDArray gradTemp(numerator.shapeInfo(), false, ctx);
          numerator.applyScalar(scalar::Divide, sumSq, &gradTemp);

          gradTemp.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        }
        else {
          // EWeighted = E * weightsBroad
          NDArray EWeighted(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);

          // EWeightedSum = sum(EWeighted)
          NDArray EWeightedSum(floatType, ctx);
          EWeighted.reduceNumber(reduce::Sum, &EWeightedSum);
          double eWeightedSumVal = EWeightedSum.e<double>(0);

          // ESum = E * sumVal
          NDArray ESum(E.shapeInfo(), false, ctx);
          E.applyScalar(scalar::Multiply, sumVal, &ESum);

          // numerator = ESum - eWeightedSumVal
          NDArray numerator(ESum.shapeInfo(), false, ctx);
          ESum.applyScalar(scalar::Subtract, eWeightedSumVal, &numerator);

          // dLdwTemp = numerator / (sumVal * sumVal)
          double sumSq = sumVal * sumVal;
          NDArray dLdwTemp(numerator.shapeInfo(), false, ctx);
          numerator.applyScalar(scalar::Divide, sumSq, &dLdwTemp);

          dLdw->assign(&dLdwTemp);
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
        dLdw->assign(zeroVal);
      } else {
        double nzwDouble = static_cast<double>(numOfNonZeroWeights);

        if (weights->isScalar()) {
          NDArray sumE(floatType, ctx);
          E.reduceNumber(reduce::Sum, &sumE);
          double val = sumE.e<double>(0) / nzwDouble;
          dLdw->assign(val);
        }
        else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          // dLdw /= numOfNonZeroWeights
          NDArray dLdwResult(dLdw->shapeInfo(), false, ctx);
          dLdw->applyScalar(scalar::Divide, nzwDouble, &dLdwResult);
          dLdw->assign(&dLdwResult);
        }
        else {
          // dLdwTemp = E / numOfNonZeroWeights
          NDArray dLdwTemp(E.shapeInfo(), false, ctx);
          E.applyScalar(scalar::Divide, nzwDouble, &dLdwTemp);
          dLdw->assign(&dLdwTemp);
        }

        // dLdp = dLdp * (weightsBroad / numOfNonZeroWeights)
        NDArray temp(weightsBroad->shapeInfo(), false, ctx);
        weightsBroad->applyScalar(scalar::Divide, nzwDouble, &temp);
        NDArray dLdpResult(dLdp->shapeInfo(), false, ctx);
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdpResult, false);
        dLdp->assign(&dLdpResult);
      }
      break;
    }
  }

  // dLdl = -dLdp
  dLdp->applyTransform(transform::Neg, dLdl);

  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

DECLARE_TYPES(mean_sqerr_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(mean_sqerr_loss_grad) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "MEAN_SQERR_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "MEAN_SQERR_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got "
               "%i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "MEAN_SQERR_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
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
