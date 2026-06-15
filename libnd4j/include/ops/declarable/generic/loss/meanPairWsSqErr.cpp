#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"

/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 24.11.2017
// @author Paul Dubs
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_mean_pairwssqerr_loss)

#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/loss.h>

#include <numeric>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// SAFETY NOTE: NDArray binary operators (operator*, operator-, etc.) return NDArray*.
// The rvalue (&&) overloads do in-place operations and return the INPUT pointer.
// This means: `NDArray* result = (*ptr) * scalar;` when ptr is dereferenced
// produces an rvalue, and `result == ptr`. Deleting both causes double-free.
//
// SAFE PATTERN: Use applyScalar/applyPairwiseTransform/reduceAlongDimension
// with explicit output arrays. Never use binary operators on heap-allocated arrays.
// All these methods take NDArray* (pointers), not NDArray& (references).

CUSTOM_OP_IMPL(mean_pairwssqerr_loss, 3, 1, false, 0, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"

  // input validation
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "MEAN_PAIRWSSQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "MEAN_PAIRWSSQERR_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but "
               "got %i instead!",
               reductionMode);

  if (labels->rankOf() == 1) {
    double zero = 0.0;
    output->assign(zero);
    return Status::OK;
  }

  std::vector<LongType> zero = {0};
  auto reductionIdx = ShapeUtils::evalDimsToExclude(labels->rankOf(), 1, zero.data());

  auto n = double(labels->sizeAt(1));

  // diffs = predictions - labels (stack-allocated, we own it)
  NDArray diffs(predictions->shapeInfo(), false, block.launchContext());
  predictions->applyPairwiseTransform(pairwise::Subtract, labels, &diffs);

  // diffsSquared = diffs * diffs
  NDArray diffsSquared(diffs.shapeInfo(), false, block.launchContext());
  diffs.applyPairwiseTransform(pairwise::Multiply, &diffs, &diffsSquared);

  // sumOfSquares = sum(diffsSquared, reductionIdx, keepDims=true)
  auto sumOfSquaresPtr = diffsSquared.reduceAlongDimension(reduce::Sum, reductionIdx, true);

  // squareOfSum = (sum(diffs, reductionIdx, keepDims=true))^2
  auto squareOfSumPtr = diffs.reduceAlongDimension(reduce::Sum, reductionIdx, true);
  squareOfSumPtr->applyScalar(scalar::Pow, 2, squareOfSumPtr);

  delete reductionIdx;

  // E = ((sumOfSquares * n) - squareOfSum) * (4 / (n * (n - 1)))
  // Step 1: sumOfSquaresTimesN = sumOfSquares * n
  NDArray sumOfSquaresTimesN(sumOfSquaresPtr->shapeInfo(), false, block.launchContext());
  sumOfSquaresPtr->applyScalar(scalar::Multiply, n, &sumOfSquaresTimesN);
  delete sumOfSquaresPtr;

  // Step 2: numerator = sumOfSquaresTimesN - squareOfSum
  NDArray numerator(sumOfSquaresTimesN.shapeInfo(), false, block.launchContext());
  sumOfSquaresTimesN.applyPairwiseTransform(pairwise::Subtract, squareOfSumPtr, &numerator);
  delete squareOfSumPtr;

  // Step 3: E = numerator * (4 / (n * (n - 1)))
  NDArray E(numerator.shapeInfo(), false, block.launchContext());
  numerator.applyScalar(scalar::Multiply, 4.0 / (n * (n - 1.0)), &E);

  // weights validation
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == E.rankOf(), 0,
               "MEAN_PAIRWSSQERR_LOSS OP: weights array should be scalar or have the same rank as results array, "
               "but got %i and %i correspondingly!",
               weights->rankOf(), E.rankOf());
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, E), 0,
               "MEAN_PAIRWSSQERR_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got "
               "weights = %s and results = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(&E).c_str());

  // perform weights broadcasting/tile to labels if needed
  NDArray* weightsBroad = weights;
  bool ownWeightsBroad = false;
  if (!weights->isScalar() && !weights->isSameShape(E)) {
    weightsBroad = new NDArray(weights->tileToShape(E.shapeInfo()));
    ownWeightsBroad = true;
  }

  {
    NDArray EWeighted(E.shapeInfo(), false, block.launchContext());
    E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);
    E.assign(&EWeighted);
  }

  switch (reductionMode) {
    case 0:  // "none"
      output->assign(&E);
      break;

    case 1: {  // "weighted_sum"
      E.reduceNumber(reduce::Sum, output);
      break;
    }
    case 2: {  // "weighted_mean"
      double sumVal = 0.0;
      if (weights->isScalar()) {
        sumVal = weights->e<double>(0) * (double)E.lengthOf();
      } else {
        NDArray sumArr(output->dataType(), block.launchContext());
        weightsBroad->reduceNumber(reduce::Sum, &sumArr);
        sumVal = sumArr.e<double>(0);
      }

      if (sumVal == 0.) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray eSum(output->dataType(), block.launchContext());
        E.reduceNumber(reduce::Sum, &eSum);
        double val = eSum.e<double>(0) / sumVal;
        output->assign(val);
      }
      break;
    }
    case 3: {  // "weighted_sum_by_nonzero_weights"
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E.lengthOf();
      } else {
        NDArray countNonZero(DataType::INT64, block.launchContext());
        weightsBroad->reduceNumber(reduce::CountNonZero, &countNonZero);
        numOfNonZeroWeights = countNonZero.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray eSum(output->dataType(), block.launchContext());
        E.reduceNumber(reduce::Sum, &eSum);
        double val = eSum.e<double>(0) / (double)numOfNonZeroWeights;
        output->assign(val);
      }
      break;
    }
  }

  if (ownWeightsBroad) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(mean_pairwssqerr_loss) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(mean_pairwssqerr_loss) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "MEAN_PAIRWSSQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
  LongType  *outShapeInfo = nullptr;

  if (INT_ARG(0) != 0)  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  else {  // in this case output has the shape as labels and logits minus last dimension
    std::vector<LongType> dimensions = {-1};
    outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(predictionsShapeInfo), &dimensions, predictionsShapeInfo,
                                                   false, true, block.getWorkspace());

    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(outShapeInfo), 0,
                 "MEAN_PAIRWSSQERR_LOSS OP: weights array should be scalar or have the same rank as output array, but "
                 "got %i and %i correspondingly!",
                 shape::rank(weightsShapeInfo), shape::rank(outShapeInfo));
    REQUIRE_TRUE(
        shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, outShapeInfo), 0,
        "MEAN_PAIRWSSQERR_LOSS OP: shapes of weights and output arrays should be broadcastable, but got weights = %s "
        "and output = %s instead!",
        ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(outShapeInfo).c_str());
  }

  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(mean_pairwssqerr_loss_grad, 3, 3, false, 0, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dpredictions
  auto dLdw = OUTPUT_VARIABLE(1);  // dL/dweights
  auto dLdl = OUTPUT_VARIABLE(2);  // dL/dlabels

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  if (reductionMode == 0) reductionMode = 1;

  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "MEAN_PAIRWSSQERR_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "MEAN_PAIRWSSQERR_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, "
               "but got %i instead!",
               reductionMode);

  auto n = double(labels->sizeAt(1));

  // diffs = predictions - labels
  NDArray diffs(predictions->shapeInfo(), false, block.launchContext());
  predictions->applyPairwiseTransform(pairwise::Subtract, labels, &diffs);

  std::vector<LongType> dims2 = {0};
  auto reductionIdx = ShapeUtils::evalDimsToExclude(labels->rankOf(), 1, dims2.data());

  // diffsSquared = diffs * diffs
  NDArray diffsSquared(diffs.shapeInfo(), false, block.launchContext());
  diffs.applyPairwiseTransform(pairwise::Multiply, &diffs, &diffsSquared);

  // sumOfSquares
  auto sumOfSquaresPtr = diffsSquared.reduceAlongDimension(reduce::Sum, reductionIdx, true);

  // squareOfSum = (sum(diffs))^2
  auto squareOfSumPtr = diffs.reduceAlongDimension(reduce::Sum, reductionIdx, true);
  squareOfSumPtr->applyScalar(scalar::Pow, 2, squareOfSumPtr);

  // E = ((sumOfSquares * n) - squareOfSum) * (4 / (n * (n - 1)))
  NDArray sumOfSquaresTimesN(sumOfSquaresPtr->shapeInfo(), false, block.launchContext());
  sumOfSquaresPtr->applyScalar(scalar::Multiply, n, &sumOfSquaresTimesN);
  delete sumOfSquaresPtr;

  NDArray eTerm1(sumOfSquaresTimesN.shapeInfo(), false, block.launchContext());
  sumOfSquaresTimesN.applyPairwiseTransform(pairwise::Subtract, squareOfSumPtr, &eTerm1);
  delete squareOfSumPtr;

  NDArray E(eTerm1.shapeInfo(), false, block.launchContext());
  eTerm1.applyScalar(scalar::Multiply, 4.0 / (n * (n - 1.0)), &E);

  // Compute gradients for predictions
  auto sumPredPtr = predictions->reduceAlongDimension(reduce::Sum, reductionIdx, true);
  auto sumLabelPtr = labels->reduceAlongDimension(reduce::Sum, reductionIdx, true);

  delete reductionIdx;

  // dLdpTemp = ((diffs * n) - sumDiff) * (8 / (n * (n - 1)))
  // where sumDiff = sumPred - sumLabel (broadcast [batch,1] -> [batch,features])
  // Use applyTrueBroadcast since sumPredPtr/sumLabelPtr are [batch,1] and diffs is [batch,features]
  NDArray diffsTimesN(diffs.shapeInfo(), false, block.launchContext());
  diffs.applyScalar(scalar::Multiply, n, &diffsTimesN);

  // sumDiff = sumPred - sumLabel  (both [batch,1])
  NDArray sumDiff(sumPredPtr->shapeInfo(), false, block.launchContext());
  sumPredPtr->applyPairwiseTransform(pairwise::Subtract, sumLabelPtr, &sumDiff);
  delete sumPredPtr;
  delete sumLabelPtr;

  // term = diffsTimesN - sumDiff (broadcast sumDiff from [batch,1] to [batch,features])
  NDArray term2(diffsTimesN.shapeInfo(), false, block.launchContext());
  diffsTimesN.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &sumDiff, &term2, false);

  term2.applyScalar(scalar::Multiply, 8.0 / (n * (n - 1.0)), dLdp);

  // weights validation
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == E.rankOf(), 0,
               "MEAN_PAIRWSSQERR_LOSS_GRAD OP: weights array should be scalar or have the same rank as results array, "
               "but got %i and %i correspondingly!",
               weights->rankOf(), E.rankOf());
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, E), 0,
               "MEAN_PAIRWSSQERR_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got "
               "weights = %s and results = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(&E).c_str());

  // perform weights broadcasting/tile to labels if needed
  NDArray* weightsBroad = weights;
  bool ownWeightsBroad = false;
  if (!weights->isScalar() && !weights->isSameShape(E)) {
    weightsBroad = new NDArray(weights->tileToShape(E.shapeInfo()));
    ownWeightsBroad = true;
  }

  switch (reductionMode) {
    case 1: {  // "none" and "weighted_sum"
      {
        NDArray dLdpW(dLdp->shapeInfo(), false, block.launchContext());
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpW, false);
        dLdp->assign(&dLdpW);
      }

      if (weights->isScalar()) {
        NDArray eSum(dLdw->dataType(), block.launchContext());
        E.reduceNumber(reduce::Sum, &eSum);
        dLdw->assign(&eSum);
      } else if (weights != weightsBroad) {
        auto axesToReduceAlong =
            ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
        E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
      } else
        dLdw->assign(&E);
      break;
    }
    case 2: {  // "weighted_mean"
      double sumVal = 0.0;
      if (weights->isScalar()) {
        sumVal = weights->e<double>(0) * (double)E.lengthOf();
      } else {
        NDArray sumArr(dLdw->dataType(), block.launchContext());
        weightsBroad->reduceNumber(reduce::Sum, &sumArr);
        sumVal = sumArr.e<double>(0);
      }

      if (sumVal == 0.) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdw->assign(zero);
      } else {
        // dLdp *= weightsBroad / sum
        NDArray weightsDivSum(weightsBroad->shapeInfo(), false, block.launchContext());
        weightsBroad->applyScalar(scalar::Divide, sumVal, &weightsDivSum);
        NDArray dLdpR(dLdp->shapeInfo(), false, block.launchContext());
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivSum, &dLdpR, false);
        dLdp->assign(&dLdpR);

        if (weights->isScalar()) {
          double zero = 0.0;
          dLdw->assign(zero);
        } else if (weights != weightsBroad) {
          auto axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());

          NDArray ETimesSum(E.shapeInfo(), false, block.launchContext());
          E.applyScalar(scalar::Multiply, sumVal, &ETimesSum);

          NDArray ETimesWeights(E.shapeInfo(), false, block.launchContext());
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &ETimesWeights, false);

          NDArray ETimesWeightsSum(dLdw->dataType(), block.launchContext());
          ETimesWeights.reduceNumber(reduce::Sum, &ETimesWeightsSum);

          NDArray num(ETimesSum.shapeInfo(), false, block.launchContext());
          ETimesSum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &ETimesWeightsSum, &num, false);

          NDArray result(num.shapeInfo(), false, block.launchContext());
          num.applyScalar(scalar::Divide, sumVal * sumVal, &result);

          result.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else {
          NDArray ETimesSum(E.shapeInfo(), false, block.launchContext());
          E.applyScalar(scalar::Multiply, sumVal, &ETimesSum);

          NDArray ETimesWeights(E.shapeInfo(), false, block.launchContext());
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &ETimesWeights, false);

          NDArray ETimesWeightsSum(dLdw->dataType(), block.launchContext());
          ETimesWeights.reduceNumber(reduce::Sum, &ETimesWeightsSum);

          NDArray num(ETimesSum.shapeInfo(), false, block.launchContext());
          ETimesSum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &ETimesWeightsSum, &num, false);

          NDArray result(num.shapeInfo(), false, block.launchContext());
          num.applyScalar(scalar::Divide, sumVal * sumVal, &result);

          dLdw->assign(&result);
        }
      }
      break;
    }
    case 3: {  // "weighted_sum_by_nonzero_weights"
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E.lengthOf();
      } else {
        NDArray countNonZero(DataType::INT64, block.launchContext());
        weightsBroad->reduceNumber(reduce::CountNonZero, &countNonZero);
        numOfNonZeroWeights = countNonZero.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdw->assign(zero);
      } else {
        double numNZW = (double)numOfNonZeroWeights;

        if (weights->isScalar()) {
          NDArray eSum(dLdw->dataType(), block.launchContext());
          E.reduceNumber(reduce::Sum, &eSum);
          double val = eSum.e<double>(0) / numNZW;
          dLdw->assign(val);
        } else if (weights != weightsBroad) {
          auto axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          NDArray dLdwR(dLdw->shapeInfo(), false, block.launchContext());
          dLdw->applyScalar(scalar::Divide, numNZW, &dLdwR);
          dLdw->assign(&dLdwR);
        } else {
          E.applyScalar(scalar::Divide, numNZW, dLdw);
        }

        NDArray weightsDivNum(weightsBroad->shapeInfo(), false, block.launchContext());
        weightsBroad->applyScalar(scalar::Divide, numNZW, &weightsDivNum);
        NDArray dLdpR(dLdp->shapeInfo(), false, block.launchContext());
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivNum, &dLdpR, false);
        dLdp->assign(&dLdpR);
      }
      break;
    }
  }

  // dLdl = -dLdp
  dLdp->applyTransform(transform::Neg, dLdl);

  if (ownWeightsBroad) delete weightsBroad;

  return Status::OK;
}

DECLARE_TYPES(mean_pairwssqerr_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(mean_pairwssqerr_loss_grad) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "MEAN_PAIRWSSQERR_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "MEAN_PAIRWSSQERR_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, "
               "but got %i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "MEAN_PAIRWSSQERR_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = "
      "%s and labels = %s instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

  LongType *dLdpShapeInfo =
      ShapeBuilders::copyShapeInfoAndType(predictionsShapeInfo, outType, false, block.getWorkspace());
  LongType *dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());
  LongType *dLdlShapeInfo =
      ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

  return SHAPELIST(dLdpShapeInfo, dLdwShapeInfo, dLdlShapeInfo);
}
}  // namespace ops
}  // namespace sd

#endif
#pragma clang diagnostic pop
