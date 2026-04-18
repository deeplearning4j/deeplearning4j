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
// Created by Yurii Shyrma on 20.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_absolute_difference_loss)
#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/loss.h>

namespace sd {
namespace ops {

// SAFETY NOTE: NDArray binary operators (operator*, operator-, etc.) return NDArray*.
// The rvalue (&&) overloads do in-place operations and return the INPUT pointer.
// This means: `NDArray* result = (*ptr) * scalar;` when ptr is dereferenced
// produces an rvalue, and `result == ptr`. Deleting both causes double-free.
//
// SAFE PATTERN: Use applyScalar/applyPairwiseTransform/reduceAlongDimension
// with explicit output arrays. Never use binary operators on heap-allocated arrays.
// All these methods take NDArray* (pointers), not NDArray& (references).

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(absolute_difference_loss, 3, 1, false, 0, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"

  // input validation
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: weights array should be scalar or have the same rank as labels array, but "
               "got %i and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got "
               "weights = %s and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, "
               "but got %i instead!",
               reductionMode);

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  bool ownWeightsBroad = false;
  if (!weights->isScalar() && !weights->isSameShape(predictions)) {
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));
    ownWeightsBroad = true;
  }

  // diff = predictions - labels, then E = |diff|
  NDArray diff(predictions->shapeInfo(), false, block.launchContext());
  predictions->applyPairwiseTransform(pairwise::Subtract, labels, &diff);
  NDArray E(diff.shapeInfo(), false, block.launchContext());
  diff.applyTransform(transform::Abs, &E);

  // EWeighted = E * weightsBroad
  NDArray EWeighted(E.shapeInfo(), false, block.launchContext());
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
      double sumVal = 0.0;
      if (weights->isScalar()) {
        sumVal = weights->e<double>(0) * (double)EWeighted.lengthOf();
      } else {
        NDArray sumArr(output->dataType(), block.launchContext());
        weightsBroad->reduceNumber(reduce::Sum, &sumArr);
        sumVal = sumArr.e<double>(0);
      }

      if (sumVal == 0.) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray eSumArr(output->dataType(), block.launchContext());
        EWeighted.reduceNumber(reduce::Sum, &eSumArr);
        double val = eSumArr.e<double>(0) / sumVal;
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
        NDArray countResultArr(DataType::INT64, block.launchContext());
        weightsBroad->reduceNumber(reduce::CountNonZero, &countResultArr);
        numOfNonZeroWeights = countResultArr.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray eSumArr(output->dataType(), block.launchContext());
        EWeighted.reduceNumber(reduce::Sum, &eSumArr);
        double val = eSumArr.e<double>(0) / (double)numOfNonZeroWeights;
        output->assign(val);
      }
      break;
    }
  }

  if (ownWeightsBroad) delete weightsBroad;

  return Status::OK;
}

DECLARE_TYPES(absolute_difference_loss) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(absolute_difference_loss) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);



  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: weights array should be scalar or have the same rank as labels array, but "
               "got %i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "ABSOLUTE_DIFFERENCE_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
      "and labels = %s instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());


  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
  LongType * outShapeInfo = nullptr;

  if (INT_ARG(0) != 0) {  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  } else {  // in this case output has the same shape as labels and predictions
    outShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(labelsShapeInfo),
                                                                         shape::rank(labelsShapeInfo),
                                                                         shape::shapeOf(labelsShapeInfo))->primary();
  }

  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(absolute_difference_loss_grad, 3, 3, false, 0, 1) {
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
               "ABSOLUTE_DIFFERENCE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: weights array should be scalar or have the same rank as labels array, but "
               "got %i and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got "
               "weights = %s and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, "
               "but got %i instead!",
               reductionMode);

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  bool ownWeightsBroad = false;
  if (!weights->isScalar() && !weights->isSameShape(predictions)) {
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));
    ownWeightsBroad = true;
  }

  // E = predictions - labels (used for sign and abs)
  NDArray E(predictions->shapeInfo(), false, block.launchContext());
  predictions->applyPairwiseTransform(pairwise::Subtract, labels, &E);

  // dE_i/dp_i = sign(p_i - y_i)
  E.applyTransform(transform::Sign, dLdp);  // dE/dp

  // E = |predictions - labels|
  E.applyTransform(transform::Abs, &E);

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

      // dLdp = sign(p - y) * weightsBroad
      NDArray dLdpWeighted(dLdp->shapeInfo(), false, block.launchContext());
      dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpWeighted, false);
      dLdp->assign(&dLdpWeighted);

      if (weights->isScalar()) {
        NDArray sumE(dLdw->dataType(), block.launchContext());
        E.reduceNumber(reduce::Sum, &sumE);
        dLdw->assign(&sumE);
      } else if (weights != weightsBroad) {
        std::vector<LongType> axesToReduceAlong =
            ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
        E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
      } else
        dLdw->assign(&E);
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
      // all elements of weightsBroad array

      double sumVal = 0.0;
      if (weights->isScalar()) {
        sumVal = weights->e<double>(0) * (double)E.lengthOf();
      } else {
        NDArray sumArr(dLdp->dataType(), block.launchContext());
        weightsBroad->reduceNumber(reduce::Sum, &sumArr);
        sumVal = sumArr.e<double>(0);
      }

      if (sumVal == 0.) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdw->assign(zero);
      } else {
        // dLdp = sign(p - y) * weightsBroad / sum
        {
          NDArray dLdpWeighted(dLdp->shapeInfo(), false, block.launchContext());
          dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpWeighted, false);
          dLdpWeighted.applyScalar(scalar::Divide, sumVal, dLdp);
        }

        if (weights->isScalar()) {
          double zero = 0.0;
          dLdw->assign(zero);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());

          // numerator = E * sum - E * weightsBroad (summed) == E * (sum - weightsBroadSum)
          // Full formula: dL/dw_i = (sum * E_i - EWeightedTotal) / sum^2, reduced over broadcast axes
          NDArray ETimesSum(E.shapeInfo(), false, block.launchContext());
          E.applyScalar(scalar::Multiply, sumVal, &ETimesSum);

          NDArray ETimesWeights(E.shapeInfo(), false, block.launchContext());
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &ETimesWeights, false);

          NDArray EWeightedSum(dLdw->dataType(), block.launchContext());
          ETimesWeights.reduceNumber(reduce::Sum, &EWeightedSum);

          NDArray numerator(ETimesSum.shapeInfo(), false, block.launchContext());
          ETimesSum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &EWeightedSum, &numerator, false);

          NDArray gradTemp(numerator.shapeInfo(), false, block.launchContext());
          numerator.applyScalar(scalar::Divide, sumVal * sumVal, &gradTemp);

          gradTemp.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else {
          // weights == weightsBroad (no broadcast needed)
          NDArray ETimesSum(E.shapeInfo(), false, block.launchContext());
          E.applyScalar(scalar::Multiply, sumVal, &ETimesSum);

          NDArray ETimesWeights(E.shapeInfo(), false, block.launchContext());
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &ETimesWeights, false);

          NDArray EWeightedSum(dLdw->dataType(), block.launchContext());
          ETimesWeights.reduceNumber(reduce::Sum, &EWeightedSum);

          NDArray numerator(ETimesSum.shapeInfo(), false, block.launchContext());
          ETimesSum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &EWeightedSum, &numerator, false);

          NDArray gradTemp(numerator.shapeInfo(), false, block.launchContext());
          numerator.applyScalar(scalar::Divide, sumVal * sumVal, &gradTemp);

          dLdw->assign(&gradTemp);
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
        NDArray countResultArr(DataType::INT64, block.launchContext());
        weightsBroad->reduceNumber(reduce::CountNonZero, &countResultArr);
        numOfNonZeroWeights = countResultArr.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdw->assign(zero);
      } else {
        double numNZW = (double)numOfNonZeroWeights;

        if (weights->isScalar()) {
          NDArray eSumArr(dLdw->dataType(), block.launchContext());
          E.reduceNumber(reduce::Sum, &eSumArr);
          double val = eSumArr.e<double>(0) / numNZW;
          dLdw->assign(val);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          NDArray dLdwDivResult(dLdw->shapeInfo(), false, block.launchContext());
          dLdw->applyScalar(scalar::Divide, numNZW, &dLdwDivResult);
          dLdw->assign(&dLdwDivResult);
        } else {
          E.applyScalar(scalar::Divide, numNZW, dLdw);
        }

        // dLdp = sign(p - y) * weightsBroad / numNZW
        NDArray weightsDivNum(weightsBroad->shapeInfo(), false, block.launchContext());
        weightsBroad->applyScalar(scalar::Divide, numNZW, &weightsDivNum);
        NDArray dLdpResult(dLdp->shapeInfo(), false, block.launchContext());
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivNum, &dLdpResult, false);
        dLdp->assign(&dLdpResult);
      }

      break;
    }
  }

  // dL/dlabels = -dL/dpredictions
  dLdp->applyTransform(transform::Neg, dLdl);

  if (ownWeightsBroad) delete weightsBroad;

  return Status::OK;
}

DECLARE_TYPES(absolute_difference_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(absolute_difference_loss_grad) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "ABSOLUTE_DIFFERENCE_LOSS OP: weights array should be scalar or have the same rank as labels array, but "
               "got %i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "ABSOLUTE_DIFFERENCE_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
      "and labels = %s instead!",
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
