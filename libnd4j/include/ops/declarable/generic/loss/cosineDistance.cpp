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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 22.11.2017
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_cosine_distance_loss)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/headers/loss.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(cosine_distance_loss, 3, 1, false, 0, 2) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);        // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  int dim = INT_ARG(1);  // axis along which sum will be made
  if (dim < 0) dim += labels->rankOf();

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "COSINE_DISTANCE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // regard 4 possible reduction modes below
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "COSINE_DISTANCE_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but "
               "got %i instead!",
               reductionMode);
  // input dimension can't be larger than labels/predictions/weights rank
  REQUIRE_TRUE(dim < labels->rankOf(), 0,
               "COSINE_DISTANCE_LOSS OP: input reduction dimension (got %i) must be < labels rank %i!", dim,
               labels->rankOf());

  if (!output->isScalar()) {
    // weights array can be single scalar or has the same shape as output, and must be broadcastable to output shape
    REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == output->rankOf(), 0,
                 "SOFTMAX_CROSS_ENTROPY_LOSS OP: weights array should be scalar or have the same rank as output array, "
                 "but got %i and %i correspondingly!",
                 weights->rankOf(), output->rankOf());
    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *output), 0,
                 "COSINE_DISTANCE_LOSS OP: shapes of weights and output arrays should be broadcastable, but got "
                 "weights = %s and output = %s instead!",
                 ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  }

  std::vector<LongType> dims;
  dims.push_back(dim);

  // Compute element-wise product of predictions and labels into a temporary array
  NDArray predLabels(predictions->shapeInfo(), false, block.launchContext());
  predictions->applyPairwiseTransform(pairwise::Multiply, labels, &predLabels);

  // Reduce along dim (keepDims=true) — returns heap-allocated result, caller must delete
  NDArray* dotProduct = predLabels.reduceAlongDimension(reduce::Sum, &dims, true);

  // E = 1 - dotProduct  (scalar::ReverseSubtract computes scalar - array)
  NDArray* E = new NDArray(dotProduct->shapeInfo(), false, block.launchContext());
  double one = 1.0;
  dotProduct->applyScalar(scalar::ReverseSubtract, one, E);
  delete dotProduct;

  // perform weights broadcasting/tile to E if it is necessary
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(E))
    weightsBroad = new NDArray(weights->tileToShape(E->shapeInfo()));

  // EWeighted = E * weightsBroad
  NDArray EWeighted(E->shapeInfo(), false, block.launchContext());
  E->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);

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
      if (weights->isScalar()) {
        // sum = weights * EWeighted.lengthOf()
        NDArray sum(output->dataType(), block.launchContext());
        double len = static_cast<double>(EWeighted.lengthOf());
        weights->applyScalar(scalar::Multiply, len, &sum);

        if (sum.e<double>(0) == 0.) {
          double zeroVal = 0.;
          output->assign(zeroVal);
        } else {
          NDArray sumE(output->dataType(), block.launchContext());
          EWeighted.reduceNumber(reduce::Sum, &sumE);
          NDArray result(output->dataType(), block.launchContext());
          sumE.applyPairwiseTransform(pairwise::Divide, &sum, &result);
          output->assign(&result);
        }
      } else {
        NDArray sum(output->dataType(), block.launchContext());
        weightsBroad->reduceNumber(reduce::Sum, &sum);

        if (sum.e<double>(0) == 0.) {
          double zeroVal = 0.;
          output->assign(zeroVal);
        } else {
          NDArray sumE(output->dataType(), block.launchContext());
          EWeighted.reduceNumber(reduce::Sum, &sumE);
          NDArray result(output->dataType(), block.launchContext());
          sumE.applyPairwiseTransform(pairwise::Divide, &sum, &result);
          output->assign(&result);
        }
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
        NDArray result(output->dataType(), block.launchContext());
        double numer = static_cast<double>(numOfNonZeroWeights);
        sumE.applyScalar(scalar::Divide, numer, &result);
        output->assign(&result);
      }
      break;
    }
  }

  STORE_RESULT(*output);

  delete E;
  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(cosine_distance_loss) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(cosine_distance_loss) {
  // labels and predictions must have the same shapes
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  int dim = INT_ARG(1);
  if (dim < 0) dim += labelsShapeInfo[0];

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "COSINE_DISTANCE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // input dimension can't be larger than labels/predictions/weights rank
  REQUIRE_TRUE(dim < labelsShapeInfo[0], 0,
               "COSINE_DISTANCE_LOSS OP: input reduction dimension (got %i) must be < labels rank %i!", dim,
               labelsShapeInfo[0]);

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

  // evaluate output shapeInfo
  LongType * outShapeInfo = nullptr;
  if (INT_ARG(0) != 0)  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  else {  // in this case output has the same shape as labels reduced  by dim axis

    std::vector<LongType> dimensions = {dim};
    outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(predictionsShapeInfo), &dimensions, predictionsShapeInfo,
                                                   outType, true, false, block.getWorkspace());

    // weights array can be single scalar or has the same rank as output, and must be broadcastable to output
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(outShapeInfo), 0,
                 "COSINE_DISTANCE_LOSS OP: weights array should be scalar or have the same rank as output array, but "
                 "got %i and %i correspondingly!",
                 shape::rank(weightsShapeInfo), shape::rank(outShapeInfo));
    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(
        shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, outShapeInfo), 0,
        "COSINE_DISTANCE_LOSS OP: shapes of weights and output arrays should be broadcastable, but got weights = %s "
        "and output = %s instead!",
        ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(outShapeInfo).c_str());
  }

  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(cosine_distance_loss_grad, 3, 3, false, 0, 2) {
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

  int dim = INT_ARG(1);  // axis along which sum will be made
  if (dim < 0) dim += labels->rankOf();

  std::vector<LongType> dimensions = {dim};

  // input validation
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, "
               "but got %i instead!",
               reductionMode);
  auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(predictions->ordering(), &dimensions, predictions->shapeInfo(),
                                                       true, false, block.getWorkspace());
  // weights array can be single scalar or has the same shape as loss, and must be broadcastable to loss shape
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == shape::rank(lossShapeInfo), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss array, but "
               "got %i and %i correspondingly!",
               weights->rankOf(), shape::rank(lossShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(weights->shapeInfo(), lossShapeInfo), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got "
               "weights = %s and loss = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());
  // input dimension can't be larger than labels/predictions/weights rank
  REQUIRE_TRUE(dim < labels->rankOf(), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: input reduction dimension (got %i) must be < labels rank %i!", dim,
               labels->rankOf());

  std::vector<LongType> dims;
  dims.push_back(dim);

  // Compute element-wise product of predictions and labels into a temporary array
  NDArray predLabels(predictions->shapeInfo(), false, block.launchContext());
  predictions->applyPairwiseTransform(pairwise::Multiply, labels, &predLabels);

  // Reduce along dim (keepDims=true) — returns heap-allocated result, caller must delete
  NDArray* dotProduct = predLabels.reduceAlongDimension(reduce::Sum, &dims, true);

  // E = 1 - dotProduct  (scalar::ReverseSubtract computes scalar - array)
  NDArray* E = new NDArray(dotProduct->shapeInfo(), false, block.launchContext());
  double one = 1.0;
  dotProduct->applyScalar(scalar::ReverseSubtract, one, E);
  delete dotProduct;

  // perform weights broadcasting/tile to E if it is necessary
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(E))
    weightsBroad = new NDArray(weights->tileToShape(E->shapeInfo()));

  // dLdp = -labels, dLdl = -predictions  (negate along cosine distance dimension)
  labels->applyTransform(transform::Neg, dLdp);
  predictions->applyTransform(transform::Neg, dLdl);

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

      // dLdp *= weightsBroad
      NDArray dLdpWeighted(dLdp->shapeInfo(), false, block.launchContext());
      dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpWeighted, false);
      dLdp->assign(&dLdpWeighted);

      // dLdl *= weightsBroad
      NDArray dLdlWeighted(dLdl->shapeInfo(), false, block.launchContext());
      dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdlWeighted, false);
      dLdl->assign(&dLdlWeighted);

      if (weights->isScalar() || weights->lengthOf() == 1) {
        NDArray sumE(dLdw->dataType(), block.launchContext());
        E->reduceNumber(reduce::Sum, &sumE);
        dLdw->assign(&sumE);
      } else {
        if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E->reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else {
          dLdw->assign(E);
        }
      }

      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
               // all elements of weightsBroad array
      if (weights->isScalar()) {
        // sum = weights * E->lengthOf()
        NDArray sum(dLdp->dataType(), block.launchContext());
        double len = static_cast<double>(E->lengthOf());
        weights->applyScalar(scalar::Multiply, len, &sum);

        if (sum.e<double>(0) == 0.) {
          double zeroVal = 0.;
          dLdp->assign(zeroVal);
          dLdl->assign(zeroVal);
          dLdw->assign(zeroVal);
        } else {
          // temp = weightsBroad / sum (scalar/scalar → fill dLdp-shaped temp with that ratio)
          double wVal = weightsBroad->e<double>(0);
          double sVal = sum.e<double>(0);
          double ratio = wVal / sVal;
          NDArray temp(dLdp->shapeInfo(), false, block.launchContext());
          temp.assign(ratio);

          // dLdp *= temp
          NDArray dLdpResult(dLdp->shapeInfo(), false, block.launchContext());
          dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdpResult, false);
          dLdp->assign(&dLdpResult);

          // dLdl *= temp
          NDArray dLdlResult(dLdl->shapeInfo(), false, block.launchContext());
          dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdlResult, false);
          dLdl->assign(&dLdlResult);

          // scalar weights: dLdw = 0
          double zeroVal = 0.;
          dLdw->assign(zeroVal);
        }
      } else {
        NDArray sum(dLdw->dataType(), block.launchContext());
        weightsBroad->reduceNumber(reduce::Sum, &sum);

        if (sum.e<double>(0) == 0.) {
          double zeroVal = 0.;
          dLdp->assign(zeroVal);
          dLdl->assign(zeroVal);
          dLdw->assign(zeroVal);
        } else {
          // temp = weightsBroad / sum  (sum is scalar, weightsBroad may be full shape)
          NDArray temp(weightsBroad->shapeInfo(), false, block.launchContext());
          weightsBroad->applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sum, &temp, false);

          // dLdp *= temp
          NDArray dLdpResult(dLdp->shapeInfo(), false, block.launchContext());
          dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdpResult, false);
          dLdp->assign(&dLdpResult);

          // dLdl *= temp
          NDArray dLdlResult(dLdl->shapeInfo(), false, block.launchContext());
          dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdlResult, false);
          dLdl->assign(&dLdlResult);

          if (weights != weightsBroad) {
            std::vector<LongType> axesToReduceAlong =
                ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
            // EWeighted = E * weightsBroad
            NDArray EWeighted(E->shapeInfo(), false, block.launchContext());
            E->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);
            NDArray EWeightedSum(sum.dataType(), block.launchContext());
            EWeighted.reduceNumber(reduce::Sum, &EWeightedSum);

            // ESum = E * sum  (sum is scalar)
            NDArray ESum(E->shapeInfo(), false, block.launchContext());
            E->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &sum, &ESum, false);

            // numerator = ESum - EWeightedSum  (EWeightedSum is scalar)
            NDArray numerator(E->shapeInfo(), false, block.launchContext());
            ESum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &EWeightedSum, &numerator, false);

            // sumSquared = sum * sum (scalar * scalar)
            NDArray sumSquared(sum.dataType(), block.launchContext());
            sum.applyPairwiseTransform(pairwise::Multiply, &sum, &sumSquared);

            // gradTemp = numerator / sumSquared  (sumSquared is scalar)
            NDArray gradTemp(E->shapeInfo(), false, block.launchContext());
            numerator.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sumSquared, &gradTemp, false);

            gradTemp.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          } else {
            // EWeighted = E * weightsBroad
            NDArray EWeighted(E->shapeInfo(), false, block.launchContext());
            E->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);
            NDArray EWeightedSum(sum.dataType(), block.launchContext());
            EWeighted.reduceNumber(reduce::Sum, &EWeightedSum);

            // ESum = E * sum  (sum is scalar)
            NDArray ESum(E->shapeInfo(), false, block.launchContext());
            E->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &sum, &ESum, false);

            // numerator = ESum - EWeightedSum  (EWeightedSum is scalar)
            NDArray numerator(E->shapeInfo(), false, block.launchContext());
            ESum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &EWeightedSum, &numerator, false);

            // sumSquared = sum * sum (scalar * scalar)
            NDArray sumSquared(sum.dataType(), block.launchContext());
            sum.applyPairwiseTransform(pairwise::Multiply, &sum, &sumSquared);

            // gradTemp = numerator / sumSquared  (sumSquared is scalar)
            NDArray gradTemp(E->shapeInfo(), false, block.launchContext());
            numerator.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sumSquared, &gradTemp, false);

            dLdw->assign(&gradTemp);
          }
        }
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
               // array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E->lengthOf();
      } else {
        NDArray countResult(DataType::INT64, block.launchContext());
        weightsBroad->reduceNumber(reduce::CountNonZero, &countResult);
        numOfNonZeroWeights = countResult.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zeroVal = 0.;
        dLdp->assign(zeroVal);
        dLdl->assign(zeroVal);
        dLdw->assign(zeroVal);
      } else {
        double numer = static_cast<double>(numOfNonZeroWeights);

        // temp = weightsBroad / numOfNonZeroWeights  (weightsBroad shape, may be scalar)
        NDArray temp(weightsBroad->shapeInfo(), false, block.launchContext());
        weightsBroad->applyScalar(scalar::Divide, numer, &temp);

        // dLdp *= temp  (broadcast if temp is scalar)
        NDArray dLdpResult(dLdp->shapeInfo(), false, block.launchContext());
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdpResult, false);
        dLdp->assign(&dLdpResult);

        // dLdl *= temp  (broadcast if temp is scalar)
        NDArray dLdlResult(dLdl->shapeInfo(), false, block.launchContext());
        dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdlResult, false);
        dLdl->assign(&dLdlResult);

        if (weights->isScalar() || weights->lengthOf() == 1) {
          NDArray sumE(dLdw->dataType(), block.launchContext());
          E->reduceNumber(reduce::Sum, &sumE);
          NDArray result(dLdw->dataType(), block.launchContext());
          sumE.applyScalar(scalar::Divide, numer, &result);
          dLdw->assign(&result);
        } else {
          if (weights != weightsBroad) {
            std::vector<LongType> axesToReduceAlong =
                ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
            E->reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
            NDArray dLdwResult(dLdw->shapeInfo(), false, block.launchContext());
            dLdw->applyScalar(scalar::Divide, numer, &dLdwResult);
            dLdw->assign(&dLdwResult);
          } else {
            NDArray result(E->shapeInfo(), false, block.launchContext());
            E->applyScalar(scalar::Divide, numer, &result);
            dLdw->assign(&result);
          }
        }
      }
      break;
    }
  }

  delete E;
  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(cosine_distance_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(cosine_distance_loss_grad) {
  /// labels and predictions must have the same shapes
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  int dim = INT_ARG(1);
  if (dim < 0) dim += labelsShapeInfo[0];

  std::vector<LongType> dimensions = {dim};

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(predictionsShapeInfo), &dimensions,
                                                       predictionsShapeInfo, true, false, block.getWorkspace());
  // weights array can be single scalar or has the same rank as loss, and must be broadcastable to loss
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(lossShapeInfo), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss array, but "
               "got %i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(lossShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, lossShapeInfo),
               0,
               "COSINE_DISTANCE_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got "
               "weights = %s and loss = %s instead!",
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());
  // input dimension can't be larger than labels/predictions/weights rank
  REQUIRE_TRUE(dim < labelsShapeInfo[0], 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: input reduction dimension (got %i) must be < labels rank %i!", dim,
               labelsShapeInfo[0]);

  auto outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

  auto dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(predictionsShapeInfo, outType, false, block.getWorkspace());
  auto dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());
  auto dLdlShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

  return SHAPELIST(CONSTANT(dLdpShapeInfo), CONSTANT(dLdwShapeInfo), CONSTANT(dLdlShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
