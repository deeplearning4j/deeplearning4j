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
#if NOT_EXCLUDED(OP_hinge_loss)

#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/loss.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(hinge_loss, 3, 1, false, 0, 1) {
  auto logits = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"

  // input validation
  REQUIRE_TRUE(labels->isSameShape(logits), 0,
               "HINGE_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "HINGE_LOSS OP: weights array should be scalar or have the same rank as labels array, but "
               "got %i and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "HINGE_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got "
               "weights = %s and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "HINGE_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, "
      "but got %i instead!",
      reductionMode);

  // perform weights broadcasting/tile to logits if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(logits))
    weightsBroad = new NDArray(weights->tileToShape(logits->shapeInfo()));

  auto ctx = block.launchContext();

  // labelsScaled = labels * 2
  NDArray labelsScaled(labels->shapeInfo(), false, ctx);
  labels->applyScalar(scalar::Multiply, 2.0, &labelsScaled);

  // labelsTransformed = labelsScaled - 1  (i.e. 2*labels - 1, maps {0,1} -> {-1,1})
  NDArray labelsTransformed(labels->shapeInfo(), false, ctx);
  labelsScaled.applyScalar(scalar::Subtract, 1.0, &labelsTransformed);

  // logitsScaled = labelsTransformed * logits
  NDArray logitsScaled(logits->shapeInfo(), false, ctx);
  labelsTransformed.applyPairwiseTransform(pairwise::Multiply, logits, &logitsScaled);

  // E = 1 - logitsScaled, then RELU(E)
  NDArray E(logits->shapeInfo(), false, ctx);
  logitsScaled.applyScalar(scalar::ReverseSubtract, 1.0, &E);
  E.applyScalar(scalar::RELU, 0.0, &E);

  // EWeighted = E * weightsBroad
  NDArray EWeighted(logits->shapeInfo(), false, ctx);
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
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum / sum_of_weights
      NDArray sum(output->dataType(), ctx);
      if (weights->isScalar()) {
        double wVal = weights->e<double>(0);
        double len = static_cast<double>(EWeighted.lengthOf());
        double prod = wVal * len;
        sum.assign(prod);
      } else {
        weightsBroad->reduceNumber(reduce::Sum, &sum);
      }

      if (sum.e<double>(0) == 0.) {
        double zero = 0.;
        output->assign(zero);
      } else {
        NDArray sumE(output->dataType(), ctx);
        EWeighted.reduceNumber(reduce::Sum, &sumE);
        // result = sumE / sum
        sumE.applyPairwiseTransform(pairwise::Divide, &sum, output);
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights"
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = EWeighted.lengthOf();
      } else {
        NDArray countScalar(DataType::INT64, ctx);
        weightsBroad->reduceNumber(reduce::CountNonZero, &countScalar);
        numOfNonZeroWeights = countScalar.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.;
        output->assign(zero);
      } else {
        NDArray sumE(output->dataType(), ctx);
        EWeighted.reduceNumber(reduce::Sum, &sumE);
        sumE.applyScalar(scalar::Divide, static_cast<double>(numOfNonZeroWeights), output);
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(hinge_loss) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(hinge_loss) {
  auto logitsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(
      shape::shapeEquals(labelsShapeInfo, logitsShapeInfo), 0,
      "HINGE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "HINGE_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i "
               "correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "HINGE_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = "
      "%s instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));
  LongType  *outShapeInfo = nullptr;

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
CUSTOM_OP_IMPL(hinge_loss_grad, 3, 3, false, 0, 1) {
  auto logits = INPUT_VARIABLE(0);
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
  REQUIRE_TRUE(
      labels->isSameShape(logits), 0,
      "HINGE_LOSS_GRAD OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "HINGE_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i "
               "and %i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "HINGE_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
               "and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "HINGE_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!",
      reductionMode);

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(logits))
    weightsBroad = new NDArray(weights->tileToShape(logits->shapeInfo()));

  auto ctx = block.launchContext();

  // labelsScaled = labels * 2
  NDArray labelsScaled(labels->shapeInfo(), false, ctx);
  labels->applyScalar(scalar::Multiply, 2.0, &labelsScaled);

  // z = labelsScaled - 1  (maps {0,1} -> {-1,1})
  NDArray z(labels->shapeInfo(), false, ctx);
  labelsScaled.applyScalar(scalar::Subtract, 1.0, &z);

  // logitsScaled = z * logits
  NDArray logitsScaled(logits->shapeInfo(), false, ctx);
  z.applyPairwiseTransform(pairwise::Multiply, logits, &logitsScaled);

  // E = 1 - logitsScaled, then RELU(E)
  NDArray E(logits->shapeInfo(), false, ctx);
  logitsScaled.applyScalar(scalar::ReverseSubtract, 1.0, &E);
  E.applyScalar(scalar::RELU, 0.0, &E);

  // gradient mask: sign(E)  — nonzero where hinge loss is active
  NDArray gradientMask(E.shapeInfo(), false, ctx);
  E.applyTransform(transform::Sign, &gradientMask);

  // dLdp = -z * gradientMask
  NDArray negZ(z.shapeInfo(), false, ctx);
  z.applyTransform(transform::Neg, &negZ);
  negZ.applyPairwiseTransform(pairwise::Multiply, &gradientMask, dLdp);

  // dLdl = -(logits * 2) * gradientMask = -logits*2 * gradientMask
  NDArray logitsTimes2(logits->shapeInfo(), false, ctx);
  logits->applyScalar(scalar::Multiply, 2.0, &logitsTimes2);
  NDArray dLdlTemp(logits->shapeInfo(), false, ctx);
  logitsTimes2.applyPairwiseTransform(pairwise::Multiply, &gradientMask, &dLdlTemp);
  dLdlTemp.applyTransform(transform::Neg, dLdl);

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum"

      // dLdp *= weightsBroad
      NDArray dLdpWeighted(dLdp->shapeInfo(), false, ctx);
      dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpWeighted, false);
      dLdp->assign(&dLdpWeighted);

      // dLdl *= weightsBroad
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

      NDArray sum(dLdw->dataType(), ctx);
      if (weights->isScalar()) {
        double wVal = weights->e<double>(0);
        double len = static_cast<double>(E.lengthOf());
        double prod = wVal * len;
        sum.assign(prod);
      } else {
        weightsBroad->reduceNumber(reduce::Sum, &sum);
      }

      if (sum.e<double>(0) == 0.) {
        double zero = 0.;
        dLdp->assign(zero);
        dLdl->assign(zero);
        dLdw->assign(zero);
      } else {
        // weightsDivSum = weightsBroad / sum  (sum is scalar)
        NDArray weightsDivSum(weightsBroad->shapeInfo(), false, ctx);
        weightsBroad->applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sum, &weightsDivSum, false);

        // dLdp = dLdp * weightsDivSum
        NDArray dLdpResult(dLdp->shapeInfo(), false, ctx);
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivSum, &dLdpResult, false);
        dLdp->assign(&dLdpResult);

        // dLdl = dLdl * weightsDivSum
        NDArray dLdlResult(dLdl->shapeInfo(), false, ctx);
        dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsDivSum, &dLdlResult, false);
        dLdl->assign(&dLdlResult);

        if (weights->isScalar()) {
          double zero = 0.;
          dLdw->assign(zero);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());

          // EWeighted = E * weightsBroad
          NDArray EWeighted(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);

          NDArray EWeightedSum(dLdw->dataType(), ctx);
          EWeighted.reduceNumber(reduce::Sum, &EWeightedSum);

          // ESum = E * sum  (sum is scalar)
          NDArray ESum(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &sum, &ESum, false);

          // numerator = ESum - EWeightedSum  (EWeightedSum is scalar)
          NDArray numerator(E.shapeInfo(), false, ctx);
          ESum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &EWeightedSum, &numerator, false);

          // sumSquared = sum * sum (scalar * scalar)
          NDArray sumSquared(sum.dataType(), ctx);
          sum.applyPairwiseTransform(pairwise::Multiply, &sum, &sumSquared);

          // gradTemp = numerator / sumSquared  (sumSquared is scalar)
          NDArray gradTemp(E.shapeInfo(), false, ctx);
          numerator.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sumSquared, &gradTemp, false);

          gradTemp.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else {
          // EWeighted = E * weightsBroad
          NDArray EWeighted(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);

          NDArray EWeightedSum(dLdw->dataType(), ctx);
          EWeighted.reduceNumber(reduce::Sum, &EWeightedSum);

          // ESum = E * sum  (sum is scalar)
          NDArray ESum(E.shapeInfo(), false, ctx);
          E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &sum, &ESum, false);

          // numerator = ESum - EWeightedSum  (EWeightedSum is scalar)
          NDArray numerator(E.shapeInfo(), false, ctx);
          ESum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &EWeightedSum, &numerator, false);

          // sumSquared = sum * sum (scalar * scalar)
          NDArray sumSquared(sum.dataType(), ctx);
          sum.applyPairwiseTransform(pairwise::Multiply, &sum, &sumSquared);

          // gradTemp = numerator / sumSquared  (sumSquared is scalar)
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
        NDArray countScalar(DataType::INT64, ctx);
        weightsBroad->reduceNumber(reduce::CountNonZero, &countScalar);
        numOfNonZeroWeights = countScalar.e<LongType>(0);
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.;
        dLdp->assign(zero);
        dLdl->assign(zero);
        dLdw->assign(zero);
      } else {
        double nnzD = static_cast<double>(numOfNonZeroWeights);

        if (weights->isScalar()) {
          NDArray sumE(dLdw->dataType(), ctx);
          E.reduceNumber(reduce::Sum, &sumE);
          sumE.applyScalar(scalar::Divide, nnzD, dLdw);
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          NDArray dLdwScaled(dLdw->shapeInfo(), false, ctx);
          dLdw->applyScalar(scalar::Divide, nnzD, &dLdwScaled);
          dLdw->assign(&dLdwScaled);
        } else {
          E.applyScalar(scalar::Divide, nnzD, dLdw);
        }

        // temp = weightsBroad / numOfNonZeroWeights
        NDArray temp(weightsBroad->shapeInfo(), false, ctx);
        weightsBroad->applyScalar(scalar::Divide, nnzD, &temp);

        // dLdp *= temp
        NDArray dLdpResult(dLdp->shapeInfo(), false, ctx);
        dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdpResult, false);
        dLdp->assign(&dLdpResult);

        // dLdl *= temp
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

DECLARE_TYPES(hinge_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(hinge_loss_grad) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "HINGE_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "HINGE_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i "
               "and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "HINGE_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
      "labels = %s instead!",
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
